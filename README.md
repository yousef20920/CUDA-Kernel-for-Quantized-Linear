# CUDA Kernel for Quantized Linear (Weight-only INT8)

A Tesla-aligned mini project focused on **inference performance**: implement a **quantized Linear layer** (INT8 weights + per-channel scales) and a **custom CUDA kernel** optimized for **batch=1** (matvec) latency. You’ll ship a clean CLI, correctness checks, and benchmarks.

This README is written so you can implement everything in **VS Code + CMake**.

---

## 0) What you’re building

A small library + CLI that computes:

[
y = xW + b
]

Where:

* `x` is FP16 or FP32 activations (shape: `[1, K]`)
* `W` is **INT8** weights (shape: `[K, N]`)
* `b` is FP32 bias (shape: `[N]`) (optional)
* Output `y` is FP16 or FP32 (shape: `[1, N]`)

**Quantization scheme (weight-only INT8):**

* Quantize per output channel (per column of `W`) with scale `s[j]`.
* Store `Wq[k, j]` as int8.
* Compute output as:

[
y[j] = (\sum_k x[k] \cdot Wq[k,j]) \cdot s[j] + b[j]
]

This is very relevant to LLM inference because Linear layers dominate runtime and **batch=1** is common during decoding.

---

## 1) Requirements

### Hardware

* **NVIDIA GPU** with CUDA support

### Software

* C++17 or newer
* **CUDA Toolkit** (nvcc)
* CMake >= 3.20
* VS Code extensions:

  * C/C++ (ms-vscode.cpptools)
  * CMake Tools (ms-vscode.cmake-tools)

### OS notes

* **Windows**: easiest with Visual Studio Build Tools + CUDA installed
* **Linux**: works out of the box with gcc/clang + CUDA
* **macOS**: NVIDIA CUDA is typically not supported on modern Macs. Use a Linux/Windows machine, a desktop with an NVIDIA GPU, or WSL2 with GPU support.

---

## 2) Deliverables (what makes this resume-worthy)

You will ship:

1. **CPU reference implementation** (correctness baseline)
2. **INT8 quantizer** (per-channel scale) + dequant math
3. **CUDA kernel** for quantized matvec (fast path)
4. **Benchmark harness** reporting latency and speedup
5. **Correctness tests** (max error vs FP32)
6. (Optional) a simple profiling mode and kernel configuration sweeps

---

## 3) Repo layout

```
qlinear-cuda/
  CMakeLists.txt
  README.md
  .vscode/
    settings.json
    launch.json
  include/
    qlinear/
      tensor.h
      quantize.h
      cpu_ref.h
      cuda_kernels.h
      benchmark.h
  src/
    main.cpp
    tensor.cpp
    quantize.cpp
    cpu_ref.cpp
    benchmark.cpp
    cuda/
      qlinear_int8_matvec.cu
      cuda_kernels.cpp
  tests/
    test_correctness.cpp
  scripts/
    run_bench.sh (optional)
  results/
    results.md
```

---

## 4) Step-by-step implementation plan

### Milestone A — Minimal CPU baseline (Day 1–2)

Implement:

* A lightweight `Tensor` wrapper for host buffers
* FP32 reference Linear:

  * `y[j] = sum_k x[k] * W_fp32[k,j] + b[j]`

**Exit criteria:**

* Running `./qlinear --mode cpu_fp32` prints output stats and passes basic checks.

---

### Milestone B — Weight-only INT8 quantization (Day 3–5)

Add:

* `quantize_per_channel(W_fp32[K,N]) -> (W_int8[K,N], scales[N])`

Suggested quantization:

* For each output channel `j`:

  * `max_abs = max_k |W[k,j]|`
  * `scale[j] = max_abs / 127.0f` (handle max_abs=0)
  * `Wq[k,j] = round(W[k,j] / scale[j])` clamped to [-127,127]

Then implement a CPU “quantized” path:

* Accumulate in float:

  * `acc = sum_k x[k] * (float)Wq[k,j]`
  * `y[j] = acc * scale[j] + b[j]`

**Exit criteria:**

* `./qlinear --mode cpu_int8` runs
* Max absolute error vs FP32 is printed and is reasonable (e.g., < 1e-2 to 1e-1 depending on sizes)

---

### Milestone C — CUDA kernel (Week 2)

Implement a custom GPU kernel for batch=1:

**Kernel goal:** compute all `N` outputs in parallel.

Recommended starting strategy:

* 1 block computes a tile of output channels `j`
* Threads in the block iterate over `k` dimension and reduce partial sums
* Use a reduction (shared memory or warp-level) to finalize each output

**Inputs on GPU:**

* `x` (FP16 or FP32)
* `Wq` (INT8)
* `scales` (FP32)
* `bias` (FP32, optional)

**Output:**

* `y` (FP16 or FP32)

**Exit criteria:**

* `./qlinear --mode gpu_int8` runs
* Correctness: compare GPU output vs CPU FP32 baseline (print max error)

---

### Milestone D — Benchmarking + tuning (Week 3)

Add a benchmark harness that:

* Runs warmup iterations
* Runs `iters` iterations and reports:

  * average latency (ms)
  * p50/p95 (optional)
  * speedup vs CPU

Also add a kernel config sweep mode:

* try different block sizes (e.g., 128/256/512 threads)
* pick the best for your GPU

**Exit criteria:**

* `./qlinear --bench --sizes 512x2048,1024x4096 --iters 200` produces a nice table

---

## 5) Technical design

### 5.1 Data layout

Use a simple contiguous layout:

* `x`: shape `[K]` (batch=1)
* `W`: shape `[K, N]`

Pick one and be consistent:

* **Column-major for W (recommended for per-output-channel quantization)**

  * Store `Wq[k,j]` so that `j` is contiguous (better when each thread computes a fixed `j`)

If you store `W` as `[N, K]` (transposed), then each output channel is a contiguous row.

**Recommendation:** store as `Wq[j*K + k]` (shape `[N, K]`).

* Then output `j` reads a contiguous chunk of `K` weights.

This will make your kernel simpler and faster.

---

### 5.2 Kernel approach (simple + effective)

**Batch=1 matvec:**

* Each output `j` is an independent dot product of length `K`.

A straightforward CUDA mapping:

* 1 block handles a group of `j` values
* Each warp computes one `j` (warp-per-output)

  * lane `t` handles `k = t, t+32, t+64, ...`
  * reduce within warp

This is a good first kernel because:

* It’s simple
* Uses warp reductions
* Works well for K in the 512–8192 range

Later optimization ideas:

* load `x` into shared memory
* vectorized loads
* use `__half` for x and output

---

### 5.3 Correctness metrics

Report:

* `max_abs_error = max_j |y_gpu[j] - y_fp32[j]|`
* `mean_abs_error` (optional)

For weight-only INT8, it’s normal that outputs differ slightly.

---

## 6) CLI spec

Your `main.cpp` should support:

* `--mode cpu_fp32 | cpu_int8 | gpu_int8`
* `--K <int>` input dimension
* `--N <int>` output dimension
* `--dtype fp32 | fp16` (for x and output; weights are int8)
* `--iters <int>` benchmark iterations
* `--warmup <int>` warmup iterations
* `--seed <int>`
* `--bias 0|1`
* `--print 0|1` print first few outputs
* `--sweep 0|1` sweep block sizes

Examples:

```bash
./build/qlinear --mode cpu_fp32 --K 1024 --N 4096 --bias 1
./build/qlinear --mode cpu_int8 --K 1024 --N 4096 --bias 1
./build/qlinear --mode gpu_int8 --K 1024 --N 4096 --bias 1 --bench --iters 200 --warmup 50
```

---

## 7) Build instructions (VS Code + CMake)

### 7.1 CMakeLists.txt (skeleton)

Your CMake should:

* build a C++ library
* compile `.cu` with CUDA
* link into a CLI

Key CMake settings:

* `enable_language(CUDA)`
* set `CMAKE_CUDA_STANDARD 17`
* set architecture or let CUDA choose default

**Tip:** start with default arch, then optionally set `CMAKE_CUDA_ARCHITECTURES` later.

---

### 7.2 VS Code `.vscode/settings.json`

```json
{
  "cmake.sourceDirectory": "${workspaceFolder}",
  "cmake.buildDirectory": "${workspaceFolder}/build",
  "cmake.configureOnOpen": true,
  "C_Cpp.default.cppStandard": "c++17"
}
```

---

## 8) Benchmarks you should run

Use these common inference-ish sizes:

### Decoding-like (batch=1)

* `K=512, N=2048`
* `K=1024, N=4096`
* `K=2048, N=8192` (if your GPU can handle it)

Benchmark table columns:

* Mode (cpu_fp32 / cpu_int8 / gpu_int8)
* K, N
* Latency (ms)
* Speedup vs cpu_fp32
* Max error vs cpu_fp32

Store in `results/results.md`.

---

## 9) Testing checklist

Minimum tests (put in `tests/test_correctness.cpp`):

* For a few sizes (e.g., 128x256, 512x2048):

  * Generate random `x`, `W`, `b`
  * Compute FP32 reference
  * Compute CPU INT8
  * Compute GPU INT8
  * Assert:

    * shapes match
    * max_abs_error < threshold (choose a reasonable threshold; start loose then tighten)

Also test corner cases:

* bias off
* all-zero weights
* small sizes (K=32, N=32)

---

## 10) Performance tuning ideas (optional but great)

Pick 1–2 to keep it manageable:

1. **Cache x in shared memory**

* `x[k]` is reused for every output `j`

2. **Use FP16 x/output**

* Keep accumulation in float

3. **Kernel config sweep**

* try block sizes and mapping strategies

4. **Memory layout experiment**

* compare `[N,K]` vs `[K,N]` storage

5. **Pinned host memory** (if you measure transfer time)

---

## 11) Common pitfalls

* **Wrong memory layout**: decide early if `Wq` is `[N,K]` or `[K,N]` and stick to it.
* **Overflow/precision**: accumulate into `float` (or `int32` then convert) to avoid issues.
* **Not warming up**: always warm up before timing kernels.
* **Timing GPU incorrectly**: use CUDA events for kernel timing.

---

## 12) “Definition of Done” (what to ship)

✅ `cpu_fp32` baseline works
✅ `cpu_int8` quantized path works
✅ `gpu_int8` CUDA kernel works
✅ Benchmarks show speedup for realistic sizes
✅ Correctness checks + tests included
✅ README includes results + how to run

---

## 13) Resume bullets (fill in your numbers)

**Quantized Linear CUDA Kernel (C++/CUDA)**

* Implemented weight-only INT8 quantization (per-channel scaling) for linear layers, reducing weight memory by ~4×.
* Wrote a custom CUDA kernel for INT8 matvec inference optimized for batch-1 latency, achieving **X× speedup** over CPU FP32 baseline.
* Built a benchmarking and correctness harness reporting latency (ms), throughput, and max error vs FP32 reference.

---

## 14) Suggested weekly timeline

* **Week 1:** CPU FP32 + CPU INT8 + correctness printing
* **Week 2:** GPU kernel + correctness checks
* **Week 3:** Bench harness + tuning + results writeup
* **Week 4 (optional):** FP16 path + shared memory optimization

---

## 15) Next steps (start here)

1. Create the folder layout
2. Implement CPU FP32 reference
3. Implement per-channel quantization + CPU INT8 path
4. Add a tiny benchmark loop

When you’re ready, implement the CUDA kernel last.

If you tell me your GPU model (or just “laptop RTX / desktop GTX”), I can suggest the best default kernel mapping (warp-per-output vs block-per-output) and initial block sizes to try.
