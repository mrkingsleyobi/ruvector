# Optimization Guide: Sublinear-Time Solver Integration

**Date**: 2026-02-20
**Classification**: Engineering Reference
**Scope**: Performance optimization strategies for solver integration

---

## 1. Executive Summary

This guide provides concrete optimization strategies for achieving maximum performance from the sublinear-time-solver integration into RuVector. Targets: 10-600x speedups across 6 critical subsystems while maintaining <2% accuracy loss. Organized by optimization tier: SIMD → Memory → Algorithm → Concurrency → Compilation → Platform.

---

## 2. SIMD Optimization Strategy

### 2.1 Architecture-Specific Kernels

The solver's hot path is SpMV (sparse matrix-vector multiply). Each architecture requires a dedicated kernel:

| Architecture | SIMD Width | f32/iteration | Key Instruction | Expected SpMV Throughput |
|-------------|-----------|--------------|-----------------|-------------------------|
| AVX-512 | 512-bit | 16 | `_mm512_i32gather_ps` | ~400M nonzeros/s |
| AVX2+FMA | 256-bit | 8×4 unrolled | `_mm256_i32gather_ps` + `_mm256_fmadd_ps` | ~250M nonzeros/s |
| NEON | 128-bit | 4×4 unrolled | Manual gather + `vfmaq_f32` | ~150M nonzeros/s |
| WASM SIMD128 | 128-bit | 4 | `f32x4_mul` + `f32x4_add` | ~80M nonzeros/s |
| Scalar | 32-bit | 1 | `fmaf` | ~40M nonzeros/s |

### 2.2 New SIMD Kernels Required

**SpMV with gather** (primary bottleneck):
```
// Pseudocode for AVX2+FMA SpMV row accumulation
for each row i:
    acc = _mm256_setzero_ps()
    for j in row_ptrs[i]..row_ptrs[i+1] step 8:
        indices = _mm256_loadu_si256(&col_indices[j])
        vals = _mm256_loadu_ps(&values[j])
        x_gathered = _mm256_i32gather_ps(x_ptr, indices, 4)
        acc = _mm256_fmadd_ps(vals, x_gathered, acc)
    y[i] = horizontal_sum(acc) + scalar_remainder
```

**Vectorized PRNG** (for Hybrid Random Walk):
```
// 4 independent xoshiro256** streams for NEON
state[4][4] = initialize_from_seed()
for each walk:
    random = xoshiro256_simd(state)  // 4 random values per call
    next_node = random % degree[current_node]
```

**SIMD reductions** (convergence checks):
```
// Max reduction for residual norm check
max_residual = horizontal_max(_mm256_abs_ps(residual_vec))
```

### 2.3 Auto-Vectorization Guidelines

For code that doesn't warrant hand-written intrinsics:

1. **Sequential access**: Iterate arrays in order (no random access in inner loop)
2. **No branches**: Use `select`/`blend` instead of `if` in hot loops
3. **Independent accumulators**: 4 separate sums, combine at end
4. **Aligned data**: Use `#[repr(align(64))]` on hot data structures
5. **Known bounds**: Use `get_unchecked()` after external bounds check

---

## 3. Memory Optimization

### 3.1 Cache-Aware Tiling

| Working Set | Cache Level | Performance | Strategy |
|------------|------------|-------------|---------|
| < 48 KB | L1 (M4 Pro: 192KB/perf) | Peak (100%) | Direct iteration, no tiling |
| < 256 KB | L2 | 80-90% of peak | Single-pass with prefetch |
| < 16 MB | L3 | 50-70% of peak | Row-block tiling |
| > 16 MB | DRAM | 20-40% of peak | Page-level tiling + prefetch |
| > available RAM | Disk | 1-5% of peak | Memory-mapped streaming |

**Tiling formula**: `TILE_ROWS = L3_SIZE / (avg_row_nnz × 12 bytes)`

For L3=16MB, avg_row_nnz=100: TILE_ROWS = 16M / 1200 ≈ 13,000 rows per tile.

### 3.2 Arena Allocator Integration

Per-solve arena eliminates malloc overhead:

```rust
// Before: ~20μs overhead per solve from allocation
let r = vec![0.0f32; n];     // malloc
let p = vec![0.0f32; n];     // malloc
let ap = vec![0.0f32; n];    // malloc
// ... solve ...
// implicit drops: 3 × free

// After: ~0.2μs overhead per solve
let mut arena = SolverArena::with_capacity(n * 12);  // One malloc
let r = arena.alloc_slice::<f32>(n);
let p = arena.alloc_slice::<f32>(n);
let ap = arena.alloc_slice::<f32>(n);
// ... solve ...
arena.reset();  // One reset (no free)
```

### 3.3 Memory-Mapped Large Matrices

For matrices > 100MB, use OS paging:

```rust
let mmap = unsafe { memmap2::Mmap::map(&file)? };
let values: &[f32] = bytemuck::cast_slice(&mmap[header_size..]);
// OS handles page faults, LRU eviction
```

### 3.4 Zero-Copy Data Paths

| Path | Mechanism | Overhead |
|------|-----------|----------|
| SoA → Solver | `&[f32]` borrow | 0 bytes |
| HNSW → CSR | Direct construction | O(n×M) one-time |
| Solver → WASM | `Float32Array::view()` | 0 bytes (shared linear memory) |
| Solver → NAPI | `napi::Buffer` | 0 bytes (shared heap) |
| Solver → REST | `serde_json::to_writer` | 1 serialization |

---

## 4. Algorithmic Optimization

### 4.1 Preconditioning Strategies

| Preconditioner | Setup Cost | Per-Iteration Cost | Condition Improvement | Best For |
|---------------|-----------|-------------------|----------------------|----------|
| None | 0 | 0 | 1x | Well-conditioned (κ < 10) |
| Diagonal (Jacobi) | O(n) | O(n) | √(d_max/d_min) | General SPD |
| Incomplete Cholesky | O(nnz) | O(nnz) | 10-100x | Moderately ill-conditioned |
| Algebraic Multigrid | O(nnz·log n) | O(nnz) | Near-optimal for Laplacians | κ > 100 |

**Recommendation**: Default to diagonal preconditioner (O(n) overhead, always helps). Escalate to AMG only when κ > 100 and n > 50K.

### 4.2 Sparsity Exploitation

Auto-detect and exploit sparsity at runtime:

```rust
fn select_path(matrix: &CsrMatrix<f32>) -> ComputePath {
    let density = matrix.density();
    if density > 0.50 { ComputePath::Dense }       // Use BLAS
    else if density > 0.05 { ComputePath::Sparse }  // CSR SpMV
    else { ComputePath::Sublinear }                  // Solver algorithms
}
```

### 4.3 Batch Amortization

TRUE preprocessing amortized over B solves:

| Preprocessing Cost | Per-Solve Cost | Break-Even B |
|-------------------|---------------|-------------|
| 425 ms (n=100K, 1%) | 0.43 ms (ε=0.1) | 634 solves |
| 42 ms (n=10K, 1%) | 0.04 ms (ε=0.1) | 63 solves |
| 4 ms (n=1K, 1%) | 0.004 ms (ε=0.1) | 6 solves |

**Rule**: Amortize TRUE preprocessing when B > preprocessing_ms / cg_solve_ms.

### 4.4 Lazy Evaluation

For single-entry queries, compute only needed entries:

```rust
// Full solve: compute all n entries
let x = solver.solve(A, b)?;  // O(nnz × iterations)

// Lazy: compute only entry (i, j)
let x_ij = solver.estimate_entry(A, i, j)?;  // O(√n / ε) via random walk
```

Speedup: n / √n = √n. For n=1M: 1000x speedup for single-entry queries.

---

## 5. Concurrency Optimization

### 5.1 Rayon Tuning

```rust
// Optimal chunk size: balance parallelism overhead vs work per chunk
let chunk_size = (n / rayon::current_num_threads()).max(1024);

problems.par_chunks(chunk_size)
    .map(|chunk| chunk.iter().map(|p| solve_single(p)).collect::<Vec<_>>())
    .flatten()
    .collect()
```

### 5.2 Thread Scaling Expectations

| Threads | Efficiency | Bottleneck |
|---------|-----------|-----------|
| 1 | 100% | N/A |
| 2 | 90-95% | Rayon overhead (~500ns/task) |
| 4 | 75-85% | Memory bandwidth |
| 8 | 55-70% | L3 cache contention |
| 16 | 40-55% | NUMA effects |

**Recommendation**: Use `num_cpus::get_physical()` threads (not logical/hyperthreaded).

### 5.3 Avoid Nested Parallelism

```rust
// BAD: Rayon inside Rayon = thread pool exhaustion
problems.par_iter().map(|p| {
    p.data.par_iter()...  // Nested Rayon → deadlock risk
});

// GOOD: Outer parallel, inner SIMD
problems.par_iter().map(|p| {
    spmv_simd(&p.matrix, &p.x, &mut p.y)  // Inner: SIMD, single thread
});
```

---

## 6. Compilation Optimization

### 6.1 Profile-Guided Optimization (PGO)

```bash
# Step 1: Build instrumented binary
RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data" cargo build --release -p ruvector-solver

# Step 2: Run representative workload
./target/release/bench_solver --profile-workload

# Step 3: Merge profiles
llvm-profdata merge -o /tmp/pgo-data/merged.profdata /tmp/pgo-data/*.profraw

# Step 4: Build optimized binary
RUSTFLAGS="-Cprofile-use=/tmp/pgo-data/merged.profdata" cargo build --release -p ruvector-solver
```

Expected improvement: 5-15% for SpMV-heavy workloads (better branch prediction, improved inlining decisions).

### 6.2 Link-Time Optimization

Already configured in Cargo.toml:

```toml
[profile.release]
opt-level = 3
lto = "fat"        # Cross-crate inlining (critical for nalgebra → solver)
codegen-units = 1  # Maximum optimization scope
strip = true       # Reduce binary size
```

### 6.3 WASM Optimization

```bash
# Build with size optimization
RUSTFLAGS="-C opt-level=s -C target-feature=+simd128" wasm-pack build --release

# Post-build optimization
wasm-opt -O3 --enable-simd pkg/solver_bg.wasm -o pkg/solver_bg.wasm
```

Expected: 10-20% size reduction, 5-10% speed improvement from wasm-opt.

---

## 7. Platform-Specific Optimization

### 7.1 Server (Linux x86_64)

- **Huge pages**: `madvise(addr, len, MADV_HUGEPAGE)` for large matrix allocations (reduces TLB misses by 10-30%)
- **NUMA-aware**: Pin solver threads to same NUMA node as matrix memory
- **CPU affinity**: `taskset -c 0-7` for dedicated solver cores
- **io_uring**: For memory-mapped matrix I/O (reduces syscall overhead)
- **AVX-512**: Prefer when available (Zen 4, Ice Lake+). Check `is_x86_feature_detected!("avx512f")`

### 7.2 Apple Silicon (macOS ARM64)

- **Unified memory**: No NUMA concerns, matrix + solver share same memory pool
- **NEON**: 4x unrolled with independent accumulators for 6-wide pipeline
- **AMX**: Apple Matrix coprocessor for dense operations (via Accelerate framework, not directly accessible from Rust yet)
- **M4 Pro specifics**: 192KB L1, 16MB L2, 48MB L3 — adjust tiling accordingly

### 7.3 Browser (WASM)

- **Memory budget**: Keep total solver allocation < 8MB
- **Web Workers**: 4 workers for batch operations, SharedArrayBuffer for zero-copy
- **SIMD128**: Always enable with `-C target-feature=+simd128` (universal support since 2021)
- **Streaming**: For large problems, stream results via ReadableStream
- **IndexedDB**: Cache TRUE preprocessing results for repeat queries

### 7.4 Cloudflare Workers (Edge WASM)

- **128MB memory**: Larger than browser, can handle n up to ~500K
- **50ms CPU limit**: Use Reflex/Retrieval lanes only; Heavy lane exceeds limit
- **No Web Workers**: Single-threaded, no parallelism
- **Cold start**: Minimize WASM initialization; pre-warm with small solve

---

## 8. Optimization Checklist

### P0 (Critical — Implement First)

- [ ] SIMD SpMV kernels for AVX2+FMA and NEON
- [ ] Arena allocator for solver temporaries
- [ ] Zero-copy data path from SoA storage to solver
- [ ] CSR matrix format with aligned storage
- [ ] Diagonal preconditioning for CG
- [ ] Feature-gated Rayon parallelism (disabled on WASM)
- [ ] Input validation at system boundaries
- [ ] Regression benchmarks in CI

### P1 (High — Implement in Phase 2)

- [ ] AVX-512 SpMV kernel
- [ ] WASM SIMD128 SpMV kernel
- [ ] Cache-aware tiling for large matrices
- [ ] Memory-mapped CSR for matrices > 100MB
- [ ] SONA adaptive routing with EWC
- [ ] Batch amortization for TRUE preprocessing
- [ ] Web Worker pool for WASM parallelism
- [ ] SharedArrayBuffer zero-copy when available

### P2 (Medium — Implement in Phase 3)

- [ ] Profile-Guided Optimization in CI
- [ ] Vectorized PRNG for random walk algorithms
- [ ] SIMD max/min/argmax reductions for convergence checks
- [ ] Mixed-precision (f32 storage, f64 accumulation) for ill-conditioned systems
- [ ] IndexedDB caching for WASM preprocessing results
- [ ] Incomplete Cholesky preconditioner
- [ ] Streaming API for large solve results

### P3 (Low — Long-term)

- [ ] Algebraic multigrid preconditioner
- [ ] Hardware-specific routing thresholds (per-ISA calibration)
- [ ] NUMA-aware memory allocation
- [ ] Huge pages for large matrix storage
- [ ] GPU offload via Metal/CUDA for dense fallback
- [ ] Distributed solver across ruvector-cluster shards

---

## 9. Performance Targets

| Operation | Server (AVX2) | Edge (NEON) | Browser (WASM) | Cloudflare |
|-----------|:---:|:---:|:---:|:---:|
| SpMV 10K×10K (1%) | < 30 μs | < 50 μs | < 200 μs | < 300 μs |
| CG solve 10K (ε=1e-6) | < 1 ms | < 2 ms | < 20 ms | < 30 ms |
| Forward Push 10K (ε=1e-4) | < 50 μs | < 100 μs | < 500 μs | < 1 ms |
| Neumann 10K (k=20) | < 600 μs | < 1 ms | < 5 ms | < 8 ms |
| BMSSP 100K (ε=1e-4) | < 50 ms | < 100 ms | N/A | < 200 ms |
| TRUE prep 100K (ε=0.1) | < 500 ms | < 1 s | N/A | < 2 s |
| TRUE solve 100K (amortized) | < 1 ms | < 2 ms | N/A | < 5 ms |
| Batch pairwise 10K | < 15 s | < 30 s | < 120 s | N/A |
| Scheduler tick | < 200 ns | < 300 ns | N/A | N/A |
| Algorithm routing | < 1 μs | < 1 μs | < 5 μs | < 5 μs |

---

## 10. Measurement Methodology

All performance claims must be validated with:

1. **Criterion.rs**: 200 samples, 5s warmup, p < 0.05 significance
2. **Multi-platform**: Results on both x86_64 (AVX2) and aarch64 (NEON)
3. **Deterministic seeds**: `random_vector(dim, seed=42)` for reproducibility
4. **Equal accuracy**: Fix ε before comparing approximate algorithms
5. **Cold + hot cache**: Report both first-run and steady-state latencies
6. **Profile.bench**: Inherits release optimization with debug symbols for profiling
7. **Regression CI**: 10% degradation threshold triggers build failure
