# Ruvector Benchmark Suite

Comprehensive performance testing and profiling tools for the Ruvector vector database.

## Quick Start

```bash
# Build benchmarks
cargo build --release

# Run all benchmarks
./scripts/run_all_benchmarks.sh

# Run specific benchmark
cargo run --release --bin ann-benchmark
```

## Available Benchmarks

### 1. ANN-Benchmarks (`ann-benchmark`)
Standard ANN benchmark compatibility (SIFT1M, GIST1M, Deep1M)

```bash
cargo run --release --bin ann-benchmark -- \
    --dataset synthetic \
    --num-vectors 100000 \
    --queries 1000
```

### 2. AgenticDB Workloads (`agenticdb-benchmark`)
Agentic AI workload simulation (Reflexion, skills, causal graphs)

```bash
cargo run --release --bin agenticdb-benchmark -- \
    --episodes 10000 \
    --queries 500
```

### 3. Latency Profiling (`latency-benchmark`)
Detailed latency analysis (p50, p95, p99, p99.9)

```bash
cargo run --release --bin latency-benchmark -- \
    --num-vectors 50000 \
    --threads "1,4,8,16"
```

### 4. Memory Profiling (`memory-benchmark`)
Memory usage at various scales with quantization effects

```bash
cargo run --release --bin memory-benchmark -- \
    --scales "1000,10000,100000"
```

### 5. System Comparison (`comparison-benchmark`)
Cross-system performance comparison

```bash
cargo run --release --bin comparison-benchmark -- \
    --num-vectors 50000
```

### 6. Performance Profiling (`profiling-benchmark`)
CPU flamegraphs and hotspot analysis

```bash
cargo run --release --features profiling --bin profiling-benchmark -- \
    --flamegraph
```

## Features

- **ANN-Benchmarks Compatible**: Standard testing format
- **AgenticDB Workloads**: Real-world agentic AI scenarios
- **Comprehensive Metrics**: QPS, latency percentiles, recall, memory
- **Flexible Configuration**: Adjustable parameters for all tests
- **Multiple Output Formats**: JSON, CSV, Markdown reports
- **Profiling Support**: Flamegraphs and performance analysis

## Documentation

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for detailed documentation including:
- Installation and setup
- Usage examples
- Result interpretation
- Performance targets
- Troubleshooting

## Scripts

- `scripts/download_datasets.sh` - Download ANN benchmark datasets
- `scripts/run_all_benchmarks.sh` - Run complete benchmark suite

## Optional Features

```toml
# Enable HDF5 dataset loading
cargo build --release --features hdf5-datasets

# Enable profiling with flamegraphs
cargo build --release --features profiling

# Build without optional features (recommended if dependencies not available)
cargo build --release --no-default-features
```

## Requirements

- Rust 1.75+
- Optional: HDF5 libraries (for real datasets)
- Optional: perf tools (for profiling on Linux)

## Results

All benchmark results are saved to `bench_results/` directory:
- `*.json` - Raw data for programmatic analysis
- `*.csv` - Tabular data for spreadsheet analysis
- `*.md` - Human-readable reports

## Performance Targets

Ruvector aims for 10-100x improvement over AgenticDB:

| Metric | Target |
|--------|--------|
| QPS (100K vectors) | >10,000 |
| Latency p99 | <5ms |
| Recall@10 | >95% |
| Memory per vector | <2KB |

## License

MIT

## Contributing

Contributions welcome! Please see main repository for guidelines.
