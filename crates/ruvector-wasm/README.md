# Ruvector WASM

High-performance Rust vector database for browsers via WebAssembly.

## Features

✅ **Complete VectorDB API**
- Insert, search, delete, batch operations
- Multiple distance metrics (Euclidean, Cosine, Dot Product, Manhattan)
- HNSW indexing for fast approximate nearest neighbor search

✅ **SIMD Acceleration**
- Automatic SIMD detection
- Dual builds (with/without SIMD)
- 2-4x speedup on supported hardware

✅ **Web Workers Integration**
- Parallel operations across multiple threads
- Worker pool management (4-8 workers)
- Promise-based async API

✅ **IndexedDB Persistence**
- Save/load database state
- Batch operations for performance
- Progressive loading with callbacks
- LRU cache for hot vectors (1000 cached)

✅ **Production Ready**
- Comprehensive error handling
- Browser console debugging
- Optimized for size (<500KB gzipped)
- Full TypeScript definitions

## Quick Start

### Installation

```bash
npm install @ruvector/wasm
```

Or build from source:

```bash
cd crates/ruvector-wasm
npm run build
```

### Basic Usage

```javascript
import init, { VectorDB } from '@ruvector/wasm';

// Initialize WASM
await init();

// Create database
const db = new VectorDB(
  384,        // dimensions
  'cosine',   // metric
  true        // use HNSW index
);

// Insert vector
const vector = new Float32Array(384).map(() => Math.random());
const id = db.insert(vector, 'my_vector', { label: 'example' });

// Search
const query = new Float32Array(384).map(() => Math.random());
const results = db.search(query, 10);  // top 10 results

console.log(results);
// [{ id: 'my_vector', score: 0.95, metadata: { label: 'example' } }, ...]
```

## Advanced Usage

### Web Workers

```javascript
import { WorkerPool } from '@ruvector/wasm/worker-pool';

const pool = new WorkerPool(
  '/worker.js',
  '/pkg/ruvector_wasm.js',
  {
    poolSize: navigator.hardwareConcurrency || 4,
    dimensions: 384,
    metric: 'cosine',
    useHnsw: true
  }
);

await pool.init();

// Parallel batch insert
const entries = [...]; // Your vectors
const ids = await pool.insertBatch(entries);

// Parallel search
const results = await pool.search(query, 10);

// Get pool statistics
const stats = pool.getStats();
console.log(`${stats.busyWorkers}/${stats.poolSize} workers busy`);

// Cleanup
pool.terminate();
```

### IndexedDB Persistence

```javascript
import { IndexedDBPersistence } from '@ruvector/wasm/indexeddb';

const persistence = new IndexedDBPersistence('my_database');
await persistence.open();

// Save vectors
await persistence.saveBatch(entries);

// Load with progress callback
await persistence.loadAll((progress) => {
  console.log(`Loaded ${progress.loaded} vectors`);

  // Insert batch into VectorDB
  if (progress.vectors.length > 0) {
    await db.insertBatch(progress.vectors);
  }

  if (progress.complete) {
    console.log('Load complete!');
  }
});

// Get statistics
const stats = await persistence.getStats();
console.log(`Total: ${stats.totalVectors}, Cache: ${stats.cacheSize}`);
console.log(`Cache hit rate: ${(stats.cacheHitRate * 100).toFixed(2)}%`);
```

## API Reference

See [WASM API Documentation](../../docs/wasm-api.md) for complete API reference.

## Examples

- [Vanilla JavaScript](../../examples/wasm-vanilla/index.html)
- [React + Web Workers](../../examples/wasm-react/)

## Build from Source

```bash
# Standard web build
npm run build

# SIMD-enabled build
npm run build:simd

# All targets (web, node, bundler)
npm run build:all

# Run tests
npm test

# Check bundle size
npm run size

# Optimize with wasm-opt
npm run optimize
```

## Performance

Benchmark on M1 MacBook Pro (Chrome 120):

| Operation | Vectors | Dimensions | Time | Throughput |
|-----------|---------|------------|------|------------|
| Insert (batch) | 10,000 | 384 | 1.2s | 8,333 ops/sec |
| Search | 100 queries | 384 | 0.5s | 200 queries/sec |
| Insert (SIMD) | 10,000 | 384 | 0.4s | 25,000 ops/sec |
| Search (SIMD) | 100 queries | 384 | 0.2s | 500 queries/sec |

## Browser Support

| Browser | Version | SIMD | Workers | IndexedDB |
|---------|---------|------|---------|-----------|
| Chrome  | 91+     | ✅   | ✅      | ✅        |
| Firefox | 89+     | ✅   | ✅      | ✅        |
| Safari  | 16.4+   | Partial | ✅   | ✅        |
| Edge    | 91+     | ✅   | ✅      | ✅        |

## Size

- Base build: ~450KB gzipped
- SIMD build: ~480KB gzipped
- With wasm-opt: ~380KB gzipped

## Troubleshooting

See [WASM Build Guide](../../docs/wasm-build-guide.md) for detailed troubleshooting.

## License

MIT
