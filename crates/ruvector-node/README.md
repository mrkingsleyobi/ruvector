# @ruvector/node

High-performance Rust vector database for Node.js with HNSW indexing, SIMD-optimized distance calculations, and zero-copy buffer sharing.

## Features

- **🚀 10-100x Faster**: Rust-powered performance with SIMD optimizations
- **🧠 HNSW Indexing**: Sub-millisecond search with 95%+ recall
- **💾 Memory Efficient**: 4-32x compression with quantization
- **⚡ Async/Await**: Non-blocking operations with Tokio
- **🔒 Type Safe**: Complete TypeScript definitions
- **🌐 Zero-Copy**: Direct Float32Array buffer sharing
- **📦 Cross-Platform**: Linux, macOS, Windows (x64 and ARM64)

## Installation

```bash
npm install @ruvector/node
```

## Quick Start

```javascript
import { VectorDB } from '@ruvector/node';

// Create a vector database
const db = new VectorDB({
  dimensions: 384,
  distanceMetric: 'Cosine',
  storagePath: './vectors.db'
});

// Insert vectors
const id = await db.insert({
  vector: new Float32Array([1.0, 2.0, 3.0, ...]),
  metadata: { text: 'example document' }
});

// Search for similar vectors
const results = await db.search({
  vector: new Float32Array([1.0, 2.0, 3.0, ...]),
  k: 10
});

console.log(results);
// [{ id, score, metadata, vector }, ...]
```

## API Reference

### VectorDB

Main vector database class.

#### Constructor

```typescript
new VectorDB(options: DbOptions)
```

**Options:**

- `dimensions` (number, required): Vector dimensions
- `distanceMetric` (string, optional): Distance metric - `'Euclidean'`, `'Cosine'`, `'DotProduct'`, `'Manhattan'`. Default: `'Cosine'`
- `storagePath` (string, optional): Path to database file. Default: `'./ruvector.db'`
- `hnswConfig` (object, optional): HNSW index configuration
  - `m` (number): Connections per node. Default: 32
  - `efConstruction` (number): Construction quality. Default: 200
  - `efSearch` (number): Search quality. Default: 100
  - `maxElements` (number): Maximum elements. Default: 10,000,000
- `quantization` (object, optional): Quantization configuration
  - `type` (string): `'none'`, `'scalar'`, `'product'`, `'binary'`. Default: `'scalar'`
  - `subspaces` (number): For product quantization. Default: 16
  - `k` (number): Codebook size. Default: 256

**Example:**

```javascript
const db = new VectorDB({
  dimensions: 384,
  distanceMetric: 'Cosine',
  storagePath: './my-vectors.db',
  hnswConfig: {
    m: 32,
    efConstruction: 200,
    efSearch: 100
  },
  quantization: {
    type: 'scalar'
  }
});
```

#### Factory Method

```typescript
VectorDB.withDimensions(dimensions: number): VectorDB
```

Create a database with default options.

```javascript
const db = VectorDB.withDimensions(384);
```

### Methods

#### insert

```typescript
async insert(entry: VectorEntry): Promise<string>
```

Insert a vector into the database. Returns the vector ID.

**Parameters:**

- `entry.vector` (Float32Array, required): Vector data
- `entry.id` (string, optional): Custom ID (auto-generated if not provided)
- `entry.metadata` (object, optional): Metadata as JSON object

**Example:**

```javascript
const id = await db.insert({
  vector: new Float32Array([1, 2, 3]),
  metadata: { text: 'example' }
});
```

#### insertBatch

```typescript
async insertBatch(entries: VectorEntry[]): Promise<string[]>
```

Insert multiple vectors in a batch. Returns an array of vector IDs.

**Example:**

```javascript
const ids = await db.insertBatch([
  { vector: new Float32Array([1, 2, 3]) },
  { vector: new Float32Array([4, 5, 6]) }
]);
```

#### search

```typescript
async search(query: SearchQuery): Promise<SearchResult[]>
```

Search for similar vectors. Returns an array of results sorted by similarity.

**Parameters:**

- `query.vector` (Float32Array, required): Query vector
- `query.k` (number, required): Number of results to return
- `query.filter` (object, optional): Metadata filters
- `query.efSearch` (number, optional): HNSW search quality override

**Returns:**

- `id` (string): Vector ID
- `score` (number): Similarity score (lower is better)
- `vector` (Float32Array, optional): Vector data
- `metadata` (object, optional): Metadata

**Example:**

```javascript
const results = await db.search({
  vector: new Float32Array([1, 2, 3]),
  k: 10,
  filter: { category: 'example' }
});

results.forEach(result => {
  console.log(`ID: ${result.id}, Score: ${result.score}`);
});
```

#### delete

```typescript
async delete(id: string): Promise<boolean>
```

Delete a vector by ID. Returns `true` if deleted, `false` if not found.

**Example:**

```javascript
const deleted = await db.delete('vector-id');
```

#### get

```typescript
async get(id: string): Promise<VectorEntry | null>
```

Get a vector by ID. Returns the vector entry or `null` if not found.

**Example:**

```javascript
const entry = await db.get('vector-id');
if (entry) {
  console.log(entry.metadata);
}
```

#### len

```typescript
async len(): Promise<number>
```

Get the number of vectors in the database.

**Example:**

```javascript
const count = await db.len();
console.log(`Database contains ${count} vectors`);
```

#### isEmpty

```typescript
async isEmpty(): Promise<boolean>
```

Check if the database is empty.

**Example:**

```javascript
if (await db.isEmpty()) {
  console.log('Database is empty');
}
```

## Performance

Ruvector achieves exceptional performance through:

- **Rust Implementation**: 2-50x faster than Python/TypeScript
- **SIMD Optimizations**: 4-16x faster distance calculations
- **HNSW Indexing**: O(log n) search complexity
- **Zero-Copy Buffers**: Direct Float32Array access
- **Async Operations**: Non-blocking with Tokio threadpool

### Benchmarks

**10,000 vectors (128D)**

- Insert: ~1,000 vectors/sec
- Search (k=10): ~1ms average latency
- QPS: ~1,000 queries/sec (single-threaded)

**1,000,000 vectors (128D)**

- Insert: ~500-1,000 vectors/sec
- Search (k=10): ~5ms average latency
- QPS: ~200-500 queries/sec

## Examples

See the [examples](./examples) directory:

- **simple.mjs**: Basic operations
- **advanced.mjs**: HNSW indexing and batch operations
- **semantic-search.mjs**: Text similarity search

Run examples:

```bash
npm run build
node examples/simple.mjs
node examples/advanced.mjs
node examples/semantic-search.mjs
```

## Use Cases

- **RAG Systems**: Retrieval-augmented generation
- **Semantic Search**: Find similar documents
- **Recommender Systems**: Content recommendations
- **Duplicate Detection**: Find similar items
- **Image Search**: Visual similarity
- **Agent Memory**: Reflexion and skill libraries

## Building from Source

```bash
# Clone the repository
git clone https://github.com/ruvnet/ruvector
cd ruvector/crates/ruvector-node

# Install dependencies
npm install

# Build
npm run build

# Run tests
npm test
```

## Cross-Platform Builds

```bash
# Build for all platforms
npm run build -- --target x86_64-unknown-linux-gnu
npm run build -- --target aarch64-unknown-linux-gnu
npm run build -- --target x86_64-apple-darwin
npm run build -- --target aarch64-apple-darwin
npm run build -- --target x86_64-pc-windows-msvc
```

## TypeScript Support

TypeScript definitions are automatically generated from Rust code:

```typescript
import { VectorDB, type VectorEntry, type SearchResult } from '@ruvector/node';

const db = new VectorDB({
  dimensions: 384,
  distanceMetric: 'Cosine'
});

const entry: VectorEntry = {
  vector: new Float32Array(384),
  metadata: { text: 'example' }
};

const id = await db.insert(entry);
const results: SearchResult[] = await db.search({
  vector: new Float32Array(384),
  k: 10
});
```

## Memory Management

Ruvector uses:

- **Arc<RwLock<>>**: Thread-safe reference counting
- **tokio::spawn_blocking**: CPU-bound work on threadpool
- **Zero-copy buffers**: Direct Float32Array access
- **Automatic cleanup**: Rust handles memory deallocation

No manual memory management required!

## Troubleshooting

### Installation fails

Make sure you have Rust installed:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Build errors

Update NAPI-RS CLI:

```bash
npm install -g @napi-rs/cli
```

### Performance issues

- Use HNSW indexing for large datasets (>10K vectors)
- Enable quantization for memory efficiency
- Adjust `efSearch` for speed/accuracy tradeoff
- Use `insertBatch` instead of individual `insert` calls

## License

MIT

## Contributing

Contributions welcome! See [CONTRIBUTING.md](../../CONTRIBUTING.md)

## Links

- [Documentation](https://github.com/ruvnet/ruvector)
- [Examples](./examples)
- [Issues](https://github.com/ruvnet/ruvector/issues)
- [Rust Crate](../ruvector-core)

## Acknowledgments

Built with:

- [NAPI-RS](https://napi.rs) - Rust bindings for Node.js
- [hnsw_rs](https://github.com/jean-pierreBoth/hnswlib-rs) - HNSW implementation
- [SimSIMD](https://github.com/ashvardanian/simsimd) - SIMD distance metrics
- [redb](https://github.com/cberner/redb) - Embedded database
