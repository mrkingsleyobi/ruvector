# ADR-032: RVF WASM Integration into npx ruvector and rvlite

**Status**: Proposed
**Date**: 2026-02-14
**Deciders**: ruv.io Team
**Supersedes**: None
**Related**: ADR-030 (RVF Cognitive Container), ADR-031 (RVCOW Branching)

---

## Context

The RuVector Format (RVF) ecosystem now ships four npm packages:

| Package | Purpose | Size |
|---------|---------|------|
| `@ruvector/rvf` | Unified TypeScript SDK with auto backend selection | - |
| `@ruvector/rvf-node` | Native N-API bindings (Rust via napi-rs) | - |
| `@ruvector/rvf-wasm` | Browser/edge WASM build | ~46 KB control plane, ~5.5 KB tile |
| `@ruvector/rvf-mcp-server` | MCP server for AI agent integration | - |

Two existing packages would benefit from RVF integration:

1. **`ruvector` (npx ruvector)** -- The main CLI and SDK package (v0.1.88). It has 28 CLI command groups (7,065 lines), depends on `@ruvector/core`, `@ruvector/attention`, `@ruvector/gnn`, `@ruvector/sona`, but has **no dependency on `@ruvector/rvf`**. It currently uses in-memory vector storage with no persistent file-backed option.

2. **`rvlite`** -- A lightweight multi-query vector database (SQL, SPARQL, Cypher) running entirely in WASM. It uses `ruvector-core` for vectors and IndexedDB for browser persistence. A Rust adapter already exists at `crates/rvf/rvf-adapters/rvlite/` wrapping `RvfStore` as `RvliteCollection`.

## Decision

Integrate `@ruvector/rvf` (and its WASM backend) into both packages in three phases:

### Phase 1: npx ruvector -- Add RVF as optional dependency + CLI command group

**Changes:**

1. **package.json** -- Add `@ruvector/rvf` as an optional dependency:
   ```json
   "optionalDependencies": {
     "@ruvector/rvf": "^0.1.0"
   }
   ```

2. **src/index.ts** -- Extend platform detection to try RVF after `@ruvector/core`:
   ```
   Detection order:
   1. @ruvector/core  (native Rust -- fastest)
   2. @ruvector/rvf   (RVF store -- persistent, file-backed)
   3. Stub fallback   (in-memory, testing only)
   ```

3. **bin/cli.js** -- Add `rvf` command group before the `mcp` command (~line 7010):
   ```
   ruvector rvf create <path>           Create a new .rvf store
   ruvector rvf ingest <path> <file>    Ingest vectors from JSON/CSV
   ruvector rvf query <path> <vector>   k-NN search
   ruvector rvf status <path>           Show store statistics
   ruvector rvf segments <path>         List all segments
   ruvector rvf derive <path> <child>   Create derived store with lineage
   ruvector rvf compact <path>          Reclaim deleted space
   ruvector rvf export <path>           Export store
   ```

4. **src/core/rvf-wrapper.ts** -- Create wrapper module exposing `RvfDatabase` through the existing core interface pattern. Exports added to `src/core/index.ts`.

5. **Hooks integration** -- Add `ruvector hooks rvf-backend` subcommand to use `.rvf` files as persistent vector memory backend for the hooks/intelligence system (replacing in-memory storage).

### Phase 2: rvlite -- RVF as storage backend for vector data

**Changes:**

1. **Rust crate (`crates/rvlite`)** -- Add optional `rvf-runtime` dependency behind a feature flag:
   ```toml
   [features]
   default = []
   rvf-backend = ["rvf-runtime", "rvf-types"]
   ```

2. **Hybrid persistence model:**
   - **Vectors**: Stored in `.rvf` file via `RvliteCollection` adapter (already exists at `rvf-adapters/rvlite/`)
   - **Metadata/Graphs**: Continue using IndexedDB JSON state (SQL tables, Cypher nodes/edges, SPARQL triples)
   - **Rationale**: RVF is optimized for vector storage with SIMD-aligned slabs and HNSW indexing. Graph and relational data are better served by the existing serialization.

3. **npm package (`npm/packages/rvlite`)** -- Add `@ruvector/rvf-wasm` as optional dependency. Extend `RvLite` TypeScript class:
   ```typescript
   // New factory method
   static async createWithRvf(config: RvLiteConfig & { rvfPath: string }): Promise<RvLite>

   // New methods
   async saveToRvf(path: string): Promise<void>
   async loadFromRvf(path: string): Promise<void>
   ```

4. **Migration utility** -- `rvlite rvf-migrate` CLI command to convert existing IndexedDB vector data into `.rvf` files.

### Phase 3: Shared WASM backend unification

1. **Single WASM build** -- Both `rvlite` and `ruvector` share `@ruvector/rvf-wasm` as the vector computation engine in browser environments, eliminating duplicate WASM binaries.

2. **MCP bridge** -- The existing `@ruvector/rvf-mcp-server` exposes all RVF operations to AI agents. Extend with rvlite-specific tools:
   ```
   rvlite_sql(storeId, query)       Execute SQL over RVF-backed store
   rvlite_cypher(storeId, query)    Execute Cypher query
   rvlite_sparql(storeId, query)    Execute SPARQL query
   ```

3. **Core export consolidation** -- `ruvector` re-exports `RvfDatabase` so downstream consumers use a single import:
   ```typescript
   import { RvfDatabase } from 'ruvector';
   ```

## API Mapping

### ruvector hooks system -> RVF

| Hooks Operation | Current Implementation | RVF Equivalent |
|----------------|----------------------|----------------|
| `hooks remember` | In-memory vector store | `RvfDatabase.ingestBatch()` |
| `hooks recall` | In-memory k-NN | `RvfDatabase.query()` |
| `hooks export` | JSON dump | `RvfDatabase.segments()` + file copy |
| `hooks stats` | Runtime counters | `RvfDatabase.status()` |

### rvlite -> RVF

| RvLite Operation | Current Implementation | RVF Equivalent |
|-----------------|----------------------|----------------|
| `insert(vector)` | `VectorDB.add()` (ruvector-core) | `RvliteCollection.add()` |
| `search(query, k)` | `VectorDB.search()` | `RvliteCollection.search()` |
| `delete(id)` | `VectorDB.remove()` | `RvliteCollection.remove()` |
| `save()` | IndexedDB serialization | `RvfStore` file (automatic) |
| `load()` | IndexedDB deserialization | `RvliteCollection.open()` |

### RVF WASM exports used

| Export | Used By | Purpose |
|--------|---------|---------|
| `rvf_store_create` | Both | Initialize in-memory store |
| `rvf_store_ingest` | Both | Batch vector ingestion |
| `rvf_store_query` | Both | k-NN search |
| `rvf_store_delete` | Both | Soft-delete vectors |
| `rvf_store_export` | ruvector | Serialize to `.rvf` bytes |
| `rvf_store_open` | rvlite | Parse `.rvf` into queryable store |
| `rvf_store_count` | Both | Live vector count |
| `rvf_store_status` | ruvector | Store statistics |

## Consequences

### Positive

- **Persistent vector storage** -- `npx ruvector` gains file-backed vector storage (`.rvf` files) for the first time, enabling hooks intelligence to survive across sessions.
- **Single format** -- Both packages read/write the same `.rvf` binary format, enabling data interchange.
- **Reduced bundle size** -- Sharing `@ruvector/rvf-wasm` (~46 KB) between packages eliminates duplicate vector engines.
- **Lineage tracking** -- `RvfDatabase.derive()` brings COW branching and provenance to both packages.
- **Cross-platform** -- RVF auto-selects N-API (Node.js) or WASM (browser) without user configuration.

### Negative

- **Optional dependency complexity** -- Both packages must gracefully handle missing `@ruvector/rvf` at runtime.
- **Dual persistence in rvlite** -- Vectors in `.rvf` files + metadata in IndexedDB adds a split-brain risk if one store is modified without the other.
- **API surface growth** -- `npx ruvector` gains 8 new CLI subcommands.

### Risks

- **IndexedDB + RVF sync** -- In rvlite's hybrid mode, crash between RVF write and IndexedDB write could leave metadata inconsistent. Mitigated by writing RVF first (append-only, crash-safe) and treating IndexedDB as rebuildable cache.
- **WASM size budget** -- Adding RVF WASM (~46 KB) to rvlite's existing WASM bundle (~850 KB) is acceptable (<6% increase).

## Implementation Files

### npx ruvector (Phase 1)

| File | Action |
|------|--------|
| `npm/packages/ruvector/package.json` | Edit -- add `@ruvector/rvf` optional dep |
| `npm/packages/ruvector/src/index.ts` | Edit -- add RVF to platform detection |
| `npm/packages/ruvector/src/core/rvf-wrapper.ts` | Create -- RVF wrapper module |
| `npm/packages/ruvector/src/core/index.ts` | Edit -- export rvf-wrapper |
| `npm/packages/ruvector/bin/cli.js` | Edit -- add `rvf` command group (~line 7010) |

### rvlite (Phase 2)

| File | Action |
|------|--------|
| `crates/rvlite/Cargo.toml` | Edit -- add optional `rvf-runtime` dep |
| `crates/rvlite/src/lib.rs` | Edit -- add RVF backend behind feature flag |
| `npm/packages/rvlite/package.json` | Edit -- add `@ruvector/rvf-wasm` optional dep |
| `npm/packages/rvlite/src/index.ts` | Edit -- add `createWithRvf()` factory |

### Shared (Phase 3)

| File | Action |
|------|--------|
| `npm/packages/rvf-mcp-server/src/server.ts` | Edit -- add rvlite query tools |

## Verification

```bash
# Phase 1: npx ruvector RVF integration
npx ruvector rvf create test.rvf --dimension 384
npx ruvector rvf ingest test.rvf --input vectors.json
npx ruvector rvf query test.rvf --vector "0.1,0.2,..." --k 10
npx ruvector rvf status test.rvf
npx ruvector hooks remember --backend rvf --store hooks.rvf "test pattern"
npx ruvector hooks recall --backend rvf --store hooks.rvf "test"

# Phase 2: rvlite RVF backend
cargo test -p rvlite --features rvf-backend
# npm test for rvlite with RVF factory

# Phase 3: Shared WASM
# Verify single @ruvector/rvf-wasm instance in node_modules
npm ls @ruvector/rvf-wasm
```
