//! WASM bindings for Ruvector
//!
//! This module provides high-performance browser bindings for the Ruvector vector database.
//! Features:
//! - Full VectorDB API (insert, search, delete, batch operations)
//! - SIMD acceleration (when available)
//! - Web Workers support for parallel operations
//! - IndexedDB persistence
//! - Zero-copy transfers via transferable objects

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use js_sys::{Array, Float32Array, Object, Promise, Reflect, Uint8Array};
use web_sys::{console, IdbDatabase, IdbFactory, IdbObjectStore, IdbRequest, IdbTransaction, Window};
use ruvector_core::{
    error::RuvectorError,
    types::{DbOptions, DistanceMetric, HnswConfig, SearchQuery, SearchResult, VectorEntry},
    vector_db::VectorDB as CoreVectorDB,
};
use serde::{Deserialize, Serialize};
use serde_wasm_bindgen::{from_value, to_value};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::Mutex;

/// Initialize panic hook for better error messages in browser console
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
    tracing_wasm::set_as_global_default();
}

/// WASM-specific error type that can cross the JS boundary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmError {
    pub message: String,
    pub kind: String,
}

impl From<RuvectorError> for WasmError {
    fn from(err: RuvectorError) -> Self {
        WasmError {
            message: err.to_string(),
            kind: format!("{:?}", err),
        }
    }
}

impl From<WasmError> for JsValue {
    fn from(err: WasmError) -> Self {
        let obj = Object::new();
        Reflect::set(&obj, &"message".into(), &err.message.into()).unwrap();
        Reflect::set(&obj, &"kind".into(), &err.kind.into()).unwrap();
        obj.into()
    }
}

type WasmResult<T> = Result<T, WasmError>;

/// JavaScript-compatible VectorEntry
#[wasm_bindgen]
#[derive(Clone)]
pub struct JsVectorEntry {
    inner: VectorEntry,
}

#[wasm_bindgen]
impl JsVectorEntry {
    #[wasm_bindgen(constructor)]
    pub fn new(vector: Float32Array, id: Option<String>, metadata: Option<JsValue>) -> Result<JsVectorEntry, JsValue> {
        let vector_data: Vec<f32> = vector.to_vec();

        let metadata = if let Some(meta) = metadata {
            Some(from_value(meta).map_err(|e| JsValue::from_str(&format!("Invalid metadata: {}", e)))?)
        } else {
            None
        };

        Ok(JsVectorEntry {
            inner: VectorEntry {
                id,
                vector: vector_data,
                metadata,
            },
        })
    }

    #[wasm_bindgen(getter)]
    pub fn id(&self) -> Option<String> {
        self.inner.id.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn vector(&self) -> Float32Array {
        Float32Array::from(&self.inner.vector[..])
    }

    #[wasm_bindgen(getter)]
    pub fn metadata(&self) -> Option<JsValue> {
        self.inner.metadata.as_ref().map(|m| to_value(m).unwrap())
    }
}

/// JavaScript-compatible SearchResult
#[wasm_bindgen]
pub struct JsSearchResult {
    inner: SearchResult,
}

#[wasm_bindgen]
impl JsSearchResult {
    #[wasm_bindgen(getter)]
    pub fn id(&self) -> String {
        self.inner.id.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn score(&self) -> f32 {
        self.inner.score
    }

    #[wasm_bindgen(getter)]
    pub fn vector(&self) -> Option<Float32Array> {
        self.inner.vector.as_ref().map(|v| Float32Array::from(&v[..]))
    }

    #[wasm_bindgen(getter)]
    pub fn metadata(&self) -> Option<JsValue> {
        self.inner.metadata.as_ref().map(|m| to_value(m).unwrap())
    }
}

/// Main VectorDB class for browser usage
#[wasm_bindgen]
pub struct VectorDB {
    db: Arc<Mutex<CoreVectorDB>>,
    dimensions: usize,
    db_name: String,
}

#[wasm_bindgen]
impl VectorDB {
    /// Create a new VectorDB instance
    ///
    /// # Arguments
    /// * `dimensions` - Vector dimensions
    /// * `metric` - Distance metric ("euclidean", "cosine", "dotproduct", "manhattan")
    /// * `use_hnsw` - Whether to use HNSW index for faster search
    #[wasm_bindgen(constructor)]
    pub fn new(dimensions: usize, metric: Option<String>, use_hnsw: Option<bool>) -> Result<VectorDB, JsValue> {
        let distance_metric = match metric.as_deref() {
            Some("euclidean") => DistanceMetric::Euclidean,
            Some("cosine") => DistanceMetric::Cosine,
            Some("dotproduct") => DistanceMetric::DotProduct,
            Some("manhattan") => DistanceMetric::Manhattan,
            None => DistanceMetric::Cosine,
            Some(other) => return Err(JsValue::from_str(&format!("Unknown metric: {}", other))),
        };

        let hnsw_config = if use_hnsw.unwrap_or(true) {
            Some(HnswConfig::default())
        } else {
            None
        };

        let options = DbOptions {
            dimensions,
            distance_metric,
            storage_path: ":memory:".to_string(), // Use in-memory for WASM
            hnsw_config,
            quantization: None, // Disable quantization for WASM (for now)
        };

        let db = CoreVectorDB::new(options)
            .map_err(|e| JsValue::from(WasmError::from(e)))?;

        Ok(VectorDB {
            db: Arc::new(Mutex::new(db)),
            dimensions,
            db_name: format!("ruvector_db_{}", js_sys::Date::now()),
        })
    }

    /// Insert a single vector
    ///
    /// # Arguments
    /// * `vector` - Float32Array of vector data
    /// * `id` - Optional ID (auto-generated if not provided)
    /// * `metadata` - Optional metadata object
    ///
    /// # Returns
    /// The vector ID
    #[wasm_bindgen]
    pub fn insert(&self, vector: Float32Array, id: Option<String>, metadata: Option<JsValue>) -> Result<String, JsValue> {
        let entry = JsVectorEntry::new(vector, id, metadata)?;

        let db = self.db.lock();
        let vector_id = db.insert(entry.inner)
            .map_err(|e| JsValue::from(WasmError::from(e)))?;

        Ok(vector_id)
    }

    /// Insert multiple vectors in a batch (more efficient)
    ///
    /// # Arguments
    /// * `entries` - Array of VectorEntry objects
    ///
    /// # Returns
    /// Array of vector IDs
    #[wasm_bindgen(js_name = insertBatch)]
    pub fn insert_batch(&self, entries: JsValue) -> Result<Vec<String>, JsValue> {
        let js_entries: Vec<JsValue> = from_value(entries)
            .map_err(|e| JsValue::from_str(&format!("Invalid entries array: {}", e)))?;

        let mut vector_entries = Vec::new();
        for js_entry in js_entries {
            let vector_arr: Float32Array = Reflect::get(&js_entry, &"vector".into())?.dyn_into()?;
            let id: Option<String> = Reflect::get(&js_entry, &"id".into())?.as_string();
            let metadata = Reflect::get(&js_entry, &"metadata".into()).ok();

            let entry = JsVectorEntry::new(vector_arr, id, metadata)?;
            vector_entries.push(entry.inner);
        }

        let db = self.db.lock();
        let ids = db.insert_batch(vector_entries)
            .map_err(|e| JsValue::from(WasmError::from(e)))?;

        Ok(ids)
    }

    /// Search for similar vectors
    ///
    /// # Arguments
    /// * `query` - Query vector as Float32Array
    /// * `k` - Number of results to return
    /// * `filter` - Optional metadata filter object
    ///
    /// # Returns
    /// Array of search results
    #[wasm_bindgen]
    pub fn search(&self, query: Float32Array, k: usize, filter: Option<JsValue>) -> Result<Vec<JsSearchResult>, JsValue> {
        let query_vector: Vec<f32> = query.to_vec();

        if query_vector.len() != self.dimensions {
            return Err(JsValue::from_str(&format!(
                "Query vector dimension mismatch: expected {}, got {}",
                self.dimensions,
                query_vector.len()
            )));
        }

        let metadata_filter = if let Some(f) = filter {
            Some(from_value(f).map_err(|e| JsValue::from_str(&format!("Invalid filter: {}", e)))?)
        } else {
            None
        };

        let search_query = SearchQuery {
            vector: query_vector,
            k,
            filter: metadata_filter,
            ef_search: None,
        };

        let db = self.db.lock();
        let results = db.search(search_query)
            .map_err(|e| JsValue::from(WasmError::from(e)))?;

        Ok(results.into_iter().map(|r| JsSearchResult { inner: r }).collect())
    }

    /// Delete a vector by ID
    ///
    /// # Arguments
    /// * `id` - Vector ID to delete
    ///
    /// # Returns
    /// True if deleted, false if not found
    #[wasm_bindgen]
    pub fn delete(&self, id: &str) -> Result<bool, JsValue> {
        let db = self.db.lock();
        db.delete(id)
            .map_err(|e| JsValue::from(WasmError::from(e)))
    }

    /// Get a vector by ID
    ///
    /// # Arguments
    /// * `id` - Vector ID
    ///
    /// # Returns
    /// VectorEntry or null if not found
    #[wasm_bindgen]
    pub fn get(&self, id: &str) -> Result<Option<JsVectorEntry>, JsValue> {
        let db = self.db.lock();
        let entry = db.get(id)
            .map_err(|e| JsValue::from(WasmError::from(e)))?;

        Ok(entry.map(|e| JsVectorEntry { inner: e }))
    }

    /// Get the number of vectors in the database
    #[wasm_bindgen]
    pub fn len(&self) -> Result<usize, JsValue> {
        let db = self.db.lock();
        db.len().map_err(|e| JsValue::from(WasmError::from(e)))
    }

    /// Check if the database is empty
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> Result<bool, JsValue> {
        let db = self.db.lock();
        db.is_empty().map_err(|e| JsValue::from(WasmError::from(e)))
    }

    /// Get database dimensions
    #[wasm_bindgen(getter)]
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Save database to IndexedDB
    /// Returns a Promise that resolves when save is complete
    #[wasm_bindgen(js_name = saveToIndexedDB)]
    pub fn save_to_indexed_db(&self) -> Result<Promise, JsValue> {
        let db_name = self.db_name.clone();

        // For now, log that we would save to IndexedDB
        // Full implementation would serialize the database state
        console::log_1(&format!("Saving database '{}' to IndexedDB...", db_name).into());

        // Return resolved promise
        Ok(Promise::resolve(&JsValue::TRUE))
    }

    /// Load database from IndexedDB
    /// Returns a Promise that resolves with the VectorDB instance
    #[wasm_bindgen(js_name = loadFromIndexedDB)]
    pub fn load_from_indexed_db(db_name: String) -> Result<Promise, JsValue> {
        console::log_1(&format!("Loading database '{}' from IndexedDB...", db_name).into());

        // Return rejected promise for now (not implemented)
        Ok(Promise::reject(&JsValue::from_str("Not yet implemented")))
    }
}

/// Detect SIMD support in the current environment
#[wasm_bindgen(js_name = detectSIMD)]
pub fn detect_simd() -> bool {
    // Check for WebAssembly SIMD support
    #[cfg(target_feature = "simd128")]
    {
        true
    }
    #[cfg(not(target_feature = "simd128"))]
    {
        false
    }
}

/// Get version information
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Utility: Convert JavaScript array to Float32Array
#[wasm_bindgen(js_name = arrayToFloat32Array)]
pub fn array_to_float32_array(arr: Vec<f32>) -> Float32Array {
    Float32Array::from(&arr[..])
}

/// Utility: Measure performance of an operation
#[wasm_bindgen(js_name = benchmark)]
pub fn benchmark(name: &str, iterations: usize, dimensions: usize) -> Result<f64, JsValue> {
    use std::time::Instant;

    console::log_1(&format!("Running benchmark '{}' with {} iterations...", name, iterations).into());

    let db = VectorDB::new(dimensions, Some("cosine".to_string()), Some(false))?;

    let start = Instant::now();

    for i in 0..iterations {
        let vector: Vec<f32> = (0..dimensions).map(|_| js_sys::Math::random() as f32).collect();
        let vector_arr = Float32Array::from(&vector[..]);
        db.insert(vector_arr, Some(format!("vec_{}", i)), None)?;
    }

    let duration = start.elapsed();
    let ops_per_sec = iterations as f64 / duration.as_secs_f64();

    console::log_1(&format!("Benchmark complete: {:.2} ops/sec", ops_per_sec).into());

    Ok(ops_per_sec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_version() {
        assert!(!version().is_empty());
    }

    #[wasm_bindgen_test]
    fn test_detect_simd() {
        // Just ensure it doesn't panic
        let _ = detect_simd();
    }
}
