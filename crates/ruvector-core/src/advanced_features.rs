//! Advanced Features for Ruvector
//!
//! This module provides advanced vector database capabilities:
//! - Enhanced Product Quantization with precomputed lookup tables
//! - Filtered Search with automatic strategy selection
//! - MMR (Maximal Marginal Relevance) for diversity
//! - Hybrid Search combining vector and keyword matching
//! - Conformal Prediction for uncertainty quantification

pub mod product_quantization;
pub mod filtered_search;
pub mod mmr;
pub mod hybrid_search;
pub mod conformal_prediction;

// Re-exports
pub use product_quantization::{EnhancedPQ, PQConfig, LookupTable};
pub use filtered_search::{FilteredSearch, FilterStrategy, FilterExpression};
pub use mmr::{MMRSearch, MMRConfig};
pub use hybrid_search::{HybridSearch, HybridConfig, BM25};
pub use conformal_prediction::{ConformalPredictor, ConformalConfig, PredictionSet};
