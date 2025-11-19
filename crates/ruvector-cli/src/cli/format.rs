//! Output formatting utilities

use colored::*;
use ruvector_core::types::{SearchResult, VectorEntry};
use serde_json;

/// Format search results for display
pub fn format_search_results(results: &[SearchResult], show_vectors: bool) -> String {
    let mut output = String::new();

    for (i, result) in results.iter().enumerate() {
        output.push_str(&format!("\n{}. {}\n", i + 1, result.id.bold()));
        output.push_str(&format!("   Score: {:.4}\n", result.score));

        if let Some(metadata) = &result.metadata {
            if !metadata.is_empty() {
                output.push_str(&format!("   Metadata: {}\n",
                    serde_json::to_string_pretty(metadata).unwrap_or_else(|_| "{}".to_string())
                ));
            }
        }

        if show_vectors {
            if let Some(vector) = &result.vector {
                let preview: Vec<f32> = vector.iter().take(5).copied().collect();
                output.push_str(&format!("   Vector (first 5): {:?}...\n", preview));
            }
        }
    }

    output
}

/// Format database statistics
pub fn format_stats(count: usize, dimensions: usize, metric: &str) -> String {
    format!(
        "\n{}\n  Vectors: {}\n  Dimensions: {}\n  Distance Metric: {}\n",
        "Database Statistics".bold().green(),
        count.to_string().cyan(),
        dimensions.to_string().cyan(),
        metric.cyan()
    )
}

/// Format error message
pub fn format_error(msg: &str) -> String {
    format!("{} {}", "Error:".red().bold(), msg)
}

/// Format success message
pub fn format_success(msg: &str) -> String {
    format!("{} {}", "✓".green().bold(), msg)
}

/// Format warning message
pub fn format_warning(msg: &str) -> String {
    format!("{} {}", "Warning:".yellow().bold(), msg)
}

/// Format info message
pub fn format_info(msg: &str) -> String {
    format!("{} {}", "ℹ".blue().bold(), msg)
}

/// Export vector entries to JSON
pub fn export_json(entries: &[VectorEntry]) -> anyhow::Result<String> {
    serde_json::to_string_pretty(entries)
        .map_err(|e| anyhow::anyhow!("Failed to serialize to JSON: {}", e))
}

/// Export vector entries to CSV
pub fn export_csv(entries: &[VectorEntry]) -> anyhow::Result<String> {
    let mut wtr = csv::Writer::from_writer(vec![]);

    // Write header
    wtr.write_record(&["id", "vector", "metadata"])?;

    // Write entries
    for entry in entries {
        wtr.write_record(&[
            entry.id.as_ref().map(|s| s.as_str()).unwrap_or(""),
            &serde_json::to_string(&entry.vector)?,
            &serde_json::to_string(&entry.metadata)?,
        ])?;
    }

    wtr.flush()?;
    String::from_utf8(wtr.into_inner()?)
        .map_err(|e| anyhow::anyhow!("Failed to convert CSV to string: {}", e))
}
