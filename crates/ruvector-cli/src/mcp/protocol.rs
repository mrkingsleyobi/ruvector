//! MCP protocol types and utilities

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// MCP request message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpRequest {
    pub jsonrpc: String,
    pub id: Option<Value>,
    pub method: String,
    pub params: Option<Value>,
}

/// MCP response message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResponse {
    pub jsonrpc: String,
    pub id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<McpError>,
}

/// MCP error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl McpError {
    pub fn new(code: i32, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            data: None,
        }
    }

    pub fn with_data(mut self, data: Value) -> Self {
        self.data = Some(data);
        self
    }
}

/// Standard MCP error codes
pub mod error_codes {
    pub const PARSE_ERROR: i32 = -32700;
    pub const INVALID_REQUEST: i32 = -32600;
    pub const METHOD_NOT_FOUND: i32 = -32601;
    pub const INVALID_PARAMS: i32 = -32602;
    pub const INTERNAL_ERROR: i32 = -32603;
}

impl McpResponse {
    pub fn success(id: Option<Value>, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn error(id: Option<Value>, error: McpError) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(error),
        }
    }
}

/// MCP Tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}

/// MCP Resource definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResource {
    pub uri: String,
    pub name: String,
    pub description: String,
    #[serde(rename = "mimeType")]
    pub mime_type: String,
}

/// MCP Prompt definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpPrompt {
    pub name: String,
    pub description: String,
    pub arguments: Option<Vec<PromptArgument>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptArgument {
    pub name: String,
    pub description: String,
    pub required: bool,
}

/// Tool call parameters for vector_db_create
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateDbParams {
    pub path: String,
    pub dimensions: usize,
    #[serde(default)]
    pub distance_metric: Option<String>,
}

/// Tool call parameters for vector_db_insert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsertParams {
    pub db_path: String,
    pub vectors: Vec<VectorInsert>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorInsert {
    pub id: Option<String>,
    pub vector: Vec<f32>,
    pub metadata: Option<Value>,
}

/// Tool call parameters for vector_db_search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchParams {
    pub db_path: String,
    pub query: Vec<f32>,
    pub k: usize,
    pub filter: Option<Value>,
}

/// Tool call parameters for vector_db_stats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsParams {
    pub db_path: String,
}

/// Tool call parameters for vector_db_backup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupParams {
    pub db_path: String,
    pub backup_path: String,
}
