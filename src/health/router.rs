use axum::{http::StatusCode, response::Json};
use serde_json::{json, Value};

/// Health check endpoint
pub async fn health_check() -> Result<Json<Value>, StatusCode> {
    let response = json!({
        "status": "healthy",
        "service": "qna-agent-poc-rust",
        "version": "0.1.0",
        "timestamp": chrono::Utc::now().to_rfc3339()
    });

    Ok(Json(response))
}