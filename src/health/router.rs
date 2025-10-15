use axum::{extract::State, http::StatusCode, response::Json};
use serde_json::{json, Value};

use crate::shared::configs::Configs;


pub async fn health_check(State(configs): State<Configs>) -> Result<Json<Value>, StatusCode> {
    let response = json!({
        "status": "healthy",
        "service": "qna-agent-poc-rust",
        "version": "0.1.0",
        "env": configs.build_env,
        "timestamp": chrono::Utc::now().to_rfc3339()
    });

    Ok(Json(response))
}