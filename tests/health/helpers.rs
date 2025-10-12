use axum::Router;
use qna_agent_poc_rust::health::router as health;
use axum::routing::get;

pub async fn create_test_app() -> Router {
    Router::new()
        .route("/health", get(health::health_check))
}