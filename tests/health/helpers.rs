use axum::Router;
use qna_agent_poc_rust::health::router as health;
use qna_agent_poc_rust::shared::configs::Configs;
use axum::routing::get;

pub async fn create_test_app() -> Router {
    let configs = Configs::new();
    
    Router::new()
        .route("/health", get(health::health_check))
        .with_state(configs)
}