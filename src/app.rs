use axum::Router;
use axum::routing::get;
use tokio::signal;
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use crate::health::router as health;
use crate::shared::configs::Configs;

pub struct App {
    routers: Vec<Router>,
    configs: Configs,
}

impl Default for App {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl App {
    pub fn new(configs: Configs) -> Self {
        let mut app = App {
            routers: Vec::new(),
            configs,
        };
        app._build_routers();
        app
    }

    fn _build_routers(&mut self) {
        let health_router = Router::new()
            .route("/health", get(health::health_check))
            .with_state(self.configs.clone());
        self.routers.push(health_router);
    }

    fn _merge_routers(self) -> Router {
        let mut app = Router::new();

        for router in self.routers {
            app = app.merge(router);
        }

        app.layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(CorsLayer::permissive()),
        )
    }
    
    async fn _shutdown_signal() {
        let ctrl_c = async {
            signal::ctrl_c()
                .await
                .expect("failed to install Ctrl+C handler");
        };

        #[cfg(unix)]
        let terminate = async {
            signal::unix::signal(signal::unix::SignalKind::terminate())
                .expect("failed to install signal handler")
                .recv()
                .await;
        };

        #[cfg(not(unix))]
        let terminate = std::future::pending::<()>();
        tokio::select! {
            _ = ctrl_c => {
                tracing::info!("Received Ctrl+C, shutting down gracefully...");
            },
            _ = terminate => {
                tracing::info!("Received terminate signal, shutting down gracefully...");
            },
        }
    }

    pub async fn run(self) {
        let addr = self.configs.get_socket_addr();
        let routers = self._merge_routers();
        tracing::info!("Server starting on {}", addr);

        let listener = tokio::net::TcpListener::bind(addr).await.unwrap();

        axum::serve(listener, routers)
            .with_graceful_shutdown(App::_shutdown_signal())
            .await
            .unwrap();

        tracing::info!("Server shut down gracefully");
    }
}