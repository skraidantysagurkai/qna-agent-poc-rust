use serde::Deserialize;
use std::env;

#[derive(Debug, Clone, Deserialize)]
pub struct Configs {
    pub server_host: String,
    pub server_port: u16,
    pub build_env: String,
}

impl Default for Configs {
    fn default() -> Self {
        Self {
            server_host: "0.0.0.0".to_string(),
            server_port: 8080,
            build_env: "dev".to_string(),
        }
    }
}

impl Configs {
    pub fn new() -> Self {
        let build_env = Self::get_build_environment();
        let env_file = Self::get_env_file_path(&build_env);

        if std::path::Path::new(&env_file).exists() {
            if let Err(e) = dotenv::from_path(&env_file) {
                tracing::warn!("Failed to load environment file {}: {}", env_file, e);
            } else {
                tracing::info!("Loaded environment from: {}", env_file);
            }
        } else {
            tracing::warn!("Environment file not found: {}, using defaults", env_file);
        }

        let config = Self {
            server_host: env::var("SERVER_HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
            server_port: env::var("SERVER_PORT")
                .unwrap_or_else(|_| "8080".to_string())
                .parse()
                .unwrap_or(8080),
            build_env: build_env.clone(),
        };

        tracing::info!("Configuration loaded for environment: {}", build_env);
        config
    }

    fn get_build_environment() -> String {
        if let Ok(env) = env::var("BUILD_ENV") {
            return env;
        }

        if cfg!(debug_assertions) {
            "dev".to_string()
        } else {
            "prod".to_string()
        }
    }

    fn get_env_file_path(build_env: &str) -> String {
        match build_env {
            "test" => ".env.test".to_string(),
            "dev" => ".env".to_string(),
            "prod" => ".env".to_string(),
            _ => ".env".to_string(),
        }
    }

    pub fn get_socket_addr(&self) -> std::net::SocketAddr {
        format!("{}:{}", self.server_host, self.server_port)
            .parse()
            .expect("Invalid socket address")
    }
}
