mod app;
mod health;
mod shared;

use app::App;
use shared::configs::Configs;

#[tokio::main]
async fn main() {
    let configs = Configs::new();

    tracing_subscriber::fmt::init();

    let app = App::new(configs);
    app.run().await;
}
