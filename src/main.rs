mod app;
mod health;

use app::App;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let app = App::new();
    app.run().await;
}
