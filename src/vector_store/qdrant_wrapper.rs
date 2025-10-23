use qdrant_client::qdrant::{CountPointsBuilder, CollectionDescription, CountResponse, SearchPointsBuilder, SearchResponse};
use qdrant_client::{Qdrant, QdrantError};

use async_openai::{
    types::{CreateEmbeddingRequestArgs, EmbeddingInput},
    Client as OpenAIClient,
};

pub struct QdrantWrapper {
    client: Qdrant,
    collection_name: String,
}

impl QdrantWrapper {
    pub fn new(collection_name: String) -> Result<Self, QdrantError> {
        let client: Qdrant = Qdrant::from_url("http://localhost:6334").build()?;
        Ok(Self { client, collection_name })
    }

    pub async fn is_store_empty(&self) -> Result<bool, QdrantError> {
        let collection_list: Vec<CollectionDescription> = self.client.list_collections().await?.collections;

        if !collection_list.iter().any(|collection| collection.name == self.collection_name) {
            return Ok(true);
        }
        let num_points: CountResponse = self.client.count(CountPointsBuilder::new(&self.collection_name).exact(false)).await?;
        if let Some(count_result) = num_points.result {
            Ok(count_result.count == 0)
        } else {
            Ok(true) // Hit when count_result is None
        }
    }

    pub async fn similarity_search(&self, query_embedding: Vec<f32>, top_k: u64) -> Result<Vec<qdrant_client::qdrant::ScoredPoint
    >, QdrantError> {
        let search_request = SearchPointsBuilder::new(
            &self.collection_name,    // Collection name
            query_embedding,
            top_k,                  // Search limit, number of results to return
        ).with_payload(true);

        let response: SearchResponse = self.client.search_points(search_request).await?;
        Ok(response.result)
    }
}


struct Embedder {
    client: OpenAIClient<async_openai::config::OpenAIConfig>,
}


impl Embedder {
    pub fn new(api_key: String) -> Self {
        let config = async_openai::config::OpenAIConfig::new().with_api_key(api_key);
        let client = OpenAIClient::with_config(config);
        Self { client }
    }

    pub async fn embed_message(&self, message: String) -> Result<Vec<f32>, async_openai::error::OpenAIError> {
        let request = CreateEmbeddingRequestArgs::default()
            .model("text-embedding-3-small")
            .input(EmbeddingInput::String(message))
            .build()?;

        let response = self.client.embeddings().create(request).await?;

        if let Some(embedding_data) = response.data.first() {
            Ok(embedding_data.embedding.clone())
        } else {
            Err(async_openai::error::OpenAIError::InvalidArgument("No embedding data returned".to_string()))
        }
    }
}


#[tokio::test]
async fn test_qdrant_wrapper_is_store_empty() {
    let qdrant_wrapper = QdrantWrapper::new("non_existent_collection".to_string()).unwrap();
    let is_empty = qdrant_wrapper.is_store_empty().await.unwrap();
    assert!(is_empty);
}

#[tokio::test]
async fn test_embedder_embed_message() {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let embedder = Embedder::new(api_key);
    let message = "Hello, world!".to_string();
    let embedding = embedder.embed_message(message).await.unwrap();
    assert!(!embedding.is_empty());
}

#[tokio::test]
async fn test_search() {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let embedder = Embedder::new(api_key);
    let qdrant_wrapper = QdrantWrapper::new("dev".to_string()).unwrap();

    let message = "Foxy Proxy".to_string();
    let embedding = embedder.embed_message(message).await.unwrap();
    let results = qdrant_wrapper.similarity_search(embedding, 5).await.unwrap();
    println!("{:#?}", results);

    assert!(!results.is_empty());
}