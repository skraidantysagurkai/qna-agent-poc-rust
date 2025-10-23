# qna-agent-poc-rust
A almost identical copy of https://github.com/skraidantysagurkai/qna-agent-poc but implemented in Rust, keep note that this is my first rust project

```bash
cargo sync
docker volume create qdrant_data
docker run -d --name vector_store -p 6333:6333 -p 6334:6334 -v qdrant_data:/qdrant/storage -e QDRANT__SERVICE__GRPC_PORT="6334" qdrant/qdrant
docker stop vector_store
docker start vector_store
```
