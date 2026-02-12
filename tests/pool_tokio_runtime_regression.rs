use anyhow::Result;
use rust_embed::{EmbeddingPool, ModelType, PoolConfig};

#[tokio::test(flavor = "current_thread")]
async fn pool_embed_batch_from_tokio_runtime_initializes_worker_model() -> Result<()> {
    let config = PoolConfig {
        cpu_workers: 1,
        gpu_workers: 0,
        model: ModelType::MiniLM,
        cache_size_per_worker: 0,
        routing_config: None,
    };

    let pool = EmbeddingPool::new(config)?;

    // Regression test: this used to panic with
    // "Cannot block the current thread from within a runtime"
    // when using tokio::oneshot::blocking_recv in pool code.
    let embeddings = pool.embed_batch(vec!["runtime regression text".to_string()])?;

    assert_eq!(embeddings.len(), 1);
    assert_eq!(embeddings[0].len(), 384);

    pool.shutdown()?;
    Ok(())
}
