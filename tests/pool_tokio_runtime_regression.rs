use anyhow::Result;
use rust_embed::{EmbeddingPool, ModelType, PoolConfig};

#[tokio::test(flavor = "current_thread")]
async fn pool_aggregate_stats_from_tokio_runtime_no_blocking_panic() -> Result<()> {
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
    // aggregate_stats exercises worker request/response waiting without requiring model inference.
    let stats = pool.aggregate_stats()?;

    assert_eq!(stats.embeddings_count, 0);

    pool.shutdown()?;
    Ok(())
}
