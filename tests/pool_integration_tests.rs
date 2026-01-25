use anyhow::Result;
use rust_embed::{EmbeddingPool, PoolConfig, ModelType};

#[test]
fn test_pool_single_worker() -> Result<()> {
    let config = PoolConfig {
        cpu_workers: 1,
        gpu_workers: 0,
        model: ModelType::MiniLM,
        cache_size_per_worker: 1000,
        routing_config: None,
    };

    let pool = EmbeddingPool::new(config)?;

    // Test single embedding
    let text = "This is a test sentence.".to_string();
    let embedding = pool.embed_text(text)?;

    // Check dimensions
    assert_eq!(embedding.len(), 384);

    // Check normalization (length should be close to 1.0)
    let norm = embedding.dot(&embedding).sqrt();
    assert!((norm - 1.0).abs() < 1e-5, "Embedding should be normalized");

    pool.shutdown()?;
    Ok(())
}

#[test]
fn test_pool_multiple_workers() -> Result<()> {
    let config = PoolConfig {
        cpu_workers: 4,
        gpu_workers: 0,
        model: ModelType::MiniLM,
        cache_size_per_worker: 1000,
        routing_config: None,
    };

    let pool = EmbeddingPool::new(config)?;
    assert_eq!(pool.worker_count(), 4);

    // Test single embedding
    let text = "Hello world".to_string();
    let embedding = pool.embed_text(text)?;

    assert_eq!(embedding.len(), 384);

    pool.shutdown()?;
    Ok(())
}

#[test]
fn test_pool_batch_embedding() -> Result<()> {
    let config = PoolConfig {
        cpu_workers: 2,
        gpu_workers: 0,
        model: ModelType::MiniLM,
        cache_size_per_worker: 1000,
    };

    let pool = EmbeddingPool::new(config)?;

    // Create test texts
    let texts: Vec<String> = vec![
        "The cat sat on the mat.".to_string(),
        "Dogs are loyal pets.".to_string(),
        "The weather is nice today.".to_string(),
        "I love programming in Rust.".to_string(),
    ];

    // Embed batch
    let embeddings = pool.embed_batch(texts)?;

    // Check results
    assert_eq!(embeddings.len(), 4);
    for embedding in &embeddings {
        assert_eq!(embedding.len(), 384);

        // Check normalization
        let norm = embedding.dot(embedding).sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    pool.shutdown()?;
    Ok(())
}

#[test]
fn test_pool_reconfiguration() -> Result<()> {
    let mut pool = EmbeddingPool::new(PoolConfig {
        cpu_workers: 2,
        gpu_workers: 0,
        model: ModelType::MiniLM,
        cache_size_per_worker: 1000,
        routing_config: None,
    })?;

    assert_eq!(pool.worker_count(), 2);

    // Scale up
    pool.reconfigure(PoolConfig {
        cpu_workers: 4,
        gpu_workers: 0,
        model: ModelType::MiniLM,
        cache_size_per_worker: 1000,
        routing_config: None,
    })?;

    assert_eq!(pool.worker_count(), 4);

    // Test that it still works
    let text = "Test after scale up".to_string();
    let embedding = pool.embed_text(text)?;
    assert_eq!(embedding.len(), 384);

    // Scale down
    pool.reconfigure(PoolConfig {
        cpu_workers: 1,
        gpu_workers: 0,
        model: ModelType::MiniLM,
        cache_size_per_worker: 1000,
        routing_config: None,
    })?;

    assert_eq!(pool.worker_count(), 1);

    // Test that it still works
    let text = "Test after scale down".to_string();
    let embedding = pool.embed_text(text)?;
    assert_eq!(embedding.len(), 384);

    pool.shutdown()?;
    Ok(())
}

#[test]
fn test_pool_statistics() -> Result<()> {
    let config = PoolConfig {
        cpu_workers: 2,
        gpu_workers: 0,
        model: ModelType::MiniLM,
        cache_size_per_worker: 1000,
    };

    let pool = EmbeddingPool::new(config)?;

    // Embed some texts
    let texts: Vec<String> = vec![
        "first text".to_string(),
        "second text".to_string(),
        "first text".to_string(), // Duplicate - should hit cache
    ];

    pool.embed_batch(texts)?;

    // Get stats
    let stats = pool.aggregate_stats()?;

    // Should have processed 3 embeddings
    assert_eq!(stats.embeddings_count, 3);

    // Should have at least 1 cache hit (the duplicate)
    assert!(stats.cache_hits > 0, "Should have cache hits for duplicate text");

    pool.shutdown()?;
    Ok(())
}

#[test]
fn test_pool_similarity() -> Result<()> {
    let config = PoolConfig::minimal();
    let pool = EmbeddingPool::new(config)?;

    let text1 = "Dogs are pets that bark.".to_string();
    let text2 = "Canines are domesticated animals.".to_string();
    let text3 = "Quantum physics is fascinating.".to_string();

    let emb1 = pool.embed_text(text1)?;
    let emb2 = pool.embed_text(text2)?;
    let emb3 = pool.embed_text(text3)?;

    // Calculate cosine similarities
    let sim12 = emb1.dot(&emb2);
    let sim13 = emb1.dot(&emb3);

    // Similar texts should have higher similarity than dissimilar texts
    assert!(
        sim12 > sim13,
        "Similar texts should have higher similarity (sim12={}, sim13={})",
        sim12,
        sim13
    );

    pool.shutdown()?;
    Ok(())
}

#[test]
fn test_pool_presets() -> Result<()> {
    // Test minimal preset
    let minimal = PoolConfig::minimal();
    assert_eq!(minimal.cpu_workers, 1);
    let pool = EmbeddingPool::new(minimal)?;
    assert_eq!(pool.worker_count(), 1);
    pool.shutdown()?;

    // Test balanced preset
    let balanced = PoolConfig::balanced();
    assert_eq!(balanced.cpu_workers, 4);
    let pool = EmbeddingPool::new(balanced)?;
    assert_eq!(pool.worker_count(), 4);
    pool.shutdown()?;

    // Test high throughput preset
    let high = PoolConfig::high_throughput();
    assert_eq!(high.cpu_workers, 8);
    let pool = EmbeddingPool::new(high)?;
    assert_eq!(pool.worker_count(), 8);
    pool.shutdown()?;

    Ok(())
}

#[test]
fn test_empty_batch() -> Result<()> {
    let config = PoolConfig::minimal();
    let pool = EmbeddingPool::new(config)?;

    let empty_texts: Vec<String> = vec![];
    let embeddings = pool.embed_batch(empty_texts)?;

    assert_eq!(embeddings.len(), 0);

    pool.shutdown()?;
    Ok(())
}
