# Caching Strategy in rust-embed

## Overview

rust-embed includes an **optional** embedding cache to avoid recomputing embeddings for duplicate texts. However, **for most document embedding pipelines, caching provides minimal value** and can be safely disabled.

## What the Cache Does

Each worker maintains a `HashMap<String, Array1<f32>>` that stores:
- **Key**: Input text (String)
- **Value**: Computed embedding (384 floats for MiniLM, 768 for ModernBERT)

When the same text is embedded again:
```rust
// First call: cache miss, compute embedding (~10ms)
let emb1 = pool.embed_text("Hello world".to_string())?;

// Second call: cache hit, instant retrieval (<1ms)
let emb2 = pool.embed_text("Hello world".to_string())?;
```

### Cache Behavior

- **cache_size_per_worker: 0** → Caching disabled entirely
- **cache_size_per_worker: 100** → Cache up to 100 unique texts
- **cache_size_per_worker: 10000** → Cache up to 10,000 unique texts

When the cache exceeds the limit, the **oldest entry** is evicted (simple FIFO, not true LRU).

### Memory Usage

**Per cache entry (MiniLM):**
- Text: ~50-200 bytes
- Embedding: 384 floats × 4 bytes = 1,536 bytes
- HashMap overhead: ~40 bytes
- **Total**: ~1,600-1,800 bytes per entry

**Total memory per worker:**
- `cache_size_per_worker × 1.6 KB`
- 100 entries = ~160 KB
- 1,000 entries = ~1.6 MB
- 10,000 entries = ~16 MB

## When to Use Caching

### ✅ High Cache Value (cache_size_per_worker: 10000+)

**Search query systems:**
```
Users search "weather forecast" 1,000 times
→ 1 computation + 999 cache hits
→ 99.9% hit rate, massive speedup
```

**Recommendation systems:**
```
Same product descriptions embedded repeatedly
→ High repetition, excellent cache benefit
```

**Real-time APIs:**
```
Common phrases like "hello", "yes", "no" appear frequently
→ Moderate hit rate, worthwhile optimization
```

### ⚠️ Low Cache Value (cache_size_per_worker: 100-500)

**Mixed workloads:**
```
Some repeated queries, mostly unique documents
→ Small cache to catch accidental duplicates
→ Minimal memory overhead
```

### ❌ No Cache Value (cache_size_per_worker: 0)

**Document embedding pipelines (PRIMARY USE CASE FOR RUST-EMBED):**
```
Upstream software sends unique messages → rust-embed embeds them
→ Each message is unique, no repetition
→ Cache hit rate: ~0%
→ Cache is just wasting memory
```

**Stream processing:**
```
Embedding unique events, logs, chat messages
→ No repetition, disable cache
```

**Batch ETL:**
```
One-time embedding of datasets
→ Documents processed once, disable cache
```

## Configuration Examples

### For Document Embedding (Recommended Default)

```rust
let config = PoolConfig {
    cpu_workers: 4,
    cache_size_per_worker: 0,  // Disable cache for unique messages
    model: ModelType::MiniLM,
    // ...
};
```

**CLI:**
```bash
rust-embed --file messages.txt --cache-size 0 --workers 4
```

### For Catching Accidental Duplicates

```rust
let config = PoolConfig {
    cpu_workers: 4,
    cache_size_per_worker: 100,  // Small cache to catch bugs upstream
    model: ModelType::MiniLM,
    // ...
};
```

### For Search Query Systems

```rust
let config = PoolConfig::search_optimized();  // 10,000 cache per worker
```

**CLI:**
```bash
rust-embed --file queries.txt --cache-size 10000 --workers 4
```

## Monitoring Cache Performance

```rust
let stats = pool.aggregate_stats()?;

println!("Total embeddings: {}", stats.embeddings_count);
println!("Cache hits: {}", stats.cache_hits);
println!("Cache misses: {}", stats.cache_misses);

let hit_rate = (stats.cache_hits as f64 / stats.embeddings_count as f64) * 100.0;
println!("Cache hit rate: {:.1}%", hit_rate);
```

**Interpretation:**
- **Hit rate > 50%**: Cache is working well, appropriate size
- **Hit rate 20-50%**: Moderate benefit, consider adjusting size
- **Hit rate < 20%**: Consider reducing cache size or disabling
- **Hit rate < 5%**: **Disable cache** (wasting memory)

## Preset Configurations

| Preset | Workers | Cache Size | Use Case |
|--------|---------|------------|----------|
| `minimal()` | 1 | 0 | Low-throughput unique messages |
| `balanced()` | 4 | 100 | Document pipelines with duplicate detection |
| `high_throughput()` | 8 | 100 | High-volume document pipelines |
| `search_optimized()` | 4 | 10,000 | Search query systems |

## Best Practices

### 1. Start with Cache Disabled

For document embedding pipelines (rust-embed's primary use case):
```rust
cache_size_per_worker: 0
```

### 2. Monitor Hit Rate in Production

Run a test batch and check cache hit rate:
```bash
rust-embed --file sample.txt --cache-size 1000 --workers 4 --verbose
# Check logs for cache hit rate
```

### 3. Adjust Based on Workload

- **Hit rate < 5%**: Disable cache (`cache_size_per_worker: 0`)
- **Hit rate 5-20%**: Keep small cache (`cache_size_per_worker: 100-500`)
- **Hit rate > 50%**: Increase cache (`cache_size_per_worker: 5000-10000`)

### 4. Per-Worker Cache Sizing

Total cache memory = `workers × cache_size_per_worker × 1.6 KB`

Example: 8 workers, 10,000 cache per worker:
```
8 × 10,000 × 1.6 KB = 128 MB total cache memory
```

Ensure this fits your memory budget alongside model memory (~100 MB per worker).

## Design Rationale

### Why Independent Per-Worker Caches?

Each worker has its own cache (no shared state):
- ✅ **Zero lock contention**: Workers never block each other
- ✅ **Lock-free performance**: No mutex overhead
- ✅ **Simple design**: No complex synchronization
- ⚠️ **Cache fragmentation**: Same text may be cached in multiple workers

For write-heavy workloads (which embedding pipelines are), **independent caches are optimal** despite potential duplication.

### Why Not Shared Cache?

A shared cache across workers would require:
- ❌ **Mutex/RwLock**: High contention on every lookup
- ❌ **Performance bottleneck**: Workers block each other
- ❌ **Complex implementation**: Requires careful synchronization
- ✅ **No duplication**: Same text cached once

For **rust-embed's unique message workload**, cache duplication is irrelevant (hit rate is ~0% anyway).

### Why Simple FIFO Eviction?

Current eviction: Remove oldest entry when limit exceeded

**Alternative: True LRU (Least Recently Used)**
- ✅ Better eviction policy for real LRU behavior
- ❌ Requires additional tracking (access timestamps or linked list)
- ❌ More complex implementation

**Decision**: FIFO is sufficient because:
1. For document pipelines: Cache disabled or minimal (eviction rare)
2. For search systems: Most queries are recent anyway (FIFO ≈ LRU)
3. Simplicity > marginal LRU improvement

## Summary

**For rust-embed's primary use case (embedding unique messages):**
- **Recommended**: `cache_size_per_worker: 0` (disabled)
- **Alternative**: `cache_size_per_worker: 100` (catch duplicates)
- **Avoid**: `cache_size_per_worker: 5000+` (wasting memory)

**For search query systems (non-primary use case):**
- **Recommended**: `cache_size_per_worker: 10000+`
- Use `PoolConfig::search_optimized()` preset

**Key insight**: Caching is powerful for **repeated queries**, but rust-embed's primary use case is **unique message embedding**, where caching provides minimal value.
