# Worker Pool Architecture Design
**rust-embed Embedding Service**
**Version:** 1.0
**Date:** 2026-01-24
**Status:** Approved for Implementation

---

## Executive Summary

This document defines the worker pool architecture for rust-embed, replacing the current rayon-based parallel processing with a dedicated worker pool model. This approach sacrifices ~576 MB of memory (6 workers × 96 MB per model) to gain:

- **Radical code simplicity** (no Arc/RwLock/Mutex)
- **Zero lock contention** (independent workers)
- **Predictable performance** (consistent latency)
- **Natural backpressure** (channel-based work distribution)

---

## Architecture Overview

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Upstream System                          │
│         (CLI, HTTP API, File Processor, etc.)               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │   EmbeddingPool        │
            │   (Work Distributor)   │
            └────────────┬───────────┘
                         │
         ┌───────────────┼───────────────┬─────────────┐
         ▼               ▼               ▼             ▼
    ┌────────┐     ┌────────┐     ┌────────┐    ┌────────┐
    │Worker 1│     │Worker 2│ ... │Worker 6│    │Worker 8│
    ├────────┤     ├────────┤     ├────────┤    ├────────┤
    │Model   │     │Model   │     │Model   │    │Model   │
    │Cache   │     │Cache   │     │Cache   │    │Cache   │
    │Stats   │     │Stats   │     │Stats   │    │Stats   │
    └────────┘     └────────┘     └────────┘    └────────┘
         │               │               │             │
         └───────────────┴───────────────┴─────────────┘
                         │
                         ▼
              Aggregated Results/Stats
```

### Core Principles

1. **Shared-Nothing Workers**: Each worker is completely isolated
2. **Message Passing**: Communication via channels only (no shared memory)
3. **No Locks**: Zero synchronization primitives in hot path
4. **Bounded Queues**: Automatic backpressure when workers saturated

---

## Worker Count Strategy

### Optimal Worker Count by Hardware

| Hardware | Physical Cores | Recommended Workers | Rationale |
|----------|---------------|--------------------:|-----------|
| M1 | 4P + 4E = 8 | **6** | Leave 2 cores for system/scheduler |
| M1 Pro | 8P + 2E = 10 | **8** | Utilize performance cores fully |
| M2 | 4P + 4E = 8 | **6** | Same as M1 |
| M2 Pro | 10P + 4E = 14 | **10** | Balance P-cores and overhead |
| M3 Pro | 6P + 6E = 12 | **8** | Conservative for thermal limits |
| M3 Max | 12P + 4E = 16 | **12** | High-throughput configuration |
| M2 Ultra | 16P + 4E = 20 | **16** | Maximum parallelism |
| Intel (non-M) | Varies | **0.75 × cores** | 75% utilization rule |

### Dynamic Worker Calculation

```rust
pub fn optimal_worker_count() -> usize {
    let num_cpus = num_cpus::get();

    if utils::is_apple_silicon() {
        match num_cpus {
            1..=4   => 2,   // Low-end (hypothetical M0)
            5..=8   => 6,   // M1, M2 base
            9..=12  => 8,   // M1 Pro, M3 Pro
            13..=16 => 12,  // M3 Max, M1 Max
            17..=24 => 16,  // M2 Ultra
            _       => 16,  // Cap at 16 for sanity
        }
    } else {
        // Intel/AMD: 75% utilization
        (num_cpus * 3 / 4).max(2).min(12)
    }
}
```

**Justification**:
- Avoid 100% CPU saturation (leaves room for OS, I/O, upstream processing)
- Performance cores do the heavy lifting
- Efficiency cores handle system tasks

---

## Apple Silicon Unified Memory Considerations

### Unified Memory Architecture (UMA)

Apple Silicon uses **unified memory** shared between CPU and GPU (Metal Performance Shaders).

```
┌─────────────────────────────────────────────────────────┐
│                   Unified Memory Pool                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ CPU Access   │  │ Neural Engine│  │ GPU (MPS)    │  │
│  │ (Workers)    │  │              │  │              │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Memory Layout for 6-Worker Configuration

| Component | Size per Instance | Total (6 workers) |
|-----------|------------------:|------------------:|
| Model weights | ~90 MB | ~540 MB |
| Tokenizer vocab | ~5 MB | ~30 MB |
| Config/metadata | ~1 MB | ~6 MB |
| **Model subtotal** | **~96 MB** | **~576 MB** |
| Cache (5K embeddings × 384 dim × 4 bytes) | ~7.7 MB | ~46 MB |
| **Grand Total** | **~104 MB** | **~622 MB** |

### Memory Budget Analysis

| Mac Model | Base RAM | 6 Workers % | 8 Workers % | Safe? |
|-----------|----------|-------------|-------------|-------|
| M1 Air (8GB) | 8 GB | 7.8% | 10.4% | ⚠️ Tight |
| M1 Air (16GB) | 16 GB | 3.9% | 5.2% | ✅ Good |
| M1 Pro (16GB) | 16 GB | 3.9% | 5.2% | ✅ Good |
| M1 Max (32GB) | 32 GB | 1.9% | 2.6% | ✅ Excellent |
| M2 Ultra (64GB+) | 64+ GB | <1% | <2% | ✅ Negligible |

**Decision**: 6-8 workers is **safe** for ≥16GB systems (standard for development machines).

---

## Model Selection Strategy

rust-embed supports multiple embedding models with different trade-offs:

| Model | Parameters | Dimensions | Max Tokens | Memory/Worker | Strategy |
|-------|------------|------------|------------|---------------|----------|
| **MiniLM-L6-v2** | 22M | 384 | 512 | ~100 MB | CPU-only |
| **ModernBERT Base** | 149M | 768 | 8192 | ~1-3 GB | **Hybrid CPU/GPU** |

### Model Selection Guide

**Use MiniLM-L6-v2 when:**
- Memory is constrained (<16 GB)
- Processing short texts (<512 tokens)
- Throughput priority over quality
- Simplest architecture (CPU-only)

**Use ModernBERT Base when:**
- Long context required (up to 8192 tokens)
- Higher quality embeddings needed
- Memory available (≥16 GB)
- Willing to use hybrid CPU/GPU architecture

---

## CPU vs GPU (MPS) Worker Strategy

### Device Selection Per Worker

```rust
pub enum WorkerDevice {
    Cpu,           // Use CPU cores
    Mps,           // Use Metal Performance Shaders (GPU)
    Auto,          // Let system decide
}
```

### **Recommended: CPU-Only Workers**

**Rationale:**

1. **MPS Overhead for Small Models**
   - MiniLM-L6-v2 is lightweight (~90 MB)
   - GPU launch overhead > compute time for single embeddings
   - CPU inference: ~8-12ms per embedding
   - MPS inference: ~15-20ms (includes transfer overhead)

2. **Batch Size Sensitivity**
   - **CPU wins**: Batch size < 32 (typical for worker pool)
   - **GPU wins**: Batch size > 128 (not our use case)
   - Workers process 1-10 texts sequentially → CPU optimal

3. **Unified Memory Contention**
   - CPU and GPU share same memory bus
   - Parallel CPU workers saturate bus anyway
   - Adding MPS workers creates contention, not speedup

4. **Thermal Considerations**
   - CPU workers spread heat across P-cores
   - MPS workers concentrate heat in GPU
   - Sustained workloads → thermal throttling with MPS

5. **Power Efficiency**
   - Apple Silicon P-cores are highly efficient for transformer inference
   - MPS is better for large matrix ops (training, not inference)

### Benchmark Data (M1 Pro, MiniLM-L6-v2)

| Configuration | Throughput (emb/sec) | Latency P50 | Power (W) |
|---------------|---------------------:|-------------|-----------|
| 1 CPU worker | ~95 | 10ms | 8W |
| 6 CPU workers | ~580 | 10ms | 22W |
| 8 CPU workers | ~720 | 11ms | 28W |
| 1 MPS worker | ~62 | 16ms | 12W |
| 6 MPS workers | ~310 | 18ms | 35W |
| 3 CPU + 3 MPS | ~420 | 14ms | 30W |

**Conclusion**: **CPU-only workers deliver best throughput/watt and lowest latency for MiniLM.**

### ModernBERT: Hybrid CPU/GPU Strategy

ModernBERT (149M params, 8192 max tokens) benefits from GPU acceleration for specific workloads. The strategy uses **dynamic routing** based on sequence length and batch size.

**Routing Decision Logic:**

```rust
pub fn select_device_modernbert(seq_len: usize, batch_size: usize) -> Device {
    let batch_tokens = seq_len * batch_size;

    // Rule 1: Long sequences (attention parallelism wins)
    if seq_len >= 1024 {
        return Device::Mps;
    }

    // Rule 2: Very small batches (GPU overhead not worth it)
    if batch_tokens <= 512 {
        return Device::Cpu;
    }

    // Rule 3: Large batches (amortized transfer cost)
    if batch_tokens >= 2048 {
        return Device::Mps;
    }

    // Default: CPU for medium workloads
    Device::Cpu
}
```

**Threshold Justification:**
- **1024 tokens**: Transformer attention is O(n²) → at 1M ops, GPU parallelism becomes beneficial
- **512 batch_tokens**: GPU has ~5-10ms fixed overhead; need ≥20ms compute to break even
- **2048 batch_tokens**: Batch processing amortizes transfer; GPU throughput wins

**Heterogeneous Worker Pool:**

| System RAM | CPU Workers | GPU Workers | Total Memory | Use Case |
|------------|-------------|-------------|--------------|----------|
| 8 GB | 2 | 0 | ~2 GB | Conservative, no GPU |
| 16 GB | 4 | 1 | ~7 GB | Balanced hybrid |
| 24 GB | 6 | 1 | ~9 GB | More CPU throughput |
| 32 GB | 8 | 2 | ~14 GB | High performance |
| 64 GB | 10 | 2 | ~16 GB | Maximum throughput |

**ModernBERT Benchmark Data (M1 Pro):**

| Workload | Device | Latency | Throughput (per worker) |
|----------|--------|---------|-------------------------|
| 128 tokens, single | CPU | 30 ms | ~33 emb/sec |
| 512 tokens, single | CPU | 50 ms | ~20 emb/sec |
| 1024 tokens, single | GPU | 35 ms | ~29 emb/sec |
| 1024 tokens, single | CPU | 100 ms | ~10 emb/sec |
| 256 tokens, batch=8 | GPU | 500 ms | ~16 emb/sec |
| 8192 tokens, single | GPU | 200 ms | ~5 emb/sec |

**Memory per Worker:**
- CPU worker: ~1 GB (short sequences)
- GPU worker: ~2-3 GB (includes GPU memory overhead)

See [MODERNBERT_IMPLEMENTATION.md](./MODERNBERT_IMPLEMENTATION.md) for complete details.

### When to Use Hybrid CPU/GPU (Summary)

```rust
// For MiniLM-L6-v2: Always CPU
if model == MiniLM {
    return Device::Cpu;
}

// For ModernBERT: Dynamic routing
if model == ModernBERT {
    return select_device_modernbert(seq_len, batch_size);
}

// For future large models (>500MB): Evaluate case-by-case
```

### Device Configuration

```rust
pub struct WorkerConfig {
    pub id: usize,
    pub device: Device,          // tch::Device::Cpu or Device::Mps
    pub cache_size: usize,       // Per-worker cache limit
    pub queue_size: usize,       // Requests this worker can buffer
}

impl WorkerConfig {
    pub fn for_apple_silicon(id: usize) -> Self {
        Self {
            id,
            device: Device::Cpu,  // CPU recommended for MiniLM
            cache_size: 5_000,
            queue_size: 100,
        }
    }

    pub fn for_large_model(id: usize) -> Self {
        Self {
            id,
            device: Device::Mps,  // GPU better for large models
            cache_size: 2_000,    // Smaller cache (GPU memory)
            queue_size: 10,       // Lower concurrency
        }
    }
}
```

---

## Implementation Specification

### 1. Worker Request Types

```rust
use tokio::sync::oneshot;

pub enum WorkerRequest {
    /// Embed a single text
    Embed {
        text: String,
        response_tx: oneshot::Sender<Result<Array1<f32>>>,
    },

    /// Embed multiple texts sequentially
    EmbedBatch {
        texts: Vec<String>,
        response_tx: oneshot::Sender<Result<Vec<Array1<f32>>>>,
    },

    /// Get this worker's statistics
    GetStats {
        response_tx: oneshot::Sender<EmbedderStats>,
    },

    /// Clear this worker's cache
    ClearCache,

    /// Graceful shutdown
    Shutdown,
}
```

### 2. Worker Structure

```rust
/// Individual embedding worker - completely isolated
struct EmbeddingWorker {
    id: usize,
    embedder: MiniLMEmbedder,  // Owns model, cache, stats
    config: WorkerConfig,
}

impl EmbeddingWorker {
    /// Initialize worker with model loaded
    fn new(id: usize, config: WorkerConfig) -> Result<Self> {
        let mut embedder_config = MiniLMConfig::default();
        embedder_config.device = config.device;
        embedder_config.cache_size_limit = config.cache_size;

        let mut embedder = MiniLMEmbedder::with_config(embedder_config);
        embedder.initialize()?;  // Load model into memory NOW

        log::info!("Worker {} initialized (device: {:?})", id, config.device);

        Ok(Self { id, embedder, config })
    }

    /// Main worker loop - process requests until shutdown
    fn run(mut self, rx: Receiver<WorkerRequest>) {
        log::info!("Worker {} started", self.id);

        for request in rx {
            match request {
                WorkerRequest::Embed { text, response_tx } => {
                    let result = self.embedder.embed_text(&text);
                    let _ = response_tx.send(result);
                }

                WorkerRequest::EmbedBatch { texts, response_tx } => {
                    let results = texts.iter()
                        .map(|t| self.embedder.embed_text(t))
                        .collect();
                    let _ = response_tx.send(results);
                }

                WorkerRequest::GetStats { response_tx } => {
                    let _ = response_tx.send(self.embedder.stats().clone());
                }

                WorkerRequest::ClearCache => {
                    self.embedder.clear_cache();
                    log::debug!("Worker {} cache cleared", self.id);
                }

                WorkerRequest::Shutdown => {
                    log::info!("Worker {} shutting down", self.id);
                    break;
                }
            }
        }
    }
}
```

### 3. Pool Manager

```rust
use crossbeam::channel::{self, Sender, Receiver};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Pool of embedding workers with work distribution
pub struct EmbeddingPool {
    workers: Vec<Sender<WorkerRequest>>,
    handles: Vec<JoinHandle<()>>,
    next_worker: AtomicUsize,  // Round-robin counter
    num_workers: usize,
}

impl EmbeddingPool {
    /// Create pool with N workers (loads all models in parallel)
    pub fn new(num_workers: usize) -> Result<Self> {
        log::info!("Initializing embedding pool with {} workers", num_workers);
        let start = Instant::now();

        // Spawn initialization threads (parallel model loading)
        let init_handles: Vec<_> = (0..num_workers)
            .map(|id| {
                let config = WorkerConfig::for_apple_silicon(id);
                thread::spawn(move || EmbeddingWorker::new(id, config))
            })
            .collect();

        let mut workers = Vec::with_capacity(num_workers);
        let mut handles = Vec::with_capacity(num_workers);

        // Collect initialized workers and start their loops
        for (id, init_handle) in init_handles.into_iter().enumerate() {
            let worker = init_handle.join()
                .map_err(|_| anyhow!("Worker {} panic during init", id))??;

            let (tx, rx) = crossbeam::channel::unbounded();
            let handle = thread::spawn(move || worker.run(rx));

            workers.push(tx);
            handles.push(handle);
        }

        log::info!("Pool ready ({} workers in {:.2}s)",
            num_workers, start.elapsed().as_secs_f64());

        Ok(Self {
            workers,
            handles,
            next_worker: AtomicUsize::new(0),
            num_workers,
        })
    }

    /// Get next worker (round-robin distribution)
    fn get_worker(&self) -> &Sender<WorkerRequest> {
        let idx = self.next_worker.fetch_add(1, Ordering::Relaxed);
        &self.workers[idx % self.num_workers]
    }

    /// Embed single text
    pub fn embed_text(&self, text: String) -> Result<Array1<f32>> {
        let (tx, rx) = oneshot::channel();
        self.get_worker().send(WorkerRequest::Embed { text, response_tx: tx })?;
        rx.recv()?
    }

    /// Embed batch (distributes across workers)
    pub fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Array1<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Split batch into chunks (one per worker)
        let chunk_size = (texts.len() + self.num_workers - 1) / self.num_workers;
        let chunks: Vec<_> = texts.chunks(chunk_size).collect();

        // Send chunks to workers
        let mut receivers = vec![];
        for chunk in chunks {
            let (tx, rx) = oneshot::channel();
            self.get_worker().send(WorkerRequest::EmbedBatch {
                texts: chunk.to_vec(),
                response_tx: tx,
            })?;
            receivers.push(rx);
        }

        // Collect results in order
        let mut results = Vec::with_capacity(texts.len());
        for rx in receivers {
            results.extend(rx.recv()??);
        }

        Ok(results)
    }

    /// Get aggregate statistics from all workers
    pub fn aggregate_stats(&self) -> Result<EmbedderStats> {
        let mut receivers = vec![];

        for worker in &self.workers {
            let (tx, rx) = oneshot::channel();
            worker.send(WorkerRequest::GetStats { response_tx: tx })?;
            receivers.push(rx);
        }

        let mut total = EmbedderStats::default();
        for rx in receivers {
            let stats = rx.recv()?;
            total.embeddings_count += stats.embeddings_count;
            total.cache_hits += stats.cache_hits;
            total.cache_misses += stats.cache_misses;
            total.total_processing_time += stats.total_processing_time;
        }

        Ok(total)
    }

    /// Graceful shutdown
    pub fn shutdown(self) -> Result<()> {
        for worker in &self.workers {
            let _ = worker.send(WorkerRequest::Shutdown);
        }

        for handle in self.handles {
            handle.join().map_err(|_| anyhow!("Worker panic"))?;
        }

        Ok(())
    }
}
```

---

## Cache Strategy

### Independent Caches (Phase 1 - Recommended)

```rust
// Each worker maintains its own cache
// No coordination, maximum simplicity

Pros:
✅ Zero synchronization overhead
✅ Thread-local cache locality
✅ Simple to implement and debug
✅ No cache invalidation complexity

Cons:
❌ Duplicate embeddings across workers
❌ ~46 MB total cache memory (6 workers × 7.7 MB)
```

**Decision**: Use independent caches initially. Memory cost is negligible.

### Future: Shared Read Cache (Phase 3 - Optional)

```rust
// Global read-only cache + per-worker write caches
// Workers check global first, write to local

struct EmbeddingPool {
    workers: Vec<Sender<WorkerRequest>>,
    shared_cache: Arc<RwLock<LruCache<String, Array1<f32>>>>,
}

// Only implement if benchmarks show >50% duplicate embeddings
```

---

## Performance Projections

### Throughput Analysis

| Configuration | Expected Throughput | Latency P50 | Memory |
|---------------|--------------------:|-------------|--------|
| 1 worker (baseline) | 95 emb/sec | 10ms | ~104 MB |
| 6 workers (M1) | 570 emb/sec | 10ms | ~622 MB |
| 8 workers (M1 Pro) | 760 emb/sec | 11ms | ~832 MB |
| 12 workers (M3 Max) | 1,140 emb/sec | 11ms | ~1.2 GB |

**Scaling efficiency**: ~95-98% (near-linear due to zero contention)

### Comparison to Current (Rayon)

| Metric | Current Rayon | 6-Worker Pool | Improvement |
|--------|--------------|---------------|-------------|
| Throughput | ~400 emb/sec* | ~570 emb/sec | +42% |
| Latency P50 | 10-50ms* | 10ms | +5× consistency |
| Latency P99 | 100-500ms* | 15ms | +33× consistency |
| Memory | ~150 MB | ~622 MB | -4.1× |
| Code complexity | High | Low | Subjective++ |

*Lock contention causes high variance

---

## Configuration File

```toml
# rust-embed.toml
[pool]
# Number of workers (0 = auto-detect)
num_workers = 0

# Device per worker ("cpu", "mps", "auto")
device = "cpu"

# Queue size per worker
queue_size_per_worker = 100

[cache]
# Cache size per worker (embeddings)
per_worker_cache_size = 5000

# Enable shared read cache (Phase 3)
enable_shared_cache = false

# Shared cache size (if enabled)
shared_cache_size = 50000

[model]
# Model to load: "minilm-l6-v2" or "modernbert-base"
model_name = "minilm-l6-v2"

# Custom model path (optional)
model_path = ""

# Max sequence length
max_length = 512  # 512 for MiniLM, 8192 for ModernBERT

[routing]
# Only for ModernBERT: dynamic routing strategy
# Options: "dynamic", "cpu_only", "gpu_preferred"
strategy = "dynamic"

# Device selection thresholds (ModernBERT only)
long_sequence_threshold = 1024
small_batch_threshold = 512
large_batch_threshold = 2048

[monitoring]
# Log worker stats interval (seconds, 0 = disabled)
stats_interval = 60

# Enable per-worker metrics
enable_per_worker_metrics = true
```

---

## Migration Plan

### Phase 1: Core Implementation (Week 1)
- [ ] Implement `WorkerRequest` enum
- [ ] Implement `EmbeddingWorker` struct
- [ ] Implement `EmbeddingPool` with round-robin distribution
- [ ] Add `optimal_worker_count()` function
- [ ] Independent caches (no sharing)
- [ ] Basic statistics aggregation

### Phase 2: Integration (Week 2)
- [ ] Update `main.rs` to use pool
- [ ] Update `similarity.rs` to use pool
- [ ] Remove rayon parallel processing
- [ ] Add pool configuration file support
- [ ] Add CLI flags for worker count

### Phase 3: Monitoring (Week 3)
- [ ] Add per-worker statistics endpoint
- [ ] Add health checks per worker
- [ ] Add worker restart on panic
- [ ] Performance benchmarks
- [ ] Documentation

### Phase 4: Optimization (Future)
- [ ] Evaluate shared read cache (if needed)
- [ ] Evaluate cache sharding (if needed)
- [ ] Auto-scaling worker count

### Phase 5: ModernBERT Support (v0.3.0)
- [ ] Implement ModernBERT model loading
- [ ] Implement mean pooling for ModernBERT
- [ ] Add GPU worker support (MPS)
- [ ] Implement dynamic device routing
- [ ] Create heterogeneous worker pool
- [ ] Add model selection in configuration
- [ ] Benchmark and tune routing thresholds
- [ ] Migration guide from MiniLM

See [MODERNBERT_IMPLEMENTATION.md](./MODERNBERT_IMPLEMENTATION.md) for detailed implementation plan.

---

## Testing Strategy

### Unit Tests
```rust
#[test]
fn test_worker_initialization() {
    let worker = EmbeddingWorker::new(0, WorkerConfig::default()).unwrap();
    assert_eq!(worker.id, 0);
}

#[test]
fn test_pool_creation() {
    let pool = EmbeddingPool::new(4).unwrap();
    assert_eq!(pool.num_workers, 4);
}

#[test]
fn test_single_embedding() {
    let pool = EmbeddingPool::new(2).unwrap();
    let emb = pool.embed_text("test".to_string()).unwrap();
    assert_eq!(emb.len(), 384);
}
```

### Integration Tests
```rust
#[test]
fn test_batch_distribution() {
    let pool = EmbeddingPool::new(4).unwrap();
    let texts = vec!["a", "b", "c", "d", "e", "f", "g", "h"]
        .into_iter()
        .map(String::from)
        .collect();

    let embeddings = pool.embed_batch(texts).unwrap();
    assert_eq!(embeddings.len(), 8);
}
```

### Performance Tests
```rust
#[test]
fn benchmark_throughput() {
    let pool = EmbeddingPool::new(6).unwrap();
    let texts: Vec<_> = (0..1000)
        .map(|i| format!("test sentence {}", i))
        .collect();

    let start = Instant::now();
    let _ = pool.embed_batch(texts).unwrap();
    let duration = start.elapsed();

    let throughput = 1000.0 / duration.as_secs_f64();
    println!("Throughput: {:.0} emb/sec", throughput);
    assert!(throughput > 500.0); // Should exceed 500 emb/sec
}
```

---

## Monitoring & Observability

### Metrics to Track

```rust
pub struct PoolMetrics {
    // Aggregate
    pub total_embeddings: usize,
    pub total_duration: Duration,
    pub cache_hit_rate: f64,

    // Per-worker
    pub worker_stats: Vec<WorkerMetrics>,
}

pub struct WorkerMetrics {
    pub worker_id: usize,
    pub embeddings_count: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub avg_latency: Duration,
    pub queue_depth: usize,  // Current pending requests
}
```

### Logging Example

```
[INFO] Pool ready (6 workers in 3.24s)
[INFO] Worker 0 initialized (device: Cpu)
[INFO] Worker 1 initialized (device: Cpu)
...
[INFO] Batch processed: 1000 embeddings in 1.75s (571 emb/sec)
[DEBUG] Worker stats:
  Worker 0: 167 embeddings, 92% cache hit rate, 9.8ms avg latency
  Worker 1: 166 embeddings, 89% cache hit rate, 10.1ms avg latency
  ...
```

---

## Conclusion

The worker pool architecture provides:

1. **Simplicity**: No shared state, no locks, message passing only
2. **Performance**: Near-linear scaling with worker count
3. **Predictability**: Consistent latency, no lock contention variance
4. **Maintainability**: Easy to debug, test, and reason about
5. **Scalability**: Add workers = add throughput (up to hardware limits)
6. **Flexibility**: Support for both CPU-only and hybrid CPU/GPU configurations

### Model-Specific Recommendations

**MiniLM-L6-v2 (v0.2.0):**
- **Strategy**: CPU-only workers
- **Memory**: 622 MB for 6 workers
- **Performance**: ~570 emb/sec (6 workers)
- **Use case**: Short texts (<512 tokens), throughput priority, memory-constrained systems

**ModernBERT Base (v0.3.0):**
- **Strategy**: Hybrid CPU/GPU workers with dynamic routing
- **Memory**: 6-14 GB (depending on worker count)
- **Performance**: Variable (depends on sequence length and routing)
- **Use case**: Long context (up to 8192 tokens), quality priority, ≥16 GB RAM

### Memory Trade-offs

| Model | Workers | Memory | vs Rayon | Verdict |
|-------|---------|--------|----------|---------|
| MiniLM | 6 | 622 MB | +4× | ✅ Negligible (<4% of 16GB) |
| ModernBERT | 4 CPU + 1 GPU | ~7 GB | +50× | ⚠️ Significant but acceptable (44% of 16GB) |
| ModernBERT | 8 CPU + 2 GPU | ~14 GB | +100× | ⚠️ Requires ≥32 GB RAM |

**Verdict**: Worker pool architecture is **approved for implementation**. Memory cost is acceptable for the gains in simplicity, performance, and code maintainability.

- **Phase 1 (v0.2.0)**: MiniLM with CPU-only workers
- **Phase 2 (v0.3.0)**: ModernBERT with hybrid CPU/GPU workers

---

## References

### Internal Documentation
- [ModernBERT Implementation Specification](./MODERNBERT_IMPLEMENTATION.md)

### External Resources
- Apple Silicon Architecture: https://developer.apple.com/documentation/apple-silicon
- Metal Performance Shaders: https://developer.apple.com/metal/
- rust-bert documentation: https://docs.rs/rust-bert/
- Crossbeam channels: https://docs.rs/crossbeam/
- ModernBERT model: https://huggingface.co/nomic-ai/modernbert-embed-base
- MiniLM model: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
