# ModernBERT Embedding Model Implementation
**rust-embed v0.2.1 / v0.3.0**
**Model:** nomic-ai/modernbert-embed-base
**Date:** 2026-01-24
**Status:** Planning

---

## Executive Summary

This document specifies the implementation of **ModernBERT Base** as a **selectable model option** in rust-embed, available alongside MiniLM-L6-v2. **Both models will be available** and callers can choose which to use via `PoolConfig.model`.

ModernBERT offers:

- **7× more parameters** (149M vs 22M)
- **2× embedding dimensions** (768 vs 384)
- **16× longer context** (8192 tokens vs 512)
- **Dynamic CPU/GPU routing** based on workload characteristics

**Key Design Decision**: ModernBERT is NOT a replacement for MiniLM. Both models are available simultaneously, and the caller selects which to use based on their requirements:

```rust
// Caller chooses MiniLM for speed
let pool = EmbeddingPool::new(PoolConfig {
    cpu_workers: 6,
    gpu_workers: 0,
    model: ModelType::MiniLM,
    cache_size_per_worker: 5000,
})?;

// OR caller chooses ModernBERT for quality/long context
let pool = EmbeddingPool::new(PoolConfig {
    cpu_workers: 4,
    gpu_workers: 1,
    model: ModelType::ModernBERT,
    cache_size_per_worker: 3000,
})?;
```

Implementation requires heterogeneous worker pools (mixed CPU and GPU workers) and intelligent device selection.

---

## Model Coexistence Strategy

### Why Both Models?

Different use cases have different requirements:

| Use Case | Optimal Model | Rationale |
|----------|---------------|-----------|
| **Short queries** (<100 tokens) | MiniLM | 3× faster, 6× less memory |
| **Long documents** (1000-8000 tokens) | ModernBERT | Only option (MiniLM limited to 512) |
| **High throughput** (millions of texts) | MiniLM | Can run 6× more workers in same RAM |
| **High quality** (semantic search) | ModernBERT | Larger model, better representations |
| **Memory constrained** (<16 GB) | MiniLM | ~600 MB for 6 workers |
| **Memory available** (≥32 GB) | ModernBERT | Can use hybrid CPU/GPU |

### Model Selection at Configuration Time

```rust
pub enum ModelType {
    /// MiniLM-L6-v2: Fast, efficient, 384 dims, 512 tokens max
    MiniLM,

    /// ModernBERT Base: High quality, 768 dims, 8192 tokens max
    ModernBERT,
}

// Caller specifies in PoolConfig
pub struct PoolConfig {
    pub cpu_workers: usize,
    pub gpu_workers: usize,
    pub model: ModelType,              // ← Model selection
    pub cache_size_per_worker: usize,
}
```

### Switching Models

**Cannot switch during reconfigure**: Model change requires loading new weights (600 MB vs 90 MB), initializing new tokenizer, and clearing caches. Must create new pool:

```rust
// Correct: Create new pool for different model
let pool_minilm = EmbeddingPool::new(PoolConfig {
    cpu_workers: 6,
    gpu_workers: 0,
    model: ModelType::MiniLM,
    cache_size_per_worker: 5000,
})?;

// Process with MiniLM
pool_minilm.embed_batch(texts)?;
pool_minilm.shutdown()?;

// Switch to ModernBERT
let pool_modernbert = EmbeddingPool::new(PoolConfig {
    cpu_workers: 4,
    gpu_workers: 1,
    model: ModelType::ModernBERT,
    cache_size_per_worker: 3000,
})?;

// Process with ModernBERT
pool_modernbert.embed_batch(texts)?;
pool_modernbert.shutdown()?;

// Incorrect: Cannot use reconfigure to switch models
// pool.reconfigure(config_with_different_model)?; // ← Will error
```

---

## Model Specifications

### Model Configuration

| Parameter | Value | vs MiniLM-L6-v2 |
|-----------|-------|-----------------|
| **Model ID** | `nomic-ai/modernbert-embed-base` | `sentence-transformers/all-MiniLM-L6-v2` |
| **Parameters** | 149M | 22M (~7× larger) |
| **Hidden Size** | 768 | 384 (2× wider) |
| **Layers** | 22 | 6 (~4× deeper) |
| **Attention Heads** | 12 | 12 (same) |
| **Intermediate Size** | 1152 | 1536 |
| **Max Position Embeddings** | 8192 | 512 (16× longer) |
| **Vocabulary Size** | 50,368 | 30,522 |
| **Output Dimensions** | 768 | 384 (2× larger) |

### Memory Requirements

| Configuration | Per Worker | Notes |
|---------------|------------|-------|
| **Weights (FP32)** | ~600 MB | Model parameters only |
| **Runtime (short sequences <512 tokens)** | ~1 GB | Includes activations + cache |
| **Runtime (long sequences ~8192 tokens)** | ~2-3 GB | Large activation memory |
| **Tokenizer vocab** | ~10 MB | Vocabulary + special tokens |
| **Total per worker** | **~1-3 GB** | Depends on sequence length |

### Comparison Table

| Metric | MiniLM-L6-v2 | ModernBERT Base | Impact |
|--------|--------------|-----------------|--------|
| Model size | 96 MB | 600 MB | +525 MB per worker |
| Runtime memory | 150 MB | 1-3 GB | +10-20× |
| Max sequence | 512 tokens | 8192 tokens | +16× context |
| Embedding dim | 384 | 768 | +2× vector size |
| CPU inference (single) | ~10 ms | ~30-50 ms | +3-5× slower |
| GPU inference (single) | N/A (not used) | ~15-25 ms | Competitive |
| GPU inference (batch=32) | N/A | ~80 ms | ~2.5 ms/item |

---

## Dynamic CPU/GPU Routing Strategy

### Core Routing Logic

ModernBERT's size and long-context support make GPU acceleration beneficial for specific workloads. The routing strategy is based on **sequence length** and **batch tokens**.

```rust
pub fn select_device(seq_len: usize, batch_size: usize) -> Device {
    let batch_tokens = seq_len * batch_size;

    // Rule 1: Long sequences always use GPU (parallel attention wins)
    if seq_len >= 1024 {
        return Device::Mps;
    }

    // Rule 2: Small batches use CPU (GPU overhead not worth it)
    if batch_tokens <= 512 {
        return Device::Cpu;
    }

    // Rule 3: Large batches use GPU (amortized transfer cost)
    if batch_tokens >= 2048 {
        return Device::Mps;
    }

    // Default: CPU for medium workloads
    Device::Cpu
}
```

### Decision Matrix

| Sequence Length | Batch Size | Batch Tokens | Device | Rationale |
|----------------|------------|--------------|--------|-----------|
| 128 | 1 | 128 | **CPU** | Tiny workload, GPU overhead > compute |
| 512 | 1 | 512 | **CPU** | Threshold, CPU slightly faster |
| 1024 | 1 | 1024 | **GPU** | Long sequence, attention parallelism wins |
| 2048 | 1 | 2048 | **GPU** | Very long sequence |
| 256 | 4 | 1024 | **CPU** | Medium batch, below threshold |
| 256 | 8 | 2048 | **GPU** | Large batch, amortized transfer |
| 512 | 8 | 4096 | **GPU** | Large batch tokens |
| 8192 | 1 | 8192 | **GPU** | Max context, GPU essential |

### Threshold Justification

**Why 1024 for sequence length?**
- Transformer attention complexity: O(n²)
- At 1024 tokens: 1M attention ops → GPU parallelism becomes beneficial
- CPU: ~50-100 ms for 1024 tokens
- GPU: ~25-35 ms for 1024 tokens (2× speedup)

**Why 512 for batch_tokens (lower bound)?**
- GPU has ~5-10 ms fixed overhead (data transfer, kernel launch)
- Need at least 20 ms of compute to amortize overhead
- 512 tokens ≈ 15-20 ms on GPU → breakeven point

**Why 2048 for batch_tokens (upper bound)?**
- Batch processing amortizes transfer overhead
- 2048 tokens = 8 items × 256 tokens or 4 items × 512 tokens
- GPU throughput: ~100 tokens/ms (batched) vs ~50 tokens/ms (CPU)

---

## Heterogeneous Worker Pool Architecture

### Worker Types

```rust
pub enum WorkerType {
    Cpu {
        cache_size: usize,
    },
    Gpu {
        cache_size: usize,
        device: Device,  // Device::Mps on Apple Silicon
    },
}

pub struct WorkerConfig {
    pub id: usize,
    pub worker_type: WorkerType,
    pub queue_size: usize,
}
```

### Hybrid Pool Configuration

```rust
pub struct HybridPoolConfig {
    pub cpu_workers: usize,
    pub gpu_workers: usize,
    pub routing_strategy: RoutingStrategy,
}

pub enum RoutingStrategy {
    /// Use routing logic to select CPU or GPU worker
    Dynamic,
    /// Round-robin across all workers
    RoundRobin,
    /// Prefer CPU, use GPU only when CPU saturated
    CpuFirst,
}
```

### Worker Allocation by System RAM

| System RAM | CPU Workers | GPU Workers | Total Memory | Rationale |
|------------|-------------|-------------|--------------|-----------|
| **8 GB** | 2 | 0 | ~2 GB | Conservative, avoid swapping |
| **16 GB** | 4 | 1 | ~7 GB | Balanced (4 GB CPU + 3 GB GPU) |
| **24 GB** | 6 | 1 | ~9 GB | More CPU throughput |
| **32 GB** | 8 | 2 | ~14 GB | High performance |
| **64 GB** | 10 | 2 | ~16 GB | Max throughput |
| **96 GB+** | 12 | 4 | ~24 GB | Server-class |

### Calculation Example (16 GB System)

```
CPU Workers: 4 × 1 GB = 4 GB
GPU Workers: 1 × 3 GB = 3 GB
OS + Other: ~4 GB
Buffer: ~5 GB
──────────────────────
Total: 16 GB ✓
```

---

## Implementation in Rust

### 1. Model Loading (rust-bert)

```rust
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder,
    SentenceEmbeddingsModel,
};
use tch::Device;

pub struct ModernBERTConfig {
    pub model_id: String,
    pub device: Device,
    pub max_length: usize,
}

impl Default for ModernBERTConfig {
    fn default() -> Self {
        Self {
            model_id: "nomic-ai/modernbert-embed-base".to_string(),
            device: Device::Cpu,
            max_length: 8192,
        }
    }
}

pub fn load_modernbert(config: ModernBERTConfig) -> Result<SentenceEmbeddingsModel> {
    log::info!("Loading ModernBERT model on device: {:?}", config.device);

    let model = SentenceEmbeddingsBuilder::remote(config.model_id)
        .with_device(config.device)
        .create_model()?;

    log::info!("ModernBERT loaded successfully");
    Ok(model)
}
```

### 2. Mean Pooling Implementation

ModernBERT requires **mean pooling** over the last hidden state (unlike MiniLM which uses CLS token).

```rust
use ndarray::{Array1, Array2};

/// Mean pooling with attention mask
pub fn mean_pooling(
    hidden_state: &Array2<f32>,      // Shape: [seq_len, hidden_size]
    attention_mask: &Array1<f32>,    // Shape: [seq_len]
) -> Array1<f32> {
    let seq_len = hidden_state.nrows();
    let hidden_size = hidden_state.ncols();

    // Expand mask to [seq_len, hidden_size]
    let mask_expanded: Array2<f32> = Array2::from_shape_fn(
        (seq_len, hidden_size),
        |(i, _)| attention_mask[i]
    );

    // Element-wise multiply: hidden_state * mask
    let masked_hidden = hidden_state * &mask_expanded;

    // Sum along sequence dimension
    let sum_embeddings = masked_hidden.sum_axis(ndarray::Axis(0));

    // Sum mask (with clamping to avoid division by zero)
    let sum_mask = mask_expanded.sum_axis(ndarray::Axis(0))
        .mapv(|x| x.max(1e-9));

    // Divide to get mean
    sum_embeddings / sum_mask
}
```

### 3. L2 Normalization

```rust
pub fn l2_normalize(embedding: &mut Array1<f32>) {
    let norm = embedding.dot(embedding).sqrt();

    if norm > 1e-12 {
        embedding.mapv_inplace(|x| x / norm);
    }
}
```

### 4. Worker with Dynamic Device Selection

```rust
pub struct ModernBERTWorker {
    id: usize,
    worker_type: WorkerType,
    model: SentenceEmbeddingsModel,
    cache: HashMap<String, Array1<f32>>,
    stats: EmbedderStats,
}

impl ModernBERTWorker {
    pub fn new_cpu(id: usize) -> Result<Self> {
        let config = ModernBERTConfig {
            device: Device::Cpu,
            ..Default::default()
        };

        let model = load_modernbert(config)?;

        Ok(Self {
            id,
            worker_type: WorkerType::Cpu { cache_size: 5000 },
            model,
            cache: HashMap::new(),
            stats: EmbedderStats::default(),
        })
    }

    pub fn new_gpu(id: usize) -> Result<Self> {
        let config = ModernBERTConfig {
            device: Device::Mps,
            ..Default::default()
        };

        let model = load_modernbert(config)?;

        Ok(Self {
            id,
            worker_type: WorkerType::Gpu {
                cache_size: 2000,
                device: Device::Mps,
            },
            model,
            cache: HashMap::new(),
            stats: EmbedderStats::default(),
        })
    }

    pub fn embed_text(&mut self, text: &str) -> Result<Array1<f32>> {
        // Check cache
        if let Some(cached) = self.cache.get(text) {
            self.stats.cache_hits += 1;
            return Ok(cached.clone());
        }

        let start = Instant::now();

        // Tokenize and encode
        let embeddings = self.model.encode(&[text.to_string()])?;

        // Convert to ndarray
        let mut embedding = Array1::from_vec(embeddings[0].clone());

        // L2 normalize
        l2_normalize(&mut embedding);

        // Update stats
        self.stats.embeddings_count += 1;
        self.stats.total_processing_time += start.elapsed();

        // Cache result
        self.cache.insert(text.to_string(), embedding.clone());

        Ok(embedding)
    }
}
```

### 5. Smart Request Router

```rust
pub struct RequestRouter {
    cpu_workers: Vec<Sender<WorkerRequest>>,
    gpu_workers: Vec<Sender<WorkerRequest>>,
    cpu_next: AtomicUsize,
    gpu_next: AtomicUsize,
}

impl RequestRouter {
    pub fn route_request(
        &self,
        text: &str,
        request: WorkerRequest,
    ) -> Result<()> {
        let seq_len = estimate_tokens(text);
        let device = select_device(seq_len, 1);

        match device {
            Device::Cpu => {
                let idx = self.cpu_next.fetch_add(1, Ordering::Relaxed);
                let worker = &self.cpu_workers[idx % self.cpu_workers.len()];
                worker.send(request)?;
            }
            Device::Mps => {
                if self.gpu_workers.is_empty() {
                    // Fallback to CPU if no GPU workers
                    let idx = self.cpu_next.fetch_add(1, Ordering::Relaxed);
                    let worker = &self.cpu_workers[idx % self.cpu_workers.len()];
                    worker.send(request)?;
                } else {
                    let idx = self.gpu_next.fetch_add(1, Ordering::Relaxed);
                    let worker = &self.gpu_workers[idx % self.gpu_workers.len()];
                    worker.send(request)?;
                }
            }
            _ => unreachable!(),
        }

        Ok(())
    }

    pub fn route_batch(
        &self,
        texts: &[String],
        request: WorkerRequest,
    ) -> Result<()> {
        if texts.is_empty() {
            return Ok(());
        }

        // Estimate batch characteristics
        let max_seq_len = texts.iter()
            .map(|t| estimate_tokens(t))
            .max()
            .unwrap_or(0);

        let device = select_device(max_seq_len, texts.len());

        // Route to appropriate worker type
        match device {
            Device::Cpu => {
                let idx = self.cpu_next.fetch_add(1, Ordering::Relaxed);
                self.cpu_workers[idx % self.cpu_workers.len()].send(request)?;
            }
            Device::Mps if !self.gpu_workers.is_empty() => {
                let idx = self.gpu_next.fetch_add(1, Ordering::Relaxed);
                self.gpu_workers[idx % self.gpu_workers.len()].send(request)?;
            }
            _ => {
                // Fallback to CPU
                let idx = self.cpu_next.fetch_add(1, Ordering::Relaxed);
                self.cpu_workers[idx % self.cpu_workers.len()].send(request)?;
            }
        }

        Ok(())
    }
}
```

### 6. Token Estimation

```rust
/// Estimate number of tokens in text
/// Rule of thumb: ~4 characters per token for English/code mix
pub fn estimate_tokens(text: &str) -> usize {
    (text.len() / 4).max(1)
}

/// Calculate total batch tokens (for routing decisions)
pub fn batch_tokens(texts: &[String]) -> usize {
    let max_len = texts.iter()
        .map(|t| estimate_tokens(t))
        .max()
        .unwrap_or(0);

    max_len * texts.len()
}
```

---

## Library Design - Explicit Configuration Required

**CRITICAL**: ModernBERT implementation follows rust-embed's library-first design:
- Worker counts are ALWAYS explicitly specified by caller
- No auto-detection of system resources
- Support dynamic reconfiguration at runtime
- Caller controls CPU/GPU worker allocation

```rust
// Caller MUST provide explicit configuration
let config = HybridPoolConfig {
    cpu_workers: 4,      // Explicit, no default
    gpu_workers: 1,      // Explicit, no default
    routing_strategy: RoutingStrategy::Dynamic,
};

let pool = ModernBERTPool::new(config)?;

// Dynamic reconfiguration supported
pool.reconfigure(HybridPoolConfig {
    cpu_workers: 8,      // Scale up
    gpu_workers: 2,
    routing_strategy: RoutingStrategy::Dynamic,
})?;
```

---

## Hybrid Pool Implementation

### Complete Pool Structure

```rust
pub struct ModernBERTPool {
    cpu_workers: Vec<Sender<WorkerRequest>>,
    gpu_workers: Vec<Sender<WorkerRequest>>,
    router: RequestRouter,
    handles: Vec<JoinHandle<()>>,
    config: HybridPoolConfig,
}

impl ModernBERTPool {
    pub fn new(config: HybridPoolConfig) -> Result<Self> {
        log::info!(
            "Initializing ModernBERT pool: {} CPU workers, {} GPU workers",
            config.cpu_workers,
            config.gpu_workers
        );

        let mut cpu_workers = Vec::new();
        let mut gpu_workers = Vec::new();
        let mut handles = Vec::new();

        // Spawn CPU workers in parallel
        let cpu_init: Vec<_> = (0..config.cpu_workers)
            .map(|id| thread::spawn(move || ModernBERTWorker::new_cpu(id)))
            .collect();

        for (id, handle) in cpu_init.into_iter().enumerate() {
            let worker = handle.join()
                .map_err(|_| anyhow!("CPU worker {} init panic", id))??;

            let (tx, rx) = crossbeam::channel::unbounded();
            let worker_handle = thread::spawn(move || worker.run(rx));

            cpu_workers.push(tx);
            handles.push(worker_handle);
        }

        // Spawn GPU workers in parallel
        let gpu_init: Vec<_> = (0..config.gpu_workers)
            .map(|id| {
                let gpu_id = config.cpu_workers + id;
                thread::spawn(move || ModernBERTWorker::new_gpu(gpu_id))
            })
            .collect();

        for (id, handle) in gpu_init.into_iter().enumerate() {
            let worker = handle.join()
                .map_err(|_| anyhow!("GPU worker {} init panic", id))??;

            let (tx, rx) = crossbeam::channel::unbounded();
            let worker_handle = thread::spawn(move || worker.run(rx));

            gpu_workers.push(tx);
            handles.push(worker_handle);
        }

        let router = RequestRouter {
            cpu_workers: cpu_workers.clone(),
            gpu_workers: gpu_workers.clone(),
            cpu_next: AtomicUsize::new(0),
            gpu_next: AtomicUsize::new(0),
        };

        log::info!("ModernBERT pool ready");

        Ok(Self {
            cpu_workers,
            gpu_workers,
            router,
            handles,
            config,
        })
    }

    pub fn embed_text(&self, text: String) -> Result<Array1<f32>> {
        let (tx, rx) = oneshot::channel();

        self.router.route_request(
            &text,
            WorkerRequest::Embed {
                text,
                response_tx: tx,
            },
        )?;

        rx.recv()?
    }

    pub fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Array1<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Split batch and route intelligently
        let chunk_size = (texts.len() + self.total_workers() - 1) / self.total_workers();

        let mut receivers = vec![];

        for chunk in texts.chunks(chunk_size) {
            let (tx, rx) = oneshot::channel();

            self.router.route_batch(
                chunk,
                WorkerRequest::EmbedBatch {
                    texts: chunk.to_vec(),
                    response_tx: tx,
                },
            )?;

            receivers.push(rx);
        }

        let mut results = Vec::with_capacity(texts.len());
        for rx in receivers {
            results.extend(rx.recv()??);
        }

        Ok(results)
    }

    fn total_workers(&self) -> usize {
        self.cpu_workers.len() + self.gpu_workers.len()
    }
}
```

---

## Configuration

### Configuration File (modernbert.toml)

```toml
[model]
model_id = "nomic-ai/modernbert-embed-base"
max_length = 8192
output_dimensions = 768

[pool]
# Number of CPU workers (0 = auto)
cpu_workers = 0

# Number of GPU workers (0 = auto)
gpu_workers = 0

# Routing strategy: "dynamic", "round_robin", "cpu_first"
routing_strategy = "dynamic"

[routing]
# Sequence length threshold for GPU (tokens)
long_sequence_threshold = 1024

# Batch tokens threshold (lower) for CPU
small_batch_threshold = 512

# Batch tokens threshold (upper) for GPU
large_batch_threshold = 2048

[cache]
# Cache size per CPU worker
cpu_cache_size = 5000

# Cache size per GPU worker
gpu_cache_size = 2000

[memory]
# Auto-calculate workers based on available RAM
auto_calculate = true

# Reserve RAM for OS/other (GB)
reserved_ram = 4.0
```

---

## Performance Projections

### Single Text Embedding

| Device | Sequence Length | Latency | Throughput (6 workers) |
|--------|----------------|---------|------------------------|
| CPU | 128 tokens | 30 ms | ~200 emb/sec |
| CPU | 512 tokens | 50 ms | ~120 emb/sec |
| CPU | 1024 tokens | 100 ms | ~60 emb/sec |
| GPU | 1024 tokens | 35 ms | ~170 emb/sec (5 GPU workers) |
| GPU | 2048 tokens | 60 ms | ~100 emb/sec |
| GPU | 8192 tokens | 200 ms | ~30 emb/sec |

### Batch Embedding (16 GB System: 4 CPU + 1 GPU)

| Batch Size | Avg Seq Len | Routed To | Latency | Throughput |
|------------|-------------|-----------|---------|------------|
| 10 | 256 | CPU | 400 ms | ~25 emb/sec |
| 32 | 256 | GPU | 500 ms | ~64 emb/sec |
| 10 | 1024 | GPU | 350 ms | ~29 emb/sec |
| 32 | 1024 | GPU | 1200 ms | ~27 emb/sec |

---

## Migration from MiniLM

### Compatibility Layer

```rust
pub enum EmbeddingModel {
    MiniLM(MiniLMEmbedder),
    ModernBERT(ModernBERTWorker),
}

impl EmbeddingModel {
    pub fn embed_text(&mut self, text: &str) -> Result<Vec<f32>> {
        match self {
            EmbeddingModel::MiniLM(embedder) => {
                let emb = embedder.embed_text(text)?;
                Ok(emb.to_vec())
            }
            EmbeddingModel::ModernBERT(embedder) => {
                let emb = embedder.embed_text(text)?;
                Ok(emb.to_vec())
            }
        }
    }

    pub fn dimension(&self) -> usize {
        match self {
            EmbeddingModel::MiniLM(_) => 384,
            EmbeddingModel::ModernBERT(_) => 768,
        }
    }
}
```

### Feature Flags

```toml
[features]
default = ["modernbert"]
minilm = []
modernbert = []
```

---

## Testing Strategy

### Unit Tests

```rust
#[test]
fn test_device_selection() {
    assert_eq!(select_device(128, 1), Device::Cpu);
    assert_eq!(select_device(1024, 1), Device::Mps);
    assert_eq!(select_device(256, 8), Device::Mps);  // 2048 batch tokens
    assert_eq!(select_device(256, 1), Device::Cpu);
}

#[test]
fn test_token_estimation() {
    assert_eq!(estimate_tokens("hello"), 1);  // 5 chars / 4 = 1
    assert_eq!(estimate_tokens("hello world"), 2);  // 11 chars / 4 = 2
    assert_eq!(estimate_tokens("a".repeat(1000)), 250);  // 1000 / 4 = 250
}

#[test]
fn test_mean_pooling() {
    let hidden = Array2::from_shape_vec((3, 4), vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
    ]).unwrap();

    let mask = Array1::from_vec(vec![1.0, 1.0, 0.0]);  // Last token is padding

    let result = mean_pooling(&hidden, &mask);

    // Mean of first two rows: (1+5)/2=3, (2+6)/2=4, (3+7)/2=5, (4+8)/2=6
    assert_eq!(result, Array1::from_vec(vec![3.0, 4.0, 5.0, 6.0]));
}
```

### Integration Tests

```rust
#[test]
fn test_hybrid_pool() {
    let config = HybridPoolConfig {
        cpu_workers: 2,
        gpu_workers: 1,
        routing_strategy: RoutingStrategy::Dynamic,
    };

    let pool = ModernBERTPool::new(config).unwrap();

    // Short text → CPU
    let emb1 = pool.embed_text("hello".to_string()).unwrap();
    assert_eq!(emb1.len(), 768);

    // Long text → GPU
    let long_text = "word ".repeat(300);
    let emb2 = pool.embed_text(long_text).unwrap();
    assert_eq!(emb2.len(), 768);
}
```

---

## Implementation Roadmap

### Implementation as Model Option (Not Replacement)

ModernBERT will be added as a **selectable model type**, not a replacement for MiniLM. Both models coexist.

### Phase 1: Model Loading and Pooling (Week 1-2)
- [ ] Add ModernBERT to ModelType enum
- [ ] Implement ModernBERT model loading via rust-bert
- [ ] Implement mean pooling function (different from MiniLM's CLS token)
- [ ] Implement L2 normalization (same as MiniLM)
- [ ] Add token estimation utilities
- [ ] Unit tests for pooling and normalization
- [ ] Validate ModernBERT can be loaded in worker

### Phase 2: CPU Worker Support (Week 2-3)
- [ ] Update EmbeddingWorker to handle both MiniLM and ModernBERT
- [ ] Add model-specific initialization logic
- [ ] Test CPU-only ModernBERT workers
- [ ] Benchmark ModernBERT CPU performance
- [ ] Ensure PoolConfig validation handles both models

### Phase 3: GPU Worker Support (Week 3-4)
- [ ] Add GPU worker initialization with Device::Mps
- [ ] Implement device selection logic (CPU vs GPU)
- [ ] Create GPU-specific worker configuration
- [ ] Test GPU inference path
- [ ] Benchmark GPU vs CPU performance by sequence length

### Phase 4: Dynamic Routing (Week 4-5)
- [ ] Implement route_worker() logic for ModernBERT
- [ ] Add sequence length estimation
- [ ] Implement routing thresholds (1024, 512, 2048)
- [ ] Test routing decisions
- [ ] Ensure MiniLM still routes to CPU only

### Phase 5: Integration and Testing (Week 5-6)
- [ ] Update CLI to support both models
- [ ] Add `--model` flag (minilm or modernbert)
- [ ] Integration tests with both models
- [ ] Performance benchmarks
- [ ] Documentation and examples

### Phase 6: Optimization (Week 6-7)
- [ ] Tune routing thresholds based on benchmarks
- [ ] Memory usage optimization
- [ ] Performance testing
- [ ] Update all documentation
- [ ] Add model selection guide

---

## Conclusion

ModernBERT Base will be added as a **selectable model option** alongside MiniLM-L6-v2, giving callers choice based on their requirements:

### Model Coexistence Benefits

1. **Flexibility**: Caller chooses model based on use case (speed vs quality, short vs long context)
2. **Resource Control**: Caller allocates resources appropriately (MiniLM = 600 MB, ModernBERT = 6-14 GB)
3. **No Forced Migration**: Existing MiniLM users unaffected
4. **Library Principle**: Configuration-time decision, not library decision

### Architecture Benefits

The hybrid CPU/GPU worker pool architecture enables:

1. **Intelligent routing** based on workload characteristics (ModernBERT only)
2. **Resource efficiency** by using GPU only when beneficial
3. **Scalability** through heterogeneous worker pools
4. **Flexibility** to adapt to available hardware

### Memory Trade-offs

| Model | Workers | Memory | Use Case |
|-------|---------|--------|----------|
| **MiniLM** | 6 CPU | 622 MB | Fast, efficient, short texts |
| **ModernBERT** | 4 CPU + 1 GPU | ~7 GB | Quality, long context (up to 8192 tokens) |
| **ModernBERT** | 8 CPU + 2 GPU | ~14 GB | Maximum quality, requires ≥32 GB RAM |

**Approved for implementation** as model option in rust-embed v0.2.1 or v0.3.0.

**Key Principle**: Both models available, caller chooses via `PoolConfig.model`.
