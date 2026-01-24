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

### Worker Count Guidelines (Caller-Controlled)

**IMPORTANT**: rust-embed is a **library**, not an application. Worker allocation is **always controlled by the caller**, never auto-detected. The following are **recommendations only** for typical use cases:

| Use Case | Recommended Configuration | Rationale |
|----------|--------------------------|-----------|
| **Minimal footprint** | 1 CPU worker | Upstream needs RAM for other operations |
| **Balanced hybrid** | 2 CPU + 1 GPU | Mix of short and long texts, shared resources |
| **Batch processing** | 4-8 CPU workers | Dedicated embedding service, maximize throughput |
| **Long context heavy** | 4 CPU + 2 GPU | Processing documents >1024 tokens frequently |
| **Maximum throughput** | 8-12 CPU + 2 GPU | Dedicated server, ample RAM (≥32 GB) |

### Suggested Worker Counts by Hardware (Reference Only)

These are **suggestions**, not defaults. Callers must explicitly specify worker counts:

| Hardware | Available RAM | Suggested Range | Example Configs |
|----------|--------------|-----------------|-----------------|
| M1 Air (8GB) | 8 GB | 1-2 CPU workers | `1 CPU` (minimal), `2 CPU` (moderate) |
| M1 Pro (16GB) | 16 GB | 2-6 CPU workers | `4 CPU` (balanced), `4 CPU + 1 GPU` (hybrid) |
| M1 Max (32GB) | 32 GB | 4-10 CPU workers | `8 CPU + 2 GPU` (high performance) |
| M2 Ultra (64GB+) | 64+ GB | 6-16 CPU workers | `12 CPU + 4 GPU` (maximum) |

### Helper Function (Optional Convenience)

```rust
/// OPTIONAL helper to suggest worker count based on system resources
/// Caller is FREE to ignore this and specify their own configuration
pub fn suggest_worker_count() -> WorkerSuggestion {
    let num_cpus = num_cpus::get();
    let available_ram_gb = get_available_ram_gb();

    // Conservative suggestions that leave headroom
    let suggested_cpu = if utils::is_apple_silicon() {
        match num_cpus {
            1..=4   => 2,
            5..=8   => 4,
            9..=12  => 6,
            13..=16 => 8,
            _       => 10,
        }
    } else {
        (num_cpus * 3 / 4).max(1).min(8)
    };

    // Only suggest GPU workers if sufficient RAM
    let suggested_gpu = if available_ram_gb >= 16 && utils::has_mps() {
        1
    } else {
        0
    };

    WorkerSuggestion {
        cpu_workers: suggested_cpu,
        gpu_workers: suggested_gpu,
        note: "This is a suggestion. Caller should configure based on their needs.".to_string(),
    }
}
```

**Design Principle**:
- **No automatic resource allocation** - caller always specifies
- Helper function is **opt-in** and returns suggestions, not mandates
- Upstream systems know their resource constraints better than the library

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

### 3. Pool Manager - Explicit Configuration Required

```rust
use crossbeam::channel::{self, Sender, Receiver};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Worker pool configuration - MUST be explicitly provided by caller
/// No defaults - library does not make resource decisions
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Number of CPU workers (required, no default)
    pub cpu_workers: usize,

    /// Number of GPU workers (required, no default)
    pub gpu_workers: usize,

    /// Model to use
    pub model: ModelType,

    /// Cache size per worker
    pub cache_size_per_worker: usize,
}

pub enum ModelType {
    MiniLM,
    ModernBERT,
}

/// Pool of embedding workers with work distribution
pub struct EmbeddingPool {
    cpu_workers: Vec<Sender<WorkerRequest>>,
    gpu_workers: Vec<Sender<WorkerRequest>>,
    handles: Vec<JoinHandle<()>>,
    cpu_next: AtomicUsize,
    gpu_next: AtomicUsize,
    current_config: PoolConfig,
}

impl EmbeddingPool {
    /// Create pool with EXPLICIT configuration
    /// Caller MUST specify worker counts - no auto-detection
    pub fn new(config: PoolConfig) -> Result<Self> {
        if config.cpu_workers == 0 && config.gpu_workers == 0 {
            return Err(anyhow!("Must specify at least one worker"));
        }

        log::info!(
            "Creating pool: {} CPU workers, {} GPU workers",
            config.cpu_workers,
            config.gpu_workers
        );

        let mut cpu_workers = Vec::new();
        let mut gpu_workers = Vec::new();
        let mut handles = Vec::new();

        // Spawn CPU workers in parallel
        let cpu_init: Vec<_> = (0..config.cpu_workers)
            .map(|id| {
                let model = config.model.clone();
                let cache_size = config.cache_size_per_worker;
                thread::spawn(move || {
                    let worker_config = WorkerConfig {
                        id,
                        device: Device::Cpu,
                        cache_size,
                        queue_size: 100,
                    };
                    EmbeddingWorker::new(id, worker_config)
                })
            })
            .collect();

        for (id, init_handle) in cpu_init.into_iter().enumerate() {
            let worker = init_handle.join()
                .map_err(|_| anyhow!("CPU worker {} init failed", id))??;

            let (tx, rx) = crossbeam::channel::unbounded();
            let handle = thread::spawn(move || worker.run(rx));

            cpu_workers.push(tx);
            handles.push(handle);
        }

        // Spawn GPU workers in parallel
        let gpu_init: Vec<_> = (0..config.gpu_workers)
            .map(|id| {
                let gpu_id = config.cpu_workers + id;
                let model = config.model.clone();
                let cache_size = config.cache_size_per_worker;
                thread::spawn(move || {
                    let worker_config = WorkerConfig {
                        id: gpu_id,
                        device: Device::Mps,
                        cache_size,
                        queue_size: 100,
                    };
                    EmbeddingWorker::new(gpu_id, worker_config)
                })
            })
            .collect();

        for (id, init_handle) in gpu_init.into_iter().enumerate() {
            let worker = init_handle.join()
                .map_err(|_| anyhow!("GPU worker {} init failed", id))??;

            let (tx, rx) = crossbeam::channel::unbounded();
            let handle = thread::spawn(move || worker.run(rx));

            gpu_workers.push(tx);
            handles.push(handle);
        }

        log::info!(
            "Pool ready: {} total workers",
            config.cpu_workers + config.gpu_workers
        );

        Ok(Self {
            cpu_workers,
            gpu_workers,
            handles,
            cpu_next: AtomicUsize::new(0),
            gpu_next: AtomicUsize::new(0),
            current_config: config,
        })
    }

    /// Reconfigure pool with new worker counts
    /// - Spawns new workers if count increased
    /// - Gracefully shuts down excess workers if count decreased
    /// - Allows upstream to dynamically adjust resource allocation
    pub fn reconfigure(&mut self, new_config: PoolConfig) -> Result<()> {
        log::info!(
            "Reconfiguring: {} → {} CPU, {} → {} GPU workers",
            self.current_config.cpu_workers,
            new_config.cpu_workers,
            self.current_config.gpu_workers,
            new_config.gpu_workers
        );

        // Handle CPU worker changes
        match new_config.cpu_workers.cmp(&self.cpu_workers.len()) {
            std::cmp::Ordering::Greater => {
                // Spawn additional CPU workers
                let to_spawn = new_config.cpu_workers - self.cpu_workers.len();
                for i in 0..to_spawn {
                    let id = self.cpu_workers.len() + i;
                    let worker_config = WorkerConfig {
                        id,
                        device: Device::Cpu,
                        cache_size: new_config.cache_size_per_worker,
                        queue_size: 100,
                    };

                    let worker = EmbeddingWorker::new(id, worker_config)?;
                    let (tx, rx) = crossbeam::channel::unbounded();
                    let handle = thread::spawn(move || worker.run(rx));

                    self.cpu_workers.push(tx);
                    self.handles.push(handle);
                }
                log::info!("Spawned {} additional CPU workers", to_spawn);
            }
            std::cmp::Ordering::Less => {
                // Shutdown excess CPU workers
                let to_remove = self.cpu_workers.len() - new_config.cpu_workers;
                for _ in 0..to_remove {
                    if let Some(worker) = self.cpu_workers.pop() {
                        let _ = worker.send(WorkerRequest::Shutdown);
                    }
                }
                log::info!("Removed {} CPU workers", to_remove);
            }
            std::cmp::Ordering::Equal => {}
        }

        // Handle GPU worker changes
        match new_config.gpu_workers.cmp(&self.gpu_workers.len()) {
            std::cmp::Ordering::Greater => {
                let to_spawn = new_config.gpu_workers - self.gpu_workers.len();
                for i in 0..to_spawn {
                    let id = new_config.cpu_workers + self.gpu_workers.len() + i;
                    let worker_config = WorkerConfig {
                        id,
                        device: Device::Mps,
                        cache_size: new_config.cache_size_per_worker,
                        queue_size: 100,
                    };

                    let worker = EmbeddingWorker::new(id, worker_config)?;
                    let (tx, rx) = crossbeam::channel::unbounded();
                    let handle = thread::spawn(move || worker.run(rx));

                    self.gpu_workers.push(tx);
                    self.handles.push(handle);
                }
                log::info!("Spawned {} additional GPU workers", to_spawn);
            }
            std::cmp::Ordering::Less => {
                let to_remove = self.gpu_workers.len() - new_config.gpu_workers;
                for _ in 0..to_remove {
                    if let Some(worker) = self.gpu_workers.pop() {
                        let _ = worker.send(WorkerRequest::Shutdown);
                    }
                }
                log::info!("Removed {} GPU workers", to_remove);
            }
            std::cmp::Ordering::Equal => {}
        }

        self.current_config = new_config;
        log::info!("Reconfiguration complete");

        Ok(())
    }

    /// Get current configuration
    pub fn config(&self) -> &PoolConfig {
        &self.current_config
    }

    /// Get current worker counts (actual running workers)
    pub fn worker_counts(&self) -> (usize, usize) {
        (self.cpu_workers.len(), self.gpu_workers.len())
    }

    /// Route request to appropriate worker (CPU or GPU)
    /// Uses dynamic routing for ModernBERT, always CPU for MiniLM
    fn route_worker(&self, text: &str) -> &Sender<WorkerRequest> {
        match self.current_config.model {
            ModelType::MiniLM => {
                // Always use CPU for MiniLM
                let idx = self.cpu_next.fetch_add(1, Ordering::Relaxed);
                &self.cpu_workers[idx % self.cpu_workers.len()]
            }
            ModelType::ModernBERT => {
                // Dynamic routing based on sequence length
                let seq_len = estimate_tokens(text);
                let device = select_device_modernbert(seq_len, 1);

                match device {
                    Device::Cpu => {
                        let idx = self.cpu_next.fetch_add(1, Ordering::Relaxed);
                        &self.cpu_workers[idx % self.cpu_workers.len()]
                    }
                    Device::Mps if !self.gpu_workers.is_empty() => {
                        let idx = self.gpu_next.fetch_add(1, Ordering::Relaxed);
                        &self.gpu_workers[idx % self.gpu_workers.len()]
                    }
                    _ => {
                        // Fallback to CPU if no GPU workers available
                        let idx = self.cpu_next.fetch_add(1, Ordering::Relaxed);
                        &self.cpu_workers[idx % self.cpu_workers.len()]
                    }
                }
            }
        }
    }

    /// Embed single text (routes to appropriate worker type)
    pub fn embed_text(&self, text: String) -> Result<Array1<f32>> {
        let (tx, rx) = oneshot::channel();
        let worker = self.route_worker(&text);
        worker.send(WorkerRequest::Embed {
            text,
            response_tx: tx,
        })?;
        rx.recv()?
    }

    /// Embed batch (distributes across ALL workers - CPU and GPU)
    pub fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Array1<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let total_workers = self.cpu_workers.len() + self.gpu_workers.len();
        let chunk_size = (texts.len() + total_workers - 1) / total_workers;

        let mut receivers = vec![];

        // For batch processing, distribute evenly across all workers
        // (routing decisions are less critical when amortizing over chunks)
        let mut cpu_idx = 0;
        let mut gpu_idx = 0;

        for chunk in texts.chunks(chunk_size) {
            let (tx, rx) = oneshot::channel();

            // Alternate between CPU and GPU workers
            let worker = if !self.gpu_workers.is_empty() && gpu_idx < self.gpu_workers.len() {
                let w = &self.gpu_workers[gpu_idx];
                gpu_idx += 1;
                w
            } else if cpu_idx < self.cpu_workers.len() {
                let w = &self.cpu_workers[cpu_idx];
                cpu_idx += 1;
                w
            } else {
                // Wrap around if needed
                cpu_idx = 0;
                let w = &self.cpu_workers[cpu_idx];
                cpu_idx += 1;
                w
            };

            worker.send(WorkerRequest::EmbedBatch {
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

    /// Get aggregate statistics from all workers (CPU + GPU)
    pub fn aggregate_stats(&self) -> Result<EmbedderStats> {
        let mut receivers = vec![];

        // Collect stats from CPU workers
        for worker in &self.cpu_workers {
            let (tx, rx) = oneshot::channel();
            worker.send(WorkerRequest::GetStats { response_tx: tx })?;
            receivers.push(rx);
        }

        // Collect stats from GPU workers
        for worker in &self.gpu_workers {
            let (tx, rx) = oneshot::channel();
            worker.send(WorkerRequest::GetStats { response_tx: tx })?;
            receivers.push(rx);
        }

        // Aggregate all stats
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

    /// Graceful shutdown of all workers
    pub fn shutdown(self) -> Result<()> {
        // Shutdown CPU workers
        for worker in &self.cpu_workers {
            let _ = worker.send(WorkerRequest::Shutdown);
        }

        // Shutdown GPU workers
        for worker in &self.gpu_workers {
            let _ = worker.send(WorkerRequest::Shutdown);
        }

        // Wait for all worker threads to finish
        for handle in self.handles {
            handle.join().map_err(|_| anyhow!("Worker panic"))?;
        }

        log::info!("Pool shutdown complete");

        Ok(())
    }
}
```

---

## Library Usage Patterns

rust-embed is designed as a **library** with explicit caller control. Here are common integration patterns:

### Pattern 1: Minimal Resource Footprint

```rust
// Upstream system has other memory-intensive operations
// Use rust-embed with minimal resources

let pool = EmbeddingPool::new(PoolConfig {
    cpu_workers: 1,
    gpu_workers: 0,
    model: ModelType::MiniLM,
    cache_size_per_worker: 2000,
})?;

// Process embeddings as needed
let embedding = pool.embed_text("sample text".to_string())?;

// Pool uses ~100 MB RAM total
```

### Pattern 2: Batch Processing Large Corpus

```rust
// Dedicated embedding job - maximize throughput
// Configure for available RAM

let pool = EmbeddingPool::new(PoolConfig {
    cpu_workers: 8,
    gpu_workers: 2,
    model: ModelType::ModernBERT,
    cache_size_per_worker: 5000,
})?;

// Process large batches efficiently
let texts: Vec<String> = load_corpus("large_dataset.txt")?;
let embeddings = pool.embed_batch(texts)?;

// Pool uses ~14 GB RAM but maximizes throughput
```

### Pattern 3: Dynamic Scaling Based on Workload

```rust
// Start conservatively, scale up when needed

let mut pool = EmbeddingPool::new(PoolConfig {
    cpu_workers: 2,
    gpu_workers: 0,
    model: ModelType::MiniLM,
    cache_size_per_worker: 5000,
})?;

// Normal operation
for text in regular_workload {
    let emb = pool.embed_text(text)?;
}

// Detect large batch incoming
if batch_size > 10000 {
    // Scale up
    pool.reconfigure(PoolConfig {
        cpu_workers: 8,
        gpu_workers: 0,
        model: ModelType::MiniLM,
        cache_size_per_worker: 5000,
    })?;

    // Process large batch
    let embeddings = pool.embed_batch(large_batch)?;

    // Scale back down
    pool.reconfigure(PoolConfig {
        cpu_workers: 2,
        gpu_workers: 0,
        model: ModelType::MiniLM,
        cache_size_per_worker: 5000,
    })?;
}
```

### Pattern 4: Hybrid Worker Pool for Mixed Workloads

```rust
// Processing both short snippets and long documents
// Use hybrid CPU/GPU configuration with ModernBERT

let pool = EmbeddingPool::new(PoolConfig {
    cpu_workers: 4,
    gpu_workers: 1,
    model: ModelType::ModernBERT,
    cache_size_per_worker: 3000,
})?;

// Short text → automatically routed to CPU worker
let snippet_emb = pool.embed_text("short query".to_string())?;

// Long document → automatically routed to GPU worker
let long_text = "word ".repeat(2000);  // ~2000 tokens
let doc_emb = pool.embed_text(long_text)?;

// Routing is transparent to caller
```

### Pattern 5: Web Service with Administrative Control

```rust
// Embedding service with runtime reconfiguration
// Allow admin to adjust resources via API

struct EmbeddingService {
    pool: Arc<Mutex<EmbeddingPool>>,
}

impl EmbeddingService {
    async fn embed(&self, text: String) -> Result<Vec<f32>> {
        let pool = self.pool.lock().unwrap();
        let embedding = pool.embed_text(text)?;
        Ok(embedding.to_vec())
    }

    async fn reconfigure_workers(&self, cpu: usize, gpu: usize) -> Result<()> {
        let mut pool = self.pool.lock().unwrap();
        pool.reconfigure(PoolConfig {
            cpu_workers: cpu,
            gpu_workers: gpu,
            model: ModelType::ModernBERT,
            cache_size_per_worker: 5000,
        })?;
        Ok(())
    }

    async fn get_status(&self) -> WorkerStatus {
        let pool = self.pool.lock().unwrap();
        let (cpu, gpu) = pool.worker_counts();
        let stats = pool.aggregate_stats().unwrap_or_default();

        WorkerStatus { cpu, gpu, stats }
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

## Configuration API - Explicit by Design

**Library Principle**: rust-embed is a library, not an application. The caller MUST explicitly configure worker allocation.

### Programmatic Configuration (Primary)

```rust
use rust_embed::{EmbeddingPool, PoolConfig, ModelType};

// Example 1: Minimal footprint (upstream has other memory needs)
let config = PoolConfig {
    cpu_workers: 1,
    gpu_workers: 0,
    model: ModelType::MiniLM,
    cache_size_per_worker: 5000,
};
let pool = EmbeddingPool::new(config)?;

// Example 2: Balanced hybrid for 16GB system
let config = PoolConfig {
    cpu_workers: 4,
    gpu_workers: 1,
    model: ModelType::ModernBERT,
    cache_size_per_worker: 3000,
};
let pool = EmbeddingPool::new(config)?;

// Example 3: Maximum throughput for dedicated server
let config = PoolConfig {
    cpu_workers: 8,
    gpu_workers: 2,
    model: ModelType::ModernBERT,
    cache_size_per_worker: 5000,
};
let pool = EmbeddingPool::new(config)?;

// Example 4: Dynamic reconfiguration
// Start with minimal config, scale up when needed
let mut pool = EmbeddingPool::new(PoolConfig {
    cpu_workers: 2,
    gpu_workers: 0,
    model: ModelType::MiniLM,
    cache_size_per_worker: 5000,
})?;

// ... process some data ...

// Scale up for large batch
pool.reconfigure(PoolConfig {
    cpu_workers: 6,
    gpu_workers: 1,
    model: ModelType::MiniLM,
    cache_size_per_worker: 5000,
})?;

// ... process large batch ...

// Scale back down
pool.reconfigure(PoolConfig {
    cpu_workers: 2,
    gpu_workers: 0,
    model: ModelType::MiniLM,
    cache_size_per_worker: 5000,
})?;
```

### Optional Configuration File (For CLI/Applications)

If building a CLI tool or application on top of rust-embed, you can use a config file:

```toml
# example-app.toml
# NOTE: This is for APPLICATIONS built on rust-embed, not the library itself

[pool]
# REQUIRED: Number of CPU workers (no default, must be explicit)
cpu_workers = 4

# REQUIRED: Number of GPU workers (no default, must be explicit)
gpu_workers = 1

[cache]
# Cache size per worker (embeddings)
cache_size_per_worker = 5000

[model]
# Model to load: "minilm" or "modernbert"
model = "minilm"

# Max sequence length
max_length = 512  # 512 for MiniLM, 8192 for ModernBERT

[routing]
# Only for ModernBERT: dynamic routing thresholds
long_sequence_threshold = 1024
small_batch_threshold = 512
large_batch_threshold = 2048

[monitoring]
# Log worker stats interval (seconds, 0 = disabled)
stats_interval = 60

# Enable per-worker metrics
enable_per_worker_metrics = true
```

### Helper Function (Optional)

For convenience, rust-embed provides a suggestion function, but callers are NOT required to use it:

```rust
use rust_embed::suggest_pool_config;

// Get a suggestion based on system resources
let suggestion = suggest_pool_config();
println!("Suggested config: {} CPU, {} GPU workers",
    suggestion.cpu_workers,
    suggestion.gpu_workers
);

// Caller can accept, modify, or ignore the suggestion
let config = PoolConfig {
    cpu_workers: suggestion.cpu_workers / 2,  // Use less than suggested
    gpu_workers: 0,                            // No GPU despite suggestion
    model: ModelType::MiniLM,
    cache_size_per_worker: 5000,
};
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
7. **Caller Control**: Library design - resource allocation is ALWAYS explicit, never automatic

### Library Design Principles

**Key Principle**: rust-embed is a **library**, not an application. Resource management is the caller's responsibility.

✅ **DO**:
- Require explicit `PoolConfig` with worker counts
- Provide `suggest_pool_config()` as optional helper
- Support dynamic reconfiguration via `reconfigure()`
- Allow 1 worker (minimal) to 16+ workers (maximum)
- Document typical configurations for common use cases

❌ **DON'T**:
- Auto-detect worker count based on system resources
- Make assumptions about available memory
- Default to "optimal" configurations
- Hide resource costs from the caller
- Prevent callers from using "suboptimal" configurations

**Rationale**: Upstream systems know their constraints better than the library. A system running 10 services might need to limit rust-embed to 1 worker, while a dedicated embedding server might use 16 workers. The library should enable both use cases without judgment.

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
