/// Worker pool implementation for parallel embedding processing
///
/// This module implements a worker pool architecture where:
/// - Each worker owns a complete model instance, cache, and statistics
/// - Workers communicate via message passing (no shared state)
/// - Resource allocation is explicitly controlled by the caller
/// - Supports dynamic reconfiguration (add/remove workers at runtime)

use crate::embedding::Embedder;
use crate::models::mini_lm::{MiniLMEmbedder, EmbedderStats};
use anyhow::{anyhow, Result};
use crossbeam::channel::{self, Sender, Receiver};
use ndarray::Array1;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread::{self, JoinHandle};
use tokio::sync::oneshot;

/// Model types supported by the pool
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    /// MiniLM-L6-v2 (22M params, 384 dims, 512 max tokens, CPU-only)
    MiniLM,
    /// ModernBERT Base (149M params, 768 dims, 8192 max tokens, CPU/GPU hybrid)
    ModernBERTBase,
    /// ModernBERT Large (395M params, 1024 dims, 8192 max tokens, CPU/GPU hybrid)
    ModernBERTLarge,
}

impl ModelType {
    /// Get the embedding dimension for this model
    pub fn dimension(&self) -> usize {
        match self {
            ModelType::MiniLM => 384,
            ModelType::ModernBERTBase => 768,
            ModelType::ModernBERTLarge => 1024,
        }
    }

    /// Get the maximum sequence length for this model
    pub fn max_sequence_length(&self) -> usize {
        match self {
            ModelType::MiniLM => 512,
            ModelType::ModernBERTBase => 8192,
            ModelType::ModernBERTLarge => 8192,
        }
    }

    /// Get the approximate memory footprint in MB
    pub fn memory_footprint_mb(&self) -> usize {
        match self {
            ModelType::MiniLM => 90,
            ModelType::ModernBERTBase => 600,
            ModelType::ModernBERTLarge => 1600,
        }
    }

    /// Check if this model supports GPU acceleration
    pub fn supports_gpu(&self) -> bool {
        match self {
            ModelType::MiniLM => false,
            ModelType::ModernBERTBase | ModelType::ModernBERTLarge => true,
        }
    }

    /// Get the HuggingFace model ID
    pub fn huggingface_id(&self) -> &'static str {
        match self {
            ModelType::MiniLM => "sentence-transformers/all-MiniLM-L6-v2",
            ModelType::ModernBERTBase => "answerdotai/ModernBERT-base",
            ModelType::ModernBERTLarge => "answerdotai/ModernBERT-large",
        }
    }
}

/// Routing configuration for dynamic CPU/GPU selection (ModernBERT only)
/// Controls when the pool routes work to CPU vs GPU workers based on sequence length
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoutingConfig {
    /// Sequence length threshold for routing to GPU (default: 1024 tokens)
    /// Sequences >= this length are sent to GPU workers (if available)
    pub long_sequence_threshold: usize,

    /// Batch size threshold for preferring CPU (default: 512)
    /// Small batches below this size may be faster on CPU due to GPU overhead
    pub small_batch_threshold: usize,

    /// Batch size threshold for preferring GPU (default: 2048)
    /// Large batches above this size benefit most from GPU parallelism
    pub large_batch_threshold: usize,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            long_sequence_threshold: 1024,
            small_batch_threshold: 512,
            large_batch_threshold: 2048,
        }
    }
}

/// Worker pool configuration - MUST be explicitly provided by caller
/// No defaults - library does not make resource decisions
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Number of CPU workers (required, no default)
    pub cpu_workers: usize,

    /// Number of GPU workers (required, must be 0 for Phase 1)
    pub gpu_workers: usize,

    /// Model to use
    pub model: ModelType,

    /// Cache size per worker (number of embeddings to cache)
    ///
    /// **Cache Behavior:**
    /// - `0`: Disable caching entirely (recommended for unique message streams)
    /// - `100-500`: Minimal cache for catching duplicate messages
    /// - `1000-5000`: Moderate cache for mixed workloads
    /// - `10000+`: Large cache for high-repetition workloads (search queries)
    ///
    /// **When to use caching:**
    /// - Search query systems: Users repeat common searches → HIGH cache value
    /// - Recommendation systems: Same items embedded repeatedly → HIGH cache value
    /// - Real-time APIs: Common phrases appear frequently → MODERATE cache value
    ///
    /// **When to disable caching (set to 0):**
    /// - Document embedding pipelines: Each message is unique → NO cache value
    /// - Stream processing: Unique events/logs → NO cache value
    /// - Batch ETL: One-time dataset embedding → NO cache value
    ///
    /// For rust-embed's primary use case (embedding unique messages from upstream),
    /// consider setting this to `0` or a small value (100-500) to catch accidental duplicates.
    pub cache_size_per_worker: usize,

    /// Optional routing configuration for CPU/GPU selection (ModernBERT only)
    /// If None, uses RoutingConfig::default()
    /// Ignored for MiniLM (CPU-only model)
    pub routing_config: Option<RoutingConfig>,
}

impl PoolConfig {
    /// Create a minimal configuration (1 CPU worker, no caching)
    /// Suitable for low-throughput unique message streams
    pub fn minimal() -> Self {
        Self {
            cpu_workers: 1,
            gpu_workers: 0,
            model: ModelType::MiniLM,
            cache_size_per_worker: 0,  // No cache for unique messages
            routing_config: None,
        }
    }

    /// Create a balanced configuration (4 CPU workers, minimal cache)
    /// Suitable for document embedding pipelines with moderate throughput
    pub fn balanced() -> Self {
        Self {
            cpu_workers: 4,
            gpu_workers: 0,
            model: ModelType::MiniLM,
            cache_size_per_worker: 100,  // Small cache to catch duplicates
            routing_config: None,
        }
    }

    /// Create a high-throughput configuration (8 CPU workers, minimal cache)
    /// Suitable for high-volume document embedding pipelines
    pub fn high_throughput() -> Self {
        Self {
            cpu_workers: 8,
            gpu_workers: 0,
            model: ModelType::MiniLM,
            cache_size_per_worker: 100,  // Small cache to catch duplicates
            routing_config: None,
        }
    }

    /// Create configuration optimized for search queries (4 CPU workers, large cache)
    /// Suitable for systems where users repeat common searches
    pub fn search_optimized() -> Self {
        Self {
            cpu_workers: 4,
            gpu_workers: 0,
            model: ModelType::MiniLM,
            cache_size_per_worker: 10_000,  // Large cache for repeated queries
            routing_config: None,
        }
    }

    /// Get effective routing configuration (returns default if not specified)
    pub fn routing_config(&self) -> RoutingConfig {
        self.routing_config.unwrap_or_default()
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.cpu_workers == 0 && self.gpu_workers == 0 {
            return Err(anyhow!("Must specify at least one worker (CPU or GPU)"));
        }

        if self.gpu_workers > 0 {
            return Err(anyhow!("GPU workers not yet supported in Phase 1 (v0.2.0)"));
        }

        if self.cpu_workers > 16 {
            log::warn!("Using {} CPU workers - this may exceed system capacity", self.cpu_workers);
        }

        // Validate routing config if provided
        if let Some(routing) = self.routing_config {
            if routing.small_batch_threshold > routing.large_batch_threshold {
                return Err(anyhow!(
                    "small_batch_threshold ({}) must be <= large_batch_threshold ({})",
                    routing.small_batch_threshold,
                    routing.large_batch_threshold
                ));
            }

            if routing.long_sequence_threshold == 0 {
                return Err(anyhow!("long_sequence_threshold must be > 0"));
            }
        }

        // Routing config only makes sense for ModernBERT (hybrid CPU/GPU model)
        if self.routing_config.is_some() && self.model == ModelType::MiniLM {
            log::warn!(
                "routing_config specified but model is MiniLM (CPU-only). Routing config will be ignored."
            );
        }

        Ok(())
    }
}

/// Request sent to a worker
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

/// Individual embedding worker - completely isolated
struct EmbeddingWorker {
    id: usize,
    embedder: MiniLMEmbedder,
}

impl EmbeddingWorker {
    /// Initialize worker with model loaded
    fn new(id: usize, cache_size: usize) -> Result<Self> {
        log::info!("Worker {} initializing...", id);

        let mut config = crate::models::mini_lm::MiniLMConfig::default();
        config.cache_size_limit = cache_size;

        let mut embedder = MiniLMEmbedder::with_config(config);
        embedder.initialize()?;

        log::info!("Worker {} initialized successfully", id);

        Ok(Self { id, embedder })
    }

    /// Main worker loop - process requests until shutdown
    fn run(mut self, rx: Receiver<WorkerRequest>) {
        log::info!("Worker {} started and ready", self.id);

        for request in rx {
            match request {
                WorkerRequest::Embed { text, response_tx } => {
                    let result = self.embedder.embed_text(&text);
                    let _ = response_tx.send(result);
                }

                WorkerRequest::EmbedBatch { texts, response_tx } => {
                    let results: Result<Vec<Array1<f32>>> = texts
                        .iter()
                        .map(|text| self.embedder.embed_text(text))
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

        log::info!("Worker {} terminated", self.id);
    }
}

/// Pool of embedding workers with work distribution
pub struct EmbeddingPool {
    workers: Vec<Sender<WorkerRequest>>,
    handles: Vec<JoinHandle<()>>,
    next_worker: AtomicUsize,
    current_config: PoolConfig,
}

impl EmbeddingPool {
    /// Create pool with EXPLICIT configuration
    /// Caller MUST specify worker counts - no auto-detection
    pub fn new(config: PoolConfig) -> Result<Self> {
        config.validate()?;

        log::info!(
            "Creating embedding pool: {} CPU workers, model: {:?}",
            config.cpu_workers,
            config.model
        );

        let start = std::time::Instant::now();

        let mut workers = Vec::new();
        let mut handles = Vec::new();

        // Spawn workers in parallel for faster initialization
        let init_handles: Vec<_> = (0..config.cpu_workers)
            .map(|id| {
                let cache_size = config.cache_size_per_worker;
                thread::spawn(move || EmbeddingWorker::new(id, cache_size))
            })
            .collect();

        // Collect initialized workers and start their loops
        for (id, init_handle) in init_handles.into_iter().enumerate() {
            let worker = init_handle
                .join()
                .map_err(|_| anyhow!("Worker {} initialization panicked", id))??;

            let (tx, rx) = crossbeam::channel::unbounded();
            let handle = thread::spawn(move || worker.run(rx));

            workers.push(tx);
            handles.push(handle);
        }

        log::info!(
            "Embedding pool ready: {} workers in {:.2}s",
            config.cpu_workers,
            start.elapsed().as_secs_f64()
        );

        Ok(Self {
            workers,
            handles,
            next_worker: AtomicUsize::new(0),
            current_config: config,
        })
    }

    /// Reconfigure pool with new worker counts and routing settings
    /// - Spawns new workers if count increased
    /// - Gracefully shuts down excess workers if count decreased
    /// - Updates routing configuration (for ModernBERT CPU/GPU routing)
    /// - Allows upstream to dynamically adjust resource allocation
    ///
    /// Cannot change: model type (requires new pool)
    /// Can change: cpu_workers, gpu_workers, cache_size_per_worker, routing_config
    pub fn reconfigure(&mut self, new_config: PoolConfig) -> Result<()> {
        new_config.validate()?;

        if new_config.model != self.current_config.model {
            return Err(anyhow!(
                "Cannot change model type during reconfiguration. Create a new pool instead."
            ));
        }

        let routing_changed = new_config.routing_config != self.current_config.routing_config;

        log::info!(
            "Reconfiguring pool: {} → {} CPU workers{}",
            self.current_config.cpu_workers,
            new_config.cpu_workers,
            if routing_changed { " (routing config updated)" } else { "" }
        );

        match new_config.cpu_workers.cmp(&self.workers.len()) {
            std::cmp::Ordering::Greater => {
                // Spawn additional workers
                let to_spawn = new_config.cpu_workers - self.workers.len();

                let init_handles: Vec<_> = (0..to_spawn)
                    .map(|i| {
                        let id = self.workers.len() + i;
                        let cache_size = new_config.cache_size_per_worker;
                        thread::spawn(move || EmbeddingWorker::new(id, cache_size))
                    })
                    .collect();

                for init_handle in init_handles {
                    let worker = init_handle
                        .join()
                        .map_err(|_| anyhow!("Worker initialization panicked"))??;

                    let (tx, rx) = crossbeam::channel::unbounded();
                    let handle = thread::spawn(move || worker.run(rx));

                    self.workers.push(tx);
                    self.handles.push(handle);
                }

                log::info!("Spawned {} additional workers", to_spawn);
            }
            std::cmp::Ordering::Less => {
                // Shutdown excess workers
                let to_remove = self.workers.len() - new_config.cpu_workers;

                for _ in 0..to_remove {
                    if let Some(worker) = self.workers.pop() {
                        let _ = worker.send(WorkerRequest::Shutdown);
                    }
                }

                log::info!("Removed {} workers", to_remove);
            }
            std::cmp::Ordering::Equal => {
                log::debug!("Worker count unchanged");
            }
        }

        self.current_config = new_config;
        log::info!("Reconfiguration complete");

        Ok(())
    }

    /// Get current configuration
    pub fn config(&self) -> &PoolConfig {
        &self.current_config
    }

    /// Get current worker count
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    /// Get next worker (round-robin distribution)
    fn get_worker(&self) -> &Sender<WorkerRequest> {
        let idx = self.next_worker.fetch_add(1, Ordering::Relaxed);
        &self.workers[idx % self.workers.len()]
    }

    /// Embed single text
    pub fn embed_text(&self, text: String) -> Result<Array1<f32>> {
        let (tx, rx) = oneshot::channel();

        self.get_worker().send(WorkerRequest::Embed {
            text,
            response_tx: tx,
        })?;

        rx.blocking_recv()?
    }

    /// Embed batch (distributes across workers)
    pub fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Array1<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Split batch into chunks (one per worker)
        let chunk_size = (texts.len() + self.workers.len() - 1) / self.workers.len();

        let mut receivers = vec![];

        for chunk in texts.chunks(chunk_size) {
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
            let chunk_results = rx.blocking_recv()??;
            results.extend(chunk_results);
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
            let stats = rx.blocking_recv()?;
            total.embeddings_count += stats.embeddings_count;
            total.cache_hits += stats.cache_hits;
            total.cache_misses += stats.cache_misses;
            total.total_processing_time += stats.total_processing_time;
        }

        Ok(total)
    }

    /// Clear cache on all workers
    pub fn clear_all_caches(&self) -> Result<()> {
        for worker in &self.workers {
            worker.send(WorkerRequest::ClearCache)?;
        }
        Ok(())
    }

    /// Graceful shutdown of all workers
    pub fn shutdown(self) -> Result<()> {
        log::info!("Shutting down pool ({} workers)...", self.workers.len());

        // Send shutdown to all workers
        for worker in &self.workers {
            let _ = worker.send(WorkerRequest::Shutdown);
        }

        // Wait for all worker threads to finish
        for handle in self.handles {
            handle
                .join()
                .map_err(|_| anyhow!("Worker panic during shutdown"))?;
        }

        log::info!("Pool shutdown complete");

        Ok(())
    }
}

impl Drop for EmbeddingPool {
    fn drop(&mut self) {
        // Attempt graceful shutdown on drop
        for worker in &self.workers {
            let _ = worker.send(WorkerRequest::Shutdown);
        }
    }
}

/// Suggestion for pool configuration based on system resources
/// This is OPTIONAL - caller is free to ignore and use their own config
#[derive(Debug, Clone)]
pub struct PoolSuggestion {
    pub cpu_workers: usize,
    pub gpu_workers: usize,
    pub cache_size_per_worker: usize,
    pub note: String,
}

/// Suggest pool configuration based on system resources
/// This is a HELPER function - caller can ignore and use their own config
pub fn suggest_pool_config() -> PoolSuggestion {
    let num_cpus = num_cpus::get();
    let available_ram_gb = estimate_available_ram_gb();

    // Conservative suggestions that leave headroom for OS and other apps
    let suggested_cpu = if crate::utils::is_apple_silicon() {
        match num_cpus {
            1..=4 => 2,
            5..=8 => 4,
            9..=12 => 6,
            13..=16 => 8,
            _ => 10,
        }
    } else {
        // Intel/AMD: Use 75% of cores
        ((num_cpus * 3) / 4).max(1).min(8)
    };

    // For document embedding pipelines (primary use case), use minimal cache
    // Caller should increase if they have high-repetition workloads (search queries)
    let cache_size = 100;  // Small cache to catch accidental duplicates

    PoolSuggestion {
        cpu_workers: suggested_cpu,
        gpu_workers: 0, // Phase 1: No GPU support yet
        cache_size_per_worker: cache_size,
        note: format!(
            "Suggestion based on {} CPUs and ~{} GB RAM. Cache set to {} for document embedding. \
             Increase to 10000+ for search query workloads. Set to 0 to disable caching.",
            num_cpus, available_ram_gb, cache_size
        ),
    }
}

/// Estimate available RAM in GB (rough estimate)
fn estimate_available_ram_gb() -> usize {
    // This is a rough estimate - actual implementation would need platform-specific code
    // For now, return a conservative default
    #[cfg(target_os = "macos")]
    {
        // On macOS, use sysctl to get memory info
        use std::process::Command;
        if let Ok(output) = Command::new("sysctl").arg("hw.memsize").output() {
            if let Ok(s) = String::from_utf8(output.stdout) {
                if let Some(value) = s.split(':').nth(1) {
                    if let Ok(bytes) = value.trim().parse::<usize>() {
                        return bytes / (1024 * 1024 * 1024);
                    }
                }
            }
        }
    }

    // Default to 16 GB if we can't determine
    16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_config_validation() {
        // Valid config
        let config = PoolConfig {
            cpu_workers: 4,
            gpu_workers: 0,
            model: ModelType::MiniLM,
            cache_size_per_worker: 5000,
            routing_config: None,
        };
        assert!(config.validate().is_ok());

        // Invalid: no workers
        let config = PoolConfig {
            cpu_workers: 0,
            gpu_workers: 0,
            model: ModelType::MiniLM,
            cache_size_per_worker: 5000,
            routing_config: None,
        };
        assert!(config.validate().is_err());

        // Invalid: GPU workers not supported yet
        let config = PoolConfig {
            cpu_workers: 2,
            gpu_workers: 1,
            model: ModelType::MiniLM,
            cache_size_per_worker: 5000,
            routing_config: None,
        };
        assert!(config.validate().is_err());

        // Invalid: small_batch_threshold > large_batch_threshold
        let config = PoolConfig {
            cpu_workers: 4,
            gpu_workers: 0,
            model: ModelType::MiniLM,
            cache_size_per_worker: 5000,
            routing_config: Some(RoutingConfig {
                long_sequence_threshold: 1024,
                small_batch_threshold: 3000,
                large_batch_threshold: 2000,
            }),
        };
        assert!(config.validate().is_err());

        // Invalid: long_sequence_threshold = 0
        let config = PoolConfig {
            cpu_workers: 4,
            gpu_workers: 0,
            model: ModelType::MiniLM,
            cache_size_per_worker: 5000,
            routing_config: Some(RoutingConfig {
                long_sequence_threshold: 0,
                small_batch_threshold: 512,
                large_batch_threshold: 2048,
            }),
        };
        assert!(config.validate().is_err());

        // Valid: custom routing config
        let config = PoolConfig {
            cpu_workers: 4,
            gpu_workers: 0,
            model: ModelType::MiniLM,
            cache_size_per_worker: 5000,
            routing_config: Some(RoutingConfig {
                long_sequence_threshold: 2048,
                small_batch_threshold: 256,
                large_batch_threshold: 4096,
            }),
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_pool_config_presets() {
        let minimal = PoolConfig::minimal();
        assert_eq!(minimal.cpu_workers, 1);
        assert_eq!(minimal.cache_size_per_worker, 0);  // No cache for unique messages

        let balanced = PoolConfig::balanced();
        assert_eq!(balanced.cpu_workers, 4);
        assert_eq!(balanced.cache_size_per_worker, 100);  // Small cache to catch duplicates

        let high = PoolConfig::high_throughput();
        assert_eq!(high.cpu_workers, 8);
        assert_eq!(high.cache_size_per_worker, 100);  // Small cache to catch duplicates

        let search = PoolConfig::search_optimized();
        assert_eq!(search.cpu_workers, 4);
        assert_eq!(search.cache_size_per_worker, 10_000);  // Large cache for repeated queries
    }

    #[test]
    fn test_cache_disabled_config() {
        // Valid: cache_size_per_worker = 0 (cache disabled)
        let config = PoolConfig {
            cpu_workers: 4,
            gpu_workers: 0,
            model: ModelType::MiniLM,
            cache_size_per_worker: 0,  // Disabled cache
            routing_config: None,
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_suggest_pool_config() {
        let suggestion = suggest_pool_config();
        assert!(suggestion.cpu_workers > 0);
        assert_eq!(suggestion.gpu_workers, 0); // Phase 1: No GPU
        assert!(suggestion.cache_size_per_worker > 0);
    }
}
