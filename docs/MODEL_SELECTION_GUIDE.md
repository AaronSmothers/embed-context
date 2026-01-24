# Model Selection Guide
**rust-embed Library**
**Date:** 2026-01-24
**Version:** v0.2.0+ (MiniLM), v0.3.0+ (ModernBERT)

---

## Overview

rust-embed supports multiple embedding models that can be selected at configuration time. **Both models are available simultaneously** - the caller chooses which to use based on their specific requirements.

```rust
// Choose your model via PoolConfig.model
pub enum ModelType {
    MiniLM,      // Fast, efficient, 384 dims
    ModernBERT,  // Quality, long context, 768 dims
}
```

---

## Quick Decision Matrix

| Your Need | Choose | Why |
|-----------|--------|-----|
| **Speed** | MiniLM | 3× faster per embedding |
| **Memory constrained** (<16 GB) | MiniLM | 6× less memory |
| **Long texts** (>512 tokens) | ModernBERT | MiniLM capped at 512 tokens |
| **Short texts** (<100 tokens) | MiniLM | Optimal for queries/snippets |
| **High throughput** | MiniLM | Can run 6× more workers |
| **High quality** | ModernBERT | Larger model, better representations |
| **GPU available** | ModernBERT | Can leverage MPS acceleration |
| **Simple deployment** | MiniLM | CPU-only, no GPU needed |

---

## Model Comparison

### Technical Specifications

| Specification | MiniLM-L6-v2 | ModernBERT Base | Ratio |
|---------------|--------------|-----------------|-------|
| **Parameters** | 22M | 149M | 7× larger |
| **Embedding Dimensions** | 384 | 768 | 2× larger |
| **Max Sequence Length** | 512 tokens | 8,192 tokens | 16× longer |
| **Model Size (FP32)** | ~90 MB | ~600 MB | 7× larger |
| **Memory per Worker** | ~100 MB | ~1-3 GB | 10-30× more |
| **CPU Latency (128 tokens)** | ~10 ms | ~30 ms | 3× slower |
| **GPU Support** | No | Yes (MPS) | N/A |

### Performance Characteristics

| Metric | MiniLM-L6-v2 | ModernBERT Base |
|--------|--------------|-----------------|
| **Throughput (6 workers)** | ~570 emb/sec | ~200 emb/sec (CPU) |
| **Memory (6 workers)** | 622 MB | 6-8 GB |
| **Best For** | Short texts, high volume | Long texts, quality |
| **Device Strategy** | CPU only | Hybrid CPU/GPU |

---

## Use Case Examples

### Use Case 1: Search Engine Query Embeddings

**Requirements:**
- High throughput (millions of queries/day)
- Short texts (5-20 words)
- Low latency (<20ms)
- Memory constrained (shared with other services)

**Recommended:**
```rust
let config = PoolConfig {
    cpu_workers: 6,
    gpu_workers: 0,
    model: ModelType::MiniLM,
    cache_size_per_worker: 10_000,
};
```

**Rationale:** MiniLM excels at short texts, provides high throughput, and uses minimal memory.

---

### Use Case 2: Document Embedding for RAG

**Requirements:**
- Long documents (500-5000 words)
- Quality is critical
- Processing batches (not real-time)
- Memory available (≥32 GB)

**Recommended:**
```rust
let config = PoolConfig {
    cpu_workers: 4,
    gpu_workers: 1,
    model: ModelType::ModernBERT,
    cache_size_per_worker: 3_000,
};
```

**Rationale:** ModernBERT handles long context (8192 tokens), produces higher quality embeddings, and GPU accelerates processing.

---

### Use Case 3: Mixed Workload (Queries + Documents)

**Requirements:**
- Both short queries and long documents
- Need to embed both types efficiently
- Memory available (≥24 GB)

**Recommended: Two Pools**
```rust
// Pool 1: MiniLM for queries
let query_pool = EmbeddingPool::new(PoolConfig {
    cpu_workers: 4,
    gpu_workers: 0,
    model: ModelType::MiniLM,
    cache_size_per_worker: 5_000,
})?;

// Pool 2: ModernBERT for documents
let doc_pool = EmbeddingPool::new(PoolConfig {
    cpu_workers: 2,
    gpu_workers: 1,
    model: ModelType::ModernBERT,
    cache_size_per_worker: 2_000,
})?;

// Route based on content type
if is_query(text) {
    query_pool.embed_text(text)?
} else {
    doc_pool.embed_text(text)?
}
```

**Rationale:** Separate pools optimize for each workload type. MiniLM for speed on queries, ModernBERT for quality on documents.

---

### Use Case 4: Batch Processing Large Corpus

**Requirements:**
- Process millions of documents
- Not real-time (batch job)
- Maximize throughput
- Memory available (≥16 GB)

**Option A: MiniLM (Maximum Throughput)**
```rust
let config = PoolConfig {
    cpu_workers: 8,
    gpu_workers: 0,
    model: ModelType::MiniLM,
    cache_size_per_worker: 5_000,
};
// Throughput: ~760 emb/sec
// Memory: ~832 MB
```

**Option B: ModernBERT (Higher Quality)**
```rust
let config = PoolConfig {
    cpu_workers: 6,
    gpu_workers: 1,
    model: ModelType::ModernBERT,
    cache_size_per_worker: 3_000,
};
// Throughput: ~250 emb/sec (mixed CPU/GPU)
// Memory: ~9 GB
```

**Decision Criteria:** If throughput is critical and quality is acceptable, use MiniLM. If quality is critical and time is available, use ModernBERT.

---

## Model Selection API

### Configuration Examples

```rust
use rust_embed::{EmbeddingPool, PoolConfig, ModelType};

// Example 1: MiniLM for speed
let pool = EmbeddingPool::new(PoolConfig {
    cpu_workers: 6,
    gpu_workers: 0,
    model: ModelType::MiniLM,
    cache_size_per_worker: 5000,
})?;

// Example 2: ModernBERT for quality (CPU only)
let pool = EmbeddingPool::new(PoolConfig {
    cpu_workers: 4,
    gpu_workers: 0,
    model: ModelType::ModernBERT,
    cache_size_per_worker: 3000,
})?;

// Example 3: ModernBERT with GPU acceleration
let pool = EmbeddingPool::new(PoolConfig {
    cpu_workers: 4,
    gpu_workers: 1,
    model: ModelType::ModernBERT,
    cache_size_per_worker: 3000,
})?;
```

### Switching Models

**Cannot switch during reconfigure**: Model change requires new pool:

```rust
// Start with MiniLM
let pool = EmbeddingPool::new(PoolConfig {
    model: ModelType::MiniLM,
    // ...
})?;

// Process with MiniLM
pool.embed_batch(texts)?;
pool.shutdown()?;

// Switch to ModernBERT: Must create new pool
let pool = EmbeddingPool::new(PoolConfig {
    model: ModelType::ModernBERT,
    // ...
})?;

// Process with ModernBERT
pool.embed_batch(texts)?;
```

---

## Memory Planning

### MiniLM Memory Requirements

| Workers | Model Memory | Cache Memory | Total |
|---------|--------------|--------------|-------|
| 1 | ~90 MB | ~7.7 MB | ~100 MB |
| 4 | ~360 MB | ~31 MB | ~400 MB |
| 6 | ~540 MB | ~46 MB | ~600 MB |
| 8 | ~720 MB | ~62 MB | ~800 MB |

**Rule of thumb**: ~100 MB per worker

### ModernBERT Memory Requirements

| Workers | Type | Model Memory | Cache Memory | Total |
|---------|------|--------------|--------------|-------|
| 1 | CPU | ~600 MB | ~10 MB | ~1 GB |
| 2 | CPU | ~1.2 GB | ~20 MB | ~2 GB |
| 4 | CPU | ~2.4 GB | ~40 MB | ~3 GB |
| 1 | GPU | ~600 MB | ~10 MB | ~2-3 GB* |
| 4 CPU + 1 GPU | Mixed | ~3 GB | ~50 MB | ~7 GB |
| 8 CPU + 2 GPU | Mixed | ~6 GB | ~100 MB | ~14 GB |

*GPU workers need extra memory for activation/intermediate tensors

**Rule of thumb**: ~1 GB per CPU worker, ~2-3 GB per GPU worker

### System RAM Recommendations

| Available RAM | MiniLM Config | ModernBERT Config |
|---------------|---------------|-------------------|
| **8 GB** | 4-6 workers | Not recommended |
| **16 GB** | 8-12 workers | 2-4 CPU workers |
| **24 GB** | 12+ workers | 4-6 CPU workers or 3 CPU + 1 GPU |
| **32 GB** | 16+ workers | 8 CPU + 2 GPU |
| **64 GB** | 32+ workers | 12 CPU + 4 GPU |

---

## Performance Tuning

### MiniLM Best Practices

1. **Worker Count**: Use 1 worker per CPU core (up to 8-12)
2. **Cache Size**: 5,000-10,000 embeddings per worker
3. **Batch Size**: 100-1000 texts per batch
4. **Device**: CPU only (GPU not beneficial)

### ModernBERT Best Practices

1. **Worker Count**: Balance CPU and GPU workers based on sequence length distribution
2. **Cache Size**: 2,000-5,000 embeddings per worker (larger embeddings)
3. **Batch Size**: 50-500 texts per batch
4. **Device**:
   - Short texts (<512 tokens) → CPU workers
   - Long texts (>1024 tokens) → GPU workers
   - Mixed workload → 4 CPU + 1-2 GPU workers

### Dynamic Routing (ModernBERT Only)

ModernBERT automatically routes requests to CPU or GPU based on sequence length:

```
seq_len < 512 tokens    → CPU (GPU overhead not worth it)
seq_len ≥ 1024 tokens   → GPU (attention parallelism wins)
512 ≤ seq_len < 1024    → CPU (default)
```

This happens transparently - caller doesn't need to manage routing.

---

## CLI Usage

### Using MiniLM

```bash
# Default: Uses MiniLM
cargo run -- --file texts.txt --workers 6 --output embeddings.bin

# Explicit MiniLM
cargo run -- --file texts.txt --model minilm --workers 6
```

### Using ModernBERT

```bash
# ModernBERT with CPU only
cargo run -- --file texts.txt --model modernbert --workers 4

# ModernBERT with GPU (not yet implemented in CLI)
cargo run -- --file texts.txt --model modernbert --workers 4 --gpu-workers 1
```

---

## Migration Guide

### From MiniLM to ModernBERT

If you're currently using MiniLM and want to switch to ModernBERT:

**What to change:**
```rust
// Before
let pool = EmbeddingPool::new(PoolConfig {
    cpu_workers: 6,
    gpu_workers: 0,
    model: ModelType::MiniLM,
    cache_size_per_worker: 5000,
})?;

// After
let pool = EmbeddingPool::new(PoolConfig {
    cpu_workers: 4,        // Fewer workers (larger model)
    gpu_workers: 1,        // Optional: Add GPU
    model: ModelType::ModernBERT,  // ← Model change
    cache_size_per_worker: 3000,   // Smaller cache (larger embeddings)
})?;
```

**What changes automatically:**
- Embedding dimensions: 384 → 768
- Output vectors are automatically normalized (same as MiniLM)
- API remains the same (no code changes needed besides config)

**What to verify:**
- Memory usage: ~600 MB → ~7 GB (example config above)
- Storage: Embeddings are 2× larger (768 vs 384 floats)
- Latency: ~10ms → ~30-50ms per embedding (depends on sequence length)

---

## Frequently Asked Questions

### Can I use both models in the same application?

Yes! Create separate pools:

```rust
let minilm_pool = EmbeddingPool::new(PoolConfig {
    model: ModelType::MiniLM,
    // ...
})?;

let modernbert_pool = EmbeddingPool::new(PoolConfig {
    model: ModelType::ModernBERT,
    // ...
})?;
```

### Can I switch models without restarting?

Yes, but you must create a new pool. Use `shutdown()` on the old pool, then create a new one with the different model.

### Does ModernBERT require a GPU?

No. ModernBERT works on CPU only. GPU (MPS on Apple Silicon) is optional and provides speedup for long sequences.

### Which model is more accurate?

ModernBERT generally produces higher quality embeddings due to its larger size (149M vs 22M parameters). However, for short texts (<100 tokens), the difference may be minimal.

### Can I mix CPU and GPU workers for MiniLM?

No. MiniLM is CPU-only. GPU workers are only supported for ModernBERT.

### How do I know which model to use?

Use the decision matrix at the top of this document. General rule:
- **Short texts + speed needed** → MiniLM
- **Long texts + quality needed** → ModernBERT

---

## References

- [Worker Pool Architecture](./WORKER_POOL_ARCHITECTURE.md)
- [ModernBERT Implementation Specification](./MODERNBERT_IMPLEMENTATION.md)
- [MiniLM Model Card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [ModernBERT Model Card](https://huggingface.co/nomic-ai/modernbert-embed-base)
