# Architecture Revision: Model Coexistence
**Date:** 2026-01-24
**Status:** Approved

---

## Summary of Changes

The architecture has been revised to support **both MiniLM and ModernBERT as selectable options** rather than sequential phases. This aligns better with library design principles and gives callers flexibility.

---

## Key Decision

**Before:** ModernBERT would replace MiniLM in Phase 2 (v0.3.0)

**After:** Both models available simultaneously, caller selects via configuration

```rust
// Caller chooses at configuration time
pub enum ModelType {
    MiniLM,      // Fast, efficient, 384 dims, CPU-only
    ModernBERT,  // Quality, long context, 768 dims, hybrid CPU/GPU
}

// Select via PoolConfig
let pool = EmbeddingPool::new(PoolConfig {
    cpu_workers: 6,
    gpu_workers: 0,
    model: ModelType::MiniLM,  // ← Caller chooses
    cache_size_per_worker: 5000,
})?;
```

---

## Rationale

### 1. Library Design Principle

rust-embed is a library, not an application. **Caller should control model selection**, not be forced into one model.

Different use cases have different optimal models:
- Short texts + speed → MiniLM
- Long texts + quality → ModernBERT
- Memory constrained → MiniLM
- GPU available → ModernBERT

### 2. No Forced Migration

Existing MiniLM users can continue using MiniLM. No breaking changes. New ModernBERT support is **additive**.

### 3. Resource Control

Callers can allocate resources appropriately:
- MiniLM: ~600 MB for 6 workers
- ModernBERT: ~7 GB for 4 CPU + 1 GPU workers

### 4. Multiple Pools Possible

Applications can run both models simultaneously:

```rust
// Pool 1: MiniLM for queries
let query_pool = EmbeddingPool::new(PoolConfig {
    model: ModelType::MiniLM,
    cpu_workers: 4,
    // ...
})?;

// Pool 2: ModernBERT for documents
let doc_pool = EmbeddingPool::new(PoolConfig {
    model: ModelType::ModernBERT,
    cpu_workers: 2,
    gpu_workers: 1,
    // ...
})?;
```

---

## Model Comparison

| Aspect | MiniLM-L6-v2 | ModernBERT Base |
|--------|--------------|-----------------|
| **Parameters** | 22M | 149M (7× larger) |
| **Dimensions** | 384 | 768 (2× larger) |
| **Max Tokens** | 512 | 8,192 (16× longer) |
| **Memory/Worker** | ~100 MB | ~1-3 GB (10-30× more) |
| **CPU Latency** | ~10 ms | ~30 ms (3× slower) |
| **GPU Support** | No | Yes (PyTorch MPS) |
| **Best For** | Short texts, speed | Long texts, quality |

---

## GPU Backend: PyTorch MPS (Not MLX)

ModernBERT GPU acceleration uses **PyTorch's Metal Performance Shaders (MPS) backend** via the tch-rs crate. This is the same PyTorch backend used throughout rust-embed.

### Why PyTorch MPS?
- **Unified backend**: Same inference engine for CPU and GPU paths
- **Shared memory**: Apple Silicon UMA enables zero-copy tensor access
- **Rust bindings**: tch-rs provides mature, well-tested Rust bindings
- **Consistency**: Matches existing MiniLM implementation

### Why Not MLX?
- **No Rust bindings**: MLX is Python/C++ only
- **Fragmentation**: Would require two separate backends
- **Maintenance burden**: Additional complexity for marginal benefit
- **PyTorch sufficiency**: MPS provides adequate GPU acceleration

**Decision**: All GPU work uses PyTorch MPS. MLX is explicitly not supported.

---

## Model Selection Criteria

### Use MiniLM When:
- ✅ Processing short texts (<512 tokens)
- ✅ Speed is priority
- ✅ Memory constrained (<16 GB)
- ✅ High throughput needed (millions/day)
- ✅ Simple deployment (no GPU)

### Use ModernBERT When:
- ✅ Long texts (>512 tokens, up to 8192)
- ✅ Quality is priority
- ✅ Memory available (≥16 GB)
- ✅ GPU available (optional but beneficial)
- ✅ Processing documents, not queries

---

## API Design

### Model Selection

```rust
// Option 1: MiniLM for speed
let pool = EmbeddingPool::new(PoolConfig {
    cpu_workers: 6,
    gpu_workers: 0,
    model: ModelType::MiniLM,
    cache_size_per_worker: 5000,
})?;

// Option 2: ModernBERT for quality
let pool = EmbeddingPool::new(PoolConfig {
    cpu_workers: 4,
    gpu_workers: 1,
    model: ModelType::ModernBERT,
    cache_size_per_worker: 3000,
})?;
```

### Switching Models

**Cannot switch during reconfigure** (different model weights, tokenizer, etc.). Must create new pool:

```rust
// Start with MiniLM
let pool = EmbeddingPool::new(config_minilm)?;
pool.embed_batch(texts)?;
pool.shutdown()?;

// Switch to ModernBERT
let pool = EmbeddingPool::new(config_modernbert)?;
pool.embed_batch(texts)?;
```

### Validation

```rust
impl PoolConfig {
    pub fn validate(&self) -> Result<()> {
        // Must have at least one worker
        if self.cpu_workers == 0 && self.gpu_workers == 0 {
            return Err(anyhow!("Must specify at least one worker"));
        }

        // Model-specific validation
        match self.model {
            ModelType::MiniLM => {
                if self.gpu_workers > 0 {
                    return Err(anyhow!("MiniLM does not support GPU workers"));
                }
            }
            ModelType::ModernBERT => {
                // GPU workers are optional but supported
            }
        }

        Ok(())
    }
}
```

---

## Implementation Roadmap

### v0.2.0 (Current - Completed)
- ✅ Worker pool architecture
- ✅ MiniLM support
- ✅ CPU-only workers
- ✅ Explicit configuration
- ✅ Dynamic reconfiguration

### v0.2.1 or v0.3.0 (Next)
- [ ] Add ModernBERT to ModelType enum
- [ ] Implement ModernBERT model loading
- [ ] Implement mean pooling for ModernBERT
- [ ] Add GPU worker support via PyTorch MPS (Device::Mps)
- [ ] Leverage Apple Silicon shared memory (UMA) for efficient GPU access
- [ ] Implement dynamic CPU/GPU routing
- [ ] Update worker initialization for both models
- [ ] CLI flag: `--model minilm` or `--model modernbert`
- [ ] Integration tests with both models
- [ ] Documentation and examples

**Note**: GPU acceleration uses PyTorch MPS backend only. MLX is not supported.

---

## Documentation Updates

### New Documents
- **MODEL_SELECTION_GUIDE.md** - Complete guide for choosing between models

### Updated Documents
- **WORKER_POOL_ARCHITECTURE.md** - Model coexistence strategy
- **MODERNBERT_IMPLEMENTATION.md** - Changed from replacement to option

### Key Sections Added
1. Model comparison tables
2. Use case examples (4 scenarios)
3. Memory planning guide
4. Model switching examples
5. Decision matrices
6. Performance tuning per model

---

## Benefits

### For Library Users

1. **Flexibility**: Choose optimal model for use case
2. **No Forced Migration**: Continue using MiniLM if it works
3. **Resource Control**: Allocate memory appropriately
4. **Mixed Workloads**: Use both models in same app

### For Library Design

1. **Proper Abstraction**: Model is a configuration parameter
2. **Backward Compatible**: Existing code works unchanged
3. **Extensible**: Easy to add more models in future
4. **Clear Contracts**: Model capabilities documented

### For Performance

1. **Optimal Selection**: Right tool for the job
2. **Memory Efficient**: Don't load large model unnecessarily
3. **Speed**: MiniLM 3× faster for short texts
4. **Quality**: ModernBERT better for long documents

---

## Example Use Cases

### Use Case 1: Search Engine (MiniLM)

```rust
// Millions of short queries per day
// Need speed and low latency
let pool = EmbeddingPool::new(PoolConfig {
    cpu_workers: 8,
    gpu_workers: 0,
    model: ModelType::MiniLM,
    cache_size_per_worker: 10_000,
})?;

// Throughput: ~760 emb/sec
// Memory: ~832 MB
```

### Use Case 2: Document RAG (ModernBERT)

```rust
// Long documents (500-5000 words)
// Quality is critical
let pool = EmbeddingPool::new(PoolConfig {
    cpu_workers: 4,
    gpu_workers: 1,
    model: ModelType::ModernBERT,
    cache_size_per_worker: 3_000,
})?;

// Handles up to 8192 tokens
// Memory: ~7 GB
```

### Use Case 3: Mixed Workload (Both Models)

```rust
// Queries + Documents
let query_pool = EmbeddingPool::new(PoolConfig {
    cpu_workers: 4,
    model: ModelType::MiniLM,
    // ...
})?;

let doc_pool = EmbeddingPool::new(PoolConfig {
    cpu_workers: 2,
    gpu_workers: 1,
    model: ModelType::ModernBERT,
    // ...
})?;

// Route based on content type
if is_query(text) {
    query_pool.embed_text(text)?
} else {
    doc_pool.embed_text(text)?
}
```

---

## Memory Planning

### MiniLM
- 1 worker: ~100 MB
- 6 workers: ~600 MB
- 8 workers: ~800 MB

**Recommendation:** Use on systems with ≥8 GB RAM

### ModernBERT
- 1 CPU worker: ~1 GB
- 4 CPU workers: ~4 GB
- 4 CPU + 1 GPU: ~7 GB
- 8 CPU + 2 GPU: ~14 GB

**Recommendation:** Use on systems with ≥16 GB RAM (32 GB for hybrid)

---

## Next Steps

1. ✅ **Architecture revised** (this document)
2. ✅ **Documentation updated** (all docs)
3. ⏳ **Implementation**:
   - Add ModernBERT support to worker pool
   - Implement GPU worker initialization
   - Add dynamic routing logic
   - Update CLI with `--model` flag
   - Integration tests

4. ⏳ **Testing**:
   - Benchmark both models
   - Tune routing thresholds
   - Performance comparisons

5. ⏳ **Release**:
   - v0.2.1 or v0.3.0 with both models
   - Migration guide
   - Blog post on model selection

---

## Conclusion

The revised architecture provides **maximum flexibility** while maintaining **library design principles**:

- ✅ Caller controls model selection
- ✅ No forced migration
- ✅ Optimal for different use cases
- ✅ Resource efficient
- ✅ Backward compatible
- ✅ Extensible (easy to add more models)

**Both MiniLM and ModernBERT are available** - caller chooses based on their requirements. This is the correct library design.
