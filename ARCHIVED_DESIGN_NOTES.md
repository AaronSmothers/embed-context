# Archived Design Notes — Worker Pool, Cache Toggle, ModernBERT

**Status:** Archived (intentionally not on `main`)
**Source commits being discarded:** `9ca8c64..f66d10b` on `main` as of 2026-05-18
**Reason for archive:** The ModernBERT-driven branch of work mixed real, working code with unimplemented stubs and aspirational documentation. Rather than carry the half-finished surface forward, `main` is being reset to the last pre-modernbert commit (`3bb00b0`, "v0.0.2 README revision"). This document captures the intended designs and the actual halfway state at the time of removal so the ideas can be revived deliberately rather than reconstructed from a tainted history.

These notes are preserved on the `archived-design-notes` branch only. They are deliberately not part of `main`'s working tree.

---

## 1. Worker Pool

### Intent

Replace `rayon::par_iter()`-based parallel embedding with an explicit, message-passing worker pool. Each worker would own its own `MiniLMEmbedder` (model + cache + stats), with no shared state and no locks on the hot path. The caller would always specify `cpu_workers` and `gpu_workers` explicitly — the library would never auto-detect resources.

Key design properties intended:

- **Shared-nothing workers**: each worker holds its own model, cache, and statistics; no `Arc<RwLock<...>>` on hot paths.
- **Message-passing**: `crossbeam::channel` for fan-out to workers, `tokio::sync::oneshot` for replies (the only piece of `tokio` actually used).
- **Round-robin distribution** as the default; pluggable routing for hybrid CPU/GPU pools.
- **Dynamic reconfiguration**: `EmbeddingPool::reconfigure(new_config)` to scale worker count up/down without tearing the pool down.
- **Backpressure** via bounded channels (planned but not implemented — current channels are unbounded).
- **Aggregate stats**: `aggregate_stats()` collects and sums per-worker counters.
- **Library principle**: configuration is mandatory, no defaults like "use all cores".

Public API surface that was envisioned (and largely landed):

- `PoolConfig { cpu_workers, gpu_workers, model, cache_size_per_worker, routing_config }`
- `EmbeddingPool::new(config)`, `embed_text`, `embed_batch`, `aggregate_stats`, `reconfigure`, `shutdown`
- Convenience presets: `PoolConfig::minimal()`, `balanced()`, `high_throughput()`, `search_optimized()`
- `suggest_pool_config()` — opt-in helper that returns a `PoolSuggestion`; caller is free to ignore.

### Actual halfway state at time of archive

What was real and worked:

- `src/pool/mod.rs` (~770 lines) implements `EmbeddingPool`, `PoolConfig`, `WorkerRequest`, `EmbeddingWorker` for the **MiniLM-only path**.
- Round-robin via `AtomicUsize` works.
- Per-worker cache, per-worker stats aggregation works.
- `reconfigure()` for CPU workers (scale up, scale down, no-op) works.
- Graceful `shutdown()` and `Drop` impl that sends shutdown messages.
- Integration tests in `tests/pool_integration_tests.rs` covering single/multi-worker, batch, reconfigure, stats, similarity, presets, empty batch.
- `src/main.rs` and `src/bin/similarity.rs` migrated to use `EmbeddingPool`.
- `crossbeam`, `num_cpus`, narrow `tokio` `sync` usage added to `Cargo.toml`. Version bumped to `0.2.0`.

What was halfway or aspirational:

- `gpu_workers` was a public field but `validate()` rejects any value > 0 with `"GPU workers not yet supported in Phase 1 (v0.2.0)"`. Every public API mentions GPU; no GPU code path actually exists.
- Worker init has `cache_size` only — no `WorkerConfig` struct, no `device` field, no `queue_size`. The architecture doc described all three.
- All channels are `crossbeam::channel::unbounded()`. The "bounded queues / natural backpressure" property is documented but not implemented.
- `route_worker()` / `RequestRouter` from the design doc do not exist. The implementation uses a single `get_worker()` round-robin method. Hybrid routing is conceptual only.
- `reconfigure()` only changes CPU worker counts. GPU worker count, cache size, and routing config are accepted in the new config but ignored or unused at the worker level.
- `estimate_available_ram_gb()` shells out to `sysctl hw.memsize` on macOS; on every other OS it returns the literal `16`.
- `suggest_pool_config()` returns hard-coded heuristics, not measurements.

### What to keep in mind if reviving

- The worker-pool design is sound and worth re-doing, but the implementation should be re-derived from first principles, not copy-pasted. The current code is workable but mixes real logic with vestigial fields named after features that never shipped.
- `tokio` dependency exists *only* for `oneshot`; consider replacing with `crossbeam::channel`'s bounded channel and a one-element `recv()` to drop `tokio` entirely.
- `tokio` was originally pulled in with `features = ["full"]`; the dependency audit narrowed it to `["sync"]`.
- Consider whether the library actually needs a worker pool. The original `par_iter` implementation was simpler and only ~150 MB. A worker pool costs ~622 MB for 6 workers and only buys ~42% throughput improvement and lower latency variance. For most consumers of the crate, that may be a bad trade.

---

## 2. Cache Toggle

### Intent

Make the per-worker embedding cache fully optional and explicit. The original design assumed callers wanted caching; this revision recognized that the **primary use case for `rust-embed` is embedding unique upstream messages**, where cache hit rate is ~0% and the cache is pure memory waste.

Key behaviors intended:

- `cache_size_per_worker: 0` → caching disabled entirely, no `HashMap` allocation, no eviction logic.
- `cache_size_per_worker: 100–500` → minimal cache to catch accidental upstream duplicates.
- `cache_size_per_worker: 10000+` → search-query-style workloads where hit rate is meaningful.
- FIFO eviction (not true LRU) when the limit is exceeded — explicitly accepted as "good enough" because in practice either the cache is disabled or the cache is large enough that eviction is rare.
- Per-worker cache only, never shared. Documented justification: shared cache would require a mutex, defeating the worker-pool's lock-free property, and cache duplication across workers is a non-issue when hit rate is near zero.
- New preset `PoolConfig::search_optimized()` that opted into the large cache, distinct from `balanced()` and `high_throughput()` which both default to `100`.
- A new doc, `docs/CACHING.md`, that walked callers through hit-rate interpretation and recommended starting with `cache_size_per_worker: 0`.

### Actual halfway state at time of archive

What was real and worked:

- `cache_size_per_worker: 0` is accepted by `PoolConfig::validate()` and tested.
- `MiniLMConfig::cache_size_limit` is wired through to the embedder.
- Presets reflect the design: `minimal()` → 0, `balanced()` → 100, `high_throughput()` → 100, `search_optimized()` → 10_000.
- `aggregate_stats()` returns aggregated `cache_hits`/`cache_misses` so callers can measure hit rate.
- `docs/CACHING.md` exists and is reasonably thorough.

What was halfway:

- The cache-disable path inside `MiniLMEmbedder` was not audited as part of this archive; the worker-pool docs assume `cache_size_limit: 0` produces a true no-op cache, but the actual `embed_text` implementation may still allocate the `HashMap` and skip insertions, which is functionally equivalent but not the "no allocation" intent stated in the design.
- `clear_all_caches()` exists on the pool but the `WorkerRequest::ClearCache` arm doesn't return a confirmation, so the caller has no way to know when cache clearing has actually happened across all workers.

### What to keep in mind if reviving

- The "cache off by default for unique-message workloads" insight is correct and worth preserving as the default mental model. Re-introduce it deliberately when next designing the API rather than carrying the current preset names along.
- `clear_all_caches()` should either be synchronous (collect oneshots back) or be removed.

---

## 3. ModernBERT

### Intent

Add ModernBERT (Base, 149M / Large, 395M) as a **selectable model alongside MiniLM**, never as a replacement. Selection happens at pool creation via `PoolConfig.model: ModelType`. Switching models at runtime is explicitly not supported — caller must shutdown the pool and create a new one.

Key design properties intended:

- `ModelType { MiniLM, ModernBERTBase, ModernBERTLarge }` with metadata methods (`dimension`, `max_sequence_length`, `memory_footprint_mb`, `supports_gpu`, `huggingface_id`).
- ModernBERT loaded via `rust-bert`'s `SentenceEmbeddingsBuilder::remote(...)` with `Device::Cpu` or `Device::Mps`.
- **Mean pooling** over the last hidden state (ModernBERT's pooling strategy), distinct from MiniLM's CLS-token approach. L2 normalization after pooling.
- **Hybrid CPU/GPU worker pool**: `cpu_workers` + `gpu_workers`, where GPU workers use `tch::Device::Mps` on Apple Silicon (UMA gives zero-copy tensor sharing).
- **Dynamic device routing** for ModernBERT only: route based on sequence length and batch tokens.
  - `seq_len >= 1024` → GPU (attention is O(n²); GPU parallelism wins).
  - `batch_tokens <= 512` → CPU (GPU launch overhead exceeds compute).
  - `batch_tokens >= 2048` → GPU (transfer cost amortized).
  - Otherwise → CPU.
- `RoutingConfig { long_sequence_threshold, small_batch_threshold, large_batch_threshold }` with defaults of `1024`, `512`, `2048`. Caller can override.
- Routing thresholds documented as informed-but-tunable; expected to be re-benchmarked once real workloads land.
- Backend is **PyTorch MPS via `tch-rs` only**. MLX explicitly rejected (no Rust bindings, would require a second backend, marginal benefit over MPS).
- Models cached locally by `rust-bert` under `~/.cache/huggingface/hub/`. First run downloads ~600 MB (Base) or ~1.6 GB (Large); subsequent runs load from cache.
- Per-model memory footprint: MiniLM ~150 MB, ModernBERT Base ~1–3 GB, ModernBERT Large ~2–5 GB depending on sequence length.

Phased plan as documented:

1. Add `ModernBERT` to `ModelType` enum, model loading via `rust-bert`, mean pooling, L2 normalize.
2. CPU-only ModernBERT workers; benchmark CPU performance.
3. GPU workers via `Device::Mps`; benchmark CPU vs GPU by sequence length.
4. Dynamic routing implementation; tune thresholds.
5. CLI integration (`--model minilm` / `--model modernbert`); end-to-end tests with both models.
6. Optimization, threshold tuning, docs.

### Actual halfway state at time of archive

What was real:

- `ModelType::ModernBERTBase` and `ModelType::ModernBERTLarge` enum variants exist on `main`.
- `ModelType` metadata methods (`dimension`, `max_sequence_length`, `memory_footprint_mb`, `supports_gpu`, `huggingface_id`) return the correct values for both ModernBERT variants.
- `RoutingConfig` struct exists with the documented defaults and validation rules.
- `PoolConfig::routing_config: Option<RoutingConfig>` exists; `validate()` warns if it's set with `ModelType::MiniLM`.
- `docs/MODERNBERT_IMPLEMENTATION.md` (~1100 lines) and `ARCHITECTURE_REVISION.md` thoroughly describe the design.
- `docs/MODEL_SELECTION_GUIDE.md` exists.

What was vapor — *no implementation, only enum stubs and docs*:

- **There is no `src/models/modernbert/` module.** `src/models/mod.rs` only declares `pub mod mini_lm;`.
- **There is no `ModernBertEmbedder` type, no model loader, no mean-pooling implementation, no L2-normalize-after-pooling helper.**
- **No GPU worker code path.** `EmbeddingWorker` only constructs `MiniLMEmbedder`. There is no branch on `ModelType` inside the worker; selecting `ModelType::ModernBERTBase` would still build a `MiniLMEmbedder` (or panic, depending on `validate()` / `WorkerConfig` reading).
- `validate()` actively rejects `gpu_workers > 0`, so the only way to use the ModernBERT enum variants is in a CPU-only pool — which would silently embed with MiniLM if the `ModelType` were ever consulted (it isn't).
- `route_worker()` / hybrid routing logic does not exist in `src/pool/mod.rs`. The single `get_worker()` is round-robin only.
- `select_device_modernbert()` / `estimate_tokens()` / `batch_tokens()` helpers from the design doc are unimplemented.
- Mean pooling and L2 normalization helpers do not exist in `src/`.
- No CLI flag `--model`. Both `main.rs` and `bin/similarity.rs` assume MiniLM.
- No integration tests for ModernBERT (they couldn't exist; there's nothing to test).

### What to keep in mind if reviving

- The **design** is complete and quite good — the routing thresholds are well-justified, the model coexistence story is consistent, and the GPU backend decision (PyTorch MPS, not MLX) is documented with rationale. The `docs/MODERNBERT_IMPLEMENTATION.md` content is worth re-reading before re-doing this work, even if the code is regenerated.
- The **implementation surface** that ships on `main` is misleading: `ModelType::ModernBERTBase` looks like a usable selector but does nothing. If/when reviving, do it as a single coherent piece — model loading + mean pooling + GPU worker + routing + tests + CLI flag — rather than landing the enum variants ahead of the embedder.
- `rust-bert` 0.21 was the assumed dependency. Verify whether it supports ModernBERT loading via `SentenceEmbeddingsBuilder::remote("answerdotai/ModernBERT-base")` before committing to it again; this was assumed in the design but never verified by running code.
- The cost/benefit of GPU workers on Apple Silicon for transformer inference is not as clear-cut as the design implies. Re-benchmark before committing to a hybrid pool.

---

## Discarded commits (for reference)

```
9ca8c64  Add worker pool architecture design document          (docs only)
be7e437  Add ModernBERT implementation spec                    (docs only)
db1617b  Update architecture: library design, explicit config  (docs only)
80a2191  Implement Phase 1 (v0.2.0): Worker Pool Architecture  (CODE — real, MiniLM-only)
e023082  Revise architecture for MiniLM + ModernBERT options   (docs only)
ccd0e61  Add architecture revision summary document            (docs only)
82a2643  Add configurable routing configuration to PoolConfig  (CODE — RoutingConfig struct, no consumers)
bc86d2d  Add cache disable option, optimize defaults           (CODE — real, useful)
f07e8ab  Add ModelType variants for ModernBERT Base and Large  (CODE — enum stubs, no embedder)
ce4c87d  Update ModernBERT implementation plan                 (docs only)
f5fa5a0  Update ModernBERT upgrade plan, fix README            (docs only)
241d446  Merge pull request #1
5b8fecd  Merge branch 'main' into claude/modernbert-...
f66d10b  Merge pull request #2
```

`main` after the reset will be at `3bb00b0` ("Revise README.md to enhance clarity and detail for version 0.0.2 release").

---

## Files removed by the reset (working-tree level)

- `ARCHITECTURE_REVISION.md`
- `PHASE1_IMPLEMENTATION.md`
- `docs/WORKER_POOL_ARCHITECTURE.md`
- `docs/MODERNBERT_IMPLEMENTATION.md`
- `docs/MODEL_SELECTION_GUIDE.md`
- `docs/CACHING.md`
- `src/pool/` (entire module)
- `tests/pool_integration_tests.rs`
- Portions of `src/lib.rs`, `src/main.rs`, `src/bin/similarity.rs` that referenced the pool API
- `crossbeam`, `num_cpus`, `tokio` from `Cargo.toml` (the latter was only ever used for `oneshot`)

The `archived-design-notes` branch retains all of these files at their last-known state, and this document is added on top.
