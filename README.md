# Rust Embeddings

> **⚠️ IMPORTANT NOTE:** The release build currently has issues with libtorch library paths. When building with `--release`, you may encounter errors like `Library not loaded: @rpath/libtorch_cpu.dylib` due to missing LC_RPATH entries. For now, use debug builds with `cargo run` which properly sets up the library paths. This issue will be addressed in future releases.

A pure Rust implementation of text embedding algorithms using the all-MiniLM-L6-v2 model from Hugging Face.

This is being built for us in Apple M architecture. Currently all M architecture from M1 forward is supported.
## Features

- Integrates the all-MiniLM-L6-v2 model using rust-bert
- Real embeddings with 384 dimensions that capture semantic meaning
- Efficient storage using Protocol Buffers
- Simple interface for embedding text and computing similarity
- Optimized for large-scale embedding tasks (supports processing 100GB+ datasets)
- Command-line interface for easy usage

## Installation

Add this to your Cargo.toml:

```toml
[dependencies]
rust_embed = "0.0.1"
```

Note: This package requires libtorch (PyTorch C++ libraries) to be installed as it uses the tch-rs crate.

## Usage

### Command Line

```bash
# Embed a single text
cargo run --bin rust_embed -- --text "This is a sample text to embed" --output embeddings.pb

# Embed multiple texts from a file (one per line)
cargo run --bin rust_embed -- --file input.txt --output embeddings.pb

# Calculate similarity between a stored embedding and a new text
cargo run --bin similarity -- --embedding-file embeddings.pb --text "This is a similar text"
```

### As a Library

```rust
use rust_embed::models::mini_lm::MiniLMEmbedder;
use rust_embed::embedding::Embedder;
use anyhow::Result;

fn main() -> Result<()> {
    // Create the embedder
    let mut embedder = MiniLMEmbedder::new();
    
    // Load necessary components
    embedder.load_or_download_tokenizer()?;
    embedder.load_or_download_model()?;
    
    // Embed a text
    let text = "This is a sample text to embed";
    let embedding = embedder.embed_text(text)?;
    
    // Use the embedding for similarity comparison
    let text2 = "This is another sample text";
    let embedding2 = embedder.embed_text(text2)?;
    
    let similarity = embedder.cosine_similarity(&embedding, &embedding2);
    println!("Similarity: {}", similarity);
    
    Ok(())
}
```

## Project Structure

- `src/embedding.rs`: Core embedding trait definition
- `src/models/mini_lm/`: Implementation of the all-MiniLM-L6-v2 model
- `src/utils.rs`: Utility functions for saving/loading embeddings
- `proto/embeddings.proto`: Protocol Buffers schema for storing embeddings
- `src/bin/`: Command-line tools for embedding and similarity

## Performance

This implementation is designed to be efficient for large-scale embedding tasks. For a 106GB dataset, it uses Protocol Buffers to minimize storage requirements and maximize serialization/deserialization speed.

## Dependencies

- rust-bert: Provides the all-MiniLM-L6-v2 model implementation
- tch: Rust bindings for the PyTorch C++ API
- tokenizers: High-performance tokenization from Hugging Face
- ndarray: N-dimensional array manipulation
- prost: Protocol Buffers implementation for Rust

## Future Improvements

- [x] ~~Implement true MiniLM model weights loading~~
- [ ] Add support for more embedding models
- [ ] Optimize for parallel processing
- [ ] Add benchmarking tools
- [ ] Implement retrieval functionality for similarity search
- [ ] Improve error handling and reporting
- [ ] Add model caching to avoid reloading for each run
- [ ] Fix release build issues with libtorch library paths

## License

This project is licensed under either of:

* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions. 
