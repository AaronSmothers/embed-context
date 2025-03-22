# Rust Embeddings

A robust, pure Rust implementation of text embedding algorithms optimized for Apple Silicon M-series chips, using the all-MiniLM-L6-v2 model from Hugging Face.

## Features

- **Apple Silicon Optimization**: Fully leverages Apple M-series chips (M1, M2, M3, M4) with Metal Performance Shaders (MPS) acceleration
- **High-quality Semantic Embeddings**: Uses the all-MiniLM-L6-v2 model to generate 384-dimensional embeddings that capture semantic meaning
- **Streamlined Model Management**: Automatically downloads and caches model files for future use
- **Efficient Storage**: Compact embedding storage using Protocol Buffers
- **Fast Parallel Processing**: Optimized for large-scale embedding tasks with rayon parallelization (needs significantly larger tests and performance profiling*)
- **Intuitive CLI**: Simple command-line interface for embedding texts and computing similarity

## Installation

Add this to your Cargo.toml:

```toml
[dependencies]
rust_embed = "0.0.2"
```

The package automatically handles downloading and configuring the necessary dependencies, including libtorch libraries optimized for your system architecture. Note that on Apple Silicon, only the arm64 version of libtorch is used for optimal performance.

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
    
    // Initialize the model (downloads/loads model and tokenizer automatically)
    embedder.initialize()?;
    
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

## Apple Silicon Optimizations

This library is specially optimized for Apple Silicon chips:

- **Unified Memory Utilization**: Leverages the unified memory architecture of M-series chips
- **MPS Acceleration**: Uses Metal Performance Shaders for accelerated tensor operations
- **Automatic Detection**: Automatically detects and configures for your hardware
- **Parallel Processing**: Uses rayon to take advantage of multi-core performance when appropriate

## Project Structure

- `src/embedding.rs`: Core embedding trait definition and functionality
- `src/models/mini_lm/mod.rs`: Implementation of the all-MiniLM-L6-v2 model with Apple Silicon optimizations
- `src/utils/mod.rs`: Utility functions for saving/loading embeddings and Apple Silicon configuration
- `src/bin/`: Command-line tools for embedding and similarity calculations
- `proto/embeddings.proto`: Protocol Buffers schema for storing embeddings efficiently

## Performance

This implementation is designed for both speed and efficiency:

- For large datasets (100GB+), embeddings are generated using parallel processing
- On Apple Silicon M-series chips, performance is significantly improved using MPS acceleration which gives us GPU acceleration (profiling CPU vs GPU vs Neural engine core usage coming soon)
- Embedding caching prevents redundant processing of identical texts

## Dependencies

- **rust-bert**: Rust implementation of transformer models including all-MiniLM-L6-v2
- **tch**: Rust bindings for PyTorch C++ API with MPS support
- **ndarray**: N-dimensional array for fast vector operations
- **rayon**: Data parallelism library for multi-core processing
- **prost**: Efficient Protocol Buffers implementation

## Completed Improvements

- [x] Implement true MiniLM model weights loading via rust-bert
- [x] Optimize for Apple Silicon with Metal Performance Shaders
- [x] Add streamlined tokenizer loading process
- [x] Implement automatic hardware detection and configuration
- [x] Optimize for parallel processing with rayon
- [x] Add model caching to avoid reloading for each run

## Future Roadmap

- [ ] Add support for more embedding models
- [ ] Implement vector database integration for efficient similarity search
- [ ] Add benchmarking tools and performance profiling
- [ ] Expose the libtorch binary for use by a future inference package
- [ ] Improve documentation with more usage examples


## License

This project is licensed under either of:

* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions. 
