pub mod embedding;
pub mod utils;  // This refers to the src/utils directory with mod.rs
pub mod models;

// Define the protobuf module
pub mod proto {
    // Include the generated rust code from the protobuf compiler
    include!(concat!(env!("OUT_DIR"), "/embeddings.rs"));
}

// Re-export commonly used items
pub use embedding::{Embedder, CachedEmbedder, EmbeddedText};
pub use models::mini_lm::MiniLMEmbedder;
pub use models::ModelConfig;

/// Version of the rust-embed library
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Detect if we're running on Apple Silicon
#[cfg(target_arch = "aarch64")]
#[cfg(target_os = "macos")]
pub const IS_APPLE_SILICON: bool = true;

/// Detect if we're running on Apple Silicon - fallback
#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
pub const IS_APPLE_SILICON: bool = false;

/// Detect if Metal Performance Shaders are available
#[cfg(target_arch = "aarch64")]
#[cfg(target_os = "macos")]
pub const HAS_MPS: bool = true;

/// Detect if Metal Performance Shaders are available - fallback
#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
pub const HAS_MPS: bool = false;

/// Initialize the library
pub fn initialize() -> anyhow::Result<()> {
    utils::initialize()
} 