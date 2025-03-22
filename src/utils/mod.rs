pub mod libtorch;

use anyhow::Result;
use std::path::Path;
use std::os::unix::fs::PermissionsExt;

/// Initialize all necessary utilities for rust-embed on Apple Silicon
pub fn initialize() -> Result<()> {
    // Check if we're running on Apple Silicon
    if libtorch::is_apple_silicon()? {
        // Set up the Apple Silicon environment
        libtorch::setup_for_apple_silicon()?;
        
        // Report MPS availability
        if libtorch::has_mps()? {
            log::info!("Metal Performance Shaders acceleration is available and enabled");
        } else {
            log::info!("Metal Performance Shaders not available, using CPU only");
        }
    } else {
        log::warn!("This version is optimized for Apple Silicon (M-series) processors");
        log::warn!("Some functionality may not work correctly on Intel Macs");
    }
    
    Ok(())
}

/// Create a wrapper binary directory with all necessary libraries for Apple Silicon
pub fn create_binary_wrapper<P: AsRef<Path>>(target_dir: P) -> Result<()> {
    let target_dir = target_dir.as_ref();
    
    // Create libtorch symlinks for packaging
    libtorch::create_libtorch_symlinks(target_dir)?;
    
    // Create shell script wrapper to set up environment variables
    let wrapper_path = target_dir.join("run_rust_embed.sh");
    let wrapper_content = format!(
        "#!/bin/bash\n\
         # Wrapper script for rust_embed on Apple Silicon\n\
         SCRIPT_DIR=\"$( cd \"$( dirname \"${{BASH_SOURCE[0]}}\" )\" && pwd )\"\n\
         export DYLD_LIBRARY_PATH=\"$SCRIPT_DIR/lib:$DYLD_LIBRARY_PATH\"\n\
         export DYLD_FALLBACK_LIBRARY_PATH=\"$SCRIPT_DIR/lib:$DYLD_FALLBACK_LIBRARY_PATH\"\n\
         export LIBTORCH=\"$SCRIPT_DIR\"\n\
         \n\
         # Enable Metal Performance Shaders if available\n\
         if [ -f \"$SCRIPT_DIR/metal/mps_available\" ]; then\n\
         export PYTORCH_ENABLE_MPS_FALLBACK=1\n\
         fi\n\
         \n\
         # Run the actual binary\n\
         \"$SCRIPT_DIR/rust_embed\" \"$@\"\n"
    );
    
    std::fs::write(&wrapper_path, wrapper_content)?;
    std::fs::set_permissions(&wrapper_path, std::fs::Permissions::from_mode(0o755))?;
    
    log::info!("Binary wrapper created in {}", target_dir.display());
    Ok(())
}

/// Returns true if running on Apple Silicon (M-series processors)
pub fn is_apple_silicon() -> bool {
    libtorch::is_apple_silicon().unwrap_or(false)
}

/// Returns true if Metal Performance Shaders (MPS) acceleration is available
pub fn has_mps() -> bool {
    libtorch::has_mps().unwrap_or(false)
}

/// Cache home directory for model storage
pub fn cache_home() -> std::path::PathBuf {
    if let Some(cache_dir) = dirs::cache_dir() {
        cache_dir.join("rust_embed")
    } else {
        std::env::temp_dir().join("rust_embed")
    }
}

/// Normalizes a vector to unit length
pub fn normalize(vec: &mut ndarray::Array1<f32>) {
    let norm = vec.dot(vec).sqrt();
    if norm > 0.0 {
        vec.mapv_inplace(|x| x / norm);
    }
}

/// Preprocesses text for embedding
pub fn preprocess_text(text: &str) -> String {
    // Simple preprocessing: trim, lowercase, collapse whitespace
    let text = text.trim().to_lowercase();
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    text
}

/// Save an embedding model to disk
pub fn save_embeddings(
    embeddings: &[ndarray::Array1<f32>],
    texts: Option<&[String]>,
    model_name: &str,
    model_version: &str,
    dimension: i32,
    path: impl AsRef<Path>,
) -> Result<()> {
    // Create a protobuf message for the embeddings
    let mut pb_embeddings = crate::proto::EmbeddingCollection::default();
    pb_embeddings.model_name = model_name.to_string();
    pb_embeddings.model_version = model_version.to_string();
    pb_embeddings.dimension = dimension;
    
    // Add the embeddings and texts to the message
    for (i, embedding) in embeddings.iter().enumerate() {
        let mut pb_embedding = crate::proto::Embedding::default();
        pb_embedding.values = embedding.iter().copied().collect();
        
        if let Some(texts) = texts {
            if i < texts.len() {
                pb_embedding.text = texts[i].clone();
            }
        }
        
        pb_embedding.timestamp = chrono::Utc::now().timestamp();
        pb_embeddings.embeddings.push(pb_embedding);
    }
    
    // Create parent directories if they don't exist
    if let Some(parent) = path.as_ref().parent() {
        std::fs::create_dir_all(parent)?;
    }
    
    // Serialize the embeddings to protobuf
    let bytes = prost::Message::encode_to_vec(&pb_embeddings);
    
    // Write the serialized embeddings to disk
    std::fs::write(path, bytes)?;
    
    Ok(())
}

/// Load embeddings from disk
pub fn load_embeddings(path: impl AsRef<Path>) -> Result<(Vec<ndarray::Array1<f32>>, Option<Vec<String>>)> {
    // Read the file
    let bytes = std::fs::read(path)?;
    
    // Deserialize the embeddings from protobuf
    let proto_embeddings: crate::proto::EmbeddingCollection = prost::Message::decode(bytes.as_slice())?;
    
    // Convert to the expected return format
    convert_proto_embeddings(proto_embeddings)
}

/// Convert a proto Embeddings to a tuple of vectors and texts
pub fn convert_proto_embeddings(proto_embeddings: crate::proto::EmbeddingCollection) 
    -> Result<(Vec<ndarray::Array1<f32>>, Option<Vec<String>>)> {
    
    let mut embeddings = Vec::with_capacity(proto_embeddings.embeddings.len());
    let mut texts = Vec::with_capacity(proto_embeddings.embeddings.len());
    let has_texts = proto_embeddings.embeddings.iter().any(|e| !e.text.is_empty());
    
    for embedding in proto_embeddings.embeddings {
        embeddings.push(ndarray::Array1::from(embedding.values));
        if has_texts {
            texts.push(embedding.text);
        }
    }
    
    let texts = if has_texts { Some(texts) } else { None };
    
    Ok((embeddings, texts))
} 