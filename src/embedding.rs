use anyhow::Result;
use ndarray::Array1;
use rayon::prelude::*;
use std::path::Path;
use std::sync::Arc;

/// The Embedder trait defines the interface for text embedding implementations.
pub trait Embedder: Clone + Send + Sync {
    /// Embeds a single text string into a vector representation.
    fn embed_text(&self, text: &str) -> Result<Array1<f32>>;
    
    /// Embeds multiple text strings into vector representations.
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Array1<f32>>> {
        // Default implementation that uses parallel processing for large batches
        if texts.len() > 10 {
            // Parallel implementation for larger batches
            texts.par_iter()
                .map(|text| self.embed_text(text))
                .collect()
        } else {
            // Sequential processing for small batches
            texts.iter()
                .map(|text| self.embed_text(text))
                .collect()
        }
    }
    
    /// Computes the cosine similarity between two embedding vectors.
    fn cosine_similarity(&self, vec1: &Array1<f32>, vec2: &Array1<f32>) -> f32 {
        let dot_product = vec1.dot(vec2);
        let norm1 = vec1.dot(vec1).sqrt();
        let norm2 = vec2.dot(vec2).sqrt();
        
        if norm1 * norm2 == 0.0 {
            return 0.0;
        }
        
        dot_product / (norm1 * norm2)
    }
    
    /// Returns the name of the model used by this embedder
    fn model_name(&self) -> &str;
    
    /// Returns the version of the model used by this embedder
    fn model_version(&self) -> &str;
    
    /// Returns the dimension of the embeddings produced by this model
    fn dimension(&self) -> usize;
    
    /// Save model to disk
    fn save_model<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        // Default implementation does nothing
        Ok(())
    }
    
    /// Load model from disk
    fn load_model<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        // Default implementation does nothing
        Ok(())
    }
    
    /// Check if model can be loaded from disk
    fn model_exists<P: AsRef<Path>>(&self, path: P) -> bool {
        path.as_ref().exists()
    }
}

/// A trait for embedders that can be cached in memory
pub trait CachedEmbedder: Embedder {
    /// Precompute and cache embeddings for a set of texts
    fn cache_embeddings(&mut self, texts: &[String]) -> Result<()>;
    
    /// Get an embedding from the cache if available
    fn get_cached_embedding(&self, text: &str) -> Option<Array1<f32>>;
    
    /// Clear the embedding cache
    fn clear_cache(&mut self);
    
    /// Returns the number of cached embeddings
    fn cache_size(&self) -> usize;
}

/// A struct to hold both the text and its embedding
#[derive(Clone)]
pub struct EmbeddedText {
    /// The original text
    pub text: String,
    
    /// The embedding vector
    pub embedding: Arc<Array1<f32>>,
}

impl EmbeddedText {
    /// Create a new EmbeddedText
    pub fn new(text: String, embedding: Array1<f32>) -> Self {
        Self {
            text,
            embedding: Arc::new(embedding),
        }
    }
    
    /// Calculate cosine similarity with another EmbeddedText
    pub fn similarity(&self, other: &EmbeddedText) -> f32 {
        let vec1 = &*self.embedding;
        let vec2 = &*other.embedding;
        
        let dot_product = vec1.dot(vec2);
        let norm1 = vec1.dot(vec1).sqrt();
        let norm2 = vec2.dot(vec2).sqrt();
        
        if norm1 * norm2 == 0.0 {
            return 0.0;
        }
        
        dot_product / (norm1 * norm2)
    }
} 