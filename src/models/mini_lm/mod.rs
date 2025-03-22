use crate::embedding::{self, EmbeddedText, Embedder};
use crate::models::ModelConfig;
use crate::utils;
use anyhow::{anyhow, Result};
use ndarray::Array1;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tch::{Device, Tensor};
use std::cell::RefCell;
use tokenizers::Tokenizer;

// Thread-local storage for model instances
thread_local! {
    static MODEL_INSTANCE: RefCell<Option<rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModel>> = RefCell::new(None);
}

// Constants for the MiniLM model
pub const MODEL_NAME: &str = "MiniLM-L6-v2";
pub const MODEL_VERSION: &str = "2.0";
pub const EMBEDDING_DIM: usize = 384;
pub const MODEL_URL: &str = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/";

/// Configuration for the MiniLM model
#[derive(Debug, Clone)]
pub struct MiniLMConfig {
    pub model_name: String,
    pub model_version: String,
    pub dimension: usize,
    pub model_path: Option<PathBuf>,
    pub device: Device,
    pub cache_embeddings: bool,
    pub cache_size_limit: usize,
    pub verify_silicon: bool,
}

impl Default for MiniLMConfig {
    fn default() -> Self {
        Self {
            model_name: MODEL_NAME.to_string(),
            model_version: MODEL_VERSION.to_string(),
            dimension: EMBEDDING_DIM,
            model_path: None,
            device: Device::Cpu,
            cache_embeddings: true,
            cache_size_limit: 10000, // Cache up to 10K embeddings
            verify_silicon: true,
        }
    }
}

impl ModelConfig for MiniLMConfig {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn model_version(&self) -> &str {
        &self.model_version
    }
}

/// Stats for the embedder
#[derive(Debug, Clone, Default)]
pub struct EmbedderStats {
    pub embeddings_count: usize,
    pub total_processing_time: Duration,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

/// MiniLM embedder implementation
#[derive(Clone)]
pub struct MiniLMEmbedder {
    config: MiniLMConfig,
    embedding_cache: HashMap<String, Array1<f32>>,
    stats: EmbedderStats,
    is_initialized: bool,
}

impl MiniLMEmbedder {
    /// Create a new embedder with default configuration
    pub fn new() -> Self {
        Self::with_config(MiniLMConfig::default())
    }

    /// Create a new embedder with custom configuration
    pub fn with_config(config: MiniLMConfig) -> Self {
        // Initialize Apple Silicon specific utilities if needed
        if config.verify_silicon && utils::is_apple_silicon() {
            utils::initialize().expect("Failed to initialize for Apple Silicon");
        }
        
        Self {
            config,
            embedding_cache: HashMap::new(),
            stats: EmbedderStats::default(),
            is_initialized: false,
        }
    }

    /// Get the model name
    pub fn model_name(&self) -> &str {
        &self.config.model_name
    }

    /// Get the model version
    pub fn model_version(&self) -> &str {
        &self.config.model_version
    }

    /// Get the embedding dimension
    pub fn dimension(&self) -> usize {
        self.config.dimension
    }

    /// Get embedder statistics
    pub fn stats(&self) -> &EmbedderStats {
        &self.stats
    }
    
    /// Initializes the model and tokenizer
    pub fn initialize(&mut self) -> Result<()> {
        if self.is_initialized {
            return Ok(());
        }
        
        // Load model which also loads the tokenizer
        self.load_or_download_model()?;
        
        self.is_initialized = true;
        Ok(())
    }
    
    /// Download and prepare the model
    pub fn load_or_download_model(&mut self) -> Result<()> {
        use rust_bert::pipelines::sentence_embeddings::{
            SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType
        };
        
        // Configure for Apple Silicon if applicable
        let device = if utils::is_apple_silicon() && utils::has_mps() {
            log::info!("Using MPS backend for model acceleration");
            Device::Mps
        } else {
            self.config.device
        };
        
        log::info!("Loading the MiniLM model...");
        
        // Use the builder pattern to create and load the model
        if let Some(model_path) = &self.config.model_path {
            // Use custom local model
            let sentence_embeddings = SentenceEmbeddingsBuilder::local(model_path.to_string_lossy().to_string())
                .with_device(device)
                .create_model()?;
                
            // Store it in thread-local storage
            MODEL_INSTANCE.with(|cell| {
                *cell.borrow_mut() = Some(sentence_embeddings);
            });
        } else {
            // Use remote model
            let model_id = SentenceEmbeddingsModelType::AllMiniLmL6V2;
            // Let rust-bert handle the tokenizer loading through the SentenceEmbeddingsBuilder
            let sentence_embeddings = SentenceEmbeddingsBuilder::remote(model_id)
                .with_device(device)
                .create_model()?;
            
            // Store it in thread-local storage
            MODEL_INSTANCE.with(|cell| {
                *cell.borrow_mut() = Some(sentence_embeddings);
            });
        }
        
        log::info!("Model loaded successfully");
        Ok(())
    }

    /// Embed a text into a vector representation
    pub fn embed_text(&mut self, text: &str) -> Result<Array1<f32>> {
        let start = Instant::now();

        // Initialize if not already done
        if !self.is_initialized {
            self.initialize()?;
        }

        // Check if in cache (if caching is enabled)
        if self.config.cache_embeddings {
            if let Some(embedding) = self.embedding_cache.get(text) {
                self.stats.cache_hits += 1;
                return Ok(embedding.clone());
            }
            self.stats.cache_misses += 1;
        }
        
        // Preprocess the text
        let processed_text = utils::preprocess_text(text);
        
        // Get model from thread-local storage or return error
        let embedding = MODEL_INSTANCE.with(|cell| -> Result<Array1<f32>> {
            let mut model_cell = cell.borrow_mut();
            
            if let Some(model) = &mut *model_cell {
                // Encode the text
                let embeddings = model.encode(&[processed_text])?;
                
                // Convert to ndarray
                let embedding = Array1::from_vec(embeddings[0].clone());
                
                // Normalize the embedding
                let mut normalized = embedding.clone();
                utils::normalize(&mut normalized);
                
                Ok(normalized)
            } else {
                Err(anyhow!("Model not initialized. Call initialize() first."))
            }
        })?;
        
        // Update statistics
        self.stats.embeddings_count += 1;
        self.stats.total_processing_time += start.elapsed();
        
        // Cache the embedding if enabled
        if self.config.cache_embeddings {
            self.embedding_cache.insert(text.to_string(), embedding.clone());
            
            // Limit cache size
            if self.embedding_cache.len() > self.config.cache_size_limit {
                if let Some(key) = self.embedding_cache.keys().next().cloned() {
                    self.embedding_cache.remove(&key);
                }
            }
        }
        
        Ok(embedding)
    }

    /// Embed multiple texts in batch
    pub fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Array1<f32>>> {
        // For Apple Silicon, use rayon for parallel processing
        if utils::is_apple_silicon() && texts.len() > 1 {
            use rayon::prelude::*;
            
            texts.par_iter()
                .map(|text| {
                    let mut local_embedder = self.clone();
                    local_embedder.embed_text(text)
                })
                .collect()
        } else {
            // Sequential processing
            texts.iter()
                .map(|text| self.embed_text(text))
                .collect()
        }
    }

    /// Calculate cosine similarity between two vectors
    pub fn cosine_similarity(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        let dot_product = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>();
        let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        
        dot_product / (norm_a * norm_b)
    }

    /// Clear the embedding cache
    pub fn clear_cache(&mut self) {
        self.embedding_cache.clear();
    }

    /// Get the number of cached embeddings
    pub fn cache_size(&self) -> usize {
        self.embedding_cache.len()
    }
    
    /// Find the most similar texts to the query
    pub fn find_similar(&mut self, query: &str, texts: &[String], top_k: usize) -> Result<Vec<(String, f32)>> {
        let query_embedding = self.embed_text(query)?;
        
        // Calculate similarities and sort
        let mut similarities: Vec<(String, f32)> = texts.iter()
            .filter_map(|text| {
                match self.embed_text(text) {
                    Ok(embedding) => {
                        let similarity = self.cosine_similarity(&query_embedding, &embedding);
                        Some((text.clone(), similarity))
                    },
                    Err(_) => None
                }
            })
            .collect();
        
        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Return top-k results
        Ok(similarities.into_iter().take(top_k).collect())
    }
}

// Implement the Embedder trait for MiniLMEmbedder
impl Embedder for MiniLMEmbedder {
    fn embed_text(&self, text: &str) -> Result<Array1<f32>> {
        // Clone self to get a mutable version since our methods require &mut self
        let mut embedder = self.clone();
        embedder.embed_text(text)
    }
    
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Array1<f32>>> {
        // Clone self to get a mutable version
        let mut embedder = self.clone();
        embedder.embed_batch(texts)
    }
    
    fn model_name(&self) -> &str {
        self.model_name()
    }
    
    fn model_version(&self) -> &str {
        self.model_version()
    }
    
    fn dimension(&self) -> usize {
        self.dimension()
    }
}

/// Helper functions
fn truncate_text(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        text.to_string()
    } else {
        format!("{}...", &text[..max_len])
    }
} 