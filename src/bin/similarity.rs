use anyhow::Result;
use clap::Parser;
use rust_embed::{
    EmbeddingPool, PoolConfig, ModelType,
    utils,
};
use std::path::PathBuf;

/// Command line arguments
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// File containing the first embedding
    #[arg(short = 'e', long)]
    embedding_file: PathBuf,
    
    /// Text to compare with the embedding
    #[arg(short, long)]
    text: String,
}

fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();
    
    // Parse command line arguments
    let args = Args::parse();
    
    // Load the embedding from file
    println!("Loading embedding from {:?}", args.embedding_file);
    let (embeddings, texts) = utils::load_embeddings(&args.embedding_file)?;
    
    if embeddings.is_empty() {
        println!("No embeddings found in the file");
        return Ok(());
    }
    
    // Create a minimal pool (1 worker is sufficient for similarity comparison)
    let pool_config = PoolConfig::minimal();
    println!("Creating embedding pool with {} worker...", pool_config.cpu_workers);
    let pool = EmbeddingPool::new(pool_config)?;

    // Output info about the model
    println!("Using MiniLM-L6-v2 model for generating embeddings.");
    println!("Embedding dimension: 384");

    // Embed the input text
    println!("Embedding text: {}", args.text);
    let new_embedding = pool.embed_text(args.text.clone())?;

    // Compute similarity (using cosine similarity formula directly)
    let dot_product = embeddings[0].dot(&new_embedding);
    let norm1 = embeddings[0].dot(&embeddings[0]).sqrt();
    let norm2 = new_embedding.dot(&new_embedding).sqrt();
    let similarity = if norm1 * norm2 == 0.0 {
        0.0
    } else {
        dot_product / (norm1 * norm2)
    };
    
    // Display results
    println!("Similarity: {:.6}", similarity);
    
    if let Some(texts) = texts {
        if !texts.is_empty() {
            println!("Original text: {}", texts[0]);
        }
    }
    
    println!("Input text: {}", args.text);

    // Graceful shutdown
    pool.shutdown()?;

    Ok(())
} 