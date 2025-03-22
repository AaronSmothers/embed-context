use anyhow::Result;
use clap::Parser;
use rust_embed::{
    models::mini_lm::MiniLMEmbedder,
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
    
    // Create the MiniLM embedder
    let mut embedder = MiniLMEmbedder::new();
    
    // Initialize the model and tokenizer
    println!("Initializing the embedder...");
    embedder.initialize()?;
    
    // Output info about the model
    println!("Using the {} model for generating embeddings.", embedder.model_name());
    println!("Embedding dimension: {}", embedder.dimension());
    
    // Embed the input text
    println!("Embedding text: {}", args.text);
    let new_embedding = embedder.embed_text(&args.text)?;
    
    // Compute similarity
    let similarity = embedder.cosine_similarity(&embeddings[0], &new_embedding);
    
    // Display results
    println!("Similarity: {:.6}", similarity);
    
    if let Some(texts) = texts {
        if !texts.is_empty() {
            println!("Original text: {}", texts[0]);
        }
    }
    
    println!("Input text: {}", args.text);
    
    Ok(())
} 