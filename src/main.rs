use anyhow::Result;
use clap::Parser;
use ndarray::s;
use rust_embed::{
    models::mini_lm::MiniLMEmbedder,
    utils,
};
use std::path::PathBuf;
use log::{info, warn, debug};

/// Command line arguments
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Text to embed
    #[arg(short, long)]
    text: Option<String>,
    
    /// File containing text to embed (one text per line)
    #[arg(short, long)]
    file: Option<PathBuf>,
    
    /// Output file for the embeddings
    #[arg(short, long)]
    output: Option<PathBuf>,
    
    /// Create a standalone binary package
    #[arg(long)]
    package: Option<PathBuf>,
    
    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    
    // Parse command line arguments
    let args = Args::parse();
    
    if args.verbose {
        log::info!("Verbose mode enabled");
    }
    
    // Initialize Apple Silicon specific utilities
    match utils::initialize() {
        Ok(_) => info!("Initialization successful"),
        Err(e) => warn!("Initialization warning: {}", e),
    }
    
    // Report architecture
    if utils::is_apple_silicon() {
        info!("Running on Apple Silicon (M-series)");
        if utils::has_mps() {
            info!("Metal Performance Shaders acceleration enabled");
        }
    } else {
        warn!("Running on non-Apple Silicon architecture");
        warn!("This build is optimized for Apple M-series processors");
    }
    
    // If packaging is requested, create a standalone binary
    if let Some(target_dir) = args.package {
        info!("Creating standalone package in {}", target_dir.display());
        utils::create_binary_wrapper(target_dir)?;
        info!("Standalone package created successfully");
        info!("Run the application using the provided shell script");
        return Ok(());
    }
    
    // Create the MiniLM embedder
    let mut embedder = MiniLMEmbedder::new();
    
    // Initialize the model (download and load both tokenizer and model)
    info!("Initializing the embedder...");
    embedder.initialize()?;
    
    // Output info about the model
    info!("Using the {} model for generating embeddings.", embedder.model_name());
    info!("Embedding dimension: {}", embedder.dimension());
    
    // Process text based on input source
    if let Some(text) = args.text {
        info!("Embedding single text: {}", text);
        let embedding = embedder.embed_text(&text)?;
        info!("Embedding size: {}", embedding.len());
        debug!("First few values: {:?}", &embedding.slice(s![..5]));
        
        // Save to file if output is specified
        if let Some(output) = &args.output {
            let text_vec = vec![text];
            utils::save_embeddings(
                &[embedding], 
                Some(&text_vec),
                embedder.model_name(),
                embedder.model_version(),
                embedder.dimension() as i32,
                output
            )?;
            info!("Embedding saved to {}", output.display());
        }
    } else if let Some(file) = args.file {
        info!("Embedding texts from file: {}", file.display());
        
        // Read file line by line
        let content = std::fs::read_to_string(file)?;
        let texts: Vec<String> = content.lines().map(|s| s.to_string()).collect();
        
        // Embed each line
        let mut embeddings = Vec::with_capacity(texts.len());
        info!("Processing {} texts", texts.len());
        
        // Use rayon for parallel processing if we have multiple texts
        use rayon::prelude::*;
        if texts.len() > 1 {
            info!("Using parallel processing for multiple texts");
            embeddings = texts.par_iter()
                .map(|text| {
                    let mut local_embedder = embedder.clone();
                    local_embedder.embed_text(text)
                })
                .filter_map(Result::ok)
                .collect();
        } else {
            for text in &texts {
                match embedder.embed_text(text) {
                    Ok(embedding) => embeddings.push(embedding),
                    Err(e) => warn!("Failed to embed text: {}", e),
                }
            }
        }
        
        info!("Successfully embedded {} of {} texts", embeddings.len(), texts.len());
        
        // Save to file if output is specified
        if let Some(output) = &args.output {
            utils::save_embeddings(
                &embeddings, 
                Some(&texts),
                embedder.model_name(),
                embedder.model_version(),
                embedder.dimension() as i32,
                output
            )?;
            info!("Embeddings saved to {}", output.display());
        }
    } else {
        warn!("Please provide either --text or --file argument");
        println!("For usage information, run with --help");
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_embedding() -> Result<()> {
        // Initialize utilities for testing
        utils::initialize()?;
        
        let mut embedder = MiniLMEmbedder::new();
        // Initialize the model for the test
        embedder.initialize()?;
        
        let text = "This is a test sentence for embedding.";
        let embedding = embedder.embed_text(text)?;
        
        // Check dimensions
        assert_eq!(embedding.len(), 384);
        
        // Check normalization (length should be close to 1.0)
        let norm = embedding.dot(&embedding).sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
        
        Ok(())
    }
    
    #[test]
    fn test_similarity() -> Result<()> {
        // Initialize utilities for testing
        utils::initialize()?;
        
        let mut embedder = MiniLMEmbedder::new();
        // Initialize the model for the test
        embedder.initialize()?;
        
        let text1 = "Dogs are pets that bark.";
        let text2 = "Canines are domesticated animals that make barking sounds.";
        let text3 = "Quantum physics explores the nature of subatomic particles.";
        
        let emb1 = embedder.embed_text(text1)?;
        let emb2 = embedder.embed_text(text2)?;
        let emb3 = embedder.embed_text(text3)?;
        
        // Similar texts should have higher similarity
        let sim12 = embedder.cosine_similarity(&emb1, &emb2);
        let sim13 = embedder.cosine_similarity(&emb1, &emb3);
        
        println!("Similarity between similar texts: {}", sim12);
        println!("Similarity between different texts: {}", sim13);
        
        // Similar texts should have higher similarity than dissimilar texts
        assert!(sim12 > sim13);
        
        Ok(())
    }
    
    #[test]
    fn test_apple_silicon_detection() {
        // This test checks if we can detect Apple Silicon
        let is_apple = utils::is_apple_silicon();
        println!("Running on Apple Silicon: {}", is_apple);
        
        // This test doesn't assert anything as it might run on different architectures
    }
}
