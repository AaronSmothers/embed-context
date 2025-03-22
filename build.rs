use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=proto");
    println!("cargo:rerun-if-changed=build.rs");
    
    // Compile Protocol Buffers
    let mut config = prost_build::Config::new();
    config.bytes(["."]);
    
    // Out dir is set by Cargo
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    
    // Create proto directory if it doesn't exist
    let proto_dir = PathBuf::from("proto");
    if !proto_dir.exists() {
        std::fs::create_dir_all(&proto_dir)?;
        
        // Create embeddings.proto file with basic schema
        let proto_content = r#"syntax = "proto3";

package embeddings;

// A single embedding vector
message Embedding {
  repeated float values = 1 [packed=true];
  string text = 2;  // Original text (optional)
  int64 timestamp = 3;  // When the embedding was created
}

// A collection of embeddings
message EmbeddingCollection {
  repeated Embedding embeddings = 1;
  string model_name = 2;  // Name of the model used
  string model_version = 3;  // Version of the model
  int32 dimension = 4;  // Dimension of each embedding vector
}
"#;
        std::fs::write(proto_dir.join("embeddings.proto"), proto_content)?;
    }
    
    // Compile the proto files
    config.out_dir(&out_dir);
    config.compile_protos(&[proto_dir.join("embeddings.proto")], &[proto_dir])?;
    
    // Detect Apple Silicon
    if cfg!(target_os = "macos") {
        let output = Command::new("uname")
            .arg("-m")
            .output()
            .expect("Failed to execute uname command");
        
        let arch = String::from_utf8_lossy(&output.stdout);
        
        if arch.trim() == "arm64" {
            println!("cargo:rustc-cfg=apple_silicon");
            println!("cargo:warning=Building for Apple Silicon (M-series)");
            
            // Check if MPS is available by compiling a small test program
            let mps_test = r#"
            #include <stdio.h>
            #include <stdlib.h>
            
            int main() {
                #if defined(__APPLE__) && defined(__arm64__)
                    printf("1\n");
                    return 0;
                #else
                    printf("0\n");
                    return 0;
                #endif
            }
            "#;
            
            let mps_test_file = out_dir.join("mps_test.c");
            std::fs::write(&mps_test_file, mps_test)?;
            
            let status = Command::new("cc")
                .arg("-o")
                .arg(out_dir.join("mps_test"))
                .arg(&mps_test_file)
                .status()
                .expect("Failed to compile MPS test");
            
            if status.success() {
                let output = Command::new(out_dir.join("mps_test"))
                    .output()
                    .expect("Failed to run MPS test");
                
                let result = String::from_utf8_lossy(&output.stdout);
                
                if result.trim() == "1" {
                    println!("cargo:rustc-cfg=has_mps");
                    println!("cargo:warning=Metal Performance Shaders (MPS) acceleration is available");
                }
            }
        } else {
            println!("cargo:warning=Building for Intel Mac (x86_64)");
            println!("cargo:warning=This build is optimized for Apple Silicon (M-series) processors");
        }
    }
    
    Ok(())
} 