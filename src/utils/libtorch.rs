use anyhow::{anyhow, Context, Result};
use std::path::{Path, PathBuf};
use std::fs::File;
use std::io::{Write, Read};
use std::sync::Once;
use std::time::Duration;
use std::process::Command;

// For Apple Silicon (M-series), we use the ARM64 version of libtorch
pub const LIBTORCH_URL_ARM64: &str = "https://download.pytorch.org/libtorch/cpu/libtorch-macos-2.0.0.zip";
pub const LIBTORCH_DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(600); // 10 minutes

static LIBTORCH_INIT: Once = Once::new();

/// Detects if running on Apple Silicon (M-series)
pub fn is_apple_silicon() -> Result<bool> {
    if cfg!(target_os = "macos") {
        #[cfg(target_arch = "aarch64")]
        {
            return Ok(true);
        }
        
        #[cfg(not(target_arch = "aarch64"))]
        {
            // Even if compiled for x86_64, check if we're running under Rosetta
            // on an Apple Silicon machine
            let output = Command::new("sysctl")
                .arg("-n")
                .arg("hw.optional.arm64")
                .output()
                .context("Failed to execute sysctl command")?;
            
            let result = String::from_utf8_lossy(&output.stdout).trim().to_string();
            return Ok(result == "1");
        }
    }
    
    Ok(false)
}

/// Check if Metal Performance Shaders (MPS) is available
pub fn has_mps() -> Result<bool> {
    if !cfg!(target_os = "macos") {
        return Ok(false);
    }
    
    if !is_apple_silicon()? {
        return Ok(false);
    }
    
    // Check if we can access Metal APIs
    // This is a basic check - in a real application, you'd use the Metal framework directly
    let output = Command::new("system_profiler")
        .arg("SPDisplaysDataType")
        .output()
        .context("Failed to execute system_profiler command")?;
    
    let result = String::from_utf8_lossy(&output.stdout);
    
    // If Metal is mentioned, it's likely available
    Ok(result.contains("Metal"))
}

/// Ensures libtorch is available for Apple Silicon, downloading it if necessary
pub fn ensure_libtorch() -> Result<PathBuf> {
    // Set up only once at runtime
    let mut libtorch_path = PathBuf::new();
    
    LIBTORCH_INIT.call_once(|| {
        if let Ok(path) = find_or_download_libtorch() {
            libtorch_path = path;
        }
    });
    
    if libtorch_path.as_os_str().is_empty() {
        find_or_download_libtorch()
    } else {
        Ok(libtorch_path)
    }
}

/// Finds an existing libtorch installation or downloads a new one
fn find_or_download_libtorch() -> Result<PathBuf> {
    // First check if we're on Apple Silicon
    if !is_apple_silicon()? {
        return Err(anyhow!("This version is optimized for Apple Silicon (M-series) processors only"));
    }
    
    // First check if LIBTORCH env var is set
    if let Ok(libtorch_path) = std::env::var("LIBTORCH") {
        let path = Path::new(&libtorch_path);
        if path.exists() && path.join("lib").join("libtorch_cpu.dylib").exists() {
            log::info!("Using libtorch from LIBTORCH env var: {}", libtorch_path);
            return Ok(path.to_path_buf());
        }
    }
    
    // Check default locations (prioritizing user locations to avoid permission issues)
    let home_dir = dirs::home_dir().context("Failed to determine home directory")?;
    let libtorch_paths = vec![
        home_dir.join("libtorch"),
        dirs::cache_dir().unwrap_or_else(|| PathBuf::from("/tmp")).join("rust_embed").join("libtorch"),
        PathBuf::from("/usr/local/libtorch"),
        PathBuf::from("/opt/homebrew/libtorch"),
    ];
    
    for path in libtorch_paths {
        if path.exists() && path.join("lib").join("libtorch_cpu.dylib").exists() {
            // Set LIBTORCH env var for future processes
            std::env::set_var("LIBTORCH", path.to_string_lossy().to_string());
            log::info!("Using libtorch from: {}", path.display());
            return Ok(path);
        }
    }
    
    // If we can't find libtorch, attempt to download it
    download_libtorch()
}

/// Downloads libtorch for Apple Silicon
fn download_libtorch() -> Result<PathBuf> {
    log::info!("Downloading libtorch for Apple Silicon (M-series)...");
    
    // Ensure we're on Apple Silicon
    if !is_apple_silicon()? {
        return Err(anyhow!("Cannot download libtorch - this version requires Apple Silicon (M-series)"));
    }
    
    let cache_dir = dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("rust_embed");
    std::fs::create_dir_all(&cache_dir).context("Failed to create cache directory")?;
    
    let zip_path = cache_dir.join("libtorch.zip");
    let extract_path = cache_dir.join("libtorch");
    
    // Only download if we don't already have it
    if !extract_path.exists() {
        log::info!("Downloading libtorch from {}", LIBTORCH_URL_ARM64);
        
        // Create a client with a timeout
        let client = reqwest::blocking::Client::builder()
            .timeout(LIBTORCH_DOWNLOAD_TIMEOUT)
            .build()?;
        
        // Download the file with progress reporting
        let mut response = client.get(LIBTORCH_URL_ARM64).send()?;
        let total_size = response.content_length().unwrap_or(0);
        
        let mut file = File::create(&zip_path)?;
        let mut downloaded: u64 = 0;
        
        let mut last_percent = 0;
        let mut buffer = [0; 8192];
        
        log::info!("Downloading libtorch ({:.1} MB)...", total_size as f64 / 1_048_576.0);
        
        while let Ok(n) = response.read(&mut buffer) {
            if n == 0 { break; }
            
            file.write_all(&buffer[..n])?;
            downloaded += n as u64;
            
            if total_size > 0 {
                let percent = (downloaded * 100 / total_size) as u8;
                if percent > last_percent && percent % 10 == 0 {
                    log::info!("Download progress: {}% ({:.1}/{:.1} MB)", 
                        percent,
                        downloaded as f64 / 1_048_576.0,
                        total_size as f64 / 1_048_576.0);
                    last_percent = percent;
                }
            }
        }
        
        // Extract the zip
        log::info!("Extracting libtorch to {}", extract_path.display());
        let file = File::open(&zip_path)?;
        let mut archive = zip::ZipArchive::new(file)?;
        archive.extract(&cache_dir)?;
        
        // Remove the zip file
        std::fs::remove_file(zip_path)?;
    }
    
    // Set the LIBTORCH env var
    std::env::set_var("LIBTORCH", extract_path.to_string_lossy().to_string());
    
    // Set up environment variables specific to Apple Silicon
    setup_apple_silicon_env(&extract_path)?;
    
    log::info!("Libtorch successfully installed to {}", extract_path.display());
    Ok(extract_path)
}

/// Set up environment variables for Apple Silicon
fn setup_apple_silicon_env(libtorch_path: &Path) -> Result<()> {
    let lib_path = libtorch_path.join("lib");
    
    // Add lib to DYLD_LIBRARY_PATH
    if let Ok(current_path) = std::env::var("DYLD_LIBRARY_PATH") {
        let new_path = format!("{}:{}", lib_path.display(), current_path);
        std::env::set_var("DYLD_LIBRARY_PATH", new_path);
    } else {
        std::env::set_var("DYLD_LIBRARY_PATH", lib_path.to_string_lossy().to_string());
    }
    
    // Set DYLD_FALLBACK_LIBRARY_PATH (more reliable than DYLD_LIBRARY_PATH in some cases)
    if let Ok(current_path) = std::env::var("DYLD_FALLBACK_LIBRARY_PATH") {
        let new_path = format!("{}:{}", lib_path.display(), current_path);
        std::env::set_var("DYLD_FALLBACK_LIBRARY_PATH", new_path);
    } else {
        std::env::set_var("DYLD_FALLBACK_LIBRARY_PATH", lib_path.to_string_lossy().to_string());
    }
    
    // Set additional variables if needed for MPS backend
    if has_mps()? {
        std::env::set_var("PYTORCH_ENABLE_MPS_FALLBACK", "1");
    }
    
    log::info!("Set dynamic library paths to include {}", lib_path.display());
    Ok(())
}

/// Creates a symbolic link to libtorch libraries in a custom location
pub fn create_libtorch_symlinks(target_dir: &Path) -> Result<()> {
    let libtorch_path = ensure_libtorch()?;
    
    // Create target directory if it doesn't exist
    std::fs::create_dir_all(target_dir).context("Failed to create target directory for symlinks")?;
    
    let lib_path = libtorch_path.join("lib");
    let target_lib_path = target_dir.join("lib");
    std::fs::create_dir_all(&target_lib_path).context("Failed to create lib directory")?;
    
    // Symlink the dylib files
    let entries = std::fs::read_dir(lib_path)?;
    
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        
        if let Some(ext) = path.extension() {
            if ext == "dylib" {
                let target = target_lib_path.join(path.file_name().unwrap());
                log::info!("Creating symlink: {} -> {}", path.display(), target.display());
                
                if target.exists() {
                    std::fs::remove_file(&target)?;
                }
                
                std::os::unix::fs::symlink(&path, &target)?;
            }
        }
    }
    
    // Create a metal directory if MPS is available
    if has_mps()? {
        let metal_path = target_dir.join("metal");
        std::fs::create_dir_all(&metal_path).context("Failed to create metal directory")?;
        
        // Create a flag file to indicate MPS is available
        let flag_file = metal_path.join("mps_available");
        std::fs::write(&flag_file, "1")?;
    }
    
    Ok(())
}

/// Fix LC_RPATH issues in macOS dylibs (important for Apple Silicon)
pub fn fix_rpath_issues() -> Result<()> {
    let libtorch_path = ensure_libtorch()?;
    let lib_path = libtorch_path.join("lib");
    
    // Check if install_name_tool is available
    if Command::new("which").arg("install_name_tool").output()?.status.success() {
        let entries = std::fs::read_dir(&lib_path)?;
        
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            
            if let Some(ext) = path.extension() {
                if ext == "dylib" {
                    log::info!("Fixing RPATH for {}", path.display());
                    
                    // Add @rpath to the library
                    Command::new("install_name_tool")
                        .arg("-id")
                        .arg(format!("@rpath/{}", path.file_name().unwrap().to_string_lossy()))
                        .arg(&path)
                        .output()?;
                    
                    // Add the lib directory to RPATH
                    Command::new("install_name_tool")
                        .arg("-add_rpath")
                        .arg(lib_path.to_string_lossy().as_ref())
                        .arg(&path)
                        .output()?;
                    
                    // For Apple Silicon, we can also add @loader_path
                    Command::new("install_name_tool")
                        .arg("-add_rpath")
                        .arg("@loader_path/")
                        .arg(&path)
                        .output()?;
                }
            }
        }
    } else {
        log::warn!("install_name_tool not found, skipping RPATH fixes");
    }
    
    Ok(())
}

/// Set up all environment variables and paths for Apple Silicon
pub fn setup_for_apple_silicon() -> Result<()> {
    // Verify we're on Apple Silicon
    if !is_apple_silicon()? {
        return Err(anyhow!("This function should only be called on Apple Silicon (M-series) Macs"));
    }
    
    // Ensure libtorch is available
    let libtorch_path = ensure_libtorch()?;
    
    // Set up dynamic library paths
    setup_apple_silicon_env(&libtorch_path)?;
    
    // Fix RPATH issues
    fix_rpath_issues()?;
    
    // Print MPS availability for diagnostics
    if has_mps()? {
        log::info!("Metal Performance Shaders (MPS) is available - will use hardware acceleration");
    } else {
        log::info!("Metal Performance Shaders (MPS) not detected - will use CPU only");
    }
    
    log::info!("Apple Silicon environment configured successfully");
    Ok(())
} 