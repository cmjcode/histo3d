//! Integrasi platform Apple (macOS dan iPadOS) lewat `objc2-foundation`
//! (pola DUCAD `apple.rs`): folder `Documents` dan `Application Support`
//! tersandbox, serta memori fisik perangkat.

use std::path::PathBuf;

use objc2_foundation::{
    NSFileManager, NSProcessInfo, NSSearchPathDirectory, NSSearchPathDomainMask,
};

fn search_dir(dir: NSSearchPathDirectory) -> Option<PathBuf> {
    let file_manager = NSFileManager::defaultManager();
    let urls = file_manager.URLsForDirectory_inDomains(dir, NSSearchPathDomainMask::UserDomainMask);
    let first = urls.firstObject()?;
    let path = first.path()?;
    Some(PathBuf::from(path.to_string()))
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".to_string()))
}

/// Folder `Documents` aplikasi. Di iPad ini folder tersandbox yang dibuka
/// ke aplikasi Files (9.3). Bila Foundation tidak memberi URL, jatuh ke
/// `$HOME/Documents`.
pub fn documents_directory() -> PathBuf {
    search_dir(NSSearchPathDirectory::DocumentDirectory).unwrap_or_else(|| home().join("Documents"))
}

/// Folder `Application Support` aplikasi.
pub fn app_support_directory() -> PathBuf {
    search_dir(NSSearchPathDirectory::ApplicationSupportDirectory)
        .unwrap_or_else(|| home().join("Library").join("Application Support"))
}

/// Memori fisik perangkat (byte), dari `NSProcessInfo`.
pub fn physical_memory_bytes() -> u64 {
    NSProcessInfo::processInfo().physicalMemory()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directories_are_not_empty() {
        assert!(documents_directory()
            .to_string_lossy()
            .contains("Documents"));
        assert!(app_support_directory()
            .to_string_lossy()
            .contains("Application Support"));
    }

    #[test]
    fn physical_memory_is_plausible() {
        // Lebih dari 256 MB di perangkat mana pun yang menjalankan tes.
        assert!(physical_memory_bytes() > 256 * 1024 * 1024);
    }
}
