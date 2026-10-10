//! Tulis atomik (aturan keras 10): tulis ke `nama.tmp` di folder yang sama,
//! `fsync`, lalu `rename`. Pembaca tidak pernah melihat berkas setengah jadi,
//! dan kegagalan di tengah jalan tidak merusak berkas lama.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::error::IoError;

/// Path sementara untuk `path`: `nama.tmp` di folder yang sama.
pub fn tmp_path(path: &Path) -> PathBuf {
    let mut name = path
        .file_name()
        .map(|s| s.to_os_string())
        .unwrap_or_default();
    name.push(".tmp");
    path.with_file_name(name)
}

/// Tulis `bytes` ke `path` secara atomik. Folder induk dibuat bila belum ada.
/// Bila gagal, berkas lama (kalau ada) tidak berubah dan `.tmp` dibersihkan.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), IoError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| IoError::io(parent, e))?;
        }
    }
    let tmp = tmp_path(path);
    let write = || -> std::io::Result<()> {
        let mut f = File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        Ok(())
    };
    if let Err(e) = write() {
        // Jangan menghapus bila `.tmp` ternyata folder milik pihak lain.
        if tmp.is_file() {
            let _ = fs::remove_file(&tmp);
        }
        return Err(IoError::io(&tmp, e));
    }
    if let Err(e) = fs::rename(&tmp, path) {
        let _ = fs::remove_file(&tmp);
        return Err(IoError::io(path, e));
    }
    Ok(())
}

/// Salin `src` ke `dst` secara atomik (lewat `dst.tmp`), tanpa memuat seluruh
/// berkas ke memori.
pub fn copy_atomic(src: &Path, dst: &Path) -> Result<u64, IoError> {
    if let Some(parent) = dst.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| IoError::io(parent, e))?;
        }
    }
    let tmp = tmp_path(dst);
    let copy = || -> std::io::Result<u64> {
        let n = fs::copy(src, &tmp)?;
        File::open(&tmp)?.sync_all()?;
        Ok(n)
    };
    let n = match copy() {
        Ok(n) => n,
        Err(e) => {
            if tmp.is_file() {
                let _ = fs::remove_file(&tmp);
            }
            return Err(IoError::io(src, e));
        }
    };
    if let Err(e) = fs::rename(&tmp, dst) {
        let _ = fs::remove_file(&tmp);
        return Err(IoError::io(dst, e));
    }
    Ok(n)
}

/// Hapus berkas bila ada; tidak ada berkas bukan galat.
pub fn remove_if_exists(path: &Path) -> Result<(), IoError> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(IoError::io(path, e)),
    }
}
