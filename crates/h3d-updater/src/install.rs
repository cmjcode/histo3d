//! Langkah `apply`: ganti berkas terpasang dengan payload hasil `fetch`.
//!
//! - Windows/Linux: tiap berkas disalin ke `.<nama>.new` di folder yang
//!   sama, berkas lama di-rename ke `<nama>.old`, lalu `.new` di-rename ke
//!   nama asli. Biner yang sedang berjalan (updater ini sendiri) boleh
//!   di-rename di Windows, jadi updater bisa mengganti dirinya. Bila satu
//!   langkah gagal, semua yang sudah diganti dikembalikan (rollback).
//! - macOS: bundel baru disalin `ditto` ke `.HISTO3D.app.new` di sebelah
//!   bundel lama, tanda tangannya dicek lagi, lalu kedua bundel ditukar
//!   dengan dua `rename`.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use anyhow::{bail, Context, Result};

use crate::stage::{app_exe_name, run, verify_codesign};

/// Tempat HISTO3D terpasang.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstallTarget {
    /// macOS: `…/HISTO3D.app`.
    Bundle(PathBuf),
    /// Windows/Linux: folder berisi `histo3d`, `h3d`, `histo3d-updater`.
    Dir(PathBuf),
}

/// Tentukan tempat terpasang dari lokasi biner updater.
pub fn detect(exe: &Path) -> Result<InstallTarget> {
    let dir = exe
        .parent()
        .context("lokasi histo3d-updater tidak diketahui")?;
    if let Some(app) = bundle_of(dir) {
        return Ok(InstallTarget::Bundle(app));
    }
    if cfg!(target_os = "macos") {
        bail!("HISTO3D tidak berjalan dari bundel .app; perbarui manual lewat berkas .dmg");
    }
    if is_dev_build(dir) {
        bail!("build pengembangan (folder target/) tidak diperbarui otomatis");
    }
    Ok(InstallTarget::Dir(dir.to_path_buf()))
}

/// `X.app/Contents/MacOS` → `X.app`.
fn bundle_of(macos_dir: &Path) -> Option<PathBuf> {
    if macos_dir.file_name()? != "MacOS" {
        return None;
    }
    let contents = macos_dir.parent()?;
    if contents.file_name()? != "Contents" {
        return None;
    }
    let app = contents.parent()?;
    app.extension()
        .is_some_and(|e| e == "app")
        .then(|| app.to_path_buf())
}

fn is_dev_build(dir: &Path) -> bool {
    let name = dir.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let parent = dir
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("");
    matches!(name, "debug" | "release") && parent == "target"
}

/// Tunggu stdin tertutup: `h3d-app` memegang ujung tulis pipa sampai
/// prosesnya keluar, jadi EOF berarti aplikasi sudah tertutup (lintas OS,
/// tanpa API PID).
pub fn wait_for_parent_exit() {
    let mut sink = Vec::new();
    let _ = std::io::stdin().read_to_end(&mut sink);
    // Windows melepas kunci berkas .exe sesaat setelah proses keluar.
    std::thread::sleep(Duration::from_millis(500));
}

/// Pasang payload ke `target`.
pub fn apply(payload: &Path, target: &InstallTarget) -> Result<()> {
    if !payload.exists() {
        bail!("payload pembaruan tidak ada: {}", payload.display());
    }
    match target {
        InstallTarget::Dir(dir) => apply_dir(payload, dir),
        InstallTarget::Bundle(app) => apply_bundle(payload, app),
    }
}

/// Buka lagi HISTO3D yang (baru) terpasang.
pub fn relaunch(target: &InstallTarget) -> Result<()> {
    match target {
        InstallTarget::Bundle(app) => {
            Command::new("open")
                .arg(app)
                .spawn()
                .context("gagal membuka HISTO3D")?;
        }
        InstallTarget::Dir(dir) => {
            Command::new(dir.join(app_exe_name()))
                .current_dir(dir)
                .spawn()
                .context("gagal membuka HISTO3D")?;
        }
    }
    Ok(())
}

/// Buang sisa pembaruan sebelumnya (`*.old` yang tidak bisa dihapus saat
/// itu karena biner masih berjalan). Hanya berkas milik HISTO3D.
pub fn cleanup_leftovers(target: &InstallTarget) {
    match target {
        InstallTarget::Dir(dir) => {
            let Ok(entries) = std::fs::read_dir(dir) else {
                return;
            };
            for e in entries.flatten() {
                let name = e.file_name().to_string_lossy().into_owned();
                if is_leftover(&name) {
                    let _ = std::fs::remove_file(e.path());
                }
            }
        }
        InstallTarget::Bundle(app) => {
            if let Some(old) = sibling(app, "", ".old") {
                if old.exists() {
                    let _ = std::fs::remove_dir_all(old);
                }
            }
        }
    }
}

fn is_leftover(name: &str) -> bool {
    let ours = |n: &str| {
        n.starts_with("histo3d")
            || n.starts_with("h3d")
            || n == "LICENSE"
            || n.starts_with("README")
    };
    if let Some(base) = name.strip_suffix(".old") {
        return ours(base);
    }
    if let Some(base) = name.strip_prefix('.').and_then(|n| n.strip_suffix(".new")) {
        return ours(base);
    }
    false
}

/// `dir/<prefix><nama><suffix>` untuk berkas/bundel `path`.
fn sibling(path: &Path, prefix: &str, suffix: &str) -> Option<PathBuf> {
    let name = path.file_name()?.to_string_lossy();
    Some(path.parent()?.join(format!("{prefix}{name}{suffix}")))
}

/// `rename` dengan percobaan ulang (Windows mengunci .exe sesaat setelah
/// proses keluar, dan antivirus kadang memegang berkas baru).
fn rename_retry(from: &Path, to: &Path) -> std::io::Result<()> {
    let mut last = None;
    for _ in 0..40 {
        match std::fs::rename(from, to) {
            Ok(()) => return Ok(()),
            Err(e) => last = Some(e),
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    Err(last.unwrap_or_else(|| std::io::Error::other("rename gagal")))
}

/// Satu berkas yang sudah diganti: tujuan dan cadangan lamanya (bila ada).
struct Replaced {
    dest: PathBuf,
    backup: Option<PathBuf>,
}

fn apply_dir(payload: &Path, dir: &Path) -> Result<()> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(payload)
        .with_context(|| format!("gagal membaca {}", payload.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file())
        .collect();
    files.sort();
    if !files.iter().any(|f| f.ends_with(app_exe_name())) {
        bail!("payload tidak berisi {}", app_exe_name());
    }

    let mut done: Vec<Replaced> = Vec::new();
    for src in &files {
        match replace_one(src, dir) {
            Ok(r) => done.push(r),
            Err(e) => {
                rollback(&done);
                return Err(e).with_context(|| {
                    format!(
                        "tidak bisa menulis ke {} (perlu izin tulis; atau perbarui manual)",
                        dir.display()
                    )
                });
            }
        }
    }
    // Cadangan dibuang; yang gagal (biner updater yang masih berjalan di
    // Windows) dibersihkan oleh `cleanup_leftovers` pada jalan berikutnya.
    for r in &done {
        if let Some(b) = &r.backup {
            let _ = std::fs::remove_file(b);
        }
    }
    Ok(())
}

fn replace_one(src: &Path, dir: &Path) -> Result<Replaced> {
    let name = src
        .file_name()
        .context("nama berkas payload kosong")?
        .to_string_lossy()
        .into_owned();
    let dest = dir.join(&name);
    let staged = dir.join(format!(".{name}.new"));
    let backup = dir.join(format!("{name}.old"));
    // `fs::copy` ikut menyalin izin (bit eksekusi di Linux).
    std::fs::copy(src, &staged).with_context(|| format!("gagal menyalin {name}"))?;
    let had = dest.exists();
    if had {
        let _ = std::fs::remove_file(&backup);
        if let Err(e) = rename_retry(&dest, &backup) {
            let _ = std::fs::remove_file(&staged);
            return Err(e).with_context(|| format!("gagal memindahkan {name} lama"));
        }
    }
    if let Err(e) = rename_retry(&staged, &dest) {
        let _ = std::fs::remove_file(&staged);
        if had {
            let _ = std::fs::rename(&backup, &dest);
        }
        return Err(e).with_context(|| format!("gagal memasang {name}"));
    }
    Ok(Replaced {
        dest,
        backup: had.then_some(backup),
    })
}

fn rollback(done: &[Replaced]) {
    for r in done.iter().rev() {
        match &r.backup {
            Some(b) => {
                let _ = std::fs::remove_file(&r.dest);
                let _ = std::fs::rename(b, &r.dest);
            }
            None => {
                let _ = std::fs::remove_file(&r.dest);
            }
        }
    }
}

fn apply_bundle(payload: &Path, app: &Path) -> Result<()> {
    let staged = sibling(app, ".", ".new").context("lokasi bundel tidak sah")?;
    let backup = sibling(app, "", ".old").context("lokasi bundel tidak sah")?;
    for p in [&staged, &backup] {
        if p.exists() {
            std::fs::remove_dir_all(p)
                .with_context(|| format!("gagal membersihkan {}", p.display()))?;
        }
    }
    run("ditto", &[payload.as_os_str(), staged.as_os_str()]).with_context(|| {
        format!(
            "tidak bisa menulis ke {} (perlu izin tulis; atau perbarui manual)",
            app.parent()
                .map(|p| p.display().to_string())
                .unwrap_or_default()
        )
    })?;
    if let Err(e) = verify_codesign(&staged) {
        let _ = std::fs::remove_dir_all(&staged);
        return Err(e);
    }
    std::fs::rename(app, &backup).context("gagal memindahkan bundel lama")?;
    if let Err(e) = std::fs::rename(&staged, app) {
        let _ = std::fs::rename(&backup, app);
        let _ = std::fs::remove_dir_all(&staged);
        return Err(e).context("gagal memasang bundel baru");
    }
    let _ = std::fs::remove_dir_all(&backup);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_bundle_and_dev_build() {
        let exe = Path::new("/Applications/HISTO3D.app/Contents/MacOS/histo3d-updater");
        assert_eq!(
            detect(exe).unwrap(),
            InstallTarget::Bundle(PathBuf::from("/Applications/HISTO3D.app"))
        );
        assert!(detect(Path::new("/x/target/release/histo3d-updater")).is_err());
        let plain = detect(Path::new("/opt/histo3d/histo3d-updater"));
        if cfg!(target_os = "macos") {
            assert!(plain.is_err());
        } else {
            assert_eq!(
                plain.unwrap(),
                InstallTarget::Dir(PathBuf::from("/opt/histo3d"))
            );
        }
    }

    #[test]
    fn leftovers_only_ours() {
        assert!(is_leftover("histo3d.old"));
        assert!(is_leftover("histo3d-updater.exe.old"));
        assert!(is_leftover(".h3d.new"));
        assert!(is_leftover("README.md.old"));
        assert!(!is_leftover("laporan.old"));
        assert!(!is_leftover("histo3d"));
    }

    fn write(p: &Path, s: &str) {
        std::fs::write(p, s).unwrap();
    }

    fn read(p: &Path) -> String {
        std::fs::read_to_string(p).unwrap()
    }

    #[test]
    fn apply_dir_replaces_all_files() {
        let tmp = tempfile::tempdir().unwrap();
        let (payload, dir) = (tmp.path().join("p"), tmp.path().join("d"));
        std::fs::create_dir_all(&payload).unwrap();
        std::fs::create_dir_all(&dir).unwrap();
        let exe = app_exe_name();
        write(&payload.join(&exe), "baru");
        write(&payload.join("h3d"), "cli-baru");
        write(&dir.join(&exe), "lama");
        write(&dir.join("data-pengguna.txt"), "jangan disentuh");

        apply(&payload, &InstallTarget::Dir(dir.clone())).unwrap();
        assert_eq!(read(&dir.join(&exe)), "baru");
        assert_eq!(read(&dir.join("h3d")), "cli-baru");
        assert_eq!(read(&dir.join("data-pengguna.txt")), "jangan disentuh");
        assert!(!dir.join(format!("{exe}.old")).exists());
        assert!(!dir.join(format!(".{exe}.new")).exists());
    }

    #[test]
    fn apply_dir_rolls_back_on_failure() {
        let tmp = tempfile::tempdir().unwrap();
        let (payload, dir) = (tmp.path().join("p"), tmp.path().join("d"));
        std::fs::create_dir_all(&payload).unwrap();
        std::fs::create_dir_all(&dir).unwrap();
        let exe = app_exe_name();
        write(&payload.join("a-first"), "baru");
        write(&payload.join(&exe), "baru");
        write(&dir.join("a-first"), "lama");
        write(&dir.join(&exe), "lama");
        // Tempat salinan sementara berupa folder → penyalinan gagal → rollback.
        std::fs::create_dir_all(dir.join(format!(".{exe}.new")).join("isi")).unwrap();

        assert!(apply(&payload, &InstallTarget::Dir(dir.clone())).is_err());
        assert_eq!(read(&dir.join("a-first")), "lama");
        assert_eq!(read(&dir.join(&exe)), "lama");
    }

    #[test]
    fn apply_rejects_payload_without_app() {
        let tmp = tempfile::tempdir().unwrap();
        let payload = tmp.path().join("p");
        std::fs::create_dir_all(&payload).unwrap();
        write(&payload.join("lain"), "x");
        let dir = tmp.path().to_path_buf();
        assert!(apply(&payload, &InstallTarget::Dir(dir)).is_err());
    }
}
