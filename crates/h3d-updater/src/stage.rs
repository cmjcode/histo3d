//! Langkah `fetch`: unduh aset + `.sha256`, cocokkan, ekstrak ke folder
//! sementara. Belum menyentuh berkas terpasang.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};
use ureq::Agent;

use crate::net;
use crate::release::{parse_sha256_file, select_asset, to_hex, ArchiveKind, Release, Selected};

/// Tim penandatangan Developer ID rilis macOS (lihat build.sh).
pub const APPLE_TEAM_ID: &str = "YD4J5Z6A4G";

/// Nama biner GUI di dalam arsip/folder terpasang.
pub fn app_exe_name() -> String {
    format!("histo3d{}", std::env::consts::EXE_SUFFIX)
}

/// Folder kerja pembaruan (di folder sementara OS).
pub fn staging_root() -> PathBuf {
    std::env::temp_dir().join("histo3d-update")
}

/// Unduh, cocokkan, dan ekstrak rilis untuk `target`. Mengembalikan aset
/// terpilih dan *payload*: folder berisi biner (Windows/Linux) atau
/// `HISTO3D.app` (macOS).
pub fn fetch(agent: &Agent, release: &Release, target: &str) -> Result<(Selected, PathBuf)> {
    let sel = select_asset(release, target)?
        .with_context(|| format!("rilis {} belum punya aset untuk {target}", release.tag_name))?;
    let dir = staging_root().join(&release.tag_name);
    if dir.exists() {
        std::fs::remove_dir_all(&dir)
            .with_context(|| format!("gagal membersihkan {}", dir.display()))?;
    }
    std::fs::create_dir_all(&dir).with_context(|| format!("gagal membuat {}", dir.display()))?;

    let sha_text = net::get_text(agent, &sel.checksum.browser_download_url)?;
    let want = parse_sha256_file(&sha_text, &sel.archive.name)?;
    let archive = dir.join(&sel.archive.name);
    let got = net::download(
        agent,
        &sel.archive.browser_download_url,
        &archive,
        sel.archive.size,
    )?;
    if got != want {
        let _ = std::fs::remove_file(&archive);
        bail!(
            "sha256 {} tidak cocok (diharapkan {}, didapat {})",
            sel.archive.name,
            to_hex(&want),
            to_hex(&got)
        );
    }

    let out = dir.join("x");
    std::fs::create_dir_all(&out).with_context(|| format!("gagal membuat {}", out.display()))?;
    extract(sel.kind, &archive, &out)?;
    let payload = locate_payload(sel.kind, &out)?;
    if sel.kind == ArchiveKind::AppZip {
        verify_codesign(&payload)?;
    }
    Ok((sel, payload))
}

/// Ekstrak memakai alat bawaan OS: `tar` (Linux; `tar.exe`/bsdtar di
/// Windows 10+ juga membaca zip) dan `ditto` di macOS (menjaga tanda tangan,
/// symlink, dan atribut bundel).
pub fn extract(kind: ArchiveKind, archive: &Path, out: &Path) -> Result<()> {
    let a = archive.as_os_str();
    let o = out.as_os_str();
    match kind {
        ArchiveKind::TarGz => run("tar", &["-xzf".as_ref(), a, "-C".as_ref(), o]),
        ArchiveKind::Zip => run("tar", &["-xf".as_ref(), a, "-C".as_ref(), o]),
        ArchiveKind::AppZip => run("ditto", &["-x".as_ref(), "-k".as_ref(), a, o]),
    }
}

/// Temukan payload di hasil ekstraksi (urutan nama tetap; aturan 9).
pub fn locate_payload(kind: ArchiveKind, out: &Path) -> Result<PathBuf> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(out)
        .with_context(|| format!("gagal membaca {}", out.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    match kind {
        ArchiveKind::AppZip => dirs
            .into_iter()
            .find(|p| p.extension().is_some_and(|e| e == "app"))
            .context("arsip tidak berisi bundel .app"),
        ArchiveKind::TarGz | ArchiveKind::Zip => {
            let exe = app_exe_name();
            if out.join(&exe).is_file() {
                return Ok(out.to_path_buf());
            }
            dirs.into_iter()
                .find(|d| d.join(&exe).is_file())
                .with_context(|| format!("arsip tidak berisi {exe}"))
        }
    }
}

/// macOS: bundel baru harus ditandatangani Developer ID tim HISTO3D.
pub fn verify_codesign(app: &Path) -> Result<()> {
    let req =
        format!("-R=anchor apple generic and certificate leaf[subject.OU] = \"{APPLE_TEAM_ID}\"");
    run(
        "codesign",
        &[
            "--verify".as_ref(),
            "--deep".as_ref(),
            "--strict".as_ref(),
            req.as_ref(),
            app.as_os_str(),
        ],
    )
    .context("tanda tangan bundel pembaruan tidak sah")
}

/// Jalankan perintah OS; galat memuat stderr-nya.
pub fn run(cmd: &str, args: &[&std::ffi::OsStr]) -> Result<()> {
    let out = Command::new(cmd)
        .args(args)
        .output()
        .with_context(|| format!("gagal menjalankan {cmd}"))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        bail!("{cmd} gagal: {}", err.trim());
    }
    Ok(())
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn extract_tar_gz_and_locate_folder() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("histo3d-0.2.0-x");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("histo3d"), b"bin").unwrap();
        std::fs::write(src.join("h3d"), b"cli").unwrap();
        let archive = tmp.path().join("a.tar.gz");
        run(
            "tar",
            &[
                "-czf".as_ref(),
                archive.as_os_str(),
                "-C".as_ref(),
                tmp.path().as_os_str(),
                "histo3d-0.2.0-x".as_ref(),
            ],
        )
        .unwrap();
        let out = tmp.path().join("out");
        std::fs::create_dir_all(&out).unwrap();
        extract(ArchiveKind::TarGz, &archive, &out).unwrap();
        let payload = locate_payload(ArchiveKind::TarGz, &out).unwrap();
        assert_eq!(payload, out.join("histo3d-0.2.0-x"));
        assert_eq!(std::fs::read(payload.join("histo3d")).unwrap(), b"bin");
    }

    #[test]
    fn locate_fails_without_binary() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("kosong")).unwrap();
        assert!(locate_payload(ArchiveKind::TarGz, tmp.path()).is_err());
        assert!(locate_payload(ArchiveKind::AppZip, tmp.path()).is_err());
    }
}
