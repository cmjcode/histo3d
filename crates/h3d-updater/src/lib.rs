//! `h3d-updater`: pembaru HISTO3D dari GitHub Release.
//!
//! Berjalan sebagai **proses terpisah** (`histo3d-updater`) agar `h3d-app`
//! tetap tanpa crate HTTP (aturan keras 11). Aplikasi hanya menjalankannya
//! saat pengguna menekan "Periksa pembaruan"; tidak ada pemeriksaan
//! otomatis, dan yang dikirim ke GitHub hanya permintaan GET tanpa data
//! proyek (K13a).
//!
//! Alur tiga langkah, masing-masing satu subperintah:
//! 1. `check`  — baca rilis terbaru, bandingkan versi ([`release`]).
//! 2. `fetch`  — unduh aset untuk OS ini + `.sha256`, cocokkan, ekstrak ke
//!    folder sementara; di macOS tanda tangan kode diperiksa ([`stage`]).
//! 3. `apply`  — tunggu aplikasi tertutup, ganti berkas terpasang dengan
//!    rollback bila gagal, lalu buka lagi ([`install`]).
//!
//! Hasil tiap langkah ditulis sebagai satu baris JSON [`Report`] di stdout.

pub mod install;
pub mod net;
pub mod release;
pub mod stage;

use std::path::PathBuf;

use serde::Serialize;

/// Repositori GitHub sumber rilis.
pub const REPO: &str = "cmjcode/histo3d";

/// Versi yang sedang terpasang (versi workspace).
pub const CURRENT_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Target triple biner ini (dari `build.rs`).
pub const TARGET: &str = env!("H3D_TARGET");

/// Laporan satu langkah untuk dibaca `h3d-app` (satu baris JSON di stdout).
/// Bentuknya dicerminkan oleh `h3d_app::update::Report`; ubah keduanya
/// bersamaan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Report {
    /// Versi terpasang sudah yang terbaru.
    UpToDate { current: String, latest: String },
    /// Ada versi baru dengan aset untuk OS ini.
    Available {
        current: String,
        latest: String,
        tag: String,
        page: String,
    },
    /// Ada versi baru, tetapi asetnya belum diunggah untuk OS ini.
    Unsupported { latest: String, page: String },
    /// Pembaruan sudah diunduh, dicocokkan, dan diekstrak.
    Staged {
        latest: String,
        tag: String,
        staged: PathBuf,
    },
    /// Pembaruan terpasang.
    Applied,
    /// Galat (pesan berbahasa Indonesia).
    Error { message: String },
}

impl Report {
    /// Satu baris JSON.
    pub fn to_line(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| {
            r#"{"status":"error","message":"gagal menyusun laporan"}"#.to_owned()
        })
    }
}

/// Versi terpasang sebagai [`release::Version`].
pub fn current_version() -> anyhow::Result<release::Version> {
    release::Version::parse(CURRENT_VERSION)
        .ok_or_else(|| anyhow::anyhow!("versi terpasang tidak sah: {CURRENT_VERSION}"))
}

/// Bandingkan rilis terbaru dengan versi terpasang (tanpa jaringan).
pub fn check_report(
    release: &release::Release,
    target: &str,
    current: release::Version,
) -> anyhow::Result<Report> {
    let latest = release::release_version(release)?;
    if latest <= current {
        return Ok(Report::UpToDate {
            current: current.to_string(),
            latest: latest.to_string(),
        });
    }
    Ok(match release::select_asset(release, target)? {
        Some(_) => Report::Available {
            current: current.to_string(),
            latest: latest.to_string(),
            tag: release.tag_name.clone(),
            page: release.html_url.clone(),
        },
        None => Report::Unsupported {
            latest: latest.to_string(),
            page: release.html_url.clone(),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use release::{Asset, Release, Version};

    fn rel(tag: &str, assets: &[&str]) -> Release {
        Release {
            tag_name: tag.to_owned(),
            html_url: "https://example.invalid/r".to_owned(),
            draft: false,
            prerelease: false,
            assets: assets
                .iter()
                .map(|n| Asset {
                    name: (*n).to_owned(),
                    browser_download_url: String::new(),
                    size: 1,
                })
                .collect(),
        }
    }

    #[test]
    fn check_compares_versions_and_assets() {
        let t = "x86_64-unknown-linux-gnu";
        let cur = Version(0, 1, 0);
        assert!(matches!(
            check_report(&rel("v0.1.0", &[]), t, cur).unwrap(),
            Report::UpToDate { .. }
        ));
        assert!(matches!(
            check_report(&rel("v0.0.9", &[]), t, cur).unwrap(),
            Report::UpToDate { .. }
        ));
        assert!(matches!(
            check_report(&rel("v0.2.0", &[]), t, cur).unwrap(),
            Report::Unsupported { .. }
        ));
        let a = "histo3d-0.2.0-x86_64-unknown-linux-gnu.tar.gz";
        let s = format!("{a}.sha256");
        assert_eq!(
            check_report(&rel("v0.2.0", &[a, &s]), t, cur).unwrap(),
            Report::Available {
                current: "0.1.0".into(),
                latest: "0.2.0".into(),
                tag: "v0.2.0".into(),
                page: "https://example.invalid/r".into(),
            }
        );
    }

    /// Bentuk JSON yang dibaca `h3d_app::update` (tes pasangannya di sana).
    #[test]
    fn report_json_shape() {
        assert_eq!(
            Report::Staged {
                latest: "0.2.0".into(),
                tag: "v0.2.0".into(),
                staged: "/tmp/x".into(),
            }
            .to_line(),
            r#"{"status":"staged","latest":"0.2.0","tag":"v0.2.0","staged":"/tmp/x"}"#
        );
        assert_eq!(Report::Applied.to_line(), r#"{"status":"applied"}"#);
        assert_eq!(
            Report::UpToDate {
                current: "0.1.0".into(),
                latest: "0.1.0".into()
            }
            .to_line(),
            r#"{"status":"up_to_date","current":"0.1.0","latest":"0.1.0"}"#
        );
    }
}
