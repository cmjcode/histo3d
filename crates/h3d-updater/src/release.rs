//! Logika murni tanpa jaringan: versi, pemilihan aset, berkas `.sha256`.

use std::fmt;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

/// Rilis GitHub (bagian yang dipakai dari `GET /repos/{repo}/releases/latest`).
#[derive(Debug, Clone, Deserialize)]
pub struct Release {
    pub tag_name: String,
    pub html_url: String,
    #[serde(default)]
    pub draft: bool,
    #[serde(default)]
    pub prerelease: bool,
    #[serde(default)]
    pub assets: Vec<Asset>,
}

/// Satu berkas yang diunggah ke rilis.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Asset {
    pub name: String,
    pub browser_download_url: String,
    #[serde(default)]
    pub size: u64,
}

/// Versi `MAYOR.MINOR.PATCH` (tag rilis `vX.Y.Z`, lihat release.yml).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Version(pub u64, pub u64, pub u64);

impl Version {
    /// Urai `1.2.3` atau `v1.2.3`; selain itu `None`.
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        let s = s.strip_prefix('v').unwrap_or(s);
        let mut it = s.split('.');
        let major = it.next()?.parse().ok()?;
        let minor = it.next()?.parse().ok()?;
        let patch = it.next()?.parse().ok()?;
        if it.next().is_some() {
            return None;
        }
        Some(Version(major, minor, patch))
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.0, self.1, self.2)
    }
}

/// Jenis arsip rilis per OS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchiveKind {
    /// Linux: `histo3d-<v>-<target>.tar.gz` berisi satu folder.
    TarGz,
    /// Windows: `histo3d-<v>-<target>.zip` berisi satu folder.
    Zip,
    /// macOS: `histo3d-<v>-<target>.app.zip` berisi `HISTO3D.app`
    /// (dibuat `ditto` di build.sh, sudah ditandatangani dan dinotarisasi).
    AppZip,
}

/// Nama aset yang diharapkan untuk versi dan target ini.
pub fn asset_name(version: Version, target: &str) -> (String, ArchiveKind) {
    let kind = if target.contains("apple-darwin") {
        ArchiveKind::AppZip
    } else if target.contains("windows") {
        ArchiveKind::Zip
    } else {
        ArchiveKind::TarGz
    };
    let ext = match kind {
        ArchiveKind::AppZip => "app.zip",
        ArchiveKind::Zip => "zip",
        ArchiveKind::TarGz => "tar.gz",
    };
    (format!("histo3d-{version}-{target}.{ext}"), kind)
}

/// Aset terpilih: arsip + berkas `.sha256` pasangannya.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selected {
    pub version: Version,
    pub archive: Asset,
    pub checksum: Asset,
    pub kind: ArchiveKind,
}

/// Versi rilis (dari tag); galat bila tag bukan `vX.Y.Z`.
pub fn release_version(release: &Release) -> Result<Version> {
    Version::parse(&release.tag_name)
        .with_context(|| format!("tag rilis tidak dikenali: {:?}", release.tag_name))
}

/// Pilih aset untuk `target`. `Ok(None)` bila rilis belum punya aset
/// (atau `.sha256`-nya) untuk OS ini — misalnya build macOS belum diunggah.
pub fn select_asset(release: &Release, target: &str) -> Result<Option<Selected>> {
    if release.draft || release.prerelease {
        bail!("rilis {} masih draf/pra-rilis", release.tag_name);
    }
    let version = release_version(release)?;
    let (name, kind) = asset_name(version, target);
    let sha_name = format!("{name}.sha256");
    let find = |n: &str| release.assets.iter().find(|a| a.name == n).cloned();
    Ok(match (find(&name), find(&sha_name)) {
        (Some(archive), Some(checksum)) => Some(Selected {
            version,
            archive,
            checksum,
            kind,
        }),
        _ => None,
    })
}

/// Baca berkas `.sha256` (`<hex>  <nama>`, `<hex> *<nama>`, atau `<hex>`
/// saja). Bila nama tercantum, harus sama dengan `expected_name`.
pub fn parse_sha256_file(text: &str, expected_name: &str) -> Result<[u8; 32]> {
    let line = text
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .context("berkas sha256 kosong")?;
    let mut parts = line.split_whitespace();
    let hex = parts.next().context("berkas sha256 kosong")?;
    if let Some(name) = parts.next() {
        let name = name.trim_start_matches('*');
        // Nama boleh berawalan folder (mis. `dist/x.tar.gz`).
        let base = name.rsplit(['/', '\\']).next().unwrap_or(name);
        if base != expected_name {
            bail!("berkas sha256 untuk {base:?}, bukan {expected_name:?}");
        }
    }
    parse_hex32(hex).with_context(|| format!("sha256 tidak sah: {hex:?}"))
}

/// Heksadesimal 64 karakter → 32 bita.
pub fn parse_hex32(hex: &str) -> Option<[u8; 32]> {
    let bytes = hex.as_bytes();
    if bytes.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, pair) in bytes.chunks_exact(2).enumerate() {
        let s = std::str::from_utf8(pair).ok()?;
        out[i] = u8::from_str_radix(s, 16).ok()?;
    }
    Some(out)
}

/// 32 bita → heksadesimal huruf kecil.
pub fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset(name: &str) -> Asset {
        Asset {
            name: name.to_owned(),
            browser_download_url: format!("https://example.invalid/{name}"),
            size: 10,
        }
    }

    fn release(tag: &str, names: &[&str]) -> Release {
        Release {
            tag_name: tag.to_owned(),
            html_url: "https://example.invalid/r".to_owned(),
            draft: false,
            prerelease: false,
            assets: names.iter().map(|n| asset(n)).collect(),
        }
    }

    #[test]
    fn version_parse_and_order() {
        assert_eq!(Version::parse("v0.1.10"), Some(Version(0, 1, 10)));
        assert_eq!(Version::parse("1.2.3"), Some(Version(1, 2, 3)));
        assert_eq!(Version::parse("v1.2"), None);
        assert_eq!(Version::parse("v1.2.3.4"), None);
        assert_eq!(Version::parse("v1.2.3-rc1"), None);
        assert!(Version(0, 1, 10) > Version(0, 1, 9));
        assert!(Version(1, 0, 0) > Version(0, 99, 99));
        assert_eq!(Version(0, 2, 0).to_string(), "0.2.0");
    }

    #[test]
    fn asset_names_follow_release_workflow() {
        let v = Version(0, 2, 0);
        assert_eq!(
            asset_name(v, "x86_64-unknown-linux-gnu"),
            (
                "histo3d-0.2.0-x86_64-unknown-linux-gnu.tar.gz".to_owned(),
                ArchiveKind::TarGz
            )
        );
        assert_eq!(
            asset_name(v, "x86_64-pc-windows-msvc"),
            (
                "histo3d-0.2.0-x86_64-pc-windows-msvc.zip".to_owned(),
                ArchiveKind::Zip
            )
        );
        assert_eq!(
            asset_name(v, "aarch64-apple-darwin"),
            (
                "histo3d-0.2.0-aarch64-apple-darwin.app.zip".to_owned(),
                ArchiveKind::AppZip
            )
        );
    }

    #[test]
    fn select_needs_archive_and_checksum() {
        let t = "x86_64-unknown-linux-gnu";
        let a = "histo3d-0.2.0-x86_64-unknown-linux-gnu.tar.gz";
        let s = "histo3d-0.2.0-x86_64-unknown-linux-gnu.tar.gz.sha256";
        let r = release(
            "v0.2.0",
            &[a, s, "histo3d-0.2.0-x86_64-pc-windows-msvc.zip"],
        );
        let sel = select_asset(&r, t).unwrap().unwrap();
        assert_eq!(sel.archive.name, a);
        assert_eq!(sel.checksum.name, s);
        assert_eq!(sel.version, Version(0, 2, 0));

        // Tanpa .sha256 → dianggap belum tersedia, bukan dipasang tanpa cek.
        assert!(select_asset(&release("v0.2.0", &[a]), t).unwrap().is_none());
        // macOS belum diunggah.
        assert!(select_asset(&r, "aarch64-apple-darwin").unwrap().is_none());
    }

    #[test]
    fn select_rejects_bad_tag_and_prerelease() {
        assert!(select_asset(&release("nightly", &[]), "x").is_err());
        let mut r = release("v0.2.0", &[]);
        r.prerelease = true;
        assert!(select_asset(&r, "x").is_err());
    }

    #[test]
    fn release_json_from_github_parses() {
        let json = r#"{
            "tag_name": "v0.2.0",
            "html_url": "https://github.com/cmjcode/histo3d/releases/tag/v0.2.0",
            "draft": false, "prerelease": false, "body": "x", "id": 1,
            "assets": [{"name": "a.zip", "browser_download_url": "https://x/a.zip",
                        "size": 1234, "content_type": "application/zip"}]
        }"#;
        let r: Release = serde_json::from_str(json).unwrap();
        assert_eq!(r.tag_name, "v0.2.0");
        assert_eq!(r.assets[0].size, 1234);
    }

    #[test]
    fn sha256_file_formats() {
        let h = "ab".repeat(32);
        let want = [0xabu8; 32];
        assert_eq!(
            parse_sha256_file(&format!("{h}  x.tar.gz\n"), "x.tar.gz").unwrap(),
            want
        );
        assert_eq!(
            parse_sha256_file(&format!("{h} *x.tar.gz"), "x.tar.gz").unwrap(),
            want
        );
        assert_eq!(
            parse_sha256_file(&format!("{h} dist/x.tar.gz"), "x.tar.gz").unwrap(),
            want
        );
        assert_eq!(
            parse_sha256_file(&format!("\n{h}\n"), "x.tar.gz").unwrap(),
            want
        );
        assert_eq!(parse_sha256_file(&h.to_uppercase(), "x").unwrap(), want);
        assert!(parse_sha256_file(&format!("{h}  y.tar.gz"), "x.tar.gz").is_err());
        assert!(parse_sha256_file("zz", "x").is_err());
        assert!(parse_sha256_file("", "x").is_err());
        assert_eq!(to_hex(&want), h);
    }
}
