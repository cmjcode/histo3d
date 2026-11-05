//! Akses jaringan (hanya HTTPS ke GitHub). Tidak ada data proyek atau
//! identitas pengguna yang dikirim: hanya GET dengan User-Agent generik.

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use ureq::Agent;

use crate::release::Release;
use crate::{CURRENT_VERSION, REPO};

/// Batas ukuran jawaban API dan berkas `.sha256` (bita).
const SMALL_BODY_LIMIT: u64 = 4 * 1024 * 1024;

/// Klien HTTP: hanya HTTPS, batas waktu sambung dan jawaban.
pub fn agent() -> Agent {
    Agent::config_builder()
        .https_only(true)
        .timeout_connect(Some(Duration::from_secs(15)))
        .timeout_recv_response(Some(Duration::from_secs(30)))
        .user_agent(format!("histo3d-updater/{CURRENT_VERSION}"))
        .build()
        .into()
}

fn describe(e: ureq::Error, what: &str) -> anyhow::Error {
    match e {
        ureq::Error::StatusCode(404) => anyhow::anyhow!("{what}: tidak ditemukan (404)"),
        ureq::Error::StatusCode(403) | ureq::Error::StatusCode(429) => {
            anyhow::anyhow!("{what}: dibatasi GitHub, coba lagi nanti")
        }
        other => anyhow::anyhow!("{what}: {other}"),
    }
}

/// Rilis stabil terbaru (`/releases/latest` melewati draf dan pra-rilis).
pub fn latest_release(agent: &Agent) -> Result<Release> {
    let url = format!("https://api.github.com/repos/{REPO}/releases/latest");
    let mut resp = agent
        .get(&url)
        .header("Accept", "application/vnd.github+json")
        .header("X-GitHub-Api-Version", "2022-11-28")
        .call()
        .map_err(|e| match e {
            ureq::Error::StatusCode(404) => anyhow::anyhow!("belum ada rilis di GitHub"),
            other => describe(other, "gagal membaca rilis terbaru"),
        })?;
    resp.body_mut()
        .with_config()
        .limit(SMALL_BODY_LIMIT)
        .read_json::<Release>()
        .context("jawaban GitHub tidak bisa dibaca")
}

/// Unduh teks kecil (berkas `.sha256`).
pub fn get_text(agent: &Agent, url: &str) -> Result<String> {
    let mut resp = agent
        .get(url)
        .call()
        .map_err(|e| describe(e, "gagal mengunduh sha256"))?;
    resp.body_mut()
        .with_config()
        .limit(SMALL_BODY_LIMIT)
        .read_to_string()
        .context("berkas sha256 tidak bisa dibaca")
}

/// Unduh `url` ke `dest` sambil menghitung SHA-256. Ditulis ke berkas
/// `.part` lalu di-rename (aturan 10). `expected_size` (0 = tidak
/// diketahui) membatasi jumlah bita yang diterima.
pub fn download(agent: &Agent, url: &str, dest: &Path, expected_size: u64) -> Result<[u8; 32]> {
    let resp = agent
        .get(url)
        .call()
        .map_err(|e| describe(e, "gagal mengunduh pembaruan"))?;
    let mut reader = resp.into_body().into_reader();
    let tmp = dest.with_extension("part");
    let mut file =
        File::create(&tmp).with_context(|| format!("gagal membuat {}", tmp.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 256 * 1024];
    let mut total: u64 = 0;
    loop {
        let n = reader.read(&mut buf).context("unduhan terputus")?;
        if n == 0 {
            break;
        }
        total += n as u64;
        if expected_size > 0 && total > expected_size {
            drop(file);
            let _ = std::fs::remove_file(&tmp);
            bail!("unduhan lebih besar dari ukuran aset ({expected_size} bita)");
        }
        hasher.update(&buf[..n]);
        file.write_all(&buf[..n])
            .with_context(|| format!("gagal menulis {}", tmp.display()))?;
    }
    file.sync_all().ok();
    drop(file);
    if expected_size > 0 && total != expected_size {
        let _ = std::fs::remove_file(&tmp);
        bail!("unduhan tidak lengkap ({total} dari {expected_size} bita)");
    }
    std::fs::rename(&tmp, dest).with_context(|| format!("gagal menyimpan {}", dest.display()))?;
    Ok(hasher.finalize().into())
}
