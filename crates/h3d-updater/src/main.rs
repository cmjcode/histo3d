//! `histo3d-updater`: pembaru HISTO3D dari GitHub Release.
//! Hanya untuk riset, bukan untuk diagnosis.
//!
//! Dijalankan `h3d-app` (tombol "Periksa pembaruan" di dialog Tentang) atau
//! langsung dari terminal (`histo3d-updater update`). Hasil tiap subperintah
//! ditulis sebagai satu baris JSON di stdout; progres di stderr.

use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use h3d_updater::{check_report, current_version, install, net, stage, Report, TARGET};

#[derive(Parser)]
#[command(
    name = "histo3d-updater",
    version,
    about = "Pembaru HISTO3D dari GitHub Release. Hanya untuk riset, bukan untuk diagnosis."
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Periksa apakah ada rilis yang lebih baru.
    Check,
    /// Unduh rilis terbaru, cocokkan sha256, dan ekstrak (belum dipasang).
    Fetch,
    /// Pasang payload hasil `fetch`.
    Apply {
        /// Folder/bundel hasil `fetch` (kolom `staged`).
        #[arg(long)]
        staged: PathBuf,
        /// Tunggu stdin tertutup (aplikasi sudah keluar) sebelum memasang.
        #[arg(long)]
        wait_stdin: bool,
        /// Buka HISTO3D lagi setelah selesai (juga bila gagal: versi lama).
        #[arg(long)]
        relaunch: bool,
    },
    /// Periksa, unduh, dan pasang sekaligus (untuk terminal).
    Update,
}

/// Tulis ke stderr tanpa panik bila pipa sudah tertutup.
fn say(msg: &str) {
    let _ = writeln!(std::io::stderr(), "histo3d-updater: {msg}");
}

fn install_target() -> Result<install::InstallTarget> {
    let exe = std::env::current_exe().context("lokasi histo3d-updater tidak diketahui")?;
    install::detect(&exe)
}

fn check() -> Result<Report> {
    say("memeriksa rilis terbaru di GitHub");
    let release = net::latest_release(&net::agent())?;
    check_report(&release, TARGET, current_version()?)
}

fn fetch() -> Result<Report> {
    let agent = net::agent();
    let release = net::latest_release(&agent)?;
    match check_report(&release, TARGET, current_version()?)? {
        Report::Available { .. } => {}
        Report::UpToDate { latest, .. } => bail!("versi {latest} sudah yang terbaru"),
        Report::Unsupported { latest, .. } => {
            bail!("versi {latest} belum tersedia untuk {TARGET}")
        }
        other => return Ok(other),
    }
    say(&format!("mengunduh {}", release.tag_name));
    let (sel, payload) = stage::fetch(&agent, &release, TARGET)?;
    say(&format!("siap dipasang: {}", payload.display()));
    Ok(Report::Staged {
        latest: sel.version.to_string(),
        tag: release.tag_name.clone(),
        staged: payload,
    })
}

fn apply(staged: PathBuf, wait_stdin: bool, relaunch: bool) -> Result<Report> {
    let target = install_target()?;
    if wait_stdin {
        install::wait_for_parent_exit();
    }
    say(&format!("memasang dari {}", staged.display()));
    let result = install::apply(&staged, &target);
    if relaunch {
        // Gagal pun aplikasi dibuka lagi: rollback sudah memulihkan versi lama.
        if let Err(e) = install::relaunch(&target) {
            say(&format!("{e:#}"));
        }
    }
    result?;
    let _ = std::fs::remove_dir_all(stage::staging_root());
    Ok(Report::Applied)
}

fn update() -> Result<Report> {
    match fetch()? {
        Report::Staged { staged, .. } => apply(staged, false, false),
        other => Ok(other),
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    if let Ok(target) = install_target() {
        install::cleanup_leftovers(&target);
    }
    let result = match cli.cmd {
        Cmd::Check => check(),
        Cmd::Fetch => fetch(),
        Cmd::Apply {
            staged,
            wait_stdin,
            relaunch,
        } => apply(staged, wait_stdin, relaunch),
        Cmd::Update => update(),
    };
    let report = result.unwrap_or_else(|e| Report::Error {
        message: format!("{e:#}"),
    });
    if let Report::Error { message } = &report {
        say(message);
    }
    let _ = writeln!(std::io::stdout(), "{}", report.to_line());
    if matches!(report, Report::Error { .. }) {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
