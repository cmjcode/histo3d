//! Pembaruan dari GitHub Release lewat proses terpisah `histo3d-updater`
//! (crate `h3d-updater`). Aplikasi ini sendiri tidak memuat crate HTTP
//! (aturan keras 11, dijaga tes `app_has_no_http_dependency`); ia hanya
//! menjalankan biner updater yang terpasang di sebelahnya dan membaca satu
//! baris JSON dari stdout-nya.
//!
//! Jaringan hanya dipakai saat pengguna menekan "Periksa pembaruan" atau
//! "Unduh pembaruan" (K13a). Memasang = jalankan `apply --wait-stdin
//! --relaunch`, lalu tutup aplikasi: updater menunggu pipa stdin tertutup
//! (aplikasi sudah keluar dan proyek sudah tersimpan di `on_exit`), mengganti
//! berkas, dan membuka HISTO3D lagi.
//!
//! Di iPadOS tidak ada updater (pembaruan lewat App Store/TestFlight), jadi
//! bagian ini tidak tampil.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, Receiver, TryRecvError};

use egui::Ui;
use h3d_i18n::t;
use serde::Deserialize;

use crate::state::AppState;
use crate::ui_util::{icon_label, icon_text_button_enabled};

/// Laporan `histo3d-updater` (cermin `h3d_updater::Report`; ubah bersamaan).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Report {
    UpToDate {
        current: String,
        latest: String,
    },
    Available {
        current: String,
        latest: String,
        tag: String,
        page: String,
    },
    Unsupported {
        latest: String,
        page: String,
    },
    Staged {
        latest: String,
        tag: String,
        staged: PathBuf,
    },
    Applied,
    Error {
        message: String,
    },
}

/// Baca baris JSON terakhir dari stdout updater. Keluaran yang tidak
/// dikenali menjadi `Report::Error`, bukan panik (aturan keras 6).
pub fn parse_report(stdout: &str) -> Report {
    stdout
        .lines()
        .rev()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .and_then(|l| serde_json::from_str(l).ok())
        .unwrap_or_else(|| Report::Error {
            message: "jawaban histo3d-updater tidak dikenali".to_owned(),
        })
}

/// Tahap pembaruan yang tampil di dialog Tentang.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum UpdatePhase {
    #[default]
    Idle,
    Checking,
    Downloading {
        latest: String,
    },
    Done(Report),
}

/// Keadaan pemeriksa pembaruan. Proses updater berjalan di thread latar;
/// hasilnya diambil tiap frame lewat [`UpdateState::poll`].
#[derive(Debug, Default)]
pub struct UpdateState {
    pub phase: UpdatePhase,
    rx: Option<Receiver<Report>>,
}

impl UpdateState {
    /// Sedang menunggu updater.
    pub fn is_busy(&self) -> bool {
        self.rx.is_some()
    }

    /// Mulai `histo3d-updater check`.
    pub fn start_check(&mut self) {
        self.spawn(UpdatePhase::Checking, "check");
    }

    /// Mulai `histo3d-updater fetch` (unduh + sha256 + ekstrak).
    pub fn start_fetch(&mut self, latest: String) {
        self.spawn(UpdatePhase::Downloading { latest }, "fetch");
    }

    fn spawn(&mut self, phase: UpdatePhase, subcommand: &'static str) {
        if self.is_busy() {
            return;
        }
        let Some(exe) = updater_path() else {
            self.phase = UpdatePhase::Done(missing_updater());
            return;
        };
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(run_updater(&exe, subcommand));
        });
        self.rx = Some(rx);
        self.phase = phase;
    }

    /// Ambil hasil updater bila sudah ada. `true` = masih berjalan.
    pub fn poll(&mut self) -> bool {
        let Some(rx) = &self.rx else {
            return false;
        };
        match rx.try_recv() {
            Ok(report) => {
                self.phase = UpdatePhase::Done(report);
                self.rx = None;
                false
            }
            Err(TryRecvError::Empty) => true,
            Err(TryRecvError::Disconnected) => {
                self.phase = UpdatePhase::Done(Report::Error {
                    message: "histo3d-updater berhenti tanpa jawaban".to_owned(),
                });
                self.rx = None;
                false
            }
        }
    }
}

fn missing_updater() -> Report {
    Report::Error {
        message: "histo3d-updater tidak ditemukan di sebelah aplikasi".to_owned(),
    }
}

/// Lokasi `histo3d-updater` di sebelah biner aplikasi (macOS:
/// `HISTO3D.app/Contents/MacOS/`). `None` di iPadOS atau bila tidak ada.
pub fn updater_path() -> Option<PathBuf> {
    if cfg!(target_os = "ios") {
        return None;
    }
    let exe = std::env::current_exe().ok()?;
    let path = exe
        .parent()?
        .join(format!("histo3d-updater{}", std::env::consts::EXE_SUFFIX));
    path.is_file().then_some(path)
}

/// Windows: jangan munculkan jendela konsol untuk updater.
fn hide_console(cmd: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    #[cfg(not(windows))]
    let _ = cmd;
}

fn run_updater(exe: &Path, subcommand: &str) -> Report {
    let mut cmd = Command::new(exe);
    cmd.arg(subcommand).stdin(Stdio::null());
    hide_console(&mut cmd);
    match cmd.output() {
        Ok(out) => parse_report(&String::from_utf8_lossy(&out.stdout)),
        Err(e) => Report::Error {
            message: format!("gagal menjalankan histo3d-updater: {e}"),
        },
    }
}

/// Jalankan `apply` untuk payload `staged`. Pemanggil lalu menutup aplikasi.
pub fn launch_install(staged: &Path) -> Result<(), String> {
    let exe = updater_path().ok_or_else(|| match missing_updater() {
        Report::Error { message } => message,
        _ => String::new(),
    })?;
    let mut cmd = Command::new(exe);
    cmd.arg("apply")
        .arg("--staged")
        .arg(staged)
        .arg("--wait-stdin")
        .arg("--relaunch")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    hide_console(&mut cmd);
    let child = cmd
        .spawn()
        .map_err(|e| format!("gagal menjalankan histo3d-updater: {e}"))?;
    // Ujung tulis stdin sengaja tetap terbuka sampai proses ini keluar:
    // EOF adalah tanda bagi updater bahwa aplikasi sudah tertutup.
    std::mem::forget(child);
    Ok(())
}

impl AppState {
    /// Bagian "Pembaruan" di dialog Tentang. Tidak tampil bila updater tidak
    /// terpasang (iPadOS, build pengembangan tanpa updater).
    pub(crate) fn update_section(&mut self, ui: &mut Ui) {
        if updater_path().is_none() {
            return;
        }
        ui.separator();
        let busy = self.update.is_busy();
        match self.update.phase.clone() {
            UpdatePhase::Idle | UpdatePhase::Done(Report::Applied) => {}
            UpdatePhase::Checking => {
                ui.horizontal(|ui| {
                    ui.add(egui::Spinner::new().size(14.0));
                    ui.label(t!("update-checking"));
                });
            }
            UpdatePhase::Downloading { latest } => {
                ui.horizontal(|ui| {
                    ui.add(egui::Spinner::new().size(14.0));
                    ui.label(t!("update-downloading", version = latest));
                });
            }
            UpdatePhase::Done(Report::UpToDate { latest, .. }) => {
                ui.label(icon_label(
                    h3d_ui::icons::CHECK,
                    &t!("update-up-to-date", version = latest),
                ));
            }
            UpdatePhase::Done(Report::Available { latest, page, .. }) => {
                ui.label(icon_label(
                    h3d_ui::icons::UPDATE,
                    &t!("update-available", version = latest.clone()),
                ));
                release_page(ui, &page);
                if icon_text_button_enabled(
                    ui,
                    !busy,
                    h3d_ui::icons::EXPORT,
                    &t!("update-download"),
                )
                .clicked()
                {
                    self.update.start_fetch(latest);
                }
            }
            UpdatePhase::Done(Report::Unsupported { latest, page }) => {
                ui.label(t!("update-unsupported", version = latest));
                release_page(ui, &page);
            }
            UpdatePhase::Done(Report::Staged { latest, staged, .. }) => {
                ui.label(icon_label(
                    h3d_ui::icons::UPDATE,
                    &t!("update-ready", version = latest),
                ));
                let job_running = self.job.is_some();
                if job_running {
                    ui.small(t!("update-busy-job"));
                } else {
                    ui.small(t!("update-install-hint"));
                }
                if icon_text_button_enabled(
                    ui,
                    !job_running,
                    h3d_ui::icons::RESTART,
                    &t!("update-install"),
                )
                .clicked()
                {
                    match launch_install(&staged) {
                        Ok(()) => {
                            self.save_now();
                            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                        Err(message) => {
                            self.update.phase = UpdatePhase::Done(Report::Error { message });
                        }
                    }
                }
                return;
            }
            UpdatePhase::Done(Report::Error { message }) => {
                ui.label(icon_label(
                    h3d_ui::icons::WARNING,
                    &t!("update-failed", error = message),
                ));
            }
        }
        if icon_text_button_enabled(ui, !busy, h3d_ui::icons::UPDATE, &t!("update-check")).clicked()
        {
            self.update.start_check();
        }
        ui.small(t!("update-network-note"));
    }
}

/// Alamat catatan rilis (bisa diseleksi dan disalin; aplikasi tidak membuka
/// peramban sendiri).
fn release_page(ui: &mut Ui, page: &str) {
    ui.horizontal_wrapped(|ui| {
        ui.small(t!("update-release-notes"));
        ui.add(egui::Label::new(egui::RichText::new(page).small()).selectable(true));
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bentuk JSON sama dengan `h3d_updater::tests::report_json_shape`.
    #[test]
    fn parses_updater_reports() {
        assert_eq!(
            parse_report(
                "progres\n{\"status\":\"staged\",\"latest\":\"0.2.0\",\"tag\":\"v0.2.0\",\"staged\":\"/tmp/x\"}\n"
            ),
            Report::Staged {
                latest: "0.2.0".into(),
                tag: "v0.2.0".into(),
                staged: "/tmp/x".into(),
            }
        );
        assert_eq!(
            parse_report(r#"{"status":"up_to_date","current":"0.1.0","latest":"0.1.0"}"#),
            Report::UpToDate {
                current: "0.1.0".into(),
                latest: "0.1.0".into(),
            }
        );
        assert_eq!(parse_report(r#"{"status":"applied"}"#), Report::Applied);
        assert!(matches!(parse_report(""), Report::Error { .. }));
        assert!(matches!(parse_report("bukan json"), Report::Error { .. }));
        assert!(matches!(
            parse_report(r#"{"status":"lain"}"#),
            Report::Error { .. }
        ));
    }

    #[test]
    fn poll_without_job_is_idle() {
        let mut s = UpdateState::default();
        assert!(!s.poll());
        assert_eq!(s.phase, UpdatePhase::Idle);
    }
}
