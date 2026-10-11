//! `h3d-app`: aplikasi GUI HISTO3D (eframe/egui + wgpu) untuk macOS dan
//! iPadOS (histo3d.md 8.11, bagian 9). Seluruh logika ada di pustaka ini
//! supaya bisa diuji tanpa jendela; `main.rs` hanya mengurai argumen.
//!
//! Tata letak modul:
//! - [`state`]: keadaan aplikasi, sesi bersama, job latar.
//! - [`shell`]: satu frame UI (bilah atas, lapisan job, dialog).
//! - [`app`]: integrasi eframe (renderer wgpu, simpan otomatis, tangkapan layar).
//! - [`platform`], [`apple`], `apple_ios`: lapisan platform.
//! - [`tiles`], [`viewer`], [`gpu`]: aliran tile dan viewer 2D.
//! - [`modes`]: satu modul per mode; [`panels`]: panel dan dialog bersama.
//! - [`geom`]: logika murni (konversi koordinat, landmark, usulan urutan).

pub mod app;
#[cfg(target_vendor = "apple")]
pub mod apple;
#[cfg(target_os = "ios")]
pub mod apple_ios;
pub mod args;
pub mod geom;
pub mod gpu;
pub mod modes;
pub mod panels;
pub mod platform;
pub mod settings;
pub mod shell;
pub mod state;
pub mod tiles;
pub mod ui_util;
pub mod viewer;

pub use app::{native_options, run, H3dApp};
pub use state::AppState;

/// Pembantu tes: bahasa aktif adalah keadaan global, jadi tes yang
/// menggantinya harus bergiliran.
#[cfg(test)]
pub mod test_util {
    use std::sync::{Mutex, MutexGuard};

    static LANG: Mutex<()> = Mutex::new(());

    /// Kunci giliran untuk tes yang menyetel bahasa.
    pub fn lang_lock() -> MutexGuard<'static, ()> {
        LANG.lock().unwrap_or_else(|p| p.into_inner())
    }
}
