//! `h3d-app`: aplikasi GUI HISTO3D (eframe/egui + wgpu) untuk macOS dan
//! iPadOS (histo3d.md 8.11, bagian 9). Seluruh logika ada di pustaka ini
//! supaya bisa diuji tanpa jendela; `main.rs` hanya mengurai argumen.
//!
//! Tata letak modul:
//! - [`state`]: keadaan aplikasi, sesi bersama, job latar, tampilan dan laci.
//! - [`shell`]: satu layar ala DUCAD — kanvas penuh (tumpukan 3D atau irisan
//!   2D) dengan bilah atas, rel alat kiri, laci kanan, bilah konteks, kartu
//!   usulan, kartu progres, dan dialog yang mengambang di atasnya.
//! - [`app`]: integrasi eframe (renderer wgpu, simpan otomatis, tangkapan layar).
//! - [`platform`], [`apple`], `apple_ios`: lapisan platform.
//! - [`tiles`], [`viewer`], [`gpu`]: aliran tile dan viewer 2D.
//! - [`features`]: potongan UI per fitur; [`panels`]: panel dan dialog bersama.
//! - [`geom`]: logika murni (konversi koordinat, landmark, usulan urutan).
//! - [`update`]: pembaruan lewat proses terpisah `histo3d-updater`.

pub mod app;
#[cfg(target_vendor = "apple")]
pub mod apple;
#[cfg(target_os = "ios")]
pub mod apple_ios;
pub mod args;
pub mod features;
pub mod geom;
pub mod gpu;
pub mod panels;
pub mod platform;
pub mod settings;
pub mod shell;
pub mod state;
pub mod tiles;
pub mod ui_util;
pub mod update;
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
