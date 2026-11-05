//! Satu modul per fitur. Tiap modul menyediakan potongan UI (kanvas, isi
//! laci, pilihan alat) yang disusun oleh [`crate::shell`] menjadi satu layar
//! ala DUCAD; tidak ada lagi "mode" terpisah.
//!
//! - [`slides`]: daftar irisan (laci Lapisan), impor, usulan urutan.
//! - [`register`]: penyejajaran — viewer pembanding, QC, landmark, geser/putar.
//! - [`segment`]: penandaan — viewer + overlay label, alat gambar, usulan.
//! - [`reconstruct`]: kanvas tumpukan 3D, picking irisan, mesh, laci Tumor 3D.
//! - [`report`]: lembar laporan dan ekspor.
//! - [`detect`]: laci Deteksi AI — model ML, cakupan, jalankan, tinjau usulan.

pub mod detect;
pub mod reconstruct;
pub mod register;
pub mod report;
pub mod segment;
pub mod slides;
