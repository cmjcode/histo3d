//! `h3d-train`: pelatihan model segmentasi H&E untuk HISTO3D dalam Rust murni
//! (histo3d.md Fase 2 versi Rust). Kerangka: `burn`. Keluaran akhir adalah
//! `model.onnx` + `model.json` yang dibaca `h3d_segment::TractSegmenter`.
//!
//! Alur: `data` (potong tile dari dataset berlabel) → `train` (U-Net kecil,
//! loss Dice + entropi silang) → `export_onnx` (lipat BatchNorm, tulis ONNX
//! opset 13) → `eval` (Dice pada tumpukan holdout lewat jalur inferensi
//! aplikasi, bukan jalur latih).
//!
//! Semua keacakan berbenih (aturan keras 9). Tidak ada `unwrap` pada berkas.

pub mod augment;
pub mod data;
pub mod error;
pub mod eval;
pub mod export_onnx;
pub mod model;
pub mod train;

pub use error::TrainError;
