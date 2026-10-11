//! `h3d-register`: registrasi berpasangan irisan histologi dan rantai tumpukan
//! (histo3d.md 8.4). Komputasi murni: tanpa I/O, tanpa GUI.
//!
//! # Ringkasan arah transformasi (aturan keras 4; histo3d.md 5.3)
//!
//! | Simbol | Arah | Satuan | Di mana |
//! |---|---|---|---|
//! | `P_{m→f}` | U irisan bergerak → U irisan tetap | µm | [`RegResult::transform`], [`RegRecord::pairwise`](h3d_core::RegRecord), [`from_landmarks`] |
//! | `T_i` | U irisan `i` → W tumpukan | µm | keluaran [`compose_chain`] (`T_ref = I`, `T_i = T_tetangga ∘ P_{i→tetangga}`) |
//! | `orient_i` | U irisan `i` → U irisan acuan | µm | [`OrderSuggestion::orient`] (hanya tebakan awal dan thumbnail) |
//! | `init` | U irisan bergerak → U irisan tetap | µm | [`Registrar::register`], [`RegInput::init`]; dari `orient_fixed⁻¹ ∘ orient_moving` |
//! | `t_inv` | piksel tujuan → piksel sumber | px | semua resampling (mundur), di dalam [`BuiltinRegistrar`] |
//!
//! Konversi piksel ↔ µm memakai pusat piksel `+0,5` dan `origin_um` = sudut
//! piksel `(0,0)` ([`RegImage::px_to_um`], [`RegImage::um_to_px`]).
//!
//! # Alur
//! 1. [`RegImage::from_rgba`] menyiapkan kanal hematoksilin + masker jaringan.
//! 2. [`BuiltinRegistrar`] : R1 kandidat momen → R2 korelasi fase → R3 ECC
//!    berjenjang (atau Nelder–Mead informasi mutual).
//! 3. [`register_stack`] mendaftarkan tiap irisan `Ok` ke tetangga ke arah
//!    acuan (paralel, deterministik) dan membuat [`RegRecord`](h3d_core::RegRecord)
//!    lewat [`qc`].
//! 4. [`compose_chain`] menyusun `T_i`; dapat diulang setelah koreksi manual
//!    ([`from_landmarks`]) tanpa registrasi ulang.
//! 5. [`suggest_order`] mengusulkan urutan dan orientasi (8.4.7); tidak pernah
//!    menimpa data tanpa persetujuan pengguna (aturan keras 8).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod builtin;
pub mod chain;
pub mod error;
pub mod landmarks;
pub mod order;
pub mod qc;
pub mod types;

pub use builtin::{coarse_align, BuiltinRegistrar, CoarseCandidate};
pub use chain::{compose_chain, neighbor_toward_reference, register_stack, RegInput};
pub use error::RegError;
pub use landmarks::{from_landmarks, tre};
pub use order::{suggest_order, OrderSuggestion, UNCERTAIN_THRESHOLD};
pub use qc::{flag, pair_residual, record_from_result, PairResidual, QcThresholds};
pub use types::{RegImage, RegMetric, RegModel, RegOptions, RegResult, Registrar};
