//! Kontrak registrasi (histo3d.md 8.4.1): citra registrasi, opsi, hasil,
//! dan trait `Registrar`.
//!
//! # Koordinat
//! Koordinat piksel bersifat kontinu dengan pusat piksel `(i, j)` di
//! `(i + 0,5, j + 0,5)` (aturan keras 5). Konversi ke ruang U irisan (µm):
//!
//! ```text
//! u = origin_um + p · mpp          (px → µm, [`RegImage::px_to_um`])
//! p = (u − origin_um) / mpp        (µm → px, [`RegImage::um_to_px`])
//! ```
//!
//! `origin_um` adalah posisi **sudut** piksel `(0, 0)` (bukan pusatnya), sehingga
//! penurunan resolusi 2× tidak mengubah `origin_um`.
//!
//! # Arah transformasi
//! [`RegResult::transform`] adalah `P_{moving→fixed}`: memetakan ruang U irisan
//! bergerak ke ruang U irisan tetap, µm → µm, bebas dari level piramida.

use h3d_core::{Affine2, Progress};
use h3d_imgproc::color::StainMatrix;
use h3d_imgproc::{downsample2_f32, hematoxylin_channel, tissue_mask, warp_affine_f32, Interp};
use ndarray::Array2;

use crate::error::RegError;

/// Citra untuk registrasi: kanal hematoksilin ternormalkan 0..1 dan masker
/// jaringan pada resolusi isotropik `mpp`.
#[derive(Clone, Debug, PartialEq)]
pub struct RegImage {
    /// Kanal hematoksilin 0..1, bentuk `(h, w)`.
    pub gray: Array2<f32>,
    /// Masker jaringan, 1 = jaringan, bentuk `(h, w)`.
    pub mask: Array2<u8>,
    /// Mikrometer per piksel citra ini (isotropik).
    pub mpp: f64,
    /// Posisi sudut piksel `(0, 0)` di ruang U (µm).
    pub origin_um: [f64; 2],
}

fn check_mpp(mpp: f64, name: &str) -> Result<(), RegError> {
    if !(mpp.is_finite() && mpp > 0.0) {
        return Err(RegError::InvalidInput(format!("{name} tidak sah: {mpp}")));
    }
    Ok(())
}

impl RegImage {
    /// Bangun dari komponen; memeriksa bentuk `gray` = bentuk `mask` dan `mpp`.
    pub fn new(
        gray: Array2<f32>,
        mask: Array2<u8>,
        mpp: f64,
        origin_um: [f64; 2],
    ) -> Result<Self, RegError> {
        check_mpp(mpp, "mpp")?;
        let (gh, gw) = gray.dim();
        let (mh, mw) = mask.dim();
        if (gh, gw) != (mh, mw) {
            return Err(RegError::ShapeMismatch {
                expected: [gh, gw],
                got: [mh, mw],
            });
        }
        if gh == 0 || gw == 0 {
            return Err(RegError::InvalidInput("citra registrasi kosong".into()));
        }
        if !origin_um.iter().all(|v| v.is_finite()) {
            return Err(RegError::InvalidInput(format!(
                "origin_um tidak finit: {origin_um:?}"
            )));
        }
        Ok(Self {
            gray,
            mask,
            mpp,
            origin_um,
        })
    }

    /// Dari RGBA (row-major, 4 byte/piksel) pada resolusi `mpp_src`:
    /// kanal hematoksilin (dekonvolusi H&E, regangan persentil 1–99) dan
    /// masker jaringan, lalu diturunkan ke `target_mpp` (lihat
    /// [`RegImage::downsample_to`]). `origin_um` = sudut piksel `(0,0)` sumber.
    pub fn from_rgba(
        rgba: &[u8],
        w: usize,
        h: usize,
        mpp_src: f64,
        origin_um: [f64; 2],
        target_mpp: f64,
    ) -> Result<RegImage, RegError> {
        check_mpp(mpp_src, "mpp_src")?;
        check_mpp(target_mpp, "target_mpp")?;
        if w == 0 || h == 0 {
            return Err(RegError::InvalidInput("ukuran citra nol".into()));
        }
        if rgba.len() != w * h * 4 {
            return Err(RegError::InvalidInput(format!(
                "panjang buffer RGBA {} ≠ {}×{}×4",
                rgba.len(),
                w,
                h
            )));
        }
        let gray = hematoxylin_channel(rgba, w, h, &StainMatrix::he())?;
        let mask = tissue_mask(rgba, w, h, mpp_src)?;
        let full = RegImage::new(gray, mask, mpp_src, origin_um)?;
        Ok(full.downsample_to(target_mpp))
    }

    /// Lebar dalam piksel.
    pub fn width(&self) -> usize {
        self.gray.dim().1
    }

    /// Tinggi dalam piksel.
    pub fn height(&self) -> usize {
        self.gray.dim().0
    }

    /// Ukuran `[w, h]` piksel.
    pub fn size_px(&self) -> [usize; 2] {
        [self.width(), self.height()]
    }

    /// Pusat citra dalam koordinat piksel kontinu.
    pub fn center_px(&self) -> [f64; 2] {
        [self.width() as f64 / 2.0, self.height() as f64 / 2.0]
    }

    /// Benar bila masker memuat ≥ 1 piksel jaringan.
    pub fn has_tissue(&self) -> bool {
        self.mask.iter().any(|&v| v != 0)
    }

    /// Piksel kontinu → µm (ruang U): `origin + p · mpp`.
    #[inline]
    pub fn px_to_um(&self, p: [f64; 2]) -> [f64; 2] {
        [
            self.origin_um[0] + p[0] * self.mpp,
            self.origin_um[1] + p[1] * self.mpp,
        ]
    }

    /// µm (ruang U) → piksel kontinu: `(u − origin) / mpp`.
    #[inline]
    pub fn um_to_px(&self, u: [f64; 2]) -> [f64; 2] {
        [
            (u[0] - self.origin_um[0]) / self.mpp,
            (u[1] - self.origin_um[1]) / self.mpp,
        ]
    }

    /// Affine **px → µm** (sama dengan [`RegImage::px_to_um`]).
    pub fn px_to_um_affine(&self) -> Affine2 {
        Affine2::translation(self.origin_um[0], self.origin_um[1])
            .compose(&Affine2::scale(self.mpp, self.mpp))
    }

    /// Affine **µm → px** (sama dengan [`RegImage::um_to_px`]).
    pub fn um_to_px_affine(&self) -> Affine2 {
        Affine2::scale(1.0 / self.mpp, 1.0 / self.mpp).compose(&Affine2::translation(
            -self.origin_um[0],
            -self.origin_um[1],
        ))
    }

    /// Turunkan (atau naikkan) resolusi ke `mpp` µm/px: rata-rata kotak 2×
    /// berulang selama `mpp_sekarang × 2 ≤ mpp`, lalu resampling bilinear ke
    /// ukuran `ceil(ukuran × mpp_sekarang / mpp)`. Masker diturunkan sebagai
    /// fraksi lalu diambangkan di 0,5. `origin_um` tidak berubah (sudut piksel
    /// `(0,0)` tetap). `mpp` tidak sah (≤ 0 / NaN) → salinan citra ini.
    pub fn downsample_to(&self, mpp: f64) -> RegImage {
        if !(mpp.is_finite() && mpp > 0.0) {
            log::warn!("downsample_to: mpp tidak sah ({mpp}); citra disalin apa adanya");
            return self.clone();
        }
        let mut gray = self.gray.clone();
        let mut maskf = self.mask.mapv(|v| if v != 0 { 1.0f32 } else { 0.0 });
        let mut cur = self.mpp;
        while cur * 2.0 <= mpp * (1.0 + 1e-9) && gray.dim().0 > 1 && gray.dim().1 > 1 {
            gray = downsample2_f32(&gray);
            maskf = downsample2_f32(&maskf);
            cur *= 2.0;
        }
        if (cur - mpp).abs() > 1e-9 * mpp {
            let (h, w) = gray.dim();
            let ow = ((w as f64 * cur / mpp).ceil() as usize).max(1);
            let oh = ((h as f64 * cur / mpp).ceil() as usize).max(1);
            // `t_inv`: piksel tujuan → piksel sumber (resampling mundur).
            let t_inv = Affine2::scale(mpp / cur, mpp / cur);
            match (
                warp_affine_f32(&gray, &t_inv, [ow, oh], Interp::Bilinear, 0.0),
                warp_affine_f32(&maskf, &t_inv, [ow, oh], Interp::Bilinear, 0.0),
            ) {
                (Ok(g), Ok(m)) => {
                    gray = g;
                    maskf = m;
                    cur = mpp;
                }
                _ => {
                    // Tidak terjangkau (ukuran keluaran ≥ 1), tetapi tidak panik.
                    log::error!("downsample_to: resampling gagal; level {cur} µm dipakai");
                }
            }
        }
        let mask = maskf.mapv(|v| u8::from(v >= 0.5));
        RegImage {
            gray,
            mask,
            mpp: cur,
            origin_um: self.origin_um,
        }
    }
}

/// Model transformasi yang dioptimasi.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum RegModel {
    /// Rotasi + translasi (3 parameter), cermin dari kandidat awal.
    Rigid,
    /// Rotasi + skala seragam + translasi (4 parameter).
    Similarity,
    /// Affine penuh (6 parameter).
    #[default]
    Affine,
}

/// Metrik kemiripan.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum RegMetric {
    /// Korelasi silang ternormalisasi (ECC untuk penghalusan).
    #[default]
    Ncc,
    /// Informasi mutual (Nelder–Mead untuk penghalusan; beda pewarnaan).
    MutualInformation,
}

/// Opsi registrasi berpasangan.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct RegOptions {
    /// Model transformasi. Bawaan `Affine`.
    pub model: RegModel,
    /// Metrik. Bawaan `Ncc`.
    pub metric: RegMetric,
    /// Izinkan kandidat cermin pada R1. Bawaan `true`.
    pub allow_flip: bool,
    /// Resolusi piramida (µm/px), kasar → halus. Bawaan `[64, 32, 16]`.
    pub levels_um: Vec<f64>,
    /// Iterasi maksimum per level. Bawaan 100.
    pub max_iterations: u32,
    /// Berhenti bila norma perubahan parameter (koordinat ternormalisasi)
    /// di bawah nilai ini. Bawaan `1e-5`.
    pub convergence_eps: f64,
}

impl Default for RegOptions {
    fn default() -> Self {
        Self {
            model: RegModel::Affine,
            metric: RegMetric::Ncc,
            allow_flip: true,
            levels_um: vec![64.0, 32.0, 16.0],
            max_iterations: 100,
            convergence_eps: 1e-5,
        }
    }
}

/// Hasil registrasi berpasangan.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegResult {
    /// `P_{moving→fixed}`: ruang U irisan bergerak → ruang U irisan tetap, µm → µm.
    pub transform: Affine2,
    /// NCC bermasker (masker kedua citra = 1) pada level terhalus.
    pub ncc: f32,
    /// Dice masker tetap vs masker bergerak yang di-warp.
    pub mask_dice: f32,
    /// `|det|` bagian linear `transform`.
    pub area_ratio: f32,
    /// Jumlah iterasi penghalusan (semua level).
    pub iterations: u32,
}

/// Backend registrasi berpasangan.
pub trait Registrar: Send + Sync {
    /// Nama metode untuk `RegRecord::method`.
    fn name(&self) -> &'static str;

    /// Daftarkan `moving` ke `fixed`. `init`, bila ada, adalah tebakan awal
    /// `P_{moving→fixed}` (µm → µm, mis. dari `orient_hint`). Hasilnya
    /// `P_{moving→fixed}` µm → µm.
    fn register(
        &self,
        fixed: &RegImage,
        moving: &RegImage,
        init: Option<Affine2>,
        opts: &RegOptions,
        progress: &dyn Progress,
    ) -> Result<RegResult, RegError>;
}
