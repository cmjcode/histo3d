//! Pemeriksaan mutu pasangan (histo3d.md 8.4.5).

use h3d_core::{Affine2, RegRecord, SlideId};
use h3d_imgproc::{gaussian_blur_f32, warp_affine_f32, warp_affine_u8, Interp};
use ndarray::Array2;

use crate::types::{RegImage, RegResult};

/// Sisi blok lokal untuk [`block_residual_um`], piksel registrasi.
pub const RESIDUAL_BLOCK_PX: usize = 32;
/// Langkah antar blok (tumpang tindih setengah blok), piksel registrasi.
pub const RESIDUAL_STEP_PX: usize = 16;
/// Jangkauan pencarian geseran sisa per blok, ± piksel registrasi. Geseran
/// yang puncaknya di tepi jangkauan dilaporkan sebesar jangkauan ini.
pub const RESIDUAL_SEARCH_PX: usize = 4;
/// Fraksi piksel blok yang harus berupa jaringan di **kedua** citra.
pub const RESIDUAL_MIN_COVERAGE: f64 = 0.6;
/// Jumlah blok sah minimum agar median dianggap bermakna.
pub const RESIDUAL_MIN_BLOCKS: usize = 4;
/// Sigma pemulusan Gauss (piksel) sebelum pencocokan blok; meredam derau
/// per piksel yang tidak berkorelasi antar irisan.
const RESIDUAL_SMOOTH_SIGMA_PX: f64 = 1.0;

/// Sisa galat registrasi satu pasangan, diukur **setelah** penjajaran
/// ([`pair_residual`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairResidual {
    /// Median besar geseran sisa per blok, µm: galat lokal pasangan ini,
    /// termasuk perubahan tekstur antar irisan dan deformasi lokal yang tidak
    /// bisa ditangkap transformasi global. Tidak menumpuk di sepanjang rantai.
    pub local_um: f64,
    /// RMS medan gerak rigid (translasi + rotasi kecil) kuadrat terkecil yang
    /// dicocokkan ke geseran sisa semua blok, µm: bagian galat yang masih
    /// bisa "dijelaskan" sebuah koreksi posisi global, yaitu galat
    /// transformasi. Bagian inilah yang menumpuk saat `T_i` dikomposisi di
    /// sepanjang rantai.
    pub fit_um: f64,
    /// Jumlah blok yang ikut dihitung.
    pub n_blocks: usize,
}

/// Sisa galat registrasi satu pasangan, diukur **setelah** penjajaran:
/// citra bergerak di-warp ke kerangka citra tetap dengan `p`
/// (`P_{moving→fixed}`, U bergerak → U tetap, µm → µm; resampling mundur
/// dengan `P⁻¹`), kerangka dibagi blok [`RESIDUAL_BLOCK_PX`] berlangkah
/// [`RESIDUAL_STEP_PX`], dan pada tiap blok yang cukup berisi jaringan di
/// kedua citra dicari geseran sisa yang memaksimalkan korelasi silang
/// ternormalisasi (rata-rata nol) dalam ± [`RESIDUAL_SEARCH_PX`], diperhalus
/// sub-piksel dengan pencocokan parabola. Lihat [`PairResidual`] untuk dua
/// ringkasannya.
///
/// Korelasi ternormalisasi dipakai (bukan korelasi fase) karena kontras
/// tekstur berubah antar irisan dan derau per piksel tidak berkorelasi;
/// korelasi fase memutihkan spektrum sehingga pada blok kecil didominasi
/// derau itu (terukur 1–4 piksel pada fantom yang galat sebenarnya 0,2
/// piksel).
///
/// Kedua citra harus beresolusi sama (dijamin gambar registrasi mesin).
/// `None` bila resolusi berbeda, `p` singular, atau blok sah kurang dari
/// [`RESIDUAL_MIN_BLOCKS`]. Deterministik (blok dipindai baris demi baris).
pub fn pair_residual(fixed: &RegImage, moving: &RegImage, p: &Affine2) -> Option<PairResidual> {
    if (fixed.mpp - moving.mpp).abs() > 1e-9 * fixed.mpp {
        return None;
    }
    let inv = p.inverse()?;
    // t_inv: piksel tetap → µm tetap → µm bergerak → piksel bergerak.
    let t_inv = moving
        .um_to_px_affine()
        .compose(&inv)
        .compose(&fixed.px_to_um_affine());
    let size = fixed.size_px();
    let mg = gaussian_blur_f32(&moving.gray, RESIDUAL_SMOOTH_SIGMA_PX);
    let wg = warp_affine_f32(&mg, &t_inv, size, Interp::Bilinear, 0.0).ok()?;
    let wm = warp_affine_u8(&moving.mask, &t_inv, size, Interp::Nearest, 0).ok()?;
    let fg = gaussian_blur_f32(&fixed.gray, RESIDUAL_SMOOTH_SIGMA_PX);
    let [w, h] = size;
    let b = RESIDUAL_BLOCK_PX;
    let r = RESIDUAL_SEARCH_PX;
    if w < b + 2 * r || h < b + 2 * r {
        return None;
    }
    let need = (RESIDUAL_MIN_COVERAGE * (b * b) as f64).ceil() as usize;
    let side = 2 * r + 1;
    // (pusat blok px, geseran px).
    let mut blocks: Vec<([f64; 2], [f64; 2])> = Vec::new();
    let mut y0 = r;
    while y0 + b + r <= h {
        let mut x0 = r;
        while x0 + b + r <= w {
            let mut both = 0usize;
            for y in y0..y0 + b {
                for x in x0..x0 + b {
                    both += usize::from(fixed.mask[[y, x]] != 0 && wm[[y, x]] != 0);
                }
            }
            if both >= need {
                // Permukaan korelasi atas geseran (dx, dy) ∈ [−r, r]².
                let mut surf = vec![f64::NEG_INFINITY; side * side];
                let mut best = (f64::NEG_INFINITY, 0usize);
                for sy in 0..side {
                    for sx in 0..side {
                        let c = block_ncc(&fg, &wg, x0, y0, x0 + sx - r, y0 + sy - r, b);
                        surf[sy * side + sx] = c;
                        if c > best.0 {
                            best = (c, sy * side + sx);
                        }
                    }
                }
                if best.0.is_finite() {
                    let (sx, sy) = (best.1 % side, best.1 / side);
                    let at = |x: usize, y: usize| surf[y * side + x];
                    // Puncak di tepi jangkauan: geseran sisa ≥ jangkauan
                    // (tanpa penghalusan sub-piksel pada sumbu itu).
                    let ox = if sx == 0 || sx == side - 1 {
                        0.0
                    } else {
                        parabola_offset(at(sx - 1, sy), best.0, at(sx + 1, sy))
                    };
                    let oy = if sy == 0 || sy == side - 1 {
                        0.0
                    } else {
                        parabola_offset(at(sx, sy - 1), best.0, at(sx, sy + 1))
                    };
                    blocks.push((
                        [(x0 + b / 2) as f64, (y0 + b / 2) as f64],
                        [sx as f64 - r as f64 + ox, sy as f64 - r as f64 + oy],
                    ));
                }
            }
            x0 += RESIDUAL_STEP_PX;
        }
        y0 += RESIDUAL_STEP_PX;
    }
    let n = blocks.len();
    if n < RESIDUAL_MIN_BLOCKS {
        return None;
    }
    let mut mags: Vec<f64> = blocks
        .iter()
        .map(|(_, d)| (d[0] * d[0] + d[1] * d[1]).sqrt())
        .collect();
    mags.sort_by(|a, c| a.total_cmp(c));
    let local_px = mags[n / 2];
    let fit_px = fitted_field_rms(&blocks, local_px);
    Some(PairResidual {
        local_um: local_px * fixed.mpp,
        fit_um: fit_px * fixed.mpp,
        n_blocks: n,
    })
}

/// RMS (piksel) medan gerak rigid kuadrat terkecil
/// `d(c) = t + ω·J·(c − c̄)` (translasi + rotasi kecil) yang dicocokkan ke
/// geseran blok `(pusat, geseran)`, dievaluasi di pusat blok. Blok pencilan
/// (`|d| > max(1 px, 3 × median)`, biasanya salah cocok) dibuang dulu.
///
/// Medan rigid — bukan affine — dipilih dengan sengaja: perubahan bentuk
/// antar irisan tampak sebagai geseran **radial** (skala semu), dan komponen
/// itu bukan galat registrasi (lihat pengaman model bebas di `builtin`).
fn fitted_field_rms(blocks: &[([f64; 2], [f64; 2])], median_mag: f64) -> f64 {
    let limit = (3.0 * median_mag).max(1.0);
    let kept: Vec<&([f64; 2], [f64; 2])> = blocks
        .iter()
        .filter(|(_, d)| (d[0] * d[0] + d[1] * d[1]).sqrt() <= limit)
        .collect();
    if kept.is_empty() {
        return 0.0;
    }
    let n = kept.len() as f64;
    let cx = kept.iter().map(|(c, _)| c[0]).sum::<f64>() / n;
    let cy = kept.iter().map(|(c, _)| c[1]).sum::<f64>() / n;
    let tx = kept.iter().map(|(_, d)| d[0]).sum::<f64>() / n;
    let ty = kept.iter().map(|(_, d)| d[1]).sum::<f64>() / n;
    let (mut num, mut r2) = (0.0f64, 0.0f64);
    for (c, d) in &kept {
        let (x, y) = (c[0] - cx, c[1] - cy);
        num += x * d[1] - y * d[0];
        r2 += x * x + y * y;
    }
    let omega = if r2 > 1e-9 { num / r2 } else { 0.0 };
    // Rata-rata |d(c)|² = |t|² + ω²·rata-rata|c − c̄|² (suku silang nol).
    (tx * tx + ty * ty + omega * omega * r2 / n).sqrt()
}

/// Korelasi silang ternormalisasi (rata-rata nol) blok `b × b` citra `f` di
/// `(fx, fy)` terhadap blok citra `m` di `(mx, my)`. Tak hingga negatif bila
/// salah satu blok tanpa variasi.
fn block_ncc(
    f: &Array2<f32>,
    m: &Array2<f32>,
    fx: usize,
    fy: usize,
    mx: usize,
    my: usize,
    b: usize,
) -> f64 {
    let n = (b * b) as f64;
    let (mut sf, mut sm, mut sff, mut smm, mut sfm) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for y in 0..b {
        for x in 0..b {
            let a = f64::from(f[[fy + y, fx + x]]);
            let c = f64::from(m[[my + y, mx + x]]);
            sf += a;
            sm += c;
            sff += a * a;
            smm += c * c;
            sfm += a * c;
        }
    }
    let vf = sff - sf * sf / n;
    let vm = smm - sm * sm / n;
    if vf <= 1e-12 || vm <= 1e-12 {
        return f64::NEG_INFINITY;
    }
    (sfm - sf * sm / n) / (vf * vm).sqrt()
}

/// Offset sub-piksel puncak parabola melalui `(l, c, r)`, dijepit ±0,5.
fn parabola_offset(l: f64, c: f64, r: f64) -> f64 {
    if !(l.is_finite() && r.is_finite()) {
        return 0.0;
    }
    let denom = l - 2.0 * c + r;
    if denom.abs() < 1e-12 {
        0.0
    } else {
        (0.5 * (l - r) / denom).clamp(-0.5, 0.5)
    }
}

/// Ambang QC. Pasangan ditandai bila `ncc < ncc_min`, `mask_dice < dice_min`,
/// atau `area_ratio` di luar `[area_ratio.0, area_ratio.1]`.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct QcThresholds {
    /// NCC minimum. Bawaan 0,5.
    pub ncc_min: f32,
    /// Dice masker minimum. Bawaan 0,8.
    pub dice_min: f32,
    /// Rentang `|det|` yang diterima. Bawaan (0,85, 1,15).
    pub area_ratio: (f32, f32),
}

impl Default for QcThresholds {
    fn default() -> Self {
        Self {
            ncc_min: 0.5,
            dice_min: 0.8,
            area_ratio: (0.85, 1.15),
        }
    }
}

/// Benar bila hasil gagal salah satu ambang (atau memuat nilai tak-finit).
pub fn flag(result: &RegResult, thr: &QcThresholds) -> bool {
    let bad_num = !result.ncc.is_finite()
        || !result.mask_dice.is_finite()
        || !result.area_ratio.is_finite()
        || !result.transform.is_finite();
    bad_num
        || result.ncc < thr.ncc_min
        || result.mask_dice < thr.dice_min
        || result.area_ratio < thr.area_ratio.0
        || result.area_ratio > thr.area_ratio.1
}

/// Bangun `RegRecord` dari hasil registrasi terhadap irisan `fixed`.
/// `result.transform` = `P_{i→fixed}` (µm → µm) disalin ke `pairwise`.
pub fn record_from_result(
    fixed: SlideId,
    method: &str,
    result: &RegResult,
    thr: &QcThresholds,
    manual: bool,
) -> RegRecord {
    RegRecord {
        fixed,
        method: method.to_string(),
        pairwise: result.transform,
        ncc: result.ncc,
        mask_dice: result.mask_dice,
        area_ratio: result.area_ratio,
        flagged: flag(result, thr),
        manual,
        residual_um: None,
        residual_fit_um: None,
    }
}
