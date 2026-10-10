//! Pemeriksaan mutu pasangan (histo3d.md 8.4.5).

use h3d_core::{RegRecord, SlideId};

use crate::types::RegResult;

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
    }
}
