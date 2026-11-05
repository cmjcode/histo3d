//! Evaluasi model ONNX pada irisan holdout lewat `h3d_segment::infer_slide`
//! (jalur inferensi aplikasi, bukan jalur latih): Dice jaringan dan tumor
//! terhadap masker manual dataset.

use h3d_core::{LabelId, NoProgress};
use h3d_segment::{infer_slide, Segmenter, TractSegmenter};
use ndarray::Array2;
use serde::Serialize;

use crate::data::{load_mask, load_rgb, SlideFiles};
use crate::TrainError;

/// Hasil evaluasi satu irisan.
#[derive(Debug, Clone, Serialize)]
pub struct SlideEval {
    /// Tumpukan.
    pub stack: String,
    /// Indeks irisan.
    pub slide: u32,
    /// Dice jaringan (kelas ≥ 1) vs `Masks_sub4`.
    pub dice_tissue: f32,
    /// Dice tumor (kelas 2) vs `TumorMasks_sub4`; `None` bila masker tidak ada.
    pub dice_tumor: Option<f32>,
    /// Fraksi piksel tumor menurut model.
    pub tumor_frac_pred: f32,
    /// Fraksi piksel tumor menurut anotasi.
    pub tumor_frac_true: Option<f32>,
    /// Tile yang dijalankan.
    pub tiles_run: usize,
    /// Rata-rata keyakinan softmax.
    pub mean_confidence: f32,
}

/// Dice dua masker biner setelah dipotong ke ukuran bersama.
fn dice_common(a: &Array2<u8>, b: &Array2<u8>) -> Result<f32, TrainError> {
    let h = a.dim().0.min(b.dim().0);
    let w = a.dim().1.min(b.dim().1);
    let a2 = a.slice(ndarray::s![..h, ..w]).to_owned();
    let b2 = b.slice(ndarray::s![..h, ..w]).to_owned();
    Ok(h3d_imgproc::dice(&a2, &b2)?)
}

/// Jalankan model pada satu irisan sub4. `mpp` adalah mpp gambar sub4; harus
/// sama dengan `spec.mpp` model (tidak ada resampling).
pub fn eval_slide(
    seg: &TractSegmenter,
    files: &SlideFiles,
    mpp: f64,
) -> Result<SlideEval, TrainError> {
    let spec = seg.spec();
    if (spec.mpp - mpp).abs() > 1e-9 {
        return Err(TrainError::InvalidParameter(format!(
            "mpp model {} ≠ mpp gambar {mpp}",
            spec.mpp
        )));
    }
    let Some(tissue_path) = &files.tissue_mask else {
        return Err(TrainError::Dataset(format!(
            "{}/im{}: masker jaringan tidak ada",
            files.stack, files.index
        )));
    };
    let image = load_rgb(&files.image)?;
    let tissue = load_mask(tissue_path)?;
    if tissue.width != image.width || tissue.height != image.height {
        return Err(TrainError::Dataset(
            "ukuran masker jaringan ≠ gambar".into(),
        ));
    }
    let (w, h) = (image.width, image.height);
    let read_tile =
        |x_um: f64, y_um: f64, tw: usize, th: usize| -> Result<Vec<u8>, h3d_segment::SegError> {
            let x0 = (x_um / mpp).round() as i64;
            let y0 = (y_um / mpp).round() as i64;
            let mut out = vec![255u8; tw * th * 4];
            for ty in 0..th {
                let sy = y0 + ty as i64;
                if sy < 0 || sy >= h as i64 {
                    continue;
                }
                for tx in 0..tw {
                    let sx = x0 + tx as i64;
                    if sx < 0 || sx >= w as i64 {
                        continue;
                    }
                    let si = (sy as usize * w + sx as usize) * 3;
                    let di = (ty * tw + tx) * 4;
                    out[di..di + 3].copy_from_slice(&image.rgb[si..si + 3]);
                }
            }
            Ok(out)
        };
    let tissue_arr = Array2::from_shape_vec((h, w), tissue.data.clone())
        .map_err(|e| TrainError::Dataset(e.to_string()))?;
    let label_map = |name: &str| -> Option<LabelId> {
        spec.classes
            .iter()
            .find(|c| c.label.as_deref() == Some(name))
            .map(|c| LabelId(c.index as u8))
    };
    let r = infer_slide(
        seg,
        &read_tile,
        [w as f64 * mpp, h as f64 * mpp],
        Some((&tissue_arr, mpp)),
        &label_map,
        mpp,
        &NoProgress,
    )?;
    let pred_tissue = r.label_map.mapv(|v| u8::from(v != 0));
    let pred_tumor = r.label_map.mapv(|v| u8::from(v == 2));
    let dice_tissue = dice_common(&pred_tissue, &tissue_arr)?;
    let n = pred_tumor.len().max(1) as f32;
    let tumor_frac_pred = pred_tumor.iter().filter(|&&v| v != 0).count() as f32 / n;
    let (dice_tumor, tumor_frac_true) = match &files.tumor_mask {
        Some(p) => {
            let tm = load_mask(p)?;
            let tm_arr = Array2::from_shape_vec((tm.height, tm.width), tm.data)
                .map_err(|e| TrainError::Dataset(e.to_string()))?;
            let frac =
                tm_arr.iter().filter(|&&v| v != 0).count() as f32 / tm_arr.len().max(1) as f32;
            (Some(dice_common(&pred_tumor, &tm_arr)?), Some(frac))
        }
        None => (None, None),
    };
    Ok(SlideEval {
        stack: files.stack.clone(),
        slide: files.index,
        dice_tissue,
        dice_tumor,
        tumor_frac_pred,
        tumor_frac_true,
        tiles_run: r.tiles_run,
        mean_confidence: r.mean_confidence,
    })
}
