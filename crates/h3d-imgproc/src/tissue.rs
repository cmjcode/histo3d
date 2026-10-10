//! Masker jaringan dari citra RGBA resolusi rendah (histo3d.md 8.3).

use ndarray::Array2;

use crate::color::od_total;
use crate::components::{component_areas, label_components};
use crate::error::ImgError;
use crate::morphology::{close, fill_holes};
use crate::threshold::otsu;

/// Radius penutupan morfologi bawaan (µm).
pub const CLOSE_RADIUS_UM: f64 = 50.0;
/// Fraksi luas minimum komponen relatif terhadap komponen terbesar.
pub const MIN_COMPONENT_FRACTION: f64 = 0.005;

/// Masker jaringan: `OD_total` → Otsu → penutupan radius ~50 µm → isi
/// lubang → buang komponen dengan luas < 0,5 % komponen terbesar.
/// `mpp_um` = ukuran piksel citra ini (µm/px, harus positif).
pub fn tissue_mask(rgba: &[u8], w: usize, h: usize, mpp_um: f64) -> Result<Array2<u8>, ImgError> {
    if !(mpp_um.is_finite() && mpp_um > 0.0) {
        return Err(ImgError::InvalidParameter(format!(
            "mpp tidak sah: {mpp_um}"
        )));
    }
    let od = od_total(rgba, w, h)?;
    let t = otsu(od.iter().copied(), 256);
    let raw = od.mapv(|v| u8::from(v > t));
    let radius_px = CLOSE_RADIUS_UM / mpp_um;
    let closed = close(&raw, radius_px);
    let filled = fill_holes(&closed);
    let (labels, n) = label_components(&filled);
    if n == 0 {
        return Ok(filled);
    }
    let areas = component_areas(&labels, n);
    let largest = areas[1..].iter().copied().max().unwrap_or(0) as f64;
    let min_area = (largest * MIN_COMPONENT_FRACTION).ceil() as usize;
    let mut out = Array2::<u8>::zeros((h, w));
    for (o, &l) in out.iter_mut().zip(labels.iter()) {
        if l != 0 && areas[l as usize] >= min_area {
            *o = 1;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_covers_tissue_not_background() {
        let (w, h) = (64usize, 48usize);
        let mut rgba = vec![255u8; w * h * 4];
        // Blok "jaringan" merah muda 20..44 × 10..38 dengan sebuah lubang kecil.
        for y in 10..38 {
            for x in 20..44 {
                if (x, y) == (30, 20) {
                    continue;
                }
                let i = (y * w + x) * 4;
                rgba[i] = 220;
                rgba[i + 1] = 150;
                rgba[i + 2] = 190;
            }
        }
        // Setitik debu kecil jauh dari jaringan.
        let i = (2 * w + 60) * 4;
        rgba[i] = 100;
        rgba[i + 1] = 100;
        rgba[i + 2] = 100;
        let m = tissue_mask(&rgba, w, h, 16.0).unwrap();
        assert_eq!(m[[20, 30]], 1, "lubang terisi");
        assert_eq!(m[[24, 32]], 1);
        assert_eq!(m[[2, 2]], 0);
        assert_eq!(m[[2, 60]], 0, "debu dibuang");
    }

    #[test]
    fn bad_mpp_rejected() {
        assert!(matches!(
            tissue_mask(&[255u8; 16], 2, 2, 0.0),
            Err(ImgError::InvalidParameter(_))
        ));
    }
}
