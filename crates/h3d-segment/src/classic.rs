//! Segmentasi klasik (histo3d.md 8.5.3): ambang pada kanal dekonvolusi
//! pewarna, lalu morfologi dan pembuangan komponen kecil. Hasil berupa
//! masker **usulan** pada resolusi masukan.

use h3d_core::{AnnotationSource, LabelId};
use h3d_imgproc::color::{deconvolve, percentile_stretch, StainMatrix};
use h3d_imgproc::components::remove_small_components;
use h3d_imgproc::morphology::{close, fill_holes, open};
use h3d_imgproc::threshold::{otsu, otsu_masked};
use ndarray::Array2;

use crate::error::SegError;
use crate::proposal::Proposal;

/// Kanal pewarna yang diambangkan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StainChannel {
    /// Hematoksilin (inti).
    #[default]
    Hematoxylin,
    /// Eosin (sitoplasma/stroma).
    Eosin,
    /// DAB (kromogen IHC).
    Dab,
}

/// Parameter segmentasi klasik.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassicParams {
    /// Kanal pewarna.
    pub channel: StainChannel,
    /// Ambang pada kanal yang diregangkan 0..1; `None` = Otsu.
    pub threshold: Option<f32>,
    /// Bila benar, ambil piksel di **bawah** ambang.
    pub invert: bool,
    /// Radius pembukaan morfologi (µm); 0 = lewati.
    pub open_radius_um: f64,
    /// Radius penutupan morfologi (µm); 0 = lewati.
    pub close_radius_um: f64,
    /// Luas komponen minimum (µm²); komponen lebih kecil dibuang.
    pub min_area_um2: f64,
    /// Isi lubang setelah morfologi.
    pub fill_holes: bool,
}

impl Default for ClassicParams {
    fn default() -> Self {
        Self {
            channel: StainChannel::Hematoxylin,
            threshold: None,
            invert: false,
            open_radius_um: 4.0,
            close_radius_um: 8.0,
            min_area_um2: 500.0,
            fill_holes: false,
        }
    }
}

impl ClassicParams {
    /// Hash FNV-1a 64-bit yang stabil atas semua parameter (untuk
    /// `AnnotationSource::Classic { params_hash }`).
    pub fn params_hash(&self) -> u64 {
        const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const PRIME: u64 = 0x0000_0100_0000_01b3;
        let mut h = OFFSET;
        let mut feed = |bytes: &[u8]| {
            for &b in bytes {
                h ^= b as u64;
                h = h.wrapping_mul(PRIME);
            }
        };
        feed(&[match self.channel {
            StainChannel::Hematoxylin => 0u8,
            StainChannel::Eosin => 1,
            StainChannel::Dab => 2,
        }]);
        match self.threshold {
            None => feed(&[0u8]),
            Some(t) => {
                feed(&[1u8]);
                feed(&t.to_bits().to_le_bytes());
            }
        }
        feed(&[self.invert as u8]);
        feed(&self.open_radius_um.to_bits().to_le_bytes());
        feed(&self.close_radius_um.to_bits().to_le_bytes());
        feed(&self.min_area_um2.to_bits().to_le_bytes());
        feed(&[self.fill_holes as u8]);
        h
    }

    /// Sumber anotasi untuk parameter ini.
    pub fn source(&self) -> AnnotationSource {
        AnnotationSource::Classic {
            params_hash: self.params_hash(),
        }
    }

    fn validate(&self) -> Result<(), SegError> {
        let ok = |v: f64| v.is_finite() && v >= 0.0;
        if !ok(self.open_radius_um) || !ok(self.close_radius_um) || !ok(self.min_area_um2) {
            return Err(SegError::InvalidParameter(
                "radius/luas harus ≥ 0 dan hingga".to_string(),
            ));
        }
        if let Some(t) = self.threshold {
            if !t.is_finite() {
                return Err(SegError::InvalidParameter("ambang NaN".to_string()));
            }
        }
        Ok(())
    }
}

/// Segmentasi klasik pada citra RGBA `w × h` berukuran piksel `mpp` (µm).
/// `tissue_mask` (bentuk `(h, w)`) membatasi Otsu dan hasil ke jaringan.
/// Mengembalikan masker usulan biner `(h, w)` pada resolusi masukan.
pub fn segment_classic(
    rgba: &[u8],
    w: usize,
    h: usize,
    mpp: f64,
    tissue_mask: Option<&Array2<u8>>,
    params: &ClassicParams,
) -> Result<Array2<u8>, SegError> {
    params.validate()?;
    if !(mpp.is_finite() && mpp > 0.0) {
        return Err(SegError::InvalidParameter(format!("mpp tidak sah: {mpp}")));
    }
    if let Some(tm) = tissue_mask {
        if tm.dim() != (h, w) {
            return Err(SegError::InvalidParameter(format!(
                "bentuk masker jaringan {:?} ≠ ({h}, {w})",
                tm.dim()
            )));
        }
    }
    let (matrix, row) = match params.channel {
        StainChannel::Hematoxylin => (StainMatrix::he(), 0usize),
        StainChannel::Eosin => (StainMatrix::he(), 1),
        StainChannel::Dab => (StainMatrix::h_dab(), 1),
    };
    let channels = deconvolve(rgba, w, h, &matrix)?;
    let ch = percentile_stretch(&channels[row], 0.01, 0.99);
    let thr = match params.threshold {
        Some(t) => t,
        None => match tissue_mask {
            Some(tm) => otsu_masked(&ch, tm),
            None => otsu(ch.iter().copied(), 256),
        },
    };
    let mut mask = Array2::<u8>::zeros((h, w));
    for ((y, x), &v) in ch.indexed_iter() {
        let hit = if params.invert { v < thr } else { v > thr };
        let in_tissue = tissue_mask.is_none_or(|tm| tm[[y, x]] != 0);
        if hit && in_tissue {
            mask[[y, x]] = 1;
        }
    }
    if params.open_radius_um > 0.0 {
        mask = open(&mask, params.open_radius_um / mpp);
    }
    if params.close_radius_um > 0.0 {
        mask = close(&mask, params.close_radius_um / mpp);
    }
    if params.fill_holes {
        mask = fill_holes(&mask);
    }
    if params.min_area_um2 > 0.0 {
        let min_px = (params.min_area_um2 / (mpp * mpp)).ceil() as usize;
        if min_px > 1 {
            mask = remove_small_components(&mask, min_px);
        }
    }
    if let Some(tm) = tissue_mask {
        for (m, &t) in mask.iter_mut().zip(tm.iter()) {
            if t == 0 {
                *m = 0;
            }
        }
    }
    Ok(mask)
}

/// Bungkus masker klasik menjadi [`Proposal`] untuk `label` dengan pojok
/// kiri-atas `origin_um`.
pub fn classic_proposal(
    mask: Array2<u8>,
    mpp: f64,
    origin_um: [f64; 2],
    label: LabelId,
    params: &ClassicParams,
) -> Proposal {
    Proposal {
        mask,
        mpp,
        origin_um,
        label,
        source: params.source(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use h3d_imgproc::metrics::dice;
    use h3d_imgproc::phantom::{ellipsoid_mask, render_histology};
    use h3d_imgproc::tissue::tissue_mask;

    #[test]
    fn params_hash_stable_and_sensitive() {
        let a = ClassicParams::default();
        let b = ClassicParams::default();
        assert_eq!(a.params_hash(), b.params_hash());
        let c = ClassicParams {
            invert: true,
            ..ClassicParams::default()
        };
        assert_ne!(a.params_hash(), c.params_hash());
        let d = ClassicParams {
            threshold: Some(0.5),
            ..ClassicParams::default()
        };
        assert_ne!(a.params_hash(), d.params_hash());
    }

    #[test]
    fn invalid_inputs_are_errors() {
        let p = ClassicParams::default();
        assert!(segment_classic(&[0u8; 16], 2, 2, 0.0, None, &p).is_err());
        assert!(segment_classic(&[0u8; 15], 2, 2, 1.0, None, &p).is_err());
        let bad = ClassicParams {
            open_radius_um: -1.0,
            ..ClassicParams::default()
        };
        assert!(segment_classic(&[0u8; 16], 2, 2, 1.0, None, &bad).is_err());
        let tm = Array2::<u8>::zeros((3, 3));
        assert!(segment_classic(&[0u8; 16], 2, 2, 1.0, Some(&tm), &p).is_err());
    }

    #[test]
    fn he_phantom_eosin_dice() {
        let mpp = 1.0;
        let (w, h) = (400usize, 300usize);
        let truth = ellipsoid_mask([w, h], [200.0, 150.0], [150.0, 100.0]);
        let rgba = render_histology(&truth, 7, mpp);
        let tm = tissue_mask(&rgba, w, h, mpp).unwrap();
        let params = ClassicParams {
            channel: StainChannel::Eosin,
            threshold: None,
            invert: false,
            open_radius_um: 2.0,
            close_radius_um: 6.0,
            min_area_um2: 200.0,
            fill_holes: true,
        };
        let mask = segment_classic(&rgba, w, h, mpp, Some(&tm), &params).unwrap();
        let d = dice(&mask, &truth).unwrap();
        assert!(d >= 0.9, "Dice {d}");
        // Tanpa masker jaringan juga harus baik.
        let mask2 = segment_classic(&rgba, w, h, mpp, None, &params).unwrap();
        let d2 = dice(&mask2, &truth).unwrap();
        assert!(d2 >= 0.9, "Dice tanpa masker {d2}");
        // Usulan tidak menyentuh LabelStore sampai diterima.
        let p = classic_proposal(mask, mpp, [0.0, 0.0], LabelId(1), &params);
        let mut store = crate::LabelStore::new(4.0, [400.0, 300.0]);
        assert!(store.is_empty());
        crate::accept_proposal(&mut store, &p, crate::PaintMode::Replace);
        assert!(store.area_px(LabelId(1)) > 0);
    }

    #[test]
    fn he_phantom_hematoxylin_selects_nuclei_only() {
        let mpp = 1.0;
        let (w, h) = (200usize, 200usize);
        let truth = ellipsoid_mask([w, h], [100.0, 100.0], [80.0, 80.0]);
        let rgba = render_histology(&truth, 3, mpp);
        let params = ClassicParams {
            channel: StainChannel::Hematoxylin,
            open_radius_um: 0.0,
            close_radius_um: 0.0,
            min_area_um2: 0.0,
            ..ClassicParams::default()
        };
        let mask = segment_classic(&rgba, w, h, mpp, None, &params).unwrap();
        let n = mask.iter().filter(|&&v| v != 0).count();
        let t = truth.iter().filter(|&&v| v != 0).count();
        assert!(n > 0 && n < t / 2, "inti {n} vs jaringan {t}");
        // Hampir semua piksel terpilih berada di dalam jaringan (derau latar
        // boleh menyumbang sedikit).
        let inside = mask
            .iter()
            .zip(truth.iter())
            .filter(|(m, tr)| **m != 0 && **tr != 0)
            .count();
        assert!(inside * 100 >= n * 95, "di dalam {inside} dari {n}");
    }
}
