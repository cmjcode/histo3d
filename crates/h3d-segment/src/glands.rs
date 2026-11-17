//! Usulan objek 2D kelenjar/struktur (histo3d.md 8.5.3, ADR 0004): hasil
//! `h3d_imgproc::segment_glands` dibungkus menjadi [`Proposal`] biner dengan
//! **pemisah** antar objek, sehingga setelah diterima ke `LabelStore` tiap
//! kelenjar tetap satu komponen terhubung tersendiri dan penautan 3D
//! (`h3d-volume::track`) dapat mengikutinya antar irisan.
//!
//! Lebar pemisah dipilih dari rasio `mpp_label / mpp` (lihat
//! [`separator_gap_px`]) agar modus pada resampling `accept_proposal` tidak
//! menyambungkan kembali dua tetangga.

use h3d_core::{AnnotationSource, LabelId};
use h3d_imgproc::{instances_to_mask, GlandParams, GlandSegmentation};

use crate::proposal::Proposal;

/// Lebar pemisah (piksel usulan) agar dua objek bertetangga tetap terpisah
/// setelah diresampling dari `mpp` ke `mpp_label` dengan aturan modus
/// "≥ setengah": jejak piksel label selebar `k = ceil(mpp_label / mpp)`
/// piksel usulan yang separuhnya objek masih menjadi 1, sehingga dibutuhkan
/// celah `k + 1`. Bila resolusi label lebih halus, cukup 1 piksel.
pub fn separator_gap_px(mpp: f64, mpp_label: f64) -> usize {
    if !(mpp.is_finite() && mpp > 0.0 && mpp_label.is_finite() && mpp_label > 0.0) {
        return 1;
    }
    let k = (mpp_label / mpp).ceil().max(1.0) as usize;
    if mpp_label > mpp {
        k + 1
    } else {
        1
    }
}

/// Bungkus segmentasi kelenjar menjadi usulan `label` dengan pojok kiri-atas
/// `origin_um` (ruang U) dan pemisah menurut `mpp_label`. Asal usulan =
/// `Classic { params_hash }` dengan hash parameter kelenjar.
pub fn gland_proposal(
    seg: &GlandSegmentation,
    mpp: f64,
    origin_um: [f64; 2],
    label: LabelId,
    params: &GlandParams,
    mpp_label: f64,
) -> Proposal {
    let gap = separator_gap_px(mpp, mpp_label);
    Proposal {
        mask: instances_to_mask(&seg.labels, gap),
        mpp,
        origin_um,
        label,
        source: AnnotationSource::Classic {
            params_hash: params.params_hash(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{accept_proposal, LabelStore, PaintMode};
    use h3d_imgproc::phantom::render_glands;
    use h3d_imgproc::{label_components, segment_glands};

    #[test]
    fn separator_width_follows_resolution_ratio() {
        assert_eq!(separator_gap_px(2.0, 4.0), 3);
        assert_eq!(separator_gap_px(2.0, 2.0), 1);
        assert_eq!(separator_gap_px(4.0, 2.0), 1);
        assert_eq!(separator_gap_px(1.0, 4.0), 5);
        assert_eq!(separator_gap_px(0.0, 4.0), 1);
    }

    #[test]
    fn accepted_proposal_keeps_every_gland_as_its_own_component() {
        let mpp = 2.0;
        let (w, h) = (400usize, 300usize);
        let (rgba, truth) = render_glands([w, h], mpp, 5);
        let params = GlandParams::default();
        let seg = segment_glands(&rgba, w, h, mpp, None, &params).unwrap();
        assert_eq!(seg.n, truth.n);
        let mpp_label = 4.0;
        let p = gland_proposal(&seg, mpp, [0.0, 0.0], LabelId(10), &params, mpp_label);
        assert!(matches!(p.source, AnnotationSource::Classic { .. }));
        let mut store = LabelStore::new(mpp_label, [w as f64 * mpp, h as f64 * mpp]);
        accept_proposal(&mut store, &p, PaintMode::Replace);
        // Baca kembali seluruh label sebagai masker biner pada mpp_label.
        let (lw, lh) = (store.size_px[0] as usize, store.size_px[1] as usize);
        let mask = ndarray::Array2::from_shape_fn((lh, lw), |(y, x)| {
            u8::from(
                store.get_um([(x as f64 + 0.5) * mpp_label, (y as f64 + 0.5) * mpp_label]) == 10,
            )
        });
        let (_, n) = label_components(&mask);
        assert_eq!(n, seg.n, "pasangan bersentuhan harus tetap dua komponen");
    }
}
