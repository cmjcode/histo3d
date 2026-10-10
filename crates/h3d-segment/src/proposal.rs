//! Usulan segmentasi (histo3d.md 8.5.3–8.5.5, aturan keras 8).
//!
//! Semua hasil otomatis (klasik, AI, interpolasi, impor) berupa [`Proposal`]
//! yang hanya masuk ke [`LabelStore`] lewat [`accept_proposal`] setelah
//! pengguna menekan "Terima".

use h3d_core::{AnnotationSource, LabelId};
use ndarray::Array2;

use crate::label_store::{LabelStore, PaintMode, TileKey};

/// Masker usulan satu label pada resolusi sendiri.
#[derive(Debug, Clone)]
pub struct Proposal {
    /// Masker biner (`!= 0` = label), bentuk `(h, w)`.
    pub mask: Array2<u8>,
    /// Ukuran piksel masker (µm).
    pub mpp: f64,
    /// Pojok kiri-atas masker di ruang U (µm).
    pub origin_um: [f64; 2],
    /// Label tujuan.
    pub label: LabelId,
    /// Asal usulan untuk `AnnotationRecord`.
    pub source: AnnotationSource,
}

impl Proposal {
    /// Luas usulan dalam µm².
    pub fn area_um2(&self) -> f64 {
        self.mask.iter().filter(|&&v| v != 0).count() as f64 * self.mpp * self.mpp
    }

    /// Benar bila masker kosong.
    pub fn is_empty(&self) -> bool {
        self.mask.iter().all(|&v| v == 0)
    }
}

/// Terima usulan: resampel masker ke `mpp_label` lalu cat ke `store` dengan
/// `mode`. Satu-satunya jalur usulan → `LabelStore`.
///
/// Resampling mundur per piksel tujuan: jejak piksel label
/// `[u0, u1)` dipetakan ke rentang piksel usulan; bila jejak meliputi
/// beberapa piksel usulan, dipakai **modus** (≥ setengah bernilai 1);
/// bila lebih kecil dari satu piksel usulan, dipakai tetangga terdekat
/// (`floor`). Mengembalikan tile yang berubah.
pub fn accept_proposal(store: &mut LabelStore, p: &Proposal, mode: PaintMode) -> Vec<TileKey> {
    let (ph, pw) = p.mask.dim();
    if ph == 0 || pw == 0 || !(p.mpp.is_finite() && p.mpp > 0.0) {
        return Vec::new();
    }
    let mpp = store.mpp_label;
    // Kotak usulan dalam piksel label.
    let ux0 = p.origin_um[0];
    let uy0 = p.origin_um[1];
    let ux1 = ux0 + pw as f64 * p.mpp;
    let uy1 = uy0 + ph as f64 * p.mpp;
    let x0 = (ux0 / mpp).floor().max(0.0);
    let y0 = (uy0 / mpp).floor().max(0.0);
    let x1 = (ux1 / mpp).ceil().min(store.size_px[0] as f64);
    let y1 = (uy1 / mpp).ceil().min(store.size_px[1] as f64);
    if !(x0 < x1 && y0 < y1) {
        return Vec::new();
    }
    let (x0, y0, x1, y1) = (x0 as u32, y0 as u32, x1 as u32, y1 as u32);
    let w = (x1 - x0) as usize;
    let h = (y1 - y0) as usize;
    let mut region = Array2::<u8>::zeros((h, w));
    for j in 0..h {
        let y = y0 + j as u32;
        let (py0, py1) = footprint(y as f64 * mpp, (y + 1) as f64 * mpp, uy0, p.mpp, ph);
        if py0 >= py1 {
            continue;
        }
        for i in 0..w {
            let x = x0 + i as u32;
            let (px0, px1) = footprint(x as f64 * mpp, (x + 1) as f64 * mpp, ux0, p.mpp, pw);
            if px0 >= px1 {
                continue;
            }
            let mut ones = 0usize;
            let mut total = 0usize;
            for py in py0..py1 {
                for px in px0..px1 {
                    total += 1;
                    if p.mask[[py, px]] != 0 {
                        ones += 1;
                    }
                }
            }
            if ones > 0 && ones * 2 >= total {
                region[[j, i]] = 1;
            }
        }
    }
    store.rasterize_mask(&region, [x0 as i64, y0 as i64], p.label, mode)
}

/// Rentang indeks piksel usulan `[a, b)` yang jejaknya beririsan dengan
/// `[lo_um, hi_um)`; bila kosong (jejak lebih kecil dari satu piksel usulan),
/// dipakai piksel yang memuat pusat jejak.
fn footprint(lo_um: f64, hi_um: f64, origin_um: f64, mpp: f64, n: usize) -> (usize, usize) {
    let a = ((lo_um - origin_um) / mpp).ceil();
    let b = ((hi_um - origin_um) / mpp).floor();
    if b > a {
        let a = a.max(0.0) as usize;
        let b = (b.max(0.0) as usize).min(n);
        return (a, b);
    }
    let c = (((lo_um + hi_um) * 0.5 - origin_um) / mpp).floor();
    if c < 0.0 || c >= n as f64 {
        return (0, 0);
    }
    let c = c as usize;
    (c, c + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn src() -> AnnotationSource {
        AnnotationSource::Classic { params_hash: 1 }
    }

    #[test]
    fn accept_finer_proposal_uses_mode() {
        // Usulan 0,5 µm/px, 400×400 px = 200×200 µm, lingkaran di tengah.
        let n = 400usize;
        let mask = h3d_imgproc::phantom::ellipsoid_mask([n, n], [200.0, 200.0], [150.0, 150.0]);
        let p = Proposal {
            mask,
            mpp: 0.5,
            origin_um: [100.0, 60.0],
            label: LabelId(4),
            source: src(),
        };
        let mut store = LabelStore::new(4.0, [1000.0, 1000.0]);
        let keys = accept_proposal(&mut store, &p, PaintMode::Replace);
        assert!(!keys.is_empty());
        let expect = p.area_um2();
        let got = store.area_um2(LabelId(4));
        assert!((got - expect).abs() / expect < 0.03, "{got} vs {expect}");
        // Pusat lingkaran: (100+100, 60+100) µm.
        assert_eq!(store.get_um([200.0, 160.0]), 4);
        assert_eq!(store.get_um([110.0, 70.0]), 0);
    }

    #[test]
    fn accept_coarser_proposal_nearest() {
        let mut mask = Array2::<u8>::zeros((4, 4));
        mask[[1, 1]] = 1;
        let p = Proposal {
            mask,
            mpp: 16.0,
            origin_um: [0.0, 0.0],
            label: LabelId(1),
            source: src(),
        };
        let mut store = LabelStore::new(4.0, [64.0, 64.0]);
        accept_proposal(&mut store, &p, PaintMode::Replace);
        assert_eq!(store.area_px(LabelId(1)), 16);
        assert_eq!(store.bbox_px(LabelId(1)), Some([4, 4, 8, 8]));
    }

    #[test]
    fn only_background_mode_respected() {
        let mut store = LabelStore::new(4.0, [100.0, 100.0]);
        store.paint_disc([50.0, 50.0], 10.0, LabelId(9), PaintMode::Replace);
        let p = Proposal {
            mask: Array2::ones((25, 25)),
            mpp: 4.0,
            origin_um: [0.0, 0.0],
            label: LabelId(1),
            source: src(),
        };
        accept_proposal(&mut store, &p, PaintMode::OnlyBackground);
        assert_eq!(store.get_um([50.0, 50.0]), 9);
        assert_eq!(store.get_um([2.0, 2.0]), 1);
    }
}
