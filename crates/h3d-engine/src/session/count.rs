//! Penghitungan objek 2D per irisan (histo3d.md 8.12.5): jumlah objek
//! tertutup satu label pada satu penampang, dari `LabelStore` atau dari
//! usulan yang belum diterima. Tidak mengubah label (aturan keras 8);
//! jumlah dilaporkan dengan rentang ketidakpastian (aturan keras 18).

use h3d_core::{LabelId, SlideId};
use h3d_imgproc::{
    count_objects_with_uncertainty, nuclei_density_per_mm2, CountEstimate, CountParams, Object2D,
};
use h3d_segment::Proposal;
use serde::{Deserialize, Serialize};

use super::Session;
use crate::error::EngineError;

/// Asal masker yang dihitung.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CountSource {
    /// Label tersimpan di `LabelStore`.
    Labels,
    /// Usulan (klasik/AI/interpolasi) yang belum diterima.
    Proposal,
}

/// Hasil penghitungan objek satu label pada satu irisan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObjectCount2D {
    /// Irisan.
    pub slide: SlideId,
    /// Label.
    pub label: LabelId,
    /// Nama label.
    pub label_name: String,
    /// Asal masker.
    pub source: CountSource,
    /// Jumlah objek beserta rentang ketidakpastian dan daftar objek.
    pub estimate: CountEstimate,
    /// Pojok kiri-atas masker yang dihitung di ruang U (µm); koordinat
    /// objek relatif terhadap titik ini.
    pub origin_um: [f64; 2],
    /// Ukuran masker yang dihitung, µm.
    pub size_um: [f64; 2],
    /// Luas jaringan di dalam wilayah hitung, µm² (dari masker jaringan
    /// registrasi); `None` bila tidak tersedia.
    pub tissue_area_um2: Option<f64>,
}

impl ObjectCount2D {
    /// Jumlah objek yang dihitung.
    pub fn n(&self) -> usize {
        self.estimate.count.n
    }

    /// Sentroid objek di ruang U (µm).
    pub fn centroid_u(&self, o: &Object2D) -> [f64; 2] {
        [
            self.origin_um[0] + o.centroid_um[0],
            self.origin_um[1] + o.centroid_um[1],
        ]
    }

    /// Densitas objek per mm² jaringan; `None` bila luas jaringan tidak
    /// diketahui atau nol.
    pub fn density_per_mm2(&self) -> Option<f64> {
        let a = self.tissue_area_um2.filter(|a| *a > 0.0)?;
        Some(nuclei_density_per_mm2(self.estimate.count.n, a))
    }
}

impl Session {
    /// Hitung objek label `label` pada irisan `slide`. `roi_um` membatasi
    /// ke kotak `[lo, hi]` di ruang U (µm); `None` = seluruh slide.
    /// Membaca `LabelStore` pada `mpp_label`; tidak mengubah apa pun.
    pub fn count_objects_2d(
        &mut self,
        slide: SlideId,
        label: LabelId,
        roi_um: Option<([f64; 2], [f64; 2])>,
        params: &CountParams,
    ) -> Result<ObjectCount2D, EngineError> {
        self.check_label(label)?;
        params.validate()?;
        let label_name = self
            .project
            .label(label)
            .map(|l| l.name.clone())
            .unwrap_or_default();
        let store = self.labels(slide)?;
        let mpp = store.mpp_label;
        let size_px = store.size_px;
        let (x0, y0, w, h) = match roi_um {
            None => (0u32, 0u32, size_px[0], size_px[1]),
            Some((lo, hi)) => {
                if !(lo[0].is_finite()
                    && lo[1].is_finite()
                    && hi[0].is_finite()
                    && hi[1].is_finite())
                {
                    return Err(EngineError::invalid("wilayah hitung tidak sah"));
                }
                let x0 = (lo[0].max(0.0) / mpp).floor() as u32;
                let y0 = (lo[1].max(0.0) / mpp).floor() as u32;
                let x1 = ((hi[0] / mpp).ceil().max(0.0) as u32).min(size_px[0]);
                let y1 = ((hi[1] / mpp).ceil().max(0.0) as u32).min(size_px[1]);
                if x1 <= x0 || y1 <= y0 {
                    return Err(EngineError::invalid("wilayah hitung di luar irisan"));
                }
                (x0, y0, x1 - x0, y1 - y0)
            }
        };
        let mask = store.mask_region(label, x0, y0, w, h);
        let estimate = count_objects_with_uncertainty(&mask, mpp, params)?;
        let origin_um = [f64::from(x0) * mpp, f64::from(y0) * mpp];
        let size_um = [f64::from(w) * mpp, f64::from(h) * mpp];
        let tissue_area_um2 = self.tissue_area_in(slide, origin_um, size_um);
        Ok(ObjectCount2D {
            slide,
            label,
            label_name,
            source: CountSource::Labels,
            estimate,
            origin_um,
            size_um,
            tissue_area_um2,
        })
    }

    /// Hitung objek pada usulan `proposal` (belum diterima) untuk irisan
    /// `slide`; label tidak disentuh.
    pub fn count_proposal_objects(
        &self,
        slide: SlideId,
        proposal: &Proposal,
        params: &CountParams,
    ) -> Result<ObjectCount2D, EngineError> {
        self.check_label(proposal.label)?;
        params.validate()?;
        let label_name = self
            .project
            .label(proposal.label)
            .map(|l| l.name.clone())
            .unwrap_or_default();
        let estimate = count_objects_with_uncertainty(&proposal.mask, proposal.mpp, params)?;
        let (h, w) = proposal.mask.dim();
        let size_um = [w as f64 * proposal.mpp, h as f64 * proposal.mpp];
        let tissue_area_um2 = self.tissue_area_in(slide, proposal.origin_um, size_um);
        Ok(ObjectCount2D {
            slide,
            label: proposal.label,
            label_name,
            source: CountSource::Proposal,
            estimate,
            origin_um: proposal.origin_um,
            size_um,
            tissue_area_um2,
        })
    }

    /// Luas jaringan (µm²) di dalam kotak `[origin, origin + size]` ruang U
    /// menurut masker jaringan citra registrasi; `None` bila citra
    /// registrasi tidak bisa dibuat.
    fn tissue_area_in(
        &self,
        slide: SlideId,
        origin_um: [f64; 2],
        size_um: [f64; 2],
    ) -> Option<f64> {
        let reg = self.registration_image(slide).ok()?;
        let (h, w) = reg.mask.dim();
        let to_px = |u: f64, o: f64| ((u - o) / reg.mpp).max(0.0);
        let x0 = to_px(origin_um[0], reg.origin_um[0]).floor() as usize;
        let y0 = to_px(origin_um[1], reg.origin_um[1]).floor() as usize;
        let x1 = (to_px(origin_um[0] + size_um[0], reg.origin_um[0]).ceil() as usize).min(w);
        let y1 = (to_px(origin_um[1] + size_um[1], reg.origin_um[1]).ceil() as usize).min(h);
        if x1 <= x0 || y1 <= y0 {
            return Some(0.0);
        }
        let n = reg
            .mask
            .slice(ndarray::s![y0..y1, x0..x1])
            .iter()
            .filter(|&&v| v != 0)
            .count();
        Some(n as f64 * reg.mpp * reg.mpp)
    }
}
