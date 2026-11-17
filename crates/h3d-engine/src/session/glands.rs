//! Usulan objek 2D kelenjar/struktur per irisan (histo3d.md 8.5.3, ADR
//! 0004): slide dibaca pada skala kerja `GlandParams::mpp_um`, disegmentasi
//! berpandu lumen (`h3d_imgproc::segment_glands`), lalu menjadi [`Proposal`]
//! dengan pemisah antar objek. Tidak mengubah label (aturan keras 8); jumlah
//! dilaporkan dengan rentang (aturan keras 18).

use h3d_core::{LabelId, Progress, SlideId};
use h3d_imgproc::{
    extract_palette, segment_glands, tissue_mask, EpitheliumSource, Gland, GlandParams, Palette,
};
use h3d_segment::Proposal;
use serde::{Deserialize, Serialize};

use super::Session;
use crate::error::EngineError;

/// Ringkasan segmentasi kelenjar satu irisan (dapat diserialkan untuk CLI
/// dan laporan).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GlandSummary {
    /// Irisan.
    pub slide: SlideId,
    /// Label tujuan.
    pub label: LabelId,
    /// Nama label.
    pub label_name: String,
    /// Skala kerja, µm/px.
    pub mpp: f64,
    /// Jumlah objek.
    pub n: usize,
    /// Batas bawah jumlah.
    pub n_min: usize,
    /// Batas atas jumlah.
    pub n_max: usize,
    /// Objek berlumen.
    pub n_with_lumen: usize,
    /// Median luas objek, µm².
    pub median_area_um2: f64,
    /// Hash parameter.
    pub params_hash: u64,
    /// Sumber peta epitel: `"warna"` (palet + toleransi) atau
    /// `"hematoksilin"`.
    pub epithelium_mode: String,
    /// Warna palet terpilih (Lab) pada mode warna.
    pub color_picks: Vec<[f32; 3]>,
    /// Toleransi ΔE pada mode warna.
    pub color_tolerance: Option<f32>,
    /// Benar bila vektor pewarna hasil estimasi Macenko.
    pub stain_estimated: bool,
    /// Vektor pewarna yang dipakai (baris H, E, residu).
    pub stains: [[f32; 3]; 3],
    /// Daftar objek (koordinat µm relatif pojok kiri-atas irisan = ruang U).
    pub glands: Vec<Gland>,
}

/// Usulan kelenjar beserta ringkasannya.
#[derive(Clone, Debug)]
pub struct GlandProposal {
    /// Usulan biner dengan pemisah antar objek; belum masuk label.
    pub proposal: Proposal,
    /// Ringkasan.
    pub summary: GlandSummary,
}

impl Session {
    /// Palet `k` warna jaringan irisan `slide` pada skala `mpp_um` (ADR 0004
    /// mode warna): k-means CIELAB deterministik di dalam masker jaringan,
    /// terurut gelap → terang. Tidak mengubah apa pun.
    pub fn slide_palette(
        &self,
        slide: SlideId,
        mpp_um: f64,
        k: usize,
        p: &dyn Progress,
    ) -> Result<Palette, EngineError> {
        if !(mpp_um.is_finite() && mpp_um > 0.0) {
            return Err(EngineError::invalid(format!("mpp tidak sah: {mpp_um}")));
        }
        p.set(0.0, "membaca irisan");
        let raster = self.read_slide_at(slide, mpp_um, p)?;
        if p.is_cancelled() {
            return Err(EngineError::Cancelled);
        }
        p.set(0.5, "palet warna");
        let tissue = tissue_mask(&raster.rgba, raster.w, raster.h, raster.mpp)?;
        let pal = extract_palette(&raster.rgba, raster.w, raster.h, Some(&tissue), k)?;
        p.set(1.0, "selesai");
        Ok(pal)
    }

    /// Segmentasi kelenjar (ADR 0004) irisan `slide` untuk `label` → usulan.
    /// Tidak mengubah label; terima lewat [`Self::accept_proposal`].
    pub fn segment_glands(
        &self,
        slide: SlideId,
        label: LabelId,
        params: &GlandParams,
        p: &dyn Progress,
    ) -> Result<GlandProposal, EngineError> {
        self.check_label(label)?;
        params.validate()?;
        p.set(0.0, "membaca irisan");
        let raster = self.read_slide_at(slide, params.mpp_um, p)?;
        if p.is_cancelled() {
            return Err(EngineError::Cancelled);
        }
        p.set(0.3, "masker jaringan");
        let tissue = tissue_mask(&raster.rgba, raster.w, raster.h, raster.mpp)?;
        p.set(0.4, "segmentasi kelenjar");
        let seg = segment_glands(
            &raster.rgba,
            raster.w,
            raster.h,
            raster.mpp,
            Some(&tissue),
            params,
        )?;
        if p.is_cancelled() {
            return Err(EngineError::Cancelled);
        }
        let proposal = h3d_segment::glands::gland_proposal(
            &seg,
            raster.mpp,
            [0.0, 0.0],
            label,
            params,
            self.project.recon.mpp_label,
        );
        let label_name = self
            .project
            .label(label)
            .map(|d| d.name.clone())
            .unwrap_or_default();
        p.set(1.0, "selesai");
        Ok(GlandProposal {
            proposal,
            summary: GlandSummary {
                slide,
                label,
                label_name,
                mpp: raster.mpp,
                n: seg.n,
                n_min: seg.n_min,
                n_max: seg.n_max,
                n_with_lumen: seg.n_with_lumen(),
                median_area_um2: seg.median_area_um2(),
                params_hash: seg.params_hash,
                epithelium_mode: match &params.epithelium {
                    EpitheliumSource::Hematoxylin => "hematoksilin".to_string(),
                    EpitheliumSource::Color { .. } => "warna".to_string(),
                },
                color_picks: match &params.epithelium {
                    EpitheliumSource::Color { picks, .. } => picks.clone(),
                    EpitheliumSource::Hematoxylin => Vec::new(),
                },
                color_tolerance: match &params.epithelium {
                    EpitheliumSource::Color { tolerance, .. } => Some(*tolerance),
                    EpitheliumSource::Hematoxylin => None,
                },
                stain_estimated: seg.stain_estimated,
                stains: seg.stains,
                glands: seg.glands,
            },
        })
    }
}
