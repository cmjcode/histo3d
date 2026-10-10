//! Model proyek (8.1) dan konfigurasi rekonstruksi/analisis.

use serde::{Deserialize, Serialize};

use crate::annotation::AnnotationRecord;
use crate::attr::AttrDef;
use crate::clinical::SampleMeta;
use crate::error::CoreError;
use crate::ids::{LabelId, SlideId};
use crate::label::LabelDef;
use crate::slide::{SlideEntry, SlideStatus};

/// Versi format `project.json` yang ditulis crate ini.
pub const FORMAT_VERSION: u32 = 1;

/// Parameter rekonstruksi (8.6).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ReconConfig {
    /// Ukuran voxel XY, µm (bawaan 16).
    pub s_xy_um: f64,
    /// `s_z ≤ k × s_xy` (bawaan k = 2).
    pub z_factor_max: f64,
    /// Lebar pita SDF dalam piksel (bawaan 8).
    pub band_px: f64,
    /// Celah maksimum sebelum objek diputus; `None` = 5 × jarak irisan median.
    pub max_gap_um: Option<f64>,
    /// Iterasi penghalusan Taubin (bawaan 10).
    pub smooth_iterations: u32,
    /// Galat decimate relatif terhadap `s_xy` (bawaan 0,5).
    pub decimate_error_factor: f64,
    /// Jarak irisan bawaan untuk `z_um` otomatis, µm.
    pub default_spacing_um: f64,
    /// Resolusi label, µm/px (bawaan 4).
    pub mpp_label: f64,
}

impl Default for ReconConfig {
    fn default() -> Self {
        Self {
            s_xy_um: 16.0,
            z_factor_max: 2.0,
            band_px: 8.0,
            max_gap_um: None,
            smooth_iterations: 10,
            decimate_error_factor: 0.5,
            default_spacing_um: 50.0,
            mpp_label: 4.0,
        }
    }
}

/// Parameter analisis spasial (8.12.6).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SpatialConfig {
    /// Ambang margin "dekat", µm (bawaan 1000).
    pub margin_close_um: f64,
    /// Pita peritumoral untuk TIL, µm (bawaan 500).
    pub peri_band_um: f64,
    /// Jarak difusi oksigen untuk proksi hipoksia, µm (bawaan 150).
    pub hypoxia_um: f64,
    /// Volume minimum fokus, µm³; `None` = (3 × s_xy)³.
    pub min_focus_um3: Option<f64>,
    /// Ambang peringatan heterogenitas IHC, poin (bawaan 20).
    pub ihc_hetero_warn: f32,
    /// Lebar bin histogram jarak pembuluh, µm (bawaan 25).
    pub vessel_hist_bin_um: f64,
}

impl Default for SpatialConfig {
    fn default() -> Self {
        Self {
            margin_close_um: 1000.0,
            peri_band_um: 500.0,
            hypoxia_um: 150.0,
            min_focus_um3: None,
            ihc_hetero_warn: 20.0,
            vessel_hist_bin_um: 25.0,
        }
    }
}

/// Manifes proyek (`project.json`). Tidak memuat `ClinicalMeta`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Project {
    /// Versi format.
    pub format_version: u32,
    /// Metadata sampel.
    pub sample: SampleMeta,
    /// Irisan.
    pub slides: Vec<SlideEntry>,
    /// Irisan acuan (`T_r` = identitas).
    pub reference: Option<SlideId>,
    /// Tabel label.
    pub labels: Vec<LabelDef>,
    /// Atribut skalar.
    pub attrs: Vec<AttrDef>,
    /// Provenance per (irisan, label).
    pub annotations: Vec<AnnotationRecord>,
    /// Konfigurasi rekonstruksi.
    pub recon: ReconConfig,
    /// Konfigurasi analisis spasial.
    pub spatial: SpatialConfig,
    /// Nama/inisial pengguna untuk provenance `Manual`.
    pub user_initials: String,
}

impl Default for Project {
    fn default() -> Self {
        Self {
            format_version: FORMAT_VERSION,
            sample: SampleMeta::default(),
            slides: Vec::new(),
            reference: None,
            labels: LabelDef::default_table(),
            attrs: Vec::new(),
            annotations: Vec::new(),
            recon: ReconConfig::default(),
            spatial: SpatialConfig::default(),
            user_initials: String::new(),
        }
    }
}

impl Project {
    /// Proyek baru dengan nama sampel.
    pub fn new(name: &str) -> Self {
        Self {
            sample: SampleMeta {
                name: name.to_owned(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Tolak versi lebih baru (aturan 8.7.1).
    pub fn check_version(&self) -> Result<(), CoreError> {
        if self.format_version > FORMAT_VERSION {
            Err(CoreError::FormatTooNew {
                found: self.format_version,
                supported: FORMAT_VERSION,
            })
        } else {
            Ok(())
        }
    }

    /// Irisan berdasarkan id.
    pub fn slide(&self, id: SlideId) -> Option<&SlideEntry> {
        self.slides.iter().find(|s| s.id == id)
    }

    /// Irisan berdasarkan id (mutable).
    pub fn slide_mut(&mut self, id: SlideId) -> Option<&mut SlideEntry> {
        self.slides.iter_mut().find(|s| s.id == id)
    }

    /// Label berdasarkan id.
    pub fn label(&self, id: LabelId) -> Option<&LabelDef> {
        self.labels.iter().find(|l| l.id == id)
    }

    /// Label berdasarkan nama (tidak peka huruf besar/kecil).
    pub fn label_by_name(&self, name: &str) -> Option<&LabelDef> {
        self.labels
            .iter()
            .find(|l| l.name.eq_ignore_ascii_case(name.trim()))
    }

    /// Id irisan baru yang belum pernah dipakai.
    pub fn next_slide_id(&self) -> SlideId {
        SlideId(self.slides.iter().map(|s| s.id.0 + 1).max().unwrap_or(0))
    }

    /// Id label baru (1..=255) yang belum dipakai; `None` bila penuh.
    pub fn next_label_id(&self) -> Option<LabelId> {
        (1u8..=255)
            .find(|v| self.labels.iter().all(|l| l.id.0 != *v))
            .map(LabelId)
    }

    /// Irisan terurut menurut `order` (seri dipecah dengan id, deterministik).
    pub fn slides_ordered(&self) -> Vec<&SlideEntry> {
        let mut v: Vec<&SlideEntry> = self.slides.iter().collect();
        v.sort_by_key(|s| (s.order, s.id));
        v
    }

    /// Irisan `Ok` terurut.
    pub fn usable_slides_ordered(&self) -> Vec<&SlideEntry> {
        self.slides_ordered()
            .into_iter()
            .filter(|s| s.status == SlideStatus::Ok)
            .collect()
    }

    /// Irisan acuan efektif: `reference` bila sah, selain itu irisan `Ok` di
    /// tengah tumpukan (8.4.4).
    pub fn effective_reference(&self) -> Option<SlideId> {
        if let Some(r) = self.reference {
            if self.slide(r).is_some_and(|s| s.status == SlideStatus::Ok) {
                return Some(r);
            }
        }
        let ok = self.usable_slides_ordered();
        ok.get(ok.len() / 2).map(|s| s.id)
    }

    /// Jarak irisan median antar irisan `Ok` berurutan, µm; `None` bila < 2 irisan.
    pub fn median_spacing_um(&self) -> Option<f64> {
        let ok = self.usable_slides_ordered();
        if ok.len() < 2 {
            return None;
        }
        let mut gaps: Vec<f64> = ok
            .windows(2)
            .map(|w| (w[1].z_um - w[0].z_um).abs())
            .collect();
        gaps.sort_by(|a, b| a.total_cmp(b));
        Some(gaps[gaps.len() / 2])
    }

    /// Hitung ulang `z_um` bawaan (`order × spacing`) hanya untuk irisan yang
    /// `z_is_default`; `z_um` isian tangan tidak disentuh (8.4.7).
    pub fn recompute_default_z(&mut self, spacing_um: f64) {
        let order_of: Vec<(SlideId, u32)> = self
            .slides_ordered()
            .iter()
            .enumerate()
            .map(|(i, s)| (s.id, i as u32))
            .collect();
        for (id, rank) in order_of {
            if let Some(s) = self.slide_mut(id) {
                if s.z_is_default {
                    s.z_um = rank as f64 * spacing_um;
                }
            }
        }
    }

    /// Benar bila `z_um` monoton naik sepanjang urutan.
    pub fn z_is_monotonic(&self) -> bool {
        self.slides_ordered()
            .windows(2)
            .all(|w| w[1].z_um > w[0].z_um)
    }

    /// Catatan provenance untuk (irisan, label).
    pub fn annotation(&self, slide: SlideId, label: LabelId) -> Option<&AnnotationRecord> {
        self.annotations
            .iter()
            .find(|a| a.slide == slide && a.label == label)
    }

    /// Sisipkan atau perbarui catatan provenance.
    pub fn upsert_annotation(&mut self, rec: AnnotationRecord) {
        if let Some(a) = self
            .annotations
            .iter_mut()
            .find(|a| a.slide == rec.slide && a.label == rec.label)
        {
            *a = rec;
        } else {
            self.annotations.push(rec);
            self.annotations.sort_by_key(|a| (a.slide, a.label));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_roundtrip_is_exact() {
        let mut p = Project::new("sampel");
        p.slides.push(SlideEntry {
            id: SlideId(3),
            mpp: [0.25, 0.25],
            size_l0: [1000, 800],
            ..Default::default()
        });
        let json = serde_json::to_string_pretty(&p).unwrap();
        let back: Project = serde_json::from_str(&json).unwrap();
        assert_eq!(p, back);
        assert!(!json.contains("clinical"));
    }

    #[test]
    fn old_project_without_new_fields_opens_with_defaults() {
        let json = r#"{"format_version":1,"slides":[],"labels":[]}"#;
        let p: Project = serde_json::from_str(json).unwrap();
        assert!(p.labels.is_empty());
        assert_eq!(p.recon, ReconConfig::default());
        assert!(p.check_version().is_ok());
    }

    #[test]
    fn newer_version_rejected() {
        let p = Project {
            format_version: FORMAT_VERSION + 1,
            ..Default::default()
        };
        assert!(matches!(
            p.check_version(),
            Err(CoreError::FormatTooNew { .. })
        ));
    }

    #[test]
    fn reference_defaults_to_middle_ok_slide() {
        let mut p = Project::default();
        for i in 0..5 {
            p.slides.push(SlideEntry {
                id: SlideId(i),
                order: i,
                status: if i == 2 {
                    SlideStatus::Damaged
                } else {
                    SlideStatus::Ok
                },
                ..Default::default()
            });
        }
        // Ok: 0,1,3,4 → tengah indeks 2 → id 3.
        assert_eq!(p.effective_reference(), Some(SlideId(3)));
    }

    #[test]
    fn recompute_z_keeps_manual_values() {
        let mut p = Project::default();
        for i in 0..3 {
            p.slides.push(SlideEntry {
                id: SlideId(i),
                order: i,
                ..Default::default()
            });
        }
        p.slides[1].z_um = 123.0;
        p.slides[1].z_is_default = false;
        p.recompute_default_z(50.0);
        assert_eq!(p.slides[0].z_um, 0.0);
        assert_eq!(p.slides[1].z_um, 123.0);
        assert_eq!(p.slides[2].z_um, 100.0);
        assert!(!p.z_is_monotonic());
    }
}
