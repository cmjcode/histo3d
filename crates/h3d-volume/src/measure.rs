//! Pengukuran per label (histo3d.md 8.6.4) dan peringatan otomatis (8.12.8).
//!
//! Volume Cavalieri adalah angka utama laporan dan hanya memakai irisan yang
//! benar-benar dianotasi (bukan interpolasi). Volume mesh hanya pembanding.

use h3d_core::LabelId;
use serde::{Deserialize, Serialize};

use crate::mesh::LabelMesh;
use crate::recon::{LabelSampler, SliceInput};

/// Peringatan otomatis dengan kunci terjemahan Fluent dan argumennya.
///
/// Kunci yang sudah ada di `h3d-i18n`: `warn-margin-close($dir)`,
/// `warn-margin-z-unassessable`, `warn-ihc-heterogeneity`,
/// `warn-mostly-interpolated`, `warn-has-draft`, `warn-z-resolution-coarse`,
/// `warn-membrane-area-score($name)`.
///
/// Kunci **baru** yang diperkenalkan crate ini (perlu ditambahkan penulis
/// i18n):
/// - `warn-volume-mismatch($percent)` — selisih volume mesh vs Cavalieri > 5 %.
/// - `warn-sxy-raised($from, $to)` — `s_xy` dinaikkan otomatis karena anggaran.
/// - `warn-gap-split($from_z, $to_z, $gap)` — celah antar irisan > `max_gap_um`,
///   objek diputus.
/// - `warn-nonuniform-z-resampled($dz)` — jarak irisan tak rata; kisi
///   diresampel ke `s_z` median untuk analisis spasial.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Warning {
    /// Tumor dekat tepi reseksi pada arah `dir`.
    MarginClose {
        /// Nama arah (anatomis bila orientasi diketahui, selain itu "+X" dsb.).
        dir: String,
    },
    /// Label `Margin` tidak ada di irisan berlabel pertama/terakhir.
    MarginZUnassessable,
    /// Rentang skor IHC antar irisan melampaui ambang.
    IhcHeterogeneity,
    /// Fraksi irisan interpolasi > 50 %.
    MostlyInterpolated,
    /// Ada anotasi `Draft`.
    HasDraft,
    /// `s_z` > 2 × `margin_close_um`.
    ZResolutionCoarse,
    /// Skor membran dihitung per luas.
    MembraneAreaScore {
        /// Nama atribut.
        name: String,
    },
    /// Selisih volume mesh vs Cavalieri > 5 %.
    VolumeMismatch {
        /// Selisih relatif dalam persen.
        percent: f64,
    },
    /// `s_xy` dinaikkan otomatis.
    SxyRaised {
        /// Nilai awal, µm.
        from: f64,
        /// Nilai akhir, µm.
        to: f64,
    },
    /// Celah antar irisan berlabel melampaui `max_gap_um`.
    GapSplit {
        /// z irisan sebelum celah.
        from_z: f64,
        /// z irisan sesudah celah.
        to_z: f64,
        /// Lebar celah, µm.
        gap: f64,
    },
    /// Kisi diresampel ke jarak irisan seragam.
    NonUniformZResampled {
        /// `s_z` median yang dipakai, µm.
        dz: f64,
    },
}

impl Warning {
    /// Kunci Fluent.
    pub fn i18n_key(&self) -> &'static str {
        match self {
            Warning::MarginClose { .. } => "warn-margin-close",
            Warning::MarginZUnassessable => "warn-margin-z-unassessable",
            Warning::IhcHeterogeneity => "warn-ihc-heterogeneity",
            Warning::MostlyInterpolated => "warn-mostly-interpolated",
            Warning::HasDraft => "warn-has-draft",
            Warning::ZResolutionCoarse => "warn-z-resolution-coarse",
            Warning::MembraneAreaScore { .. } => "warn-membrane-area-score",
            Warning::VolumeMismatch { .. } => "warn-volume-mismatch",
            Warning::SxyRaised { .. } => "warn-sxy-raised",
            Warning::GapSplit { .. } => "warn-gap-split",
            Warning::NonUniformZResampled { .. } => "warn-nonuniform-z-resampled",
        }
    }

    /// Argumen Fluent `(nama, nilai)` dalam urutan tetap.
    pub fn args(&self) -> Vec<(&'static str, String)> {
        match self {
            Warning::MarginClose { dir } => vec![("dir", dir.clone())],
            Warning::MembraneAreaScore { name } => vec![("name", name.clone())],
            Warning::VolumeMismatch { percent } => vec![("percent", format!("{percent:.1}"))],
            Warning::SxyRaised { from, to } => {
                vec![("from", format!("{from}")), ("to", format!("{to}"))]
            }
            Warning::GapSplit { from_z, to_z, gap } => vec![
                ("from_z", format!("{from_z}")),
                ("to_z", format!("{to_z}")),
                ("gap", format!("{gap}")),
            ],
            Warning::NonUniformZResampled { dz } => vec![("dz", format!("{dz}"))],
            _ => Vec::new(),
        }
    }
}

/// Hasil pengukuran satu label (8.6.4).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LabelMeasurement {
    /// Label.
    pub label: LabelId,
    /// Volume Cavalieri, µm³ (angka utama).
    pub volume_cavalieri_um3: f64,
    /// Volume mesh bertanda, µm³ (pembanding).
    pub volume_mesh_um3: f64,
    /// Volume terkoreksi susut `V × f³` bila `shrinkage_linear` diberikan.
    pub volume_corrected_um3: Option<f64>,
    /// Luas permukaan mesh, µm².
    pub surface_area_um2: f64,
    /// Kotak pembatas `(min, max)` mesh, µm.
    pub bbox_um: Option<([f64; 3], [f64; 3])>,
    /// Diameter maksimum mesh, µm.
    pub max_diameter_um: f64,
    /// Jumlah komponen terhubung mesh.
    pub n_components: usize,
    /// Jumlah irisan yang benar-benar dianotasi (bukan interpolasi).
    pub n_slices_annotated: usize,
    /// Jumlah irisan berstatus `Reviewed`.
    pub n_slices_reviewed: usize,
    /// Peringatan.
    pub warnings: Vec<Warning>,
}

/// Volume Cavalieri dari pasangan `(z, luas)` (µm, µm²): `V = Σ A_i × h_i`
/// dengan `h_i = (z_{i+1} − z_{i−1}) / 2`; di ujung `h` = jarak ke
/// satu-satunya tetangga. Satu irisan → 0 (tidak ada tebal yang diketahui).
/// Masukan diurutkan menurut `z` terlebih dahulu.
pub fn cavalieri(areas_um2: &[(f64, f64)]) -> f64 {
    let mut a: Vec<(f64, f64)> = areas_um2
        .iter()
        .copied()
        .filter(|(z, s)| z.is_finite() && s.is_finite())
        .collect();
    a.sort_by(|p, q| p.0.total_cmp(&q.0));
    let n = a.len();
    if n < 2 {
        return 0.0;
    }
    let mut v = 0.0;
    for i in 0..n {
        let h = if i == 0 {
            a[1].0 - a[0].0
        } else if i == n - 1 {
            a[n - 1].0 - a[n - 2].0
        } else {
            (a[i + 1].0 - a[i - 1].0) / 2.0
        };
        v += a[i].1 * h;
    }
    v
}

/// Luas label (µm²) dengan mencacah piksel label pada resolusi `mpp_label`
/// di dalam kotak pembatasnya: `jumlah piksel × mpp_label²`.
pub fn label_area_um2(sampler: &dyn LabelSampler, label: u8) -> f64 {
    use rayon::prelude::*;
    let mpp = sampler.mpp_label();
    let Some(bb) = sampler.bbox_um(label) else {
        return 0.0;
    };
    if mpp.is_nan() || mpp <= 0.0 {
        return 0.0;
    }
    let x0 = (bb[0] / mpp).floor() as i64;
    let y0 = (bb[1] / mpp).floor() as i64;
    let x1 = (bb[2] / mpp).ceil() as i64;
    let y1 = (bb[3] / mpp).ceil() as i64;
    if x1 <= x0 || y1 <= y0 {
        return 0.0;
    }
    let count: u64 = (y0..y1)
        .into_par_iter()
        .map(|py| {
            let uy = (py as f64 + 0.5) * mpp;
            let mut c = 0u64;
            for px in x0..x1 {
                let ux = (px as f64 + 0.5) * mpp;
                if sampler.label_at_um([ux, uy]) == label {
                    c += 1;
                }
            }
            c
        })
        .sum();
    count as f64 * mpp * mpp
}

/// Pasangan `(z, luas)` untuk Cavalieri: hanya irisan yang memuat `label`
/// dengan anotasi **bukan** interpolasi.
pub fn cavalieri_areas(inputs: &[SliceInput<'_>], label: u8) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    for s in inputs {
        let annotated = s
            .annotated_labels
            .iter()
            .any(|&(l, interp)| l == label && !interp);
        if annotated {
            out.push((s.z_um, label_area_um2(s.labels, label)));
        }
    }
    out
}

/// Pengukuran lengkap satu label dari mesh dan masukan irisan.
/// `shrinkage_linear` = faktor koreksi susut linear `f` (opsional).
pub fn measure_label(
    mesh: &LabelMesh,
    inputs: &[SliceInput<'_>],
    label: u8,
    shrinkage_linear: Option<f64>,
) -> LabelMeasurement {
    let areas = cavalieri_areas(inputs, label);
    let v_cav = cavalieri(&areas);
    let v_mesh = mesh.signed_volume_um3();
    let mut warnings = Vec::new();
    if v_cav > 0.0 {
        let pct = ((v_mesh - v_cav) / v_cav).abs() * 100.0;
        if pct > 5.0 {
            warnings.push(Warning::VolumeMismatch { percent: pct });
        }
    }
    let n_total = inputs
        .iter()
        .filter(|s| s.annotated_labels.iter().any(|&(l, _)| l == label))
        .count();
    let n_annotated = areas.len();
    let n_interp = n_total.saturating_sub(n_annotated);
    if n_total > 0 && n_interp * 2 > n_total {
        warnings.push(Warning::MostlyInterpolated);
    }
    let n_reviewed = inputs
        .iter()
        .filter(|s| {
            s.annotated_labels
                .iter()
                .any(|&(l, interp)| l == label && !interp)
                && s.reviewed_labels.contains(&label)
        })
        .count();
    if n_reviewed < n_annotated {
        warnings.push(Warning::HasDraft);
    }
    LabelMeasurement {
        label: LabelId(label),
        volume_cavalieri_um3: v_cav,
        volume_mesh_um3: v_mesh,
        volume_corrected_um3: shrinkage_linear
            .filter(|f| f.is_finite() && *f > 0.0)
            .map(|f| v_cav * f * f * f),
        surface_area_um2: mesh.surface_area_um2(),
        bbox_um: mesh.bbox(),
        max_diameter_um: mesh.max_diameter_um(),
        n_components: mesh.connected_components(),
        n_slices_annotated: n_annotated,
        n_slices_reviewed: n_reviewed,
        warnings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cavalieri_ellipsoid_analytic_areas() {
        // Ellipsoid rx=ry=1000, rz=1000; A(z) = π·rx·ry·(1 − z²/rz²).
        let r = 1000.0f64;
        let n = 41;
        let areas: Vec<(f64, f64)> = (0..n)
            .map(|i| {
                let z = -r + 2.0 * r * i as f64 / (n - 1) as f64;
                (
                    z,
                    std::f64::consts::PI * r * r * (1.0 - (z / r).powi(2)).max(0.0),
                )
            })
            .collect();
        let v = cavalieri(&areas);
        let truth = 4.0 / 3.0 * std::f64::consts::PI * r * r * r;
        assert!(((v - truth) / truth).abs() < 0.01, "{v} vs {truth}");
        // Urutan acak tidak mengubah hasil.
        let mut shuffled = areas.clone();
        shuffled.reverse();
        assert_eq!(cavalieri(&shuffled), v);
        // Dua irisan: h = jarak tetangga di kedua ujung.
        assert_eq!(cavalieri(&[(0.0, 2.0), (10.0, 4.0)]), 60.0);
        assert_eq!(cavalieri(&[(0.0, 2.0)]), 0.0);
    }

    #[test]
    fn warning_keys_and_args() {
        let w = Warning::SxyRaised {
            from: 16.0,
            to: 24.0,
        };
        assert_eq!(w.i18n_key(), "warn-sxy-raised");
        assert_eq!(w.args()[0], ("from", "16".to_owned()));
        assert!(Warning::HasDraft.args().is_empty());
    }
}
