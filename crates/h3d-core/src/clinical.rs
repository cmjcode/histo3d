//! Metadata sampel dan data klinis pseudonim (8.6.6, 8.12.2, 8.12.9).
//!
//! `ClinicalMeta` **tidak pernah** diserialisasi ke `project.json`:
//! `SampleMeta::clinical` ditandai `#[serde(skip)]` sehingga hanya bisa
//! ditulis ke `clinical.json` secara eksplisit oleh `h3d-io`.

use serde::{Deserialize, Serialize};

use crate::error::CoreError;

/// Arah anatomis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
#[allow(missing_docs)]
pub enum AnatDir {
    Anterior,
    Posterior,
    Superior,
    Inferior,
    Medial,
    Lateral,
    Proximal,
    Distal,
    Left,
    Right,
    #[default]
    Unknown,
}

impl AnatDir {
    /// Arah lawan.
    pub fn opposite(self) -> AnatDir {
        match self {
            AnatDir::Anterior => AnatDir::Posterior,
            AnatDir::Posterior => AnatDir::Anterior,
            AnatDir::Superior => AnatDir::Inferior,
            AnatDir::Inferior => AnatDir::Superior,
            AnatDir::Medial => AnatDir::Lateral,
            AnatDir::Lateral => AnatDir::Medial,
            AnatDir::Proximal => AnatDir::Distal,
            AnatDir::Distal => AnatDir::Proximal,
            AnatDir::Left => AnatDir::Right,
            AnatDir::Right => AnatDir::Left,
            AnatDir::Unknown => AnatDir::Unknown,
        }
    }

    /// Indeks pasangan (0..5) atau `None` untuk `Unknown`.
    fn pair_index(self) -> Option<u8> {
        Some(match self {
            AnatDir::Anterior | AnatDir::Posterior => 0,
            AnatDir::Superior | AnatDir::Inferior => 1,
            AnatDir::Medial | AnatDir::Lateral => 2,
            AnatDir::Proximal | AnatDir::Distal => 3,
            AnatDir::Left | AnatDir::Right => 4,
            AnatDir::Unknown => return None,
        })
    }

    /// Nama singkat Inggris untuk laporan (mis. "anterior").
    pub fn as_str(self) -> &'static str {
        match self {
            AnatDir::Anterior => "anterior",
            AnatDir::Posterior => "posterior",
            AnatDir::Superior => "superior",
            AnatDir::Inferior => "inferior",
            AnatDir::Medial => "medial",
            AnatDir::Lateral => "lateral",
            AnatDir::Proximal => "proximal",
            AnatDir::Distal => "distal",
            AnatDir::Left => "left",
            AnatDir::Right => "right",
            AnatDir::Unknown => "unknown",
        }
    }

    /// Semua nilai untuk UI.
    pub const ALL: [AnatDir; 11] = [
        AnatDir::Anterior,
        AnatDir::Posterior,
        AnatDir::Superior,
        AnatDir::Inferior,
        AnatDir::Medial,
        AnatDir::Lateral,
        AnatDir::Proximal,
        AnatDir::Distal,
        AnatDir::Left,
        AnatDir::Right,
        AnatDir::Unknown,
    ];
}

/// Nama anatomis untuk arah +X, +Y, +Z ruang W. Arah negatif = lawannya.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct SpecimenOrientation {
    /// Arah +X.
    pub pos_x: AnatDir,
    /// Arah +Y.
    pub pos_y: AnatDir,
    /// Arah +Z (arah tumpukan).
    pub pos_z: AnatDir,
}

/// Sumbu kisi tumpukan beserta tandanya.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum SignedAxis {
    PosX,
    NegX,
    PosY,
    NegY,
    PosZ,
    NegZ,
}

impl SignedAxis {
    /// Semua enam arah, urutan tetap.
    pub const ALL: [SignedAxis; 6] = [
        SignedAxis::PosX,
        SignedAxis::NegX,
        SignedAxis::PosY,
        SignedAxis::NegY,
        SignedAxis::PosZ,
        SignedAxis::NegZ,
    ];

    /// Teks mentah, mis. "+X".
    pub fn as_str(self) -> &'static str {
        match self {
            SignedAxis::PosX => "+X",
            SignedAxis::NegX => "-X",
            SignedAxis::PosY => "+Y",
            SignedAxis::NegY => "-Y",
            SignedAxis::PosZ => "+Z",
            SignedAxis::NegZ => "-Z",
        }
    }

    /// Dari vektor arah: sumbu dengan komponen absolut terbesar.
    pub fn dominant(v: [f64; 3]) -> SignedAxis {
        let ax = v[0].abs();
        let ay = v[1].abs();
        let az = v[2].abs();
        if ax >= ay && ax >= az {
            if v[0] >= 0.0 {
                SignedAxis::PosX
            } else {
                SignedAxis::NegX
            }
        } else if ay >= az {
            if v[1] >= 0.0 {
                SignedAxis::PosY
            } else {
                SignedAxis::NegY
            }
        } else if v[2] >= 0.0 {
            SignedAxis::PosZ
        } else {
            SignedAxis::NegZ
        }
    }
}

impl SpecimenOrientation {
    /// Tiga sumbu harus berasal dari pasangan anatomis berbeda; `Unknown` boleh.
    pub fn validate(&self) -> Result<(), CoreError> {
        let axes = [("+X", self.pos_x), ("+Y", self.pos_y), ("+Z", self.pos_z)];
        for i in 0..3 {
            for j in (i + 1)..3 {
                if let (Some(a), Some(b)) = (axes[i].1.pair_index(), axes[j].1.pair_index()) {
                    if a == b {
                        return Err(CoreError::InvalidOrientation {
                            a: axes[i].0.to_owned(),
                            b: axes[j].0.to_owned(),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// Nama anatomis untuk sumbu bertanda; `Unknown` → `None` (laporan
    /// lalu memakai teks mentah "+X").
    pub fn name_of(&self, axis: SignedAxis) -> Option<AnatDir> {
        let d = match axis {
            SignedAxis::PosX => self.pos_x,
            SignedAxis::NegX => self.pos_x.opposite(),
            SignedAxis::PosY => self.pos_y,
            SignedAxis::NegY => self.pos_y.opposite(),
            SignedAxis::PosZ => self.pos_z,
            SignedAxis::NegZ => self.pos_z.opposite(),
        };
        if d == AnatDir::Unknown {
            None
        } else {
            Some(d)
        }
    }
}

/// Jenis kelamin.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
#[allow(missing_docs)]
pub enum Sex {
    Female,
    Male,
    Other,
    #[default]
    Unknown,
}

/// Lateralitas.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[allow(missing_docs)]
pub enum Laterality {
    Left,
    Right,
    Bilateral,
    NotApplicable,
}

/// Prosedur pengambilan spesimen.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[allow(missing_docs)]
pub enum Procedure {
    Biopsy,
    Excision,
    Resection,
    Other(String),
}

/// Grade; `system` teks bebas dari daftar saran (WHO, Nottingham, Gleason, …).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Grade {
    /// Sistem grade.
    pub system: String,
    /// Nilai, mis. "2" atau "3+4".
    pub value: String,
}

/// pTNM.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Ptnm {
    /// Kategori T.
    pub t: String,
    /// Kategori N.
    pub n: String,
    /// Kategori M.
    pub m: String,
    /// Edisi AJCC.
    pub edition: u8,
    /// "p" atau "yp".
    pub prefix: String,
}

/// Terapi neoadjuvan.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Therapy {
    /// Diberikan atau tidak.
    pub given: bool,
    /// Regimen bila diketahui.
    pub regimen: Option<String>,
}

/// Hasil molekuler.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct MolecularResult {
    /// Gen, mis. "KRAS".
    pub gene: String,
    /// Hasil, mis. "G12D".
    pub result: String,
    /// Metode bila diketahui.
    pub method: Option<String>,
}

/// Data klinis pseudonim (8.12.2). Hanya hidup di `clinical.json`.
/// **Tidak pernah** menjadi masukan perhitungan (aturan keras 18).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ClinicalMeta {
    /// Pseudonim; bukan nama, bukan nomor rekam medis.
    pub subject_pseudo_id: String,
    /// Usia dalam tahun.
    pub age_years: Option<u8>,
    /// Jenis kelamin.
    pub sex: Option<Sex>,
    /// Topografi ICD-O, mis. "C50.4".
    pub topography_icdo: Option<String>,
    /// Morfologi ICD-O, mis. "8500/3".
    pub morphology_icdo: Option<String>,
    /// Lateralitas.
    pub laterality: Option<Laterality>,
    /// Prosedur.
    pub procedure: Option<Procedure>,
    /// Grade.
    pub grade: Option<Grade>,
    /// pTNM.
    pub ptnm: Option<Ptnm>,
    /// Neoadjuvan.
    pub neoadjuvant: Option<Therapy>,
    /// Hasil molekuler.
    pub molecular: Vec<MolecularResult>,
    /// Bulan spesimen, "YYYY-MM"; sengaja tanpa hari.
    pub specimen_month: Option<String>,
}

impl ClinicalMeta {
    /// Benar bila `specimen_month` kosong atau berbentuk `YYYY-MM`.
    pub fn month_format_ok(&self) -> bool {
        match &self.specimen_month {
            None => true,
            Some(s) => {
                let b = s.as_bytes();
                b.len() == 7
                    && b[..4].iter().all(u8::is_ascii_digit)
                    && b[4] == b'-'
                    && b[5..].iter().all(u8::is_ascii_digit)
            }
        }
    }

    /// Daftar nama field, untuk tes properti "tidak bocor ke project.json".
    pub const FIELD_NAMES: [&'static str; 12] = [
        "subject_pseudo_id",
        "age_years",
        "sex",
        "topography_icdo",
        "morphology_icdo",
        "laterality",
        "procedure",
        "grade",
        "ptnm",
        "neoadjuvant",
        "molecular",
        "specimen_month",
    ];
}

/// Metadata sampel (8.1, 8.6.6, 8.12.2).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct SampleMeta {
    /// Nama sampel.
    pub name: String,
    /// Catatan bebas.
    pub notes: String,
    /// Faktor susut linear (8.6.6); laporan memuat terukur dan terkoreksi `V × f³`.
    pub shrinkage_linear: Option<f64>,
    /// Orientasi anatomis.
    pub orientation: Option<SpecimenOrientation>,
    /// Dimuat dari `clinical.json`; `None` bila berkas tidak ada. Tidak ikut
    /// `project.json` (aturan keras 17).
    #[serde(skip)]
    pub clinical: Option<ClinicalMeta>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orientation_rejects_same_pair() {
        let o = SpecimenOrientation {
            pos_x: AnatDir::Anterior,
            pos_y: AnatDir::Posterior,
            pos_z: AnatDir::Superior,
        };
        assert!(matches!(
            o.validate(),
            Err(CoreError::InvalidOrientation { .. })
        ));
        let ok = SpecimenOrientation {
            pos_x: AnatDir::Anterior,
            pos_y: AnatDir::Superior,
            pos_z: AnatDir::Unknown,
        };
        assert!(ok.validate().is_ok());
        assert_eq!(ok.name_of(SignedAxis::NegX), Some(AnatDir::Posterior));
        assert_eq!(ok.name_of(SignedAxis::PosZ), None);
    }

    #[test]
    fn clinical_is_skipped_in_sample_meta() {
        let s = SampleMeta {
            name: "x".into(),
            clinical: Some(ClinicalMeta {
                subject_pseudo_id: "P-001".into(),
                ..Default::default()
            }),
            ..Default::default()
        };
        let json = serde_json::to_string(&s).unwrap();
        assert!(!json.contains("P-001"));
        assert!(!json.contains("clinical"));
        for f in ClinicalMeta::FIELD_NAMES {
            assert!(!json.contains(f), "field {f} bocor");
        }
    }

    #[test]
    fn month_format() {
        let mut c = ClinicalMeta::default();
        assert!(c.month_format_ok());
        c.specimen_month = Some("2025-11".into());
        assert!(c.month_format_ok());
        c.specimen_month = Some("2025-11-03".into());
        assert!(!c.month_format_ok());
    }

    #[test]
    fn dominant_axis() {
        assert_eq!(SignedAxis::dominant([0.1, -0.9, 0.2]), SignedAxis::NegY);
        assert_eq!(SignedAxis::dominant([0.0, 0.0, 1.0]), SignedAxis::PosZ);
    }
}
