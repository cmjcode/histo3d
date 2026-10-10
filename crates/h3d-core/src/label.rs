//! Definisi label dan jenisnya (8.1, 8.12.4).

use serde::{Deserialize, Serialize};

use crate::ids::LabelId;

/// Jenis pembuluh.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum VesselKind {
    /// Pembuluh darah.
    Blood,
    /// Pembuluh limfe.
    Lymph,
    /// Tidak diketahui.
    #[default]
    Unknown,
}

/// Jenis label; menentukan analisis spasial mana yang berlaku (8.12.6).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum LabelKind {
    /// Komponen tumor apa pun (subtipe lewat `name` + `group`).
    Tumor,
    /// Nekrosis.
    Necrosis,
    /// Stroma.
    Stroma,
    /// Jaringan normal.
    Normal,
    /// Pembuluh.
    Vessel(VesselKind),
    /// Saraf.
    Nerve,
    /// Kapsul / serosa / fasia.
    Capsule,
    /// Tepi reseksi (permukaan luar spesimen, biasanya bertinta).
    Margin,
    /// Permukaan epitel/mukosa, acuan kedalaman invasi.
    Surface,
    /// Garis acuan lain (mis. lapisan granular untuk Breslow).
    Reference,
    /// Lainnya.
    #[default]
    Other,
}

impl LabelKind {
    /// Label jenis ini dipakai sebagai sumber medan jarak, bukan mesh
    /// tertutup (8.12.4).
    pub fn is_distance_source(self) -> bool {
        matches!(
            self,
            LabelKind::Margin | LabelKind::Surface | LabelKind::Reference
        )
    }

    /// Nama kunci terjemahan (`label-kind-*`).
    pub fn i18n_key(self) -> &'static str {
        match self {
            LabelKind::Tumor => "label-kind-tumor",
            LabelKind::Necrosis => "label-kind-necrosis",
            LabelKind::Stroma => "label-kind-stroma",
            LabelKind::Normal => "label-kind-normal",
            LabelKind::Vessel(_) => "label-kind-vessel",
            LabelKind::Nerve => "label-kind-nerve",
            LabelKind::Capsule => "label-kind-capsule",
            LabelKind::Margin => "label-kind-margin",
            LabelKind::Surface => "label-kind-surface",
            LabelKind::Reference => "label-kind-reference",
            LabelKind::Other => "label-kind-other",
        }
    }

    /// Semua jenis untuk pemilih UI.
    pub const ALL: [LabelKind; 13] = [
        LabelKind::Tumor,
        LabelKind::Necrosis,
        LabelKind::Stroma,
        LabelKind::Normal,
        LabelKind::Vessel(VesselKind::Blood),
        LabelKind::Vessel(VesselKind::Lymph),
        LabelKind::Vessel(VesselKind::Unknown),
        LabelKind::Nerve,
        LabelKind::Capsule,
        LabelKind::Margin,
        LabelKind::Surface,
        LabelKind::Reference,
        LabelKind::Other,
    ];
}

/// Definisi satu label.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LabelDef {
    /// Pengenal (nilai piksel).
    pub id: LabelId,
    /// Nama tampil, mis. "Tumor".
    pub name: String,
    /// Warna RGBA.
    pub color: [u8; 4],
    /// Tampil di overlay.
    pub visible: bool,
    /// Ikut dibuat mesh-nya.
    pub reconstruct: bool,
    /// Jenis (8.12.4).
    pub kind: LabelKind,
    /// Grup agregasi di laporan (8.12.4).
    pub group: Option<String>,
}

impl Default for LabelDef {
    fn default() -> Self {
        Self {
            id: LabelId(1),
            name: String::new(),
            color: [255, 0, 0, 255],
            visible: true,
            reconstruct: true,
            kind: LabelKind::Other,
            group: None,
        }
    }
}

impl LabelDef {
    /// Label baru dengan nama dan warna; `reconstruct` mengikuti `kind`.
    pub fn new(id: LabelId, name: &str, color: [u8; 4], kind: LabelKind) -> Self {
        Self {
            id,
            name: name.to_owned(),
            color,
            visible: true,
            reconstruct: !kind.is_distance_source(),
            kind,
            group: None,
        }
    }

    /// Nama grup efektif: `group` bila ada, selain itu `name`.
    pub fn group_name(&self) -> &str {
        self.group.as_deref().unwrap_or(&self.name)
    }

    /// Tabel label bawaan untuk proyek baru.
    pub fn default_table() -> Vec<LabelDef> {
        vec![
            LabelDef::new(LabelId(1), "Tumor", [220, 40, 40, 255], LabelKind::Tumor),
            LabelDef::new(LabelId(2), "Stroma", [60, 160, 60, 255], LabelKind::Stroma),
            LabelDef::new(
                LabelId(3),
                "Nekrosis",
                [140, 100, 40, 255],
                LabelKind::Necrosis,
            ),
            LabelDef::new(LabelId(4), "Normal", [80, 120, 220, 255], LabelKind::Normal),
            LabelDef::new(
                LabelId(5),
                "Pembuluh darah",
                [230, 60, 160, 255],
                LabelKind::Vessel(VesselKind::Blood),
            ),
            LabelDef::new(LabelId(6), "Saraf", [240, 200, 40, 255], LabelKind::Nerve),
            LabelDef::new(
                LabelId(7),
                "Kapsul",
                [120, 80, 200, 255],
                LabelKind::Capsule,
            ),
            LabelDef::new(
                LabelId(8),
                "Tepi reseksi",
                [20, 20, 20, 255],
                LabelKind::Margin,
            ),
            LabelDef::new(
                LabelId(9),
                "Permukaan",
                [40, 200, 220, 255],
                LabelKind::Surface,
            ),
        ]
    }
}
