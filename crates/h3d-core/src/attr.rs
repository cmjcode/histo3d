//! Definisi atribut skalar per lokasi (8.12.5).

use serde::{Deserialize, Serialize};

use crate::ids::{AttrId, SlideId};
use crate::slide::IhcScoring;

/// Jenis atribut.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AttrKind {
    /// Skor IHC dari satu irisan IHC.
    IhcScore {
        /// Irisan IHC sumber.
        slide: SlideId,
        /// Skema skor.
        scoring: IhcScoring,
    },
    /// Densitas inti, sel/mm².
    NucleiDensity,
    /// Densitas limfosit, limfosit/mm².
    TilDensity,
    /// Lainnya.
    Custom,
}

/// Asal atribut.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AttrSource {
    /// Dihitung aplikasi; `params_hash` atas semua parameter (tercetak di laporan).
    Computed {
        /// Nama metode, mis. "ihc-nuclear-v1".
        method: String,
        /// Hash parameter.
        params_hash: u64,
    },
    /// Diimpor dari berkas luar.
    Imported {
        /// Nama berkas.
        file: String,
    },
}

/// Colormap untuk tampilan skalar.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
#[allow(missing_docs)]
pub enum Colormap {
    #[default]
    Viridis,
    Magma,
    Coolwarm,
    Grey,
}

impl Colormap {
    /// Semua colormap.
    pub const ALL: [Colormap; 4] = [
        Colormap::Viridis,
        Colormap::Magma,
        Colormap::Coolwarm,
        Colormap::Grey,
    ];

    /// Warna RGB untuk `t ∈ [0, 1]` (interpolasi linear antar titik kunci).
    pub fn sample(self, t: f32) -> [u8; 3] {
        let t = if t.is_finite() {
            t.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let keys: &[[f32; 3]] = match self {
            Colormap::Viridis => &[
                [0.267, 0.005, 0.329],
                [0.283, 0.141, 0.458],
                [0.254, 0.265, 0.530],
                [0.207, 0.372, 0.553],
                [0.164, 0.471, 0.558],
                [0.128, 0.567, 0.551],
                [0.135, 0.659, 0.518],
                [0.267, 0.749, 0.441],
                [0.478, 0.821, 0.317],
                [0.741, 0.873, 0.150],
                [0.993, 0.906, 0.144],
            ],
            Colormap::Magma => &[
                [0.001, 0.000, 0.014],
                [0.088, 0.058, 0.218],
                [0.254, 0.065, 0.436],
                [0.417, 0.108, 0.506],
                [0.578, 0.149, 0.504],
                [0.740, 0.191, 0.459],
                [0.880, 0.272, 0.382],
                [0.966, 0.419, 0.350],
                [0.993, 0.593, 0.395],
                [0.997, 0.771, 0.524],
                [0.987, 0.991, 0.750],
            ],
            Colormap::Coolwarm => &[
                [0.230, 0.299, 0.754],
                [0.406, 0.538, 0.934],
                [0.602, 0.731, 0.999],
                [0.788, 0.846, 0.939],
                [0.865, 0.865, 0.865],
                [0.967, 0.810, 0.740],
                [0.969, 0.661, 0.549],
                [0.887, 0.453, 0.345],
                [0.706, 0.016, 0.150],
            ],
            Colormap::Grey => &[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        };
        let n = keys.len() - 1;
        let x = t * n as f32;
        let i = (x.floor() as usize).min(n - 1);
        let f = x - i as f32;
        let a = keys[i];
        let b = keys[i + 1];
        let mut out = [0u8; 3];
        for k in 0..3 {
            out[k] = ((a[k] + (b[k] - a[k]) * f) * 255.0)
                .round()
                .clamp(0.0, 255.0) as u8;
        }
        out
    }

    /// Tabel 256 × RGBA untuk tekstur shader.
    pub fn lut256(self) -> Vec<[u8; 4]> {
        (0..256)
            .map(|i| {
                let c = self.sample(i as f32 / 255.0);
                [c[0], c[1], c[2], 255]
            })
            .collect()
    }
}

/// Definisi atribut (tersimpan di proyek; nilainya di berkas `.atr`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AttrDef {
    /// Pengenal.
    pub id: AttrId,
    /// Nama, mis. "Ki-67 % positif".
    pub name: String,
    /// Satuan, mis. "%", "sel/mm²", "µm".
    pub unit: String,
    /// Jenis.
    pub kind: AttrKind,
    /// Asal.
    pub source: AttrSource,
    /// Colormap tampilan.
    pub colormap: Colormap,
    /// Rentang tampilan; `None` = p2–p98 data.
    pub range: Option<[f32; 2]>,
}

impl Default for AttrDef {
    fn default() -> Self {
        Self {
            id: AttrId(0),
            name: String::new(),
            unit: String::new(),
            kind: AttrKind::Custom,
            source: AttrSource::Imported {
                file: String::new(),
            },
            colormap: Colormap::Viridis,
            range: None,
        }
    }
}

impl AttrDef {
    /// Nama berkas aman untuk `.atr`/NIfTI: huruf, angka, `-`, `_` saja.
    pub fn file_stem(&self) -> String {
        let s: String = self
            .name
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() {
                    c.to_ascii_lowercase()
                } else {
                    '_'
                }
            })
            .collect();
        let trimmed = s.trim_matches('_');
        if trimmed.is_empty() {
            format!("attr{}", self.id.0)
        } else {
            trimmed.to_owned()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colormap_endpoints() {
        assert_eq!(Colormap::Grey.sample(0.0), [0, 0, 0]);
        assert_eq!(Colormap::Grey.sample(1.0), [255, 255, 255]);
        assert_eq!(Colormap::Viridis.lut256().len(), 256);
        // NaN → awal skala, tidak panik.
        let _ = Colormap::Magma.sample(f32::NAN);
    }

    #[test]
    fn file_stem_sanitizes() {
        let d = AttrDef {
            name: "Ki-67 % positif".into(),
            ..Default::default()
        };
        assert_eq!(d.file_stem(), "ki_67___positif");
    }
}
