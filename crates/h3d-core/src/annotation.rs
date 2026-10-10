//! Provenance anotasi per (irisan, label) (8.12.7).

use serde::{Deserialize, Serialize};

use crate::ids::{LabelId, SlideId};

/// Asal sebuah anotasi.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AnnotationSource {
    /// Digambar tangan.
    Manual {
        /// Nama/inisial pengguna.
        by: String,
        /// Bila menyunting hasil otomatis, nama sumber awalnya ("ai", "classic", …).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        edited_from: Option<String>,
    },
    /// Segmentasi klasik.
    Classic {
        /// Hash parameter.
        params_hash: u64,
    },
    /// Model AI.
    Ai {
        /// Nama model.
        model: String,
        /// Versi model.
        version: String,
    },
    /// Interpolasi bentuk antara dua irisan bertanda.
    Interpolated {
        /// Irisan sumber `[a, b]`.
        from: [SlideId; 2],
    },
    /// Impor dari berkas.
    Imported {
        /// Nama berkas.
        file: String,
    },
}

impl AnnotationSource {
    /// Nama singkat untuk laporan.
    pub fn short_name(&self) -> &'static str {
        match self {
            AnnotationSource::Manual { .. } => "manual",
            AnnotationSource::Classic { .. } => "classic",
            AnnotationSource::Ai { .. } => "ai",
            AnnotationSource::Interpolated { .. } => "interpolated",
            AnnotationSource::Imported { .. } => "imported",
        }
    }

    /// Benar bila hasil interpolasi (tidak dipakai Cavalieri, 8.6.4).
    pub fn is_interpolated(&self) -> bool {
        matches!(self, AnnotationSource::Interpolated { .. })
    }
}

/// Status tinjauan.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ReviewStatus {
    /// Belum ditinjau.
    #[default]
    Draft,
    /// Sudah ditinjau.
    Reviewed {
        /// Peninjau.
        by: String,
        /// Waktu ISO-8601 zona lokal.
        at: String,
    },
}

impl ReviewStatus {
    /// Benar bila sudah ditinjau.
    pub fn is_reviewed(&self) -> bool {
        matches!(self, ReviewStatus::Reviewed { .. })
    }
}

/// Catatan provenance satu (irisan, label).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnnotationRecord {
    /// Irisan.
    pub slide: SlideId,
    /// Label.
    pub label: LabelId,
    /// Asal.
    pub source: AnnotationSource,
    /// Waktu ISO-8601 zona lokal.
    pub at: String,
    /// Status tinjauan.
    #[serde(default)]
    pub review: ReviewStatus,
    /// Rata-rata probabilitas argmax untuk AI; `None` untuk manual.
    #[serde(default)]
    pub confidence: Option<f32>,
}
