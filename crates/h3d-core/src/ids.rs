//! Pengenal dasar.

use serde::{Deserialize, Serialize};

/// Pengenal irisan di dalam satu proyek. Tidak pernah dipakai ulang setelah
/// irisan dihapus.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SlideId(pub u32);

/// Label raster. `0` = latar, `1..=255` = label pengguna.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LabelId(pub u8);

impl LabelId {
    /// Latar belakang (tanpa label).
    pub const BACKGROUND: LabelId = LabelId(0);

    /// Benar bila label ini adalah latar.
    pub fn is_background(self) -> bool {
        self.0 == 0
    }
}

/// Pengenal atribut skalar (8.12.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AttrId(pub u16);

impl std::fmt::Display for SlideId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::fmt::Display for LabelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::fmt::Display for AttrId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
