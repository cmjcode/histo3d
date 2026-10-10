//! `h3d-core`: tipe dasar HISTO3D (histo3d.md 8.1 dan 8.12).
//!
//! Crate ini tidak punya I/O dan tidak bergantung pada GUI. Semua panjang
//! dalam mikrometer (`f64`), sudut dalam radian. Lihat histo3d.md bagian 5
//! untuk sistem koordinat; ringkasnya: `T_i` memetakan ruang U (slide) ke
//! ruang W (tumpukan), dan resampling selalu memakai `T_i⁻¹`.

pub mod affine;
pub mod annotation;
pub mod attr;
pub mod budgets;
pub mod clinical;
pub mod error;
pub mod ids;
pub mod label;
pub mod progress;
pub mod project;
pub mod slide;
pub mod transform;
pub mod undo;

pub use affine::{Affine2, AffineParts};
pub use annotation::{AnnotationRecord, AnnotationSource, ReviewStatus};
pub use attr::{AttrDef, AttrKind, AttrSource, Colormap};
pub use budgets::Budgets;
pub use clinical::{
    AnatDir, ClinicalMeta, Grade, Laterality, MolecularResult, Procedure, Ptnm, SampleMeta, Sex,
    SpecimenOrientation, Therapy,
};
pub use error::CoreError;
pub use ids::{AttrId, LabelId, SlideId};
pub use label::{LabelDef, LabelKind, VesselKind};
pub use progress::{NoProgress, Progress};
pub use project::{Project, ReconConfig, SpatialConfig, FORMAT_VERSION};
pub use slide::{
    Chromogen, IhcPanel, IhcScoring, OrderSource, PreanalyticMeta, RegRecord, SlideEntry,
    SlideSourceRef, SlideStatus, Stain, StainPattern,
};
pub use transform::{DisplacementGrid, SlideTransform};
pub use undo::{Command, UndoStack};

/// Kalimat wajib di semua laporan dan layar Tentang (aturan keras 16).
pub const RESEARCH_ONLY_NOTICE: &str = "Hanya untuk riset, bukan untuk diagnosis";

pub use clinical::SignedAxis;
pub use progress::{SharedProgress, SubProgress};
