//! `h3d-engine`: lapisan headless HISTO3D (histo3d.md 8.8) yang dipakai CLI
//! dan GUI. Tidak bergantung pada GUI maupun jaringan (dijaga tes
//! `engine_has_no_gui_dependency`).
//!
//! - [`Session`]: proyek terbuka — impor, registrasi, label + undo, usulan,
//!   rekonstruksi, pengukuran, atribut, analisis spasial, ekspor.
//! - [`job`]: job latar dengan progres dan pembatalan.
//! - [`figure`]: perasteran mungil untuk gambar laporan (PNG).
//! - [`clock`]: jam yang bisa disuntik agar keluaran deterministik.
//! - [`phantom`]: data uji sintetis ber-kebenaran-dasar (bagian 11.1).
//! - [`tools`]: metadata slide dan thumbnail tanpa proyek.
//!
//! **Determinisme.** Tidak ada acak tak berbenih; hasil paralel digabung
//! dalam urutan tetap; satu-satunya pemakaian jam adalah cap waktu
//! provenance, lewat [`clock::Clock`].

#![warn(missing_docs)]

pub mod clock;
pub mod error;
pub mod figure;
pub mod hash;
pub mod job;
pub mod phantom;
pub mod session;
pub mod tools;

pub use clock::{clock_from_env, Clock, FixedClock, SystemClock, FIXED_TIME_ENV};
pub use error::EngineError;
pub use job::{spawn_job, spawn_session_job, JobHandle, SharedSession};
pub use session::{
    AttrParams, ExportKind, ImportOptions, MarginFigure, PendingProposal, QcRow, RegSummary,
    RegisterRun, Session, APP_VERSION, FIGURE_MARKER_RADIUS, FIGURE_MAX_SIDE, REG_IMAGE_MPP,
};

// Tipe crate bawah yang muncul di API publik, agar pemakai (CLI/GUI) cukup
// bergantung pada `h3d-engine`.
pub use h3d_io::report::fmt_uncertainty;
pub use h3d_io::strings::{direction_text, warning_text};
pub use h3d_io::{to_stable_json_of, ClinicalScope, Report, ReportFigure};
pub use h3d_register::{OrderSuggestion, QcThresholds, RegMetric, RegModel, RegOptions};
pub use h3d_segment::{ClassicParams, ImportSummary, LabelStore, PaintMode, Proposal};
pub use h3d_volume::{LabelMesh, SpatialAnalysis, VoxelMask};
