//! `h3d-io`: bundel proyek `.h3d` dan ekspor (histo3d.md 8.7, 8.12.9).
//!
//! - [`bundle::ProjectBundle`]: tata letak folder, baca/tulis `project.json`
//!   (berversi, bermigrasi), `clinical.json` (terpisah), `.lbl`, `.atr`.
//! - [`atomic::write_atomic`]: semua berkas proyek ditulis `.tmp` lalu
//!   `rename` (aturan keras 10).
//! - Ekspor sebagai fungsi murni yang menghasilkan byte/teks: [`glb`],
//!   [`stl`], [`obj`], [`nifti`], [`geojson`], [`analysis_json`], [`report`].
//!
//! Data klinis hanya hidup di `clinical.json` dan tidak pernah masuk
//! `project.json` maupun ekspor tanpa `with_clinical` (aturan keras 17).

#![warn(missing_docs)]

pub mod analysis_json;
pub mod atomic;
pub mod bundle;
pub mod error;
pub mod geojson;
pub mod glb;
pub mod meshconv;
pub mod nifti;
pub mod obj;
pub mod report;
pub mod stl;
pub mod strings;
pub mod vox_cache;

pub use analysis_json::{
    decode_analysis, encode_analysis, to_stable_json, to_stable_json_of, AnalysisFile,
};
pub use atomic::{copy_atomic, remove_if_exists, write_atomic};
pub use bundle::{encode_project, slide_stem, ProjectBundle};
pub use error::IoError;
pub use geojson::export_slide_geojson;
pub use glb::{encode_glb, parse_glb, scalar_attribute_name, GlbMesh, GlbScalar};
pub use nifti::{decode_nifti, encode_f32_nifti, encode_label_nifti, gunzip, NiftiOutput};
pub use obj::encode_obj;
pub use report::{
    render_csv, render_markdown, AttrReport, AttrSlideRow, ClinicalScope, GridInfo, GroupMember,
    GroupReport, LabelReport, ProvenanceReport, Report, SlideSummary,
};
pub use stl::{decode_stl, encode_stl, stl_file_name};
pub use vox_cache::{decode_mesh, decode_vox, encode_mesh, encode_vox};
