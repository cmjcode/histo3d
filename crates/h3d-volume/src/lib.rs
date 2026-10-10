//! `h3d-volume`: rekonstruksi 3D, pengukuran, dan analisis spasial HISTO3D
//! (histo3d.md 8.6 dan 8.12.6). Komputasi murni: tanpa I/O, tanpa GUI.
//!
//! # Konvensi
//! - Panjang µm `f64`; kisi [`StackGrid`] dengan pusat voxel di `+0,5`
//!   (`w = origin + (i + 0,5) × s_xy`, `z = z_k`).
//! - Buffer 3D berindeks `x + nx·(y + ny·z)`.
//! - SDF **negatif di dalam**; segitiga berorientasi ke luar (volume
//!   bertanda positif).
//! - `SliceInput::transform` = `T_i` (U → W); resampling memakai `T_i⁻¹`.
//! - Analisis spasial bekerja pada [`VoxelMask`], bukan mesh (aturan 19);
//!   semua jarak dilaporkan dengan [`Uncertainty`] (aturan 18).
//! - Deterministik: urutan pindai tetap `(z, y, x)`, seri → indeks terkecil.

pub mod error;
pub mod grid;
pub mod measure;
pub mod mesh;
pub mod phantom3d;
pub mod recon;
pub mod spatial;

pub use error::VolError;
pub use grid::StackGrid;
pub use measure::{
    cavalieri, cavalieri_areas, label_area_um2, measure_label, LabelMeasurement, Warning,
};
pub use mesh::{max_diameter_of_points, sample_trilinear, LabelMesh, MeshScalars};
pub use phantom3d::{
    generate as generate_phantom, ArrayLabelSampler, Phantom, PhantomShape, PhantomSlice,
    PhantomSpec, PhantomTruth,
};
pub use recon::{
    build_label_stack, build_voxel_mask, extract_mesh, layer_sources, plan_grid, reconstruct_label,
    resample_masks, GridPlan, LabelSampler, LabelStack, LayerSource, SdfProvider, SliceInput,
    SphereSdf, VoxelMask,
};
pub use spatial::{
    analyze, attr_stats, capsule_penetration, contact, depth, distance_uncertainty, foci,
    fractions, hypoxia, margin, project_fields_to_mesh, resample_uniform_z, staging_size,
    AttrStats, AttrStatsEntry, AttrVolume, CapsuleResult, ContactEntry, ContactResult, DepthResult,
    FociResult, Focus, FractionRow, FractionsResult, GridSummary, HypoxiaResult, MarginResult,
    MarginStatus, Patch, SizeResult, SpatialAnalysis, Uncertainty,
};
