//! h3d-render — pipeline wgpu untuk HISTO3D (histo3d.md 8.10).
//!
//! Crate ini hanya menerima data siap-render: posisi `f32` dalam milimeter,
//! tile RGBA/label yang sudah didekode, dan indeks. Ia tidak tahu apa pun
//! tentang proyek, registrasi, atau berkas slide.
//!
//! Konvensi (histo3d.md 5.1–5.2):
//! - Ruang render R: `r = (w − pusat) / 1000`, sumbu y **dibalik** agar sistem
//!   tangan-kanan dengan Z ke atas = arah tumpukan. Lihat
//!   [`mesh::world_um_to_render_mm`] dan kebalikannya
//!   [`mesh::render_to_world_um`].
//! - Viewer 2D bekerja dalam µm (`Camera2D`), konversi ke GPU memakai `f32`
//!   relatif terhadap pusat pandangan.
//! - Semua affine 2D (`h3d_core::Affine2`) yang diterima crate ini memetakan
//!   ruang U (slide, µm) → ruang W (tumpukan, µm), yaitu `T_i` (5.3).

pub mod camera;
pub mod colormap;
pub mod compare;
pub mod error;
pub mod mesh;
pub mod overlay;
pub mod scene;
pub mod slices3d;
pub mod tiles;

// Re-export wgpu milik egui_wgpu supaya seluruh workspace memakai versi
// wgpu yang sama persis dengan egui (mismatch versi = error tipe misterius).
pub use egui_wgpu::wgpu;

pub use camera::{Camera2D, OrbitCamera, View2D, ViewPreset};
pub use colormap::{LutTexture, Palette256};
pub use compare::{CompareMode, CompareParams};
pub use error::RenderError;
pub use mesh::{
    pick_triangle, ray_from_screen, render_to_world_um, sort_back_to_front, upload_mesh,
    world_um_to_render_mm, GpuMesh, MeshId, MeshInstance, MeshVertex, PickHit, Ray, NODATA_SCALAR,
};
pub use overlay::{default_palette, LabelOverlayDraw};
pub use scene::{
    H3dPaintCallback, Lighting, RenderBudgets, Scene, Scene2D, Scene3D, SceneRenderer,
};
pub use slices3d::{slice_model_matrix, SliceDraw, SliceId, SliceTexture};
pub use tiles::{
    GpuTile, GpuTileCache, TileCacheIndex, TileDraw, TileKey, TileLayerDraw, TileTextureId,
    TileTextureKind, UploadStatus, DEFAULT_MAX_UPLOADS_PER_FRAME, DEFAULT_STALE_AFTER_FRAMES,
};
