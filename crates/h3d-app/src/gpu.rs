//! Pegangan GPU: `SceneRenderer` hidup di `callback_resources` egui-wgpu
//! (pola DUCAD). Semua kode UI menerima `Option<&Gpu>`; tanpa GPU (tes
//! headless) tidak ada paint callback yang ditambahkan.

use h3d_core::Budgets;
use h3d_render::wgpu;
use h3d_render::{RenderBudgets, SceneRenderer, TileKey};

use crate::tiles::TileSink;

/// Format depth yang dipasang eframe untuk `NativeOptions::depth_buffer = 32`.
/// Pipeline 2D dan 3D `SceneRenderer` harus dibuat dengan format yang sama
/// dengan render pass egui (catatan modul `h3d_render::scene`).
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Bagi anggaran GPU `Budgets` menjadi cache tile gambar (7/8) dan cache
/// tile label (1/8). Angka byte hanya berasal dari `Budgets` (aturan 15).
pub fn render_budgets(b: &Budgets) -> RenderBudgets {
    let label = b.tile_cache_gpu_bytes / 8;
    RenderBudgets::new(b.tile_cache_gpu_bytes - label, label)
}

/// Keadaan render egui-wgpu + akses ke `SceneRenderer`.
#[derive(Clone)]
pub struct Gpu {
    rs: egui_wgpu::RenderState,
}

impl Gpu {
    /// Buat `SceneRenderer` dan daftarkan di `callback_resources`.
    pub fn install(rs: &egui_wgpu::RenderState, budgets: &Budgets) -> Self {
        let renderer = SceneRenderer::new(
            &rs.device,
            &rs.queue,
            rs.target_format,
            Some(DEPTH_FORMAT),
            render_budgets(budgets),
        );
        rs.renderer.write().callback_resources.insert(renderer);
        Gpu { rs: rs.clone() }
    }

    /// Jalankan `f` dengan `SceneRenderer`, device, dan queue. `None` bila
    /// renderer belum terdaftar (tidak seharusnya terjadi setelah `install`).
    pub fn with<R>(
        &self,
        f: impl FnOnce(&mut SceneRenderer, &wgpu::Device, &wgpu::Queue) -> R,
    ) -> Option<R> {
        let mut guard = self.rs.renderer.write();
        let renderer = guard.callback_resources.get_mut::<SceneRenderer>()?;
        Some(f(renderer, &self.rs.device, &self.rs.queue))
    }

    /// Kosongkan cache tile GPU (peringatan memori, 9.4).
    pub fn clear_tile_caches(&self) {
        self.with(|r, _, _| {
            r.tiles.clear();
            r.labels.clear();
        });
    }
}

/// `SceneRenderer` sebagai tujuan unggah tile untuk [`crate::tiles`].
pub struct GpuSink<'a> {
    /// Renderer.
    pub renderer: &'a mut SceneRenderer,
    /// Device wgpu.
    pub device: &'a wgpu::Device,
    /// Queue wgpu.
    pub queue: &'a wgpu::Queue,
}

fn upload_ok(status: Result<h3d_render::UploadStatus, h3d_render::RenderError>) -> bool {
    match status {
        Ok(s) => matches!(
            s,
            h3d_render::UploadStatus::Uploaded | h3d_render::UploadStatus::AlreadyPresent
        ),
        Err(e) => {
            log::warn!("unggah tile gagal: {e}");
            false
        }
    }
}

impl TileSink for GpuSink<'_> {
    fn has_tile(&self, key: &TileKey) -> bool {
        self.renderer.has_tile(key)
    }

    fn upload_tile(&mut self, key: TileKey, w: u32, h: u32, rgba: &[u8]) -> bool {
        upload_ok(
            self.renderer
                .upload_tile(self.device, self.queue, key, w, h, rgba),
        )
    }

    fn has_label_tile(&self, key: &TileKey) -> bool {
        self.renderer.has_label_tile(key)
    }

    fn upload_label_tile(&mut self, key: TileKey, w: u32, h: u32, labels: &[u8]) -> bool {
        upload_ok(
            self.renderer
                .upload_label_tile(self.device, self.queue, key, w, h, labels),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_budgets_never_exceed_gpu_budget() {
        for b in [Budgets::MACOS, Budgets::IPAD] {
            let r = render_budgets(&b);
            assert_eq!(
                r.tile_cache_bytes + r.label_cache_bytes,
                b.tile_cache_gpu_bytes
            );
            assert!(r.label_cache_bytes > 0);
        }
    }
}
