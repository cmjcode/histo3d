//! Pipeline tile 2D dan cache tekstur GPU.
//!
//! Pembukuan cache ([`TileCacheIndex`]) dipisah dari kepemilikan tekstur
//! ([`GpuTileCache`]) supaya kebijakan LRU/eviksi dapat diuji tanpa GPU.
//!
//! Kebijakan: LRU berbatas **byte** (`max_bytes` berasal dari `Budgets`
//! aplikasi, tidak pernah ditulis di sini); tile yang tidak terlihat lebih
//! lama dibuang lebih dulu; tile yang terlihat pada frame berjalan tidak
//! pernah dibuang; unggahan dibatasi `max_uploads_per_frame` per frame.

use std::collections::HashMap;

use crate::error::RenderError;
use crate::wgpu;
use h3d_core::Affine2;

/// Batas unggahan tile per frame bawaan (histo3d.md 8.10).
pub const DEFAULT_MAX_UPLOADS_PER_FRAME: usize = 8;
/// Jumlah frame tak terlihat setelah itu tile dianggap basi.
pub const DEFAULT_STALE_AFTER_FRAMES: u64 = 60;

/// Kunci tile: `(slide, level, tx, ty)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileKey {
    /// Indeks slide (aplikasi yang memetakan `SlideId` → `u32`).
    pub slide: u32,
    /// Level piramida.
    pub level: u32,
    /// Kolom tile.
    pub tx: u32,
    /// Baris tile.
    pub ty: u32,
}

/// Pengenal tekstur tile = kuncinya.
pub type TileTextureId = TileKey;

/// Satu quad tile untuk digambar.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileDraw {
    /// Tekstur yang dipakai (harus sudah ada di cache; bila tidak, tile dilewati).
    pub key: TileKey,
    /// `[x0, y0, x1, y1]` dalam µm **ruang U irisan** (sebelum `T_i`), yaitu
    /// `piksel_level0 × mpp`. Bila lapisan memakai affine identitas, ini
    /// sekaligus ruang W.
    pub rect_um: [f32; 4],
    /// `[u0, v0, u1, v1]`; `[0,0,1,1]` untuk tile penuh.
    pub uv: [f32; 4],
}

impl TileDraw {
    /// Tile penuh (`uv` = `[0,0,1,1]`).
    pub fn new(key: TileKey, rect_um: [f32; 4]) -> Self {
        Self {
            key,
            rect_um,
            uv: [0.0, 0.0, 1.0, 1.0],
        }
    }
}

/// Lapisan tile RGBA dengan satu affine.
#[derive(Clone, Debug, PartialEq)]
pub struct TileLayerDraw {
    /// `T_i`: ruang U irisan (µm) → ruang W (µm).
    pub affine: Affine2,
    /// Tile-tile lapisan.
    pub tiles: Vec<TileDraw>,
    /// Opasitas 0..1.
    pub opacity: f32,
}

impl TileLayerDraw {
    /// Lapisan dengan affine identitas dan opasitas penuh.
    pub fn new(tiles: Vec<TileDraw>) -> Self {
        Self {
            affine: Affine2::IDENTITY,
            tiles,
            opacity: 1.0,
        }
    }
}

/// Data per-instance quad tile di GPU.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct TileInstance {
    pub rect: [f32; 4],
    pub uv: [f32; 4],
}

const TILE_INSTANCE_ATTRS: [wgpu::VertexAttribute; 2] =
    wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4];

/// Layout buffer instance tile (dipakai pipeline tile dan overlay).
pub(crate) fn tile_instance_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<TileInstance>() as u64,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &TILE_INSTANCE_ATTRS,
    }
}

/// Alasan penyisipan ditolak oleh [`TileCacheIndex`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InsertRejected {
    /// Kunci sudah ada (disentuh sebagai terlihat).
    AlreadyPresent,
    /// Kuota unggahan frame ini habis.
    UploadBudgetExhausted,
    /// Tile lebih besar dari seluruh anggaran.
    TooLarge,
    /// Semua tile lain terlihat pada frame ini; tidak ada yang bisa dibuang.
    NoEvictableRoom,
}

#[derive(Clone, Copy, Debug)]
struct IndexEntry {
    bytes: u64,
    last_seen: u64,
}

/// Pembukuan cache (tanpa tekstur).
#[derive(Clone, Debug)]
pub struct TileCacheIndex {
    max_bytes: u64,
    used_bytes: u64,
    frame: u64,
    stale_after_frames: u64,
    max_uploads_per_frame: usize,
    uploads_this_frame: usize,
    entries: HashMap<TileKey, IndexEntry>,
}

impl TileCacheIndex {
    /// Buat indeks dengan anggaran `max_bytes` (dari `Budgets`).
    pub fn new(max_bytes: u64, stale_after_frames: u64, max_uploads_per_frame: usize) -> Self {
        Self {
            max_bytes,
            used_bytes: 0,
            frame: 1,
            stale_after_frames: stale_after_frames.max(1),
            max_uploads_per_frame: max_uploads_per_frame.max(1),
            uploads_this_frame: 0,
            entries: HashMap::new(),
        }
    }

    /// Mulai frame baru: naikkan penghitung dan reset kuota unggahan.
    pub fn begin_frame(&mut self) {
        self.frame += 1;
        self.uploads_this_frame = 0;
    }

    /// Nomor frame berjalan.
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Jumlah tile.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Benar bila kosong.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Byte terpakai.
    pub fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    /// Anggaran byte.
    pub fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    /// Sisa kuota unggahan frame ini.
    pub fn uploads_remaining(&self) -> usize {
        self.max_uploads_per_frame
            .saturating_sub(self.uploads_this_frame)
    }

    /// Benar bila `key` ada.
    pub fn contains(&self, key: &TileKey) -> bool {
        self.entries.contains_key(key)
    }

    /// Tandai `key` terlihat pada frame ini. Mengembalikan `false` bila tidak ada.
    pub fn touch(&mut self, key: &TileKey) -> bool {
        match self.entries.get_mut(key) {
            Some(e) => {
                e.last_seen = self.frame;
                true
            }
            None => false,
        }
    }

    /// Kandidat eviksi: tidak terlihat frame ini, terlama dulu (urutan tetap).
    fn eviction_order(&self) -> Vec<(TileKey, u64)> {
        let mut v: Vec<(TileKey, u64)> = self
            .entries
            .iter()
            .filter(|(_, e)| e.last_seen < self.frame)
            .map(|(k, e)| (*k, e.bytes))
            .collect();
        v.sort_by_key(|(k, _)| (self.entries[k].last_seen, *k));
        v
    }

    /// Daftarkan tile baru berukuran `bytes`; menghitung kuota unggahan.
    /// Mengembalikan kunci yang dibuang untuk memberi ruang (urutan tetap).
    pub fn insert(&mut self, key: TileKey, bytes: u64) -> Result<Vec<TileKey>, InsertRejected> {
        if self.entries.contains_key(&key) {
            self.touch(&key);
            return Err(InsertRejected::AlreadyPresent);
        }
        if bytes > self.max_bytes {
            return Err(InsertRejected::TooLarge);
        }
        if self.uploads_this_frame >= self.max_uploads_per_frame {
            return Err(InsertRejected::UploadBudgetExhausted);
        }
        let mut evicted = Vec::new();
        if self.used_bytes + bytes > self.max_bytes {
            let mut freed = 0u64;
            let needed = self.used_bytes + bytes - self.max_bytes;
            for (k, b) in self.eviction_order() {
                if freed >= needed {
                    break;
                }
                freed += b;
                evicted.push(k);
            }
            if freed < needed {
                return Err(InsertRejected::NoEvictableRoom);
            }
            for k in &evicted {
                self.remove(k);
            }
        }
        self.entries.insert(
            key,
            IndexEntry {
                bytes,
                last_seen: self.frame,
            },
        );
        self.used_bytes += bytes;
        self.uploads_this_frame += 1;
        Ok(evicted)
    }

    /// Hapus `key`; mengembalikan ukurannya.
    pub fn remove(&mut self, key: &TileKey) -> Option<u64> {
        let e = self.entries.remove(key)?;
        self.used_bytes -= e.bytes;
        Some(e.bytes)
    }

    /// Kunci yang tidak terlihat selama ≥ `stale_after_frames` frame (urutan tetap).
    pub fn stale_keys(&self) -> Vec<TileKey> {
        let mut v: Vec<TileKey> = self
            .entries
            .iter()
            .filter(|(_, e)| self.frame.saturating_sub(e.last_seen) >= self.stale_after_frames)
            .map(|(k, _)| *k)
            .collect();
        v.sort();
        v
    }

    /// Ubah anggaran; bila mengecil, buang tile terlama hingga muat.
    pub fn set_max_bytes(&mut self, max_bytes: u64) -> Vec<TileKey> {
        self.max_bytes = max_bytes;
        let mut evicted = Vec::new();
        if self.used_bytes > max_bytes {
            // Saat mengecilkan anggaran, tile frame ini pun boleh dibuang.
            let mut all: Vec<(TileKey, u64, u64)> = self
                .entries
                .iter()
                .map(|(k, e)| (*k, e.last_seen, e.bytes))
                .collect();
            all.sort_by_key(|(k, seen, _)| (*seen, *k));
            for (k, _, _) in all {
                if self.used_bytes <= max_bytes {
                    break;
                }
                self.remove(&k);
                evicted.push(k);
            }
        }
        evicted
    }

    /// Kosongkan.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.used_bytes = 0;
    }
}

/// Jenis tekstur tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileTextureKind {
    /// Gambar `Rgba8UnormSrgb`, 4 byte/piksel, disampel linear.
    Rgba,
    /// Label `R8Uint`, 1 byte/piksel, dibaca `textureLoad`.
    LabelU8,
}

impl TileTextureKind {
    /// Format wgpu.
    pub fn format(self) -> wgpu::TextureFormat {
        match self {
            TileTextureKind::Rgba => wgpu::TextureFormat::Rgba8UnormSrgb,
            TileTextureKind::LabelU8 => wgpu::TextureFormat::R8Uint,
        }
    }

    /// Byte per piksel.
    pub fn bytes_per_pixel(self) -> u32 {
        match self {
            TileTextureKind::Rgba => 4,
            TileTextureKind::LabelU8 => 1,
        }
    }
}

/// Tekstur tile di GPU.
pub struct GpuTile {
    /// Tekstur.
    pub texture: wgpu::Texture,
    /// View.
    pub view: wgpu::TextureView,
    /// Bind group grup 1 (tekstur [+ sampler]).
    pub bind_group: wgpu::BindGroup,
    /// Lebar piksel.
    pub width: u32,
    /// Tinggi piksel.
    pub height: u32,
}

/// Hasil `upload`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UploadStatus {
    /// Diunggah frame ini.
    Uploaded,
    /// Sudah ada (disentuh).
    AlreadyPresent,
    /// Kuota unggahan frame ini habis; coba lagi frame berikutnya.
    BudgetExhausted,
    /// Tidak ada ruang yang bisa dibebaskan (semua tile terlihat).
    CacheFull,
}

/// Cache tekstur tile GPU.
pub struct GpuTileCache {
    index: TileCacheIndex,
    kind: TileTextureKind,
    layout: wgpu::BindGroupLayout,
    sampler: Option<wgpu::Sampler>,
    tiles: HashMap<TileKey, GpuTile>,
    max_dim: u32,
}

impl GpuTileCache {
    /// Buat cache. `max_bytes` wajib berasal dari `Budgets` aplikasi.
    pub fn new(
        device: &wgpu::Device,
        kind: TileTextureKind,
        max_bytes: u64,
        stale_after_frames: u64,
        max_uploads_per_frame: usize,
    ) -> Self {
        let (layout, sampler) = match kind {
            TileTextureKind::Rgba => {
                let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("h3d-tile-rgba"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });
                let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("h3d-tile-sampler"),
                    address_mode_u: wgpu::AddressMode::ClampToEdge,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                    ..Default::default()
                });
                (layout, Some(sampler))
            }
            TileTextureKind::LabelU8 => {
                let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("h3d-tile-label"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Uint,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    }],
                });
                (layout, None)
            }
        };
        Self {
            index: TileCacheIndex::new(max_bytes, stale_after_frames, max_uploads_per_frame),
            kind,
            layout,
            sampler,
            tiles: HashMap::new(),
            max_dim: device.limits().max_texture_dimension_2d,
        }
    }

    /// Jenis tekstur.
    pub fn kind(&self) -> TileTextureKind {
        self.kind
    }

    /// Layout bind group grup 1 untuk pipeline.
    pub fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    /// Pembukuan (baca saja).
    pub fn index(&self) -> &TileCacheIndex {
        &self.index
    }

    /// Mulai frame baru (reset kuota unggahan).
    pub fn begin_frame(&mut self) {
        self.index.begin_frame();
    }

    /// Benar bila `key` ada.
    pub fn contains(&self, key: &TileKey) -> bool {
        self.tiles.contains_key(key)
    }

    /// Tandai terlihat frame ini.
    pub fn touch(&mut self, key: &TileKey) -> bool {
        self.index.touch(key)
    }

    /// Ambil tekstur.
    pub fn get(&self, key: &TileKey) -> Option<&GpuTile> {
        self.tiles.get(key)
    }

    /// Unggah tile `width × height` dengan `data` (`bytes_per_pixel` × w × h
    /// byte). Menghormati batas dimensi perangkat, anggaran byte, dan kuota
    /// unggahan per frame.
    pub fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        key: TileKey,
        width: u32,
        height: u32,
        data: &[u8],
    ) -> Result<UploadStatus, RenderError> {
        if width == 0 || height == 0 || width > self.max_dim || height > self.max_dim {
            return Err(RenderError::TextureTooLarge {
                width,
                height,
                max: self.max_dim,
            });
        }
        let bpp = self.kind.bytes_per_pixel();
        let expected = (width as usize) * (height as usize) * bpp as usize;
        if data.len() != expected {
            return Err(RenderError::DataSizeMismatch {
                expected,
                actual: data.len(),
            });
        }
        let bytes = expected as u64;
        let evicted = match self.index.insert(key, bytes) {
            Ok(ev) => ev,
            Err(InsertRejected::AlreadyPresent) => return Ok(UploadStatus::AlreadyPresent),
            Err(InsertRejected::UploadBudgetExhausted) => return Ok(UploadStatus::BudgetExhausted),
            Err(InsertRejected::NoEvictableRoom) => return Ok(UploadStatus::CacheFull),
            Err(InsertRejected::TooLarge) => {
                return Err(RenderError::TileExceedsBudget {
                    bytes,
                    max_bytes: self.index.max_bytes(),
                })
            }
        };
        for k in &evicted {
            self.tiles.remove(k);
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("h3d-tile"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.kind.format(),
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * bpp),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&view),
        }];
        if let Some(s) = &self.sampler {
            entries.push(wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(s),
            });
        }
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("h3d-tile"),
            layout: &self.layout,
            entries: &entries,
        });
        self.tiles.insert(
            key,
            GpuTile {
                texture,
                view,
                bind_group,
                width,
                height,
            },
        );
        Ok(UploadStatus::Uploaded)
    }

    /// Hapus satu tile.
    pub fn remove(&mut self, key: &TileKey) -> bool {
        self.index.remove(key);
        self.tiles.remove(key).is_some()
    }

    /// Buang semua tile basi (tidak terlihat ≥ N frame). Mengembalikan jumlahnya.
    pub fn prune_stale(&mut self) -> usize {
        let stale = self.index.stale_keys();
        for k in &stale {
            self.remove(k);
        }
        stale.len()
    }

    /// Ubah anggaran byte.
    pub fn set_max_bytes(&mut self, max_bytes: u64) {
        for k in self.index.set_max_bytes(max_bytes) {
            self.tiles.remove(&k);
        }
    }

    /// Kosongkan.
    pub fn clear(&mut self) {
        self.index.clear();
        self.tiles.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(i: u32) -> TileKey {
        TileKey {
            slide: 0,
            level: 0,
            tx: i,
            ty: 0,
        }
    }

    #[test]
    fn lru_evicts_oldest_unseen_first() {
        let mut idx = TileCacheIndex::new(300, 10, 100);
        assert!(idx.insert(key(1), 100).unwrap().is_empty());
        idx.begin_frame();
        assert!(idx.insert(key(2), 100).unwrap().is_empty());
        idx.begin_frame();
        assert!(idx.insert(key(3), 100).unwrap().is_empty());
        assert_eq!(idx.used_bytes(), 300);
        // Frame baru: tile 1 paling lama tidak terlihat → dibuang dulu.
        idx.begin_frame();
        let ev = idx.insert(key(4), 100).unwrap();
        assert_eq!(ev, vec![key(1)]);
        assert!(!idx.contains(&key(1)));
        assert_eq!(idx.used_bytes(), 300);
    }

    #[test]
    fn touched_tiles_survive_and_touch_reorders() {
        let mut idx = TileCacheIndex::new(200, 10, 100);
        idx.insert(key(1), 100).unwrap();
        idx.begin_frame();
        idx.insert(key(2), 100).unwrap();
        idx.begin_frame();
        // Sentuh tile 1 (lebih tua) pada frame sebelumnya → tile 2 jadi yang terlama.
        assert!(idx.touch(&key(1)));
        idx.begin_frame();
        let ev = idx.insert(key(3), 100).unwrap();
        assert_eq!(ev, vec![key(2)]);
    }

    #[test]
    fn visible_this_frame_is_never_evicted() {
        let mut idx = TileCacheIndex::new(200, 10, 100);
        idx.insert(key(1), 100).unwrap();
        idx.insert(key(2), 100).unwrap();
        // Keduanya terlihat frame ini → tidak ada ruang.
        assert_eq!(
            idx.insert(key(3), 100),
            Err(InsertRejected::NoEvictableRoom)
        );
        assert_eq!(idx.len(), 2);
        idx.begin_frame();
        idx.touch(&key(2));
        let ev = idx.insert(key(3), 100).unwrap();
        assert_eq!(ev, vec![key(1)]);
    }

    #[test]
    fn rejects_too_large_and_duplicates() {
        let mut idx = TileCacheIndex::new(100, 10, 100);
        assert_eq!(idx.insert(key(1), 101), Err(InsertRejected::TooLarge));
        idx.insert(key(1), 50).unwrap();
        assert_eq!(idx.insert(key(1), 50), Err(InsertRejected::AlreadyPresent));
        assert_eq!(idx.used_bytes(), 50);
    }

    #[test]
    fn upload_budget_per_frame() {
        let mut idx = TileCacheIndex::new(1_000_000, 10, 2);
        assert!(idx.insert(key(1), 1).is_ok());
        assert!(idx.insert(key(2), 1).is_ok());
        assert_eq!(
            idx.insert(key(3), 1),
            Err(InsertRejected::UploadBudgetExhausted)
        );
        assert_eq!(idx.uploads_remaining(), 0);
        idx.begin_frame();
        assert!(idx.insert(key(3), 1).is_ok());
        assert_eq!(idx.uploads_remaining(), 1);
    }

    #[test]
    fn stale_keys_after_n_frames() {
        let mut idx = TileCacheIndex::new(1000, 3, 100);
        idx.insert(key(1), 1).unwrap();
        idx.begin_frame();
        idx.insert(key(2), 1).unwrap();
        idx.begin_frame();
        idx.begin_frame();
        // key1 tidak terlihat 3 frame → basi; key2 baru 2 frame.
        assert_eq!(idx.stale_keys(), vec![key(1)]);
        idx.begin_frame();
        assert_eq!(idx.stale_keys(), vec![key(1), key(2)]);
    }

    #[test]
    fn shrinking_budget_evicts_until_fit() {
        let mut idx = TileCacheIndex::new(300, 10, 100);
        idx.insert(key(1), 100).unwrap();
        idx.begin_frame();
        idx.insert(key(2), 100).unwrap();
        idx.begin_frame();
        idx.insert(key(3), 100).unwrap();
        let ev = idx.set_max_bytes(150);
        assert_eq!(ev, vec![key(1), key(2)]);
        assert_eq!(idx.used_bytes(), 100);
        idx.remove(&key(3));
        assert!(idx.is_empty());
    }

    #[test]
    fn tile_instance_is_32_bytes() {
        assert_eq!(std::mem::size_of::<TileInstance>(), 32);
        assert_eq!(tile_instance_layout().array_stride, 32);
    }
}
