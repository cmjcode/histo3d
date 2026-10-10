//! `SceneRenderer`: semua pipeline wgpu, sumber daya GPU, dan integrasi
//! paint-callback egui (`H3dPaintCallback`).
//!
//! **Yang dipanggil aplikasi per frame** (di luar callback, dengan
//! `device`/`queue` dari `WgpuRenderState` dan `SceneRenderer` dari
//! `callback_resources`):
//! 1. `renderer.begin_frame()` — reset kuota unggahan tile.
//! 2. `renderer.upload_tile(...)` / `upload_label_tile(...)` untuk tile yang
//!    sudah didekode; periksa `UploadStatus` (kuota habis → coba frame depan).
//! 3. Susun [`Scene`] hanya dengan tile yang `has_tile(...)`; tile yang belum
//!    ada diganti tile induk yang tersedia (pemilihan level di aplikasi).
//! 4. `ui.painter().add(H3dPaintCallback::new(scene).into_egui(rect))`.
//!
//! Matriks: 2D — `T_i` (U µm → W µm) di uniform lapisan, pandangan W µm → NDC
//! dihitung di shader dari `View2D`. 3D — `view_proj` dari `OrbitCamera`
//! (ruang R mm → clip); mesh sudah dalam mm sejak `upload_mesh`.

use std::collections::HashMap;

use glam::Vec3;
use h3d_core::Colormap;

use crate::camera::{OrbitCamera, View2D};
use crate::colormap::{lut_bind_group_layout, LutTexture, Palette256};
use crate::compare::{compare_passes, CompareParams, CompareSource, LayerMode};
use crate::error::RenderError;
use crate::mesh::{mesh_vertex_layout, sort_back_to_front, GpuMesh, MeshId, MeshInstance};
use crate::overlay::{create_overlay_pipeline, default_palette, LabelOverlayDraw};
use crate::slices3d::{slice_model_matrix, slice_texture_layout, SliceDraw, SliceId, SliceTexture};
use crate::tiles::{
    tile_instance_layout, GpuTileCache, TileDraw, TileInstance, TileKey, TileLayerDraw,
    TileTextureKind, UploadStatus,
};
use crate::wgpu;
use h3d_core::Affine2;

/// Anggaran render; angka byte **wajib** diambil dari `h3d_core::Budgets`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenderBudgets {
    /// Cache tile RGBA (byte).
    pub tile_cache_bytes: u64,
    /// Cache tile label (byte).
    pub label_cache_bytes: u64,
    /// Unggahan tile maksimum per frame (per cache).
    pub max_uploads_per_frame: usize,
    /// Tile dianggap basi setelah tak terlihat N frame.
    pub stale_after_frames: u64,
}

impl RenderBudgets {
    /// Dari anggaran byte yang sudah dibagi aplikasi; batas unggahan bawaan.
    pub fn new(tile_cache_bytes: u64, label_cache_bytes: u64) -> Self {
        Self {
            tile_cache_bytes,
            label_cache_bytes,
            max_uploads_per_frame: crate::tiles::DEFAULT_MAX_UPLOADS_PER_FRAME,
            stale_after_frames: crate::tiles::DEFAULT_STALE_AFTER_FRAMES,
        }
    }
}

/// Pencahayaan 3D: dua lampu arah + ambient.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lighting {
    /// Arah ke lampu utama (ruang R).
    pub dir0: [f32; 3],
    /// Intensitas lampu utama.
    pub intensity0: f32,
    /// Arah ke lampu pengisi.
    pub dir1: [f32; 3],
    /// Intensitas lampu pengisi.
    pub intensity1: f32,
    /// Ambient RGB.
    pub ambient: [f32; 3],
}

impl Default for Lighting {
    fn default() -> Self {
        Self {
            dir0: [0.4, 0.3, 0.85],
            intensity0: 0.8,
            dir1: [-0.6, -0.4, 0.3],
            intensity1: 0.35,
            ambient: [0.25, 0.25, 0.25],
        }
    }
}

/// Adegan 2D (viewer slide / pembanding registrasi).
#[derive(Clone, Debug, PartialEq)]
pub struct Scene2D {
    /// Pandangan (dari `Camera2D::view(rect)`).
    pub view: View2D,
    /// Lapisan tile dasar (biasanya `None` saat `compare` aktif).
    pub base: Option<TileLayerDraw>,
    /// Overlay label di atas dasar/pembanding.
    pub label_layers: Vec<LabelOverlayDraw>,
    /// Pembanding registrasi.
    pub compare: Option<CompareParams>,
}

/// Adegan 3D (rekonstruksi).
#[derive(Clone, Debug, PartialEq)]
pub struct Scene3D {
    /// Kamera orbit (mm, ruang R).
    pub camera: OrbitCamera,
    /// Rasio lebar/tinggi viewport.
    pub aspect: f32,
    /// Pusat volume (µm) yang dipakai saat `upload_mesh`; dipakai untuk irisan.
    pub center_um: [f64; 3],
    /// Mesh yang digambar.
    pub meshes: Vec<MeshInstance>,
    /// Irisan thumbnail di 3D.
    pub slices: Vec<SliceDraw>,
    /// Bidang potong `[nx, ny, nz, d]` (ruang R mm): fragmen dengan
    /// `n·p − d > 0` dibuang. `None` = tidak memotong.
    pub clip_plane: Option<[f32; 4]>,
    /// Pencahayaan.
    pub lighting: Lighting,
}

/// Snapshot adegan yang dibawa paint callback.
#[derive(Clone, Debug, PartialEq)]
pub enum Scene {
    /// Viewer 2D.
    TwoD(Scene2D),
    /// Viewport 3D.
    ThreeD(Scene3D),
}

// ---------- uniform ----------

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct LayerUniform {
    affine0: [f32; 4],
    affine1: [f32; 4],
    view: [f32; 4],
    viewport: [f32; 4],
    params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Globals3D {
    view_proj: [[f32; 4]; 4],
    eye: [f32; 4],
    light0: [f32; 4],
    light1: [f32; 4],
    ambient: [f32; 4],
    clip_plane: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MeshParamsUniform {
    color: [f32; 4],
    params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SliceParamsUniform {
    model: [[f32; 4]; 4],
    alpha: [f32; 4],
}

/// Bidang potong nonaktif: normal nol, d sangat besar → tidak pernah memotong.
const CLIP_DISABLED: [f32; 4] = [0.0, 0.0, 0.0, 1.0e30];

fn layer_uniform(
    affine: &Affine2,
    view: &View2D,
    opacity: f32,
    mode: u32,
    checker_px: f32,
    ppp: f32,
) -> LayerUniform {
    let [a, b, tx, c, d, ty] = affine.m;
    LayerUniform {
        affine0: [a as f32, b as f32, tx as f32, 0.0],
        affine1: [c as f32, d as f32, ty as f32, 0.0],
        view: [
            view.center_um[0],
            view.center_um[1],
            view.um_per_px.max(1e-9),
            0.0,
        ],
        viewport: [
            view.viewport_size_pt[0],
            view.viewport_size_pt[1],
            opacity.clamp(0.0, 1.0),
            mode as f32,
        ],
        params: [
            checker_px * ppp,
            view.viewport_min_pt[0] * ppp,
            view.viewport_min_pt[1] * ppp,
            0.0,
        ],
    }
}

// ---------- sumber daya per slot ----------

struct UniformSlot {
    buf: wgpu::Buffer,
    bind: wgpu::BindGroup,
}

fn ensure_uniform(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    pool: &mut Vec<UniformSlot>,
    idx: usize,
    size: u64,
    label: &str,
) {
    while pool.len() <= idx {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            }],
        });
        pool.push(UniformSlot { buf, bind });
    }
}

#[derive(Default)]
struct GrowBuffer {
    buf: Option<wgpu::Buffer>,
    capacity: u64,
}

impl GrowBuffer {
    fn write(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, bytes: &[u8]) {
        let needed = bytes.len() as u64;
        if self.buf.is_none() || self.capacity < needed {
            let capacity = needed.max(self.capacity * 2).max(256);
            self.buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("h3d-tile-instances"),
                size: capacity,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.capacity = capacity;
        }
        if let Some(buf) = &self.buf {
            if !bytes.is_empty() {
                queue.write_buffer(buf, 0, bytes);
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LayerKind {
    Tile,
    TileAdditive,
    Solid,
    Overlay,
}

struct LayerCmd {
    kind: LayerKind,
    ubo: usize,
    /// `(kunci tile, indeks instance)`.
    tiles: Vec<(TileKey, u32)>,
}

struct MeshCmd {
    mesh: MeshId,
    ubo: usize,
    colormap: Colormap,
}

struct SliceCmd {
    slice: SliceId,
    ubo: usize,
}

enum Prepared {
    None,
    TwoD(Vec<LayerCmd>),
    ThreeD {
        opaque: Vec<MeshCmd>,
        transparent: Vec<MeshCmd>,
        slices: Vec<SliceCmd>,
    },
}

struct FrameSlot {
    layer_ubos: Vec<UniformSlot>,
    instance_bufs: Vec<GrowBuffer>,
    globals: Vec<UniformSlot>,
    mesh_ubos: Vec<UniformSlot>,
    slice_ubos: Vec<UniformSlot>,
    prepared: Prepared,
}

impl FrameSlot {
    fn new() -> Self {
        Self {
            layer_ubos: Vec::new(),
            instance_bufs: Vec::new(),
            globals: Vec::new(),
            mesh_ubos: Vec::new(),
            slice_ubos: Vec::new(),
            prepared: Prepared::None,
        }
    }
}

/// Lapisan 2D yang sudah diurai, sebelum ditulis ke GPU.
struct PlannedLayer<'a> {
    kind: LayerKind,
    uniform: LayerUniform,
    tiles: &'a [TileDraw],
}

// ---------- renderer ----------

/// Renderer adegan yang hidup di `egui_wgpu::CallbackResources`.
pub struct SceneRenderer {
    depth_format: Option<wgpu::TextureFormat>,
    max_texture_dim: u32,

    layer_layout: wgpu::BindGroupLayout,
    lut_layout: wgpu::BindGroupLayout,
    globals_layout: wgpu::BindGroupLayout,
    mesh_params_layout: wgpu::BindGroupLayout,
    slice_params_layout: wgpu::BindGroupLayout,
    slice_tex_layout: wgpu::BindGroupLayout,
    slice_sampler: wgpu::Sampler,

    tile_pipeline: wgpu::RenderPipeline,
    tile_additive_pipeline: wgpu::RenderPipeline,
    solid_pipeline: wgpu::RenderPipeline,
    overlay_pipeline: wgpu::RenderPipeline,
    mesh_opaque_pipeline: wgpu::RenderPipeline,
    mesh_transparent_pipeline: wgpu::RenderPipeline,
    slice_pipeline: wgpu::RenderPipeline,

    /// Cache tile RGBA.
    pub tiles: GpuTileCache,
    /// Cache tile label (`R8Uint`).
    pub labels: GpuTileCache,
    palette: LutTexture,
    colormaps: HashMap<Colormap, LutTexture>,
    meshes: HashMap<MeshId, GpuMesh>,
    slices: HashMap<SliceId, SliceTexture>,

    slots: Vec<FrameSlot>,
}

fn uniform_layout(
    device: &wgpu::Device,
    label: &str,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

const ADDITIVE_BLEND: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
};

impl SceneRenderer {
    /// Buat renderer. `color_format` = format target egui
    /// (`WgpuRenderState::target_format`); `depth_format` = format depth yang
    /// dipasang eframe (`NativeOptions::depth_buffer` → `Some(Depth32Float)`),
    /// `None` bila tidak ada (mesh 3D hanya mengandalkan urutan gambar).
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        depth_format: Option<wgpu::TextureFormat>,
        budgets: RenderBudgets,
    ) -> Self {
        let max_texture_dim = device.limits().max_texture_dimension_2d;

        let layer_layout = uniform_layout(device, "h3d-layer", wgpu::ShaderStages::VERTEX_FRAGMENT);
        let lut_layout = lut_bind_group_layout(device);
        let globals_layout =
            uniform_layout(device, "h3d-globals3d", wgpu::ShaderStages::VERTEX_FRAGMENT);
        let mesh_params_layout =
            uniform_layout(device, "h3d-mesh-params", wgpu::ShaderStages::FRAGMENT);
        let slice_params_layout = uniform_layout(
            device,
            "h3d-slice-params",
            wgpu::ShaderStages::VERTEX_FRAGMENT,
        );
        let slice_tex_layout = slice_texture_layout(device);
        let slice_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("h3d-slice-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let tiles = GpuTileCache::new(
            device,
            TileTextureKind::Rgba,
            budgets.tile_cache_bytes,
            budgets.stale_after_frames,
            budgets.max_uploads_per_frame,
        );
        let labels = GpuTileCache::new(
            device,
            TileTextureKind::LabelU8,
            budgets.label_cache_bytes,
            budgets.stale_after_frames,
            budgets.max_uploads_per_frame,
        );

        // --- pipeline tile 2D ---
        let tile_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("h3d-tiles"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/tiles.wgsl").into()),
        });
        let tile_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("h3d-tiles"),
            bind_group_layouts: &[Some(&layer_layout), Some(tiles.layout())],
            immediate_size: 0,
        });
        let solid_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("h3d-solid"),
            bind_group_layouts: &[Some(&layer_layout)],
            immediate_size: 0,
        });
        // Pipeline 2D ikut render pass egui, yang bisa membawa depth attachment:
        // format harus cocok, tanpa tulis/uji depth.
        let depth_2d = depth_format.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: Default::default(),
            bias: Default::default(),
        });
        let make_tile_pipeline = |label: &str,
                                  layout: &wgpu::PipelineLayout,
                                  vs: &str,
                                  fs: &str,
                                  blend: wgpu::BlendState,
                                  instanced: bool| {
            let buffers: &[Option<wgpu::VertexBufferLayout>] = if instanced {
                &[Some(tile_instance_layout())]
            } else {
                &[]
            };
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &tile_shader,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers,
                },
                fragment: Some(wgpu::FragmentState {
                    module: &tile_shader,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: depth_2d.clone(),
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let tile_pipeline = make_tile_pipeline(
            "h3d-tile",
            &tile_layout,
            "vs_tile",
            "fs_tile",
            wgpu::BlendState::ALPHA_BLENDING,
            true,
        );
        let tile_additive_pipeline = make_tile_pipeline(
            "h3d-tile-additive",
            &tile_layout,
            "vs_tile",
            "fs_tile",
            ADDITIVE_BLEND,
            true,
        );
        let solid_pipeline = make_tile_pipeline(
            "h3d-solid",
            &solid_layout,
            "vs_fullscreen",
            "fs_solid",
            wgpu::BlendState::REPLACE,
            false,
        );

        let overlay_pipeline = create_overlay_pipeline(
            device,
            color_format,
            depth_2d,
            &layer_layout,
            labels.layout(),
            &lut_layout,
        );

        // --- pipeline mesh 3D ---
        let mesh_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("h3d-mesh"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mesh.wgsl").into()),
        });
        let mesh_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("h3d-mesh"),
            bind_group_layouts: &[
                Some(&globals_layout),
                Some(&mesh_params_layout),
                Some(&lut_layout),
            ],
            immediate_size: 0,
        });
        let depth_state = |write: bool| {
            depth_format.map(|format| wgpu::DepthStencilState {
                format,
                depth_write_enabled: Some(write),
                depth_compare: Some(if write {
                    wgpu::CompareFunction::Less
                } else {
                    wgpu::CompareFunction::LessEqual
                }),
                stencil: Default::default(),
                bias: Default::default(),
            })
        };
        let make_mesh_pipeline = |label: &str, write_depth: bool| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&mesh_layout),
                vertex: wgpu::VertexState {
                    module: &mesh_shader,
                    entry_point: Some("vs_mesh"),
                    compilation_options: Default::default(),
                    buffers: &[Some(mesh_vertex_layout())],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &mesh_shader,
                    entry_point: Some("fs_mesh"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: depth_state(write_depth),
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let mesh_opaque_pipeline = make_mesh_pipeline("h3d-mesh-opaque", true);
        let mesh_transparent_pipeline = make_mesh_pipeline("h3d-mesh-transparent", false);

        // --- pipeline irisan di 3D ---
        let slice_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("h3d-slices"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/slices.wgsl").into()),
        });
        let slice_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("h3d-slices"),
            bind_group_layouts: &[
                Some(&globals_layout),
                Some(&slice_params_layout),
                Some(&slice_tex_layout),
            ],
            immediate_size: 0,
        });
        let slice_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("h3d-slice"),
            layout: Some(&slice_layout),
            vertex: wgpu::VertexState {
                module: &slice_shader,
                entry_point: Some("vs_slice"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &slice_shader,
                entry_point: Some("fs_slice"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: depth_state(false),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let palette = LutTexture::from_palette(device, queue, &lut_layout, &default_palette());

        Self {
            depth_format,
            max_texture_dim,
            layer_layout,
            lut_layout,
            globals_layout,
            mesh_params_layout,
            slice_params_layout,
            slice_tex_layout,
            slice_sampler,
            tile_pipeline,
            tile_additive_pipeline,
            solid_pipeline,
            overlay_pipeline,
            mesh_opaque_pipeline,
            mesh_transparent_pipeline,
            slice_pipeline,
            tiles,
            labels,
            palette,
            colormaps: HashMap::new(),
            meshes: HashMap::new(),
            slices: HashMap::new(),
            slots: Vec::new(),
        }
    }

    /// Format depth yang dipakai pipeline 3D.
    pub fn depth_format(&self) -> Option<wgpu::TextureFormat> {
        self.depth_format
    }

    /// `device.limits().max_texture_dimension_2d`.
    pub fn max_texture_dimension(&self) -> u32 {
        self.max_texture_dim
    }

    // ----- per frame -----

    /// Mulai frame: reset kuota unggahan kedua cache tile.
    pub fn begin_frame(&mut self) {
        self.tiles.begin_frame();
        self.labels.begin_frame();
    }

    /// Unggah tile RGBA8 (`w × h × 4` byte).
    pub fn upload_tile(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        key: TileKey,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) -> Result<UploadStatus, RenderError> {
        self.tiles.upload(device, queue, key, width, height, rgba)
    }

    /// Unggah tile label `u8` (`w × h` byte).
    pub fn upload_label_tile(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        key: TileKey,
        width: u32,
        height: u32,
        labels: &[u8],
    ) -> Result<UploadStatus, RenderError> {
        self.labels
            .upload(device, queue, key, width, height, labels)
    }

    /// Benar bila tile RGBA ada di GPU.
    pub fn has_tile(&self, key: &TileKey) -> bool {
        self.tiles.contains(key)
    }

    /// Benar bila tile label ada di GPU.
    pub fn has_label_tile(&self, key: &TileKey) -> bool {
        self.labels.contains(key)
    }

    // ----- sumber daya -----

    /// Perbarui palet label (indeks → RGBA).
    pub fn set_palette(&self, queue: &wgpu::Queue, palette: &Palette256) {
        self.palette.write_palette(queue, palette);
    }

    /// Daftarkan mesh (lihat [`crate::mesh::upload_mesh`]); mengganti yang lama.
    pub fn insert_mesh(&mut self, id: MeshId, mesh: GpuMesh) -> Option<GpuMesh> {
        self.meshes.insert(id, mesh)
    }

    /// Hapus mesh.
    pub fn remove_mesh(&mut self, id: MeshId) -> Option<GpuMesh> {
        self.meshes.remove(&id)
    }

    /// Mesh terdaftar (mis. untuk picking).
    pub fn mesh(&self, id: MeshId) -> Option<&GpuMesh> {
        self.meshes.get(&id)
    }

    /// Daftar id mesh (urutan tetap).
    pub fn mesh_ids(&self) -> Vec<MeshId> {
        let mut v: Vec<MeshId> = self.meshes.keys().copied().collect();
        v.sort();
        v
    }

    /// Unggah thumbnail irisan RGBA8 untuk tampilan 3D.
    pub fn upload_slice(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: SliceId,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) -> Result<(), RenderError> {
        let tex = SliceTexture::upload(
            device,
            queue,
            &self.slice_tex_layout,
            &self.slice_sampler,
            width,
            height,
            rgba,
            self.max_texture_dim,
        )?;
        self.slices.insert(id, tex);
        Ok(())
    }

    /// Hapus thumbnail irisan.
    pub fn remove_slice(&mut self, id: SliceId) -> bool {
        self.slices.remove(&id).is_some()
    }

    /// Benar bila thumbnail irisan ada.
    pub fn has_slice(&self, id: SliceId) -> bool {
        self.slices.contains_key(&id)
    }

    fn ensure_colormap(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, cm: Colormap) {
        if !self.colormaps.contains_key(&cm) {
            let lut = LutTexture::from_colormap(device, queue, &self.lut_layout, cm);
            self.colormaps.insert(cm, lut);
        }
    }

    fn ensure_slot(&mut self, slot: usize) {
        while self.slots.len() <= slot {
            self.slots.push(FrameSlot::new());
        }
    }

    // ----- prepare / paint -----

    /// Tulis uniform dan susun perintah gambar untuk `scene` pada `slot`.
    /// Dipanggil dari `CallbackTrait::prepare`. Slot berbeda untuk viewport
    /// berbeda dalam satu frame.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen: &egui_wgpu::ScreenDescriptor,
        slot: usize,
        scene: &Scene,
    ) {
        self.ensure_slot(slot);
        match scene {
            Scene::TwoD(s) => self.prepare_2d(device, queue, screen.pixels_per_point, slot, s),
            Scene::ThreeD(s) => self.prepare_3d(device, queue, slot, s),
        }
    }

    fn prepare_2d(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        ppp: f32,
        slot: usize,
        scene: &Scene2D,
    ) {
        let view = &scene.view;
        let mut planned: Vec<PlannedLayer<'_>> = Vec::new();
        if let Some(base) = &scene.base {
            planned.push(PlannedLayer {
                kind: LayerKind::Tile,
                uniform: layer_uniform(
                    &base.affine,
                    view,
                    base.opacity,
                    LayerMode::Normal as u32,
                    0.0,
                    ppp,
                ),
                tiles: &base.tiles,
            });
        }
        if let Some(cmp) = &scene.compare {
            for pass in compare_passes(cmp.mode, cmp.fixed.opacity, cmp.moving.opacity) {
                let (layer, kind): (Option<&TileLayerDraw>, LayerKind) = match pass.source {
                    CompareSource::Black => (None, LayerKind::Solid),
                    CompareSource::Fixed => (Some(&cmp.fixed), LayerKind::Tile),
                    CompareSource::Moving => (
                        Some(&cmp.moving),
                        if pass.mode == LayerMode::TintMoving {
                            LayerKind::TileAdditive
                        } else {
                            LayerKind::Tile
                        },
                    ),
                };
                let affine = layer.map(|l| l.affine).unwrap_or(Affine2::IDENTITY);
                planned.push(PlannedLayer {
                    kind,
                    uniform: layer_uniform(
                        &affine,
                        view,
                        pass.opacity,
                        pass.mode as u32,
                        pass.checker_px,
                        ppp,
                    ),
                    tiles: layer.map(|l| l.tiles.as_slice()).unwrap_or(&[]),
                });
            }
        }
        for lbl in &scene.label_layers {
            planned.push(PlannedLayer {
                kind: LayerKind::Overlay,
                uniform: layer_uniform(
                    &lbl.affine,
                    view,
                    lbl.opacity,
                    u32::from(lbl.outline),
                    0.0,
                    ppp,
                ),
                tiles: &lbl.tiles,
            });
        }

        let mut cmds = Vec::with_capacity(planned.len());
        let fs = &mut self.slots[slot];
        for (li, layer) in planned.iter().enumerate() {
            ensure_uniform(
                device,
                &self.layer_layout,
                &mut fs.layer_ubos,
                li,
                std::mem::size_of::<LayerUniform>() as u64,
                "h3d-layer",
            );
            queue.write_buffer(
                &fs.layer_ubos[li].buf,
                0,
                bytemuck::bytes_of(&layer.uniform),
            );
            let mut instances: Vec<TileInstance> = Vec::new();
            let mut tiles: Vec<(TileKey, u32)> = Vec::new();
            let cache = if layer.kind == LayerKind::Overlay {
                &mut self.labels
            } else {
                &mut self.tiles
            };
            for t in layer.tiles {
                if cache.touch(&t.key) {
                    tiles.push((t.key, instances.len() as u32));
                    instances.push(TileInstance {
                        rect: t.rect_um,
                        uv: t.uv,
                    });
                }
            }
            while fs.instance_bufs.len() <= li {
                fs.instance_bufs.push(GrowBuffer::default());
            }
            if !instances.is_empty() {
                fs.instance_bufs[li].write(device, queue, bytemuck::cast_slice(&instances));
            }
            cmds.push(LayerCmd {
                kind: layer.kind,
                ubo: li,
                tiles,
            });
        }
        fs.prepared = Prepared::TwoD(cmds);
    }

    fn prepare_3d(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        slot: usize,
        scene: &Scene3D,
    ) {
        for inst in &scene.meshes {
            self.ensure_colormap(device, queue, inst.colormap);
        }
        let view_proj = scene.camera.view_proj(scene.aspect.max(0.01));
        let eye = scene.camera.eye();
        let l = &scene.lighting;
        let n0 = Vec3::from(l.dir0).normalize_or_zero();
        let n1 = Vec3::from(l.dir1).normalize_or_zero();
        let globals = Globals3D {
            view_proj: view_proj.to_cols_array_2d(),
            eye: [eye.x, eye.y, eye.z, 1.0],
            light0: [n0.x, n0.y, n0.z, l.intensity0],
            light1: [n1.x, n1.y, n1.z, l.intensity1],
            ambient: [l.ambient[0], l.ambient[1], l.ambient[2], 0.0],
            clip_plane: scene.clip_plane.unwrap_or(CLIP_DISABLED),
        };

        let fs = &mut self.slots[slot];
        ensure_uniform(
            device,
            &self.globals_layout,
            &mut fs.globals,
            0,
            std::mem::size_of::<Globals3D>() as u64,
            "h3d-globals3d",
        );
        queue.write_buffer(&fs.globals[0].buf, 0, bytemuck::bytes_of(&globals));

        let mut opaque = Vec::new();
        let mut transparent: Vec<(MeshCmd, [f32; 3])> = Vec::new();
        let mut n = 0usize;
        for inst in scene.meshes.iter().filter(|m| m.visible) {
            let Some(mesh) = self.meshes.get(&inst.mesh) else {
                continue;
            };
            ensure_uniform(
                device,
                &self.mesh_params_layout,
                &mut fs.mesh_ubos,
                n,
                std::mem::size_of::<MeshParamsUniform>() as u64,
                "h3d-mesh-params",
            );
            let opacity = inst.opacity.clamp(0.0, 1.0);
            let u = MeshParamsUniform {
                color: inst.color,
                params: [
                    opacity,
                    if inst.scalar_mode { 1.0 } else { 0.0 },
                    inst.scalar_range[0],
                    inst.scalar_range[1],
                ],
            };
            queue.write_buffer(&fs.mesh_ubos[n].buf, 0, bytemuck::bytes_of(&u));
            let cmd = MeshCmd {
                mesh: inst.mesh,
                ubo: n,
                colormap: inst.colormap,
            };
            if opacity >= 1.0 {
                opaque.push(cmd);
            } else {
                transparent.push((cmd, mesh.bounds_center_mm()));
            }
            n += 1;
        }
        let centers: Vec<[f32; 3]> = transparent.iter().map(|(_, c)| *c).collect();
        let order = sort_back_to_front(&centers, eye);
        let mut transparent_cmds: Vec<Option<MeshCmd>> =
            transparent.into_iter().map(|(c, _)| Some(c)).collect();
        let transparent: Vec<MeshCmd> = order
            .into_iter()
            .filter_map(|i| transparent_cmds[i].take())
            .collect();

        let mut slices = Vec::new();
        for (si, d) in scene.slices.iter().enumerate() {
            if !self.slices.contains_key(&d.slice) {
                continue;
            }
            ensure_uniform(
                device,
                &self.slice_params_layout,
                &mut fs.slice_ubos,
                si,
                std::mem::size_of::<SliceParamsUniform>() as u64,
                "h3d-slice-params",
            );
            let u = SliceParamsUniform {
                model: slice_model_matrix(&d.affine, d.size_um, d.z_um, scene.center_um)
                    .to_cols_array_2d(),
                alpha: [d.alpha.clamp(0.0, 1.0), 0.0, 0.0, 0.0],
            };
            queue.write_buffer(&fs.slice_ubos[si].buf, 0, bytemuck::bytes_of(&u));
            slices.push(SliceCmd {
                slice: d.slice,
                ubo: si,
            });
        }

        fs.prepared = Prepared::ThreeD {
            opaque,
            transparent,
            slices,
        };
    }

    /// Gambar perintah yang disiapkan `prepare` pada `slot`. Dipanggil dari
    /// `CallbackTrait::paint`; egui sudah mengatur viewport dan scissor.
    pub fn paint(&self, slot: usize, rpass: &mut wgpu::RenderPass<'_>) {
        let Some(fs) = self.slots.get(slot) else {
            return;
        };
        match &fs.prepared {
            Prepared::None => {}
            Prepared::TwoD(cmds) => self.paint_2d(fs, cmds, rpass),
            Prepared::ThreeD {
                opaque,
                transparent,
                slices,
            } => self.paint_3d(fs, opaque, transparent, slices, rpass),
        }
    }

    fn paint_2d(&self, fs: &FrameSlot, cmds: &[LayerCmd], rpass: &mut wgpu::RenderPass<'_>) {
        for cmd in cmds {
            rpass.set_bind_group(0, &fs.layer_ubos[cmd.ubo].bind, &[]);
            if cmd.kind == LayerKind::Solid {
                rpass.set_pipeline(&self.solid_pipeline);
                rpass.draw(0..6, 0..1);
                continue;
            }
            if cmd.tiles.is_empty() {
                continue;
            }
            let Some(ibuf) = fs.instance_bufs.get(cmd.ubo).and_then(|b| b.buf.as_ref()) else {
                continue;
            };
            rpass.set_vertex_buffer(0, ibuf.slice(..));
            let cache = match cmd.kind {
                LayerKind::Tile => {
                    rpass.set_pipeline(&self.tile_pipeline);
                    &self.tiles
                }
                LayerKind::TileAdditive => {
                    rpass.set_pipeline(&self.tile_additive_pipeline);
                    &self.tiles
                }
                LayerKind::Overlay => {
                    rpass.set_pipeline(&self.overlay_pipeline);
                    rpass.set_bind_group(2, &self.palette.bind_group, &[]);
                    &self.labels
                }
                LayerKind::Solid => unreachable!("ditangani di atas"),
            };
            for (key, i) in &cmd.tiles {
                if let Some(tile) = cache.get(key) {
                    rpass.set_bind_group(1, &tile.bind_group, &[]);
                    rpass.draw(0..6, *i..*i + 1);
                }
            }
        }
    }

    fn paint_meshes(&self, fs: &FrameSlot, cmds: &[MeshCmd], rpass: &mut wgpu::RenderPass<'_>) {
        for cmd in cmds {
            let (Some(mesh), Some(cm)) = (
                self.meshes.get(&cmd.mesh),
                self.colormaps.get(&cmd.colormap),
            ) else {
                continue;
            };
            rpass.set_bind_group(1, &fs.mesh_ubos[cmd.ubo].bind, &[]);
            rpass.set_bind_group(2, &cm.bind_group, &[]);
            rpass.set_vertex_buffer(0, mesh.vbuf.slice(..));
            rpass.set_index_buffer(mesh.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..mesh.index_count, 0, 0..1);
        }
    }

    fn paint_3d(
        &self,
        fs: &FrameSlot,
        opaque: &[MeshCmd],
        transparent: &[MeshCmd],
        slices: &[SliceCmd],
        rpass: &mut wgpu::RenderPass<'_>,
    ) {
        let Some(globals) = fs.globals.first() else {
            return;
        };
        rpass.set_bind_group(0, &globals.bind, &[]);

        // 1. Mesh opak (tulis depth).
        rpass.set_pipeline(&self.mesh_opaque_pipeline);
        self.paint_meshes(fs, opaque, rpass);

        // 2. Irisan (depth test, tanpa tulis depth).
        if !slices.is_empty() {
            rpass.set_pipeline(&self.slice_pipeline);
            for cmd in slices {
                let Some(tex) = self.slices.get(&cmd.slice) else {
                    continue;
                };
                rpass.set_bind_group(1, &fs.slice_ubos[cmd.ubo].bind, &[]);
                rpass.set_bind_group(2, &tex.bind_group, &[]);
                rpass.draw(0..6, 0..1);
            }
        }

        // 3. Mesh tembus pandang, belakang ke depan, tanpa tulis depth.
        if !transparent.is_empty() {
            rpass.set_pipeline(&self.mesh_transparent_pipeline);
            self.paint_meshes(fs, transparent, rpass);
        }
    }
}

/// Paint callback egui yang membawa snapshot adegan dan menggambarnya ke
/// viewport. `SceneRenderer` harus sudah dimasukkan ke
/// `renderer.callback_resources` saat aplikasi mulai.
pub struct H3dPaintCallback {
    /// Snapshot adegan.
    pub scene: Scene,
    /// Slot sumber daya per-viewport; viewport berbeda dalam satu frame harus
    /// memakai slot berbeda.
    pub slot: usize,
}

impl H3dPaintCallback {
    /// Callback untuk slot 0.
    pub fn new(scene: Scene) -> Self {
        Self { scene, slot: 0 }
    }

    /// Callback untuk slot tertentu.
    pub fn with_slot(scene: Scene, slot: usize) -> Self {
        Self { scene, slot }
    }

    /// Bungkus menjadi `egui::PaintCallback` untuk `ui.painter().add(...)`.
    pub fn into_egui(self, rect: egui::Rect) -> egui::PaintCallback {
        egui_wgpu::Callback::new_paint_callback(rect, self)
    }
}

impl egui_wgpu::CallbackTrait for H3dPaintCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(renderer) = callback_resources.get_mut::<SceneRenderer>() {
            renderer.prepare(device, queue, screen_descriptor, self.slot, &self.scene);
        } else {
            log::warn!("SceneRenderer belum terdaftar di CallbackResources");
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        if let Some(renderer) = callback_resources.get::<SceneRenderer>() {
            renderer.paint(self.slot, render_pass);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate_wgsl(name: &str, src: &str) {
        use wgpu::naga;
        let module = naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|e| panic!("{name}: gagal parse WGSL: {}", e.emit_to_string(src)));
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: validasi WGSL gagal: {e:?}"));
    }

    #[test]
    fn all_shaders_parse_and_validate() {
        validate_wgsl("tiles.wgsl", include_str!("shaders/tiles.wgsl"));
        validate_wgsl("overlay.wgsl", include_str!("shaders/overlay.wgsl"));
        validate_wgsl("mesh.wgsl", include_str!("shaders/mesh.wgsl"));
        validate_wgsl("slices.wgsl", include_str!("shaders/slices.wgsl"));
    }

    #[test]
    fn uniform_sizes_are_16_byte_multiples() {
        assert_eq!(std::mem::size_of::<LayerUniform>(), 80);
        assert_eq!(std::mem::size_of::<Globals3D>(), 144);
        assert_eq!(std::mem::size_of::<MeshParamsUniform>(), 32);
        assert_eq!(std::mem::size_of::<SliceParamsUniform>(), 80);
    }

    #[test]
    fn layer_uniform_packs_affine_rows_and_view() {
        let t = Affine2 {
            m: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let view = View2D {
            center_um: [100.0, 200.0],
            um_per_px: 0.5,
            viewport_min_pt: [10.0, 20.0],
            viewport_size_pt: [800.0, 600.0],
        };
        let u = layer_uniform(&t, &view, 0.5, LayerMode::CheckerOdd as u32, 32.0, 2.0);
        assert_eq!(u.affine0, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(u.affine1, [4.0, 5.0, 6.0, 0.0]);
        assert_eq!(u.view, [100.0, 200.0, 0.5, 0.0]);
        assert_eq!(u.viewport, [800.0, 600.0, 0.5, 2.0]);
        // Ukuran kotak dan asal viewport dikalikan pixels_per_point.
        assert_eq!(u.params, [64.0, 20.0, 40.0, 0.0]);
    }

    #[test]
    fn clip_plane_disabled_never_cuts() {
        // dot(0, p) − 1e30 < 0 untuk semua p berhingga.
        let p = [1.0e6f32, -1.0e6, 1.0e6];
        let v = CLIP_DISABLED[0] * p[0] + CLIP_DISABLED[1] * p[1] + CLIP_DISABLED[2] * p[2]
            - CLIP_DISABLED[3];
        assert!(v < 0.0);
    }
}
