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
use crate::mesh::{
    mesh_vertex_layout, sort_back_to_front, GpuMesh, MeshId, MeshInstance, MeshMaterial,
};
use crate::overlay::{create_overlay_pipeline, default_palette, LabelOverlayDraw};
use crate::slices3d::{
    slice_model_matrix, slice_model_matrix_scaled, slice_texture_layout, SliceDraw, SliceId,
    SliceTexture,
};
use crate::tiles::{
    tile_instance_layout, GpuTileCache, TileDraw, TileInstance, TileKey, TileLayerDraw, TileMask,
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
    /// Ambient RGB dari "langit" (permukaan menghadap +Z).
    pub ambient: [f32; 3],
    /// Ambient RGB dari "tanah" (permukaan menghadap −Z); sama dengan
    /// `ambient` = ambient seragam.
    pub ground: [f32; 3],
}

impl Default for Lighting {
    fn default() -> Self {
        Self {
            dir0: [0.4, 0.3, 0.85],
            intensity0: 0.8,
            dir1: [-0.6, -0.4, 0.3],
            intensity1: 0.35,
            ambient: [0.25, 0.25, 0.25],
            ground: [0.25, 0.25, 0.25],
        }
    }
}

impl Lighting {
    /// Pencahayaan studio untuk tampilan ilustrasi jaringan: lampu kunci
    /// hangat, pengisi dingin, ambient langit kebiruan dan tanah keunguan.
    pub fn studio() -> Self {
        Self {
            dir0: [0.45, 0.35, 0.82],
            intensity0: 0.85,
            dir1: [-0.7, -0.3, 0.25],
            intensity1: 0.4,
            ambient: [0.34, 0.36, 0.44],
            ground: [0.20, 0.14, 0.22],
        }
    }
}

/// Grid tak terbatas di bidang horizontal `z = z_mm` (ruang R, mm, setelah
/// `z_scale`). Jarak garis menyesuaikan zoom berjenjang kelipatan 10
/// (… 0,1 / 1 / 10 mm …) dengan peralihan halus; garis menerus sampai
/// cakrawala lalu memudar. Hanya hiasan tampilan, tidak pernah dipakai untuk
/// pengukuran.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grid3D {
    /// Tinggi bidang grid (mm, ruang R berenggang).
    pub z_mm: f32,
    /// Jarak garis terkecil yang masih digambar (piksel fisik); jenjang yang
    /// lebih rapat dipudarkan.
    pub min_cell_px: f32,
    /// Jarak pudar penuh, kelipatan jarak kamera ke target.
    pub fade_distance: f32,
    /// Warna garis minor (RGBA linear, alfa = kekuatan).
    pub minor: [f32; 4],
    /// Warna garis mayor.
    pub major: [f32; 4],
    /// Warna sumbu X (garis `y = 0`).
    pub axis_x: [f32; 4],
    /// Warna sumbu Y (garis `x = 0`).
    pub axis_y: [f32; 4],
}

impl Default for Grid3D {
    fn default() -> Self {
        Self {
            z_mm: 0.0,
            min_cell_px: 14.0,
            fade_distance: 40.0,
            minor: [0.55, 0.55, 0.58, 0.22],
            major: [0.65, 0.65, 0.68, 0.50],
            axis_x: [0.85, 0.30, 0.30, 0.75],
            axis_y: [0.40, 0.75, 0.35, 0.75],
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
    /// Bidang potong `[nx, ny, nz, d]` (ruang R mm, **setelah** `z_scale`):
    /// fragmen dengan `n·p − d > 0` dibuang. `None` = tidak memotong.
    pub clip_plane: Option<[f32; 4]>,
    /// Pencahayaan.
    pub lighting: Lighting,
    /// Faktor renggang sumbu Z, hanya untuk tampilan: `r.z` mesh dan irisan
    /// dikalikan nilai ini di shader (ruang R berpusat di `center_um`, jadi
    /// perenggangan terjadi mengelilingi pusat volume). `1.0` = skala
    /// sebenarnya. Tidak pernah memengaruhi pengukuran.
    pub z_scale: f32,
    /// Grid tak terbatas; `None` = tidak digambar.
    pub grid: Option<Grid3D>,
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
    /// `(lebar irisan µm, tinggi irisan µm, masker aktif 0/1, 0)`.
    mask: [f32; 4],
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
    /// `x` = faktor renggang Z (lihat [`Scene3D::z_scale`]).
    params: [f32; 4],
    /// `rgb` = ambient tanah.
    ground: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MeshParamsUniform {
    color: [f32; 4],
    params: [f32; 4],
    /// `xyz` = geseran tampilan (mm, ruang R, setelah renggang Z), `w` = sorot 0..1.
    offset: [f32; 4],
    /// `(kilau, eksponen, difus terbungkus, oklusi)`.
    material: [f32; 4],
    /// `rgb` = warna rim, `w` = kekuatan.
    rim: [f32; 4],
    /// `(ukuran sel mm, kekuatan, campuran inti, warna pulasan 0/1)`.
    cells: [f32; 4],
}

/// Uniform material dari [`MeshMaterial`]; nilai tak finit → bawaan aman.
fn material_uniform(m: &MeshMaterial) -> ([f32; 4], [f32; 4], [f32; 4]) {
    let f = |v: f32, lo: f32, hi: f32, d: f32| if v.is_finite() { v.clamp(lo, hi) } else { d };
    let material = [
        f(m.specular, 0.0, 1.0, 0.0),
        f(m.shininess, 1.0, 512.0, 32.0),
        f(m.wrap, 0.0, 1.0, 0.0),
        f(m.occlusion, 0.0, 1.0, 0.0),
    ];
    let rim = [
        f(m.rim_color[0], 0.0, 1.0, 1.0),
        f(m.rim_color[1], 0.0, 1.0, 1.0),
        f(m.rim_color[2], 0.0, 1.0, 1.0),
        f(m.rim, 0.0, 1.0, 0.0),
    ];
    let (size, strength, nuc) = match m.cells {
        Some(c) if c.size_mm.is_finite() && c.size_mm > 0.0 => (
            c.size_mm,
            f(c.strength, 0.0, 1.0, 0.0),
            f(c.nucleus_mix, 0.0, 1.0, 0.0),
        ),
        _ => (0.0, 0.0, 0.0),
    };
    // `w` > 0,5 = mode warna pulasan; nilainya = penguat saturasi.
    let stain = if m.stain_colors {
        f(m.stain_saturation, 0.6, 3.0, 1.0)
    } else {
        0.0
    };
    let cells = [size, strength, nuc, stain];
    (material, rim, cells)
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SliceParamsUniform {
    model: [[f32; 4]; 4],
    alpha: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GridUniform {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    /// `xyz` = mata (mm), `w` = jarak kamera ke target (mm).
    eye: [f32; 4],
    /// `(z_mm, min_cell_px, fade_distance, 0)`.
    params: [f32; 4],
    minor: [f32; 4],
    major: [f32; 4],
    axis_x: [f32; 4],
    axis_y: [f32; 4],
}

/// Uniform grid dari kamera; `None` bila matriks pandang-proyeksi tidak
/// dapat dibalik atau parameter tidak finit (grid lalu dilewati).
fn grid_uniform(camera: &OrbitCamera, aspect: f32, grid: &Grid3D) -> Option<GridUniform> {
    let view_proj = camera.view_proj(aspect.max(0.01));
    // Invers dihitung dalam f64: rasio near/far kamera (1 : 100 000) membuat
    // invers f32 kehilangan presisi di translasi.
    let vp64 = view_proj.as_dmat4();
    let det = vp64.determinant();
    if !det.is_finite() || det.abs() < f64::MIN_POSITIVE {
        return None;
    }
    let inv = vp64.inverse().as_mat4();
    let eye = camera.eye();
    let params = [
        grid.z_mm,
        grid.min_cell_px.max(1.0),
        grid.fade_distance.max(1.0),
        0.0,
    ];
    let finite = inv.is_finite() && eye.is_finite() && params.iter().all(|v| v.is_finite());
    finite.then(|| GridUniform {
        view_proj: view_proj.to_cols_array_2d(),
        inv_view_proj: inv.to_cols_array_2d(),
        eye: [eye.x, eye.y, eye.z, camera.distance],
        params,
        minor: grid.minor,
        major: grid.major,
        axis_x: grid.axis_x,
        axis_y: grid.axis_y,
    })
}

/// Bidang potong nonaktif: normal nol, d sangat besar → tidak pernah memotong.
const CLIP_DISABLED: [f32; 4] = [0.0, 0.0, 0.0, 1.0e30];

/// Faktor renggang Z yang aman untuk shader: nilai tak hingga, NaN, nol,
/// atau negatif jatuh ke `1.0` (skala sebenarnya).
pub fn sanitize_z_scale(z_scale: f32) -> f32 {
    if z_scale.is_finite() && z_scale > 0.0 {
        z_scale
    } else {
        1.0
    }
}

/// Parameter masker untuk uniform: nonaktif bila `None` atau ukuran tidak
/// sah (tidak finit / tidak positif).
fn mask_params(mask: Option<&TileMask>) -> [f32; 4] {
    match mask {
        Some(m) if m.size_um.iter().all(|v| v.is_finite() && *v > 0.0) => {
            [m.size_um[0] as f32, m.size_um[1] as f32, 1.0, 0.0]
        }
        _ => [1.0, 1.0, 0.0, 0.0],
    }
}

fn layer_uniform(
    affine: &Affine2,
    view: &View2D,
    opacity: f32,
    mode: u32,
    checker_px: f32,
    ppp: f32,
    mask: Option<&TileMask>,
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
        mask: mask_params(mask),
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
    /// Tekstur masker jaringan (grup 2) bila ada di `tile_masks`.
    mask: Option<SliceId>,
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
        /// Grid tak terbatas siap digambar (uniform di `FrameSlot::grid`).
        grid: bool,
    },
}

struct FrameSlot {
    layer_ubos: Vec<UniformSlot>,
    instance_bufs: Vec<GrowBuffer>,
    globals: Vec<UniformSlot>,
    mesh_ubos: Vec<UniformSlot>,
    slice_ubos: Vec<UniformSlot>,
    grid: Vec<UniformSlot>,
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
            grid: Vec::new(),
            prepared: Prepared::None,
        }
    }
}

/// Lapisan 2D yang sudah diurai, sebelum ditulis ke GPU.
struct PlannedLayer<'a> {
    kind: LayerKind,
    uniform: LayerUniform,
    tiles: &'a [TileDraw],
    mask: Option<SliceId>,
}

// ---------- renderer ----------

/// Renderer adegan yang hidup di `egui_wgpu::CallbackResources`.
pub struct SceneRenderer {
    depth_format: Option<wgpu::TextureFormat>,
    sample_count: u32,
    max_texture_dim: u32,

    layer_layout: wgpu::BindGroupLayout,
    lut_layout: wgpu::BindGroupLayout,
    globals_layout: wgpu::BindGroupLayout,
    mesh_params_layout: wgpu::BindGroupLayout,
    slice_params_layout: wgpu::BindGroupLayout,
    slice_tex_layout: wgpu::BindGroupLayout,
    slice_sampler: wgpu::Sampler,
    grid_layout: wgpu::BindGroupLayout,

    tile_pipeline: wgpu::RenderPipeline,
    tile_additive_pipeline: wgpu::RenderPipeline,
    solid_pipeline: wgpu::RenderPipeline,
    overlay_pipeline: wgpu::RenderPipeline,
    mesh_opaque_pipeline: wgpu::RenderPipeline,
    mesh_transparent_pipeline: wgpu::RenderPipeline,
    slice_pipeline: wgpu::RenderPipeline,
    grid_pipeline: wgpu::RenderPipeline,

    /// Cache tile RGBA.
    pub tiles: GpuTileCache,
    /// Cache tile label (`R8Uint`).
    pub labels: GpuTileCache,
    palette: LutTexture,
    colormaps: HashMap<Colormap, LutTexture>,
    meshes: HashMap<MeshId, GpuMesh>,
    slices: HashMap<SliceId, SliceTexture>,
    /// Masker jaringan lapisan tile 2D (alfa), per irisan.
    tile_masks: HashMap<SliceId, SliceTexture>,
    /// Masker 1×1 opak: dipasang di grup 2 bila lapisan tanpa masker
    /// (shader tetap mengabaikannya lewat `Layer.mask.z`).
    mask_none: SliceTexture,

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
    /// `sample_count` = jumlah sampel MSAA render pass egui
    /// (`NativeOptions::multisampling`, minimal 1); semua pipeline memakai
    /// nilai yang sama dengan render pass.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        depth_format: Option<wgpu::TextureFormat>,
        sample_count: u32,
        budgets: RenderBudgets,
    ) -> Self {
        let max_texture_dim = device.limits().max_texture_dimension_2d;
        let sample_count = sample_count.max(1);
        let multisample = wgpu::MultisampleState {
            count: sample_count,
            mask: !0,
            alpha_to_coverage_enabled: false,
        };

        let layer_layout = uniform_layout(device, "h3d-layer", wgpu::ShaderStages::VERTEX_FRAGMENT);
        let lut_layout = lut_bind_group_layout(device);
        let globals_layout =
            uniform_layout(device, "h3d-globals3d", wgpu::ShaderStages::VERTEX_FRAGMENT);
        let mesh_params_layout = uniform_layout(
            device,
            "h3d-mesh-params",
            wgpu::ShaderStages::VERTEX_FRAGMENT,
        );
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
            bind_group_layouts: &[
                Some(&layer_layout),
                Some(tiles.layout()),
                Some(&slice_tex_layout),
            ],
            immediate_size: 0,
        });
        let mask_none = SliceTexture::upload(
            device,
            queue,
            &slice_tex_layout,
            &slice_sampler,
            1,
            1,
            &[255, 255, 255, 255],
            max_texture_dim,
        )
        .expect("tekstur 1×1 selalu muat");
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
                multisample,
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
            multisample,
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
                multisample,
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
            multisample,
            multiview_mask: None,
            cache: None,
        });

        // --- grid tak terbatas ---
        // Diuji depth tanpa menulisnya: mesh opak menutupi grid, grid tidak
        // menutupi irisan/mesh tembus pandang yang digambar sesudahnya.
        let grid_layout = uniform_layout(device, "h3d-grid", wgpu::ShaderStages::VERTEX_FRAGMENT);
        let grid_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("h3d-grid"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/grid.wgsl").into()),
        });
        let grid_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("h3d-grid"),
            bind_group_layouts: &[Some(&grid_layout)],
            immediate_size: 0,
        });
        let grid_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("h3d-grid"),
            layout: Some(&grid_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &grid_shader,
                entry_point: Some("vs_grid"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &grid_shader,
                entry_point: Some("fs_grid"),
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
            multisample,
            multiview_mask: None,
            cache: None,
        });

        let palette = LutTexture::from_palette(device, queue, &lut_layout, &default_palette());

        Self {
            depth_format,
            sample_count,
            max_texture_dim,
            layer_layout,
            lut_layout,
            globals_layout,
            mesh_params_layout,
            slice_params_layout,
            slice_tex_layout,
            slice_sampler,
            grid_layout,
            tile_pipeline,
            tile_additive_pipeline,
            solid_pipeline,
            overlay_pipeline,
            mesh_opaque_pipeline,
            mesh_transparent_pipeline,
            slice_pipeline,
            grid_pipeline,
            tiles,
            labels,
            palette,
            colormaps: HashMap::new(),
            meshes: HashMap::new(),
            slices: HashMap::new(),
            tile_masks: HashMap::new(),
            mask_none,
            slots: Vec::new(),
        }
    }

    /// Format depth yang dipakai pipeline 3D.
    pub fn depth_format(&self) -> Option<wgpu::TextureFormat> {
        self.depth_format
    }

    /// Jumlah sampel MSAA semua pipeline.
    pub fn sample_count(&self) -> u32 {
        self.sample_count
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

    /// Unggah masker jaringan lapisan tile 2D: citra RGBA8 `width × height`
    /// yang menutupi seluruh irisan, kanal alfa = jaringan (lihat
    /// [`TileMask`]). Masker lama dengan `id` sama diganti.
    pub fn upload_tile_mask(
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
        self.tile_masks.insert(id, tex);
        Ok(())
    }

    /// Hapus masker jaringan lapisan tile.
    pub fn remove_tile_mask(&mut self, id: SliceId) -> bool {
        self.tile_masks.remove(&id).is_some()
    }

    /// Benar bila masker jaringan `id` ada.
    pub fn has_tile_mask(&self, id: SliceId) -> bool {
        self.tile_masks.contains_key(&id)
    }

    /// Buang semua masker jaringan (peringatan memori / ganti proyek).
    pub fn clear_tile_masks(&mut self) {
        self.tile_masks.clear();
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
        // Masker hanya dipakai bila teksturnya sudah diunggah.
        let mask_of = |l: &TileLayerDraw| -> Option<TileMask> {
            l.mask.filter(|m| self.tile_masks.contains_key(&m.texture))
        };
        if let Some(base) = &scene.base {
            let mask = mask_of(base);
            planned.push(PlannedLayer {
                kind: LayerKind::Tile,
                uniform: layer_uniform(
                    &base.affine,
                    view,
                    base.opacity,
                    LayerMode::Normal as u32,
                    0.0,
                    ppp,
                    mask.as_ref(),
                ),
                tiles: &base.tiles,
                mask: mask.map(|m| m.texture),
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
                let mask = layer.and_then(mask_of);
                planned.push(PlannedLayer {
                    kind,
                    uniform: layer_uniform(
                        &affine,
                        view,
                        pass.opacity,
                        pass.mode as u32,
                        pass.checker_px,
                        ppp,
                        mask.as_ref(),
                    ),
                    tiles: layer.map(|l| l.tiles.as_slice()).unwrap_or(&[]),
                    mask: mask.map(|m| m.texture),
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
                    None,
                ),
                tiles: &lbl.tiles,
                mask: None,
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
                mask: layer.mask,
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
            params: [sanitize_z_scale(scene.z_scale), 0.0, 0.0, 0.0],
            ground: [l.ground[0], l.ground[1], l.ground[2], 0.0],
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
            let (material, rim, cells) = material_uniform(&inst.material);
            let u = MeshParamsUniform {
                material,
                rim,
                cells,
                color: inst.color,
                params: [
                    opacity,
                    if inst.scalar_mode { 1.0 } else { 0.0 },
                    inst.scalar_range[0],
                    inst.scalar_range[1],
                ],
                offset: [
                    inst.offset_mm[0],
                    inst.offset_mm[1],
                    inst.offset_mm[2],
                    inst.highlight.clamp(0.0, 1.0),
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
                // Pusat untuk urutan belakang-ke-depan ikut geseran tampilan.
                let c = mesh.bounds_center_mm();
                transparent.push((
                    cmd,
                    [
                        c[0] + inst.offset_mm[0],
                        c[1] + inst.offset_mm[1],
                        c[2] + inst.offset_mm[2],
                    ],
                ));
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

        // Irisan memakai alpha blending (latar transparan dari masker
        // jaringan), jadi digambar belakang ke depan terhadap mata; pusat
        // quad dihitung dengan `z_scale` yang sama seperti di shader.
        let mut slices = Vec::new();
        let mut slice_centers = Vec::new();
        let z_scale = sanitize_z_scale(scene.z_scale);
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
            let model = slice_model_matrix(&d.affine, d.size_um, d.z_um, scene.center_um);
            let u = SliceParamsUniform {
                model: model.to_cols_array_2d(),
                alpha: [
                    d.alpha.clamp(0.0, 1.0),
                    if d.glass.is_finite() {
                        d.glass.clamp(0.0, 1.0)
                    } else {
                        0.0
                    },
                    0.0,
                    0.0,
                ],
            };
            queue.write_buffer(&fs.slice_ubos[si].buf, 0, bytemuck::bytes_of(&u));
            let c = slice_model_matrix_scaled(&model, z_scale)
                .project_point3(glam::Vec3::new(0.5, 0.5, 0.0));
            slice_centers.push(c.to_array());
            slices.push(SliceCmd {
                slice: d.slice,
                ubo: si,
            });
        }
        let order = sort_back_to_front(&slice_centers, eye);
        let mut slice_cmds: Vec<Option<SliceCmd>> = slices.into_iter().map(Some).collect();
        let slices: Vec<SliceCmd> = order
            .into_iter()
            .filter_map(|i| slice_cmds[i].take())
            .collect();

        let grid = match scene
            .grid
            .as_ref()
            .and_then(|g| grid_uniform(&scene.camera, scene.aspect, g))
        {
            Some(u) => {
                ensure_uniform(
                    device,
                    &self.grid_layout,
                    &mut fs.grid,
                    0,
                    std::mem::size_of::<GridUniform>() as u64,
                    "h3d-grid",
                );
                queue.write_buffer(&fs.grid[0].buf, 0, bytemuck::bytes_of(&u));
                true
            }
            None => false,
        };

        fs.prepared = Prepared::ThreeD {
            opaque,
            transparent,
            slices,
            grid,
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
                grid,
            } => self.paint_3d(fs, opaque, transparent, slices, *grid, rpass),
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
            let mask = cmd
                .mask
                .and_then(|id| self.tile_masks.get(&id))
                .unwrap_or(&self.mask_none);
            let cache = match cmd.kind {
                LayerKind::Tile => {
                    rpass.set_pipeline(&self.tile_pipeline);
                    rpass.set_bind_group(2, &mask.bind_group, &[]);
                    &self.tiles
                }
                LayerKind::TileAdditive => {
                    rpass.set_pipeline(&self.tile_additive_pipeline);
                    rpass.set_bind_group(2, &mask.bind_group, &[]);
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
        grid: bool,
        rpass: &mut wgpu::RenderPass<'_>,
    ) {
        let Some(globals) = fs.globals.first() else {
            return;
        };
        let grid_slot = fs.grid.first().filter(|_| grid);
        // Tanpa depth buffer grid digambar paling awal supaya mesh menimpanya.
        let grid_first = self.depth_format.is_none();
        if let (Some(g), true) = (grid_slot, grid_first) {
            rpass.set_pipeline(&self.grid_pipeline);
            rpass.set_bind_group(0, &g.bind, &[]);
            rpass.draw(0..3, 0..1);
        }
        rpass.set_bind_group(0, &globals.bind, &[]);

        // 1. Mesh opak (tulis depth).
        rpass.set_pipeline(&self.mesh_opaque_pipeline);
        self.paint_meshes(fs, opaque, rpass);

        // 1b. Grid (uji depth terhadap mesh opak, tanpa tulis depth).
        if let (Some(g), false) = (grid_slot, grid_first) {
            rpass.set_pipeline(&self.grid_pipeline);
            rpass.set_bind_group(0, &g.bind, &[]);
            rpass.draw(0..3, 0..1);
            rpass.set_bind_group(0, &globals.bind, &[]);
        }

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
        validate_wgsl("grid.wgsl", include_str!("shaders/grid.wgsl"));
    }

    #[test]
    fn grid_uniform_layout_and_inverse() {
        assert_eq!(std::mem::size_of::<GridUniform>(), 224);
        let cam = OrbitCamera::default();
        let g = Grid3D {
            z_mm: -1.5,
            ..Default::default()
        };
        let u = grid_uniform(&cam, 1.6, &g).expect("kamera bawaan dapat dibalik");
        assert_eq!(u.params[0], -1.5);
        assert_eq!(u.eye[3], cam.distance);
        // Seperti shader: sinar dari depth 0 ke depth 1 dipotong dengan bidang
        // grid. Titik tengah layar harus mengenai bidang tepat di bawah target
        // kamera (target bawaan di asal, pitch 30°).
        let inv = glam::Mat4::from_cols_array_2d(&u.inv_view_proj);
        let near = inv.project_point3(glam::Vec3::new(0.0, 0.0, 0.0));
        let far = inv.project_point3(glam::Vec3::new(0.0, 0.0, 1.0));
        let t = (g.z_mm - near.z) / (far.z - near.z);
        assert!((0.0..=1.0).contains(&t));
        let hit = near + t * (far - near);
        let eye = cam.eye();
        // Mata → target → bidang segaris: titik kena = mata + (target − mata) × k.
        let k = (g.z_mm - eye.z) / (cam.target.z - eye.z);
        let expect = eye + (cam.target - eye) * k;
        assert!(
            (hit - expect).length() < 1e-3 * cam.distance,
            "{hit} vs {expect}"
        );
    }

    #[test]
    fn grid_uniform_rejects_non_finite() {
        let cam = OrbitCamera::default();
        let g = Grid3D {
            z_mm: f32::NAN,
            ..Default::default()
        };
        assert!(grid_uniform(&cam, 1.0, &g).is_none());
    }

    #[test]
    fn uniform_sizes_are_16_byte_multiples() {
        assert_eq!(std::mem::size_of::<LayerUniform>(), 96);
        assert_eq!(std::mem::size_of::<Globals3D>(), 176);
        assert_eq!(sanitize_z_scale(f32::NAN), 1.0);
        assert_eq!(sanitize_z_scale(0.0), 1.0);
        assert_eq!(sanitize_z_scale(-3.0), 1.0);
        assert_eq!(sanitize_z_scale(4.0), 4.0);
        assert_eq!(std::mem::size_of::<MeshParamsUniform>(), 96);
        assert_eq!(std::mem::size_of::<SliceParamsUniform>(), 80);
    }

    #[test]
    fn material_uniform_defaults_match_technical_look_and_clamp() {
        let (m, rim, cells) = material_uniform(&MeshMaterial::default());
        assert_eq!(m, [0.0, 32.0, 0.0, 0.0]);
        assert_eq!(rim[3], 0.0);
        assert_eq!(cells, [0.0; 4]);
        let wild = MeshMaterial {
            specular: f32::NAN,
            shininess: 0.0,
            wrap: 9.0,
            occlusion: -1.0,
            rim_color: [2.0, f32::INFINITY, 0.5],
            rim: 0.4,
            stain_colors: true,
            stain_saturation: f32::NAN,
            cells: Some(crate::mesh::CellBumps {
                size_mm: 0.1,
                strength: 3.0,
                nucleus_mix: 0.5,
            }),
        };
        let (m, rim, cells) = material_uniform(&wild);
        assert_eq!(m, [0.0, 1.0, 1.0, 0.0]);
        assert_eq!(rim, [1.0, 1.0, 0.5, 0.4]);
        assert_eq!(cells, [0.1, 1.0, 0.5, 1.0]);
        // Ukuran sel tak sah → efek mati.
        let off = MeshMaterial {
            cells: Some(crate::mesh::CellBumps {
                size_mm: f32::NAN,
                strength: 1.0,
                nucleus_mix: 1.0,
            }),
            ..MeshMaterial::default()
        };
        assert_eq!(material_uniform(&off).2, [0.0; 4]);
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
        let u = layer_uniform(
            &t,
            &view,
            0.5,
            LayerMode::CheckerOdd as u32,
            32.0,
            2.0,
            None,
        );
        assert_eq!(u.affine0, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(u.affine1, [4.0, 5.0, 6.0, 0.0]);
        assert_eq!(u.view, [100.0, 200.0, 0.5, 0.0]);
        assert_eq!(u.viewport, [800.0, 600.0, 0.5, 2.0]);
        // Ukuran kotak dan asal viewport dikalikan pixels_per_point.
        assert_eq!(u.params, [64.0, 20.0, 40.0, 0.0]);
        // Tanpa masker: nonaktif, ukuran aman (bukan nol).
        assert_eq!(u.mask, [1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn mask_params_rejects_invalid_size() {
        let ok = TileMask {
            texture: SliceId(3),
            size_um: [2000.0, 1500.0],
        };
        assert_eq!(mask_params(Some(&ok)), [2000.0, 1500.0, 1.0, 0.0]);
        for bad in [
            [0.0, 10.0],
            [10.0, f64::NAN],
            [-1.0, 5.0],
            [f64::INFINITY, 1.0],
        ] {
            let m = TileMask {
                texture: SliceId(3),
                size_um: bad,
            };
            assert_eq!(mask_params(Some(&m))[2], 0.0, "{bad:?}");
        }
        assert_eq!(mask_params(None)[2], 0.0);
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
