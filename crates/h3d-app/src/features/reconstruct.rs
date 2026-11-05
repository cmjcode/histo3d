//! Kanvas tumpukan 3D (8.11, 8.12.10): irisan sebagai quad bertekstur pada
//! `z_i` dengan `T_i`, mesh tumor di atasnya, picking irisan (klik = pilih,
//! klik ganda = buka 2D) dan mesh (nilai skalar + irisan terdekat), sorot
//! hover/terpilih, "renggangkan tumpukan" (hanya tampilan), pratinjau
//! susunan usulan yang tembus pandang, bidang potong, lapisan skalar dengan
//! legenda, dan ViewCube. Isi laci "Tumor 3D" (parameter rekonstruksi,
//! visibilitas/opasitas label, bidang potong, lapisan skalar) juga di sini.
//!
//! Selama job berjalan sesi terkunci; kanvas tetap menggambar dari cache
//! `ReconState::slices` dan mesh yang sudah di GPU.

use std::collections::{BTreeMap, BTreeSet};

use egui::Ui;
use glam::{Mat4, Vec3};
use h3d_core::{Colormap, LabelId, SlideId, SlideStatus};
use h3d_engine::{LabelMesh, Session};
use h3d_i18n::t;
use h3d_render::{
    pick_slice, ray_from_screen, render_to_world_um, slice_corners_mm, slice_model_matrix,
    slice_model_matrix_scaled, upload_mesh, world_um_to_render_mm, H3dPaintCallback, MeshId,
    MeshInstance, OrbitCamera, Ray, Scene, Scene3D, SliceDraw, SliceId, ViewPreset,
};
use h3d_ui::{ViewCube, ViewCubeAction};

use crate::features::slides::OrderPreview;
use crate::gpu::Gpu;
use crate::state::AppState;
use crate::ui_util::icon_label;

/// Sisi terpanjang thumbnail irisan di 3D (piksel).
pub const SLICE_TEX_SIDE: u32 = 512;
/// Thumbnail irisan yang diunggah ke GPU per frame.
const SLICE_UPLOADS_PER_FRAME: usize = 2;
/// Rentang penggeser renggang Z.
pub const Z_SCALE_RANGE: std::ops::RangeInclusive<f32> = 1.0..=400.0;
/// Tebal tumpukan yang dituju renggang otomatis, relatif terhadap sisi XY
/// terpanjang (0,5 = setengah lebar).
const AUTO_THICKNESS_RATIO: f32 = 0.5;

/// Hasil klik pada mesh.
#[derive(Clone, Debug, PartialEq)]
pub struct PickInfo {
    /// Label mesh yang kena.
    pub label: LabelId,
    /// Titik kena (µm, ruang W).
    pub point_um: [f64; 3],
    /// Nilai lapisan skalar di titik itu dan satuannya (bila ada lapisan).
    pub value: Option<(f32, String)>,
    /// Irisan terdekat menurut Z.
    pub nearest_slide: Option<SlideId>,
}

/// Satu irisan di kanvas (cache dari sesi; dipakai juga saat sesi terkunci).
#[derive(Clone, Debug, PartialEq)]
pub struct SliceItem {
    /// Irisan.
    pub id: SlideId,
    /// Nomor urut 1-basis.
    pub number: u32,
    /// Data gambar (`T_i`, ukuran, z).
    pub draw: SliceDraw,
    /// Gagal QC registrasi.
    pub flagged: bool,
    /// Belum punya `T_i` (bukan acuan, belum terdaftar).
    pub unregistered: bool,
    /// Bagian dari pratinjau usulan (bukan susunan proyek).
    pub preview: bool,
}

/// Kejadian dari kanvas untuk cangkang.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CanvasEvent {
    /// Tidak ada.
    None,
    /// Irisan dipilih (klik).
    Selected(SlideId),
    /// Irisan diminta dibuka di 2D (klik ganda).
    OpenSlice(SlideId),
}

/// Keadaan kanvas 3D dan rekonstruksi.
pub struct ReconState {
    /// Kamera orbit (mm, ruang R).
    pub camera: OrbitCamera,
    /// Mesh terakhir (salinan CPU untuk nilai skalar dan statistik).
    pub meshes: Vec<LabelMesh>,
    /// Mesh perlu dimuat ulang dari sesi.
    pub needs_reload: bool,
    /// Mesh perlu diunggah ulang ke GPU (lapisan skalar diganti).
    pub needs_upload: bool,
    /// Pusat volume (µm, W) yang dipakai saat mengunggah mesh dan membangun
    /// matriks irisan.
    pub center_um: [f64; 3],
    /// Kotak pembatas mesh (mm, ruang R, sebelum renggang Z).
    pub bounds_mm: Option<(Vec3, Vec3)>,
    /// Kamera sudah dipaskan ke isi kanvas.
    pub fitted: bool,
    /// Label yang disembunyikan.
    pub hidden: BTreeSet<LabelId>,
    /// Opasitas per label (bawaan 1).
    pub opacity: BTreeMap<LabelId, f32>,
    /// Bidang potong aktif.
    pub clip_enabled: bool,
    /// Sumbu bidang potong (0 = X, 1 = Y, 2 = Z ruang R).
    pub clip_axis: usize,
    /// Posisi bidang potong 0..1 sepanjang kotak pembatas.
    pub clip_offset: f32,
    /// Tampilkan irisan di 3D.
    pub show_slices: bool,
    /// Tampilkan mesh di 3D.
    pub show_meshes: bool,
    /// Tampilkan grid lantai tak terbatas di 3D.
    pub show_grid: bool,
    /// Alfa irisan di 3D.
    pub slice_alpha: f32,
    /// Sembunyikan latar kaca/putih irisan di 3D (alfa dari masker
    /// jaringan; tampilan saja, citra tidak diubah).
    pub hide_background: bool,
    /// Irisan yang disembunyikan dari tumpukan (hanya tampilan).
    pub hidden_slices: BTreeSet<SlideId>,
    /// Irisan yang thumbnailnya sudah diunggah.
    pub uploaded_slices: BTreeSet<SlideId>,
    /// Cache irisan yang digambar frame terakhir (dipakai saat sesi terkunci).
    pub slices: Vec<SliceItem>,
    /// Irisan di bawah kursor.
    pub hovered_slice: Option<SlideId>,
    /// Faktor renggang Z (tampilan saja; 1 = skala sebenarnya).
    pub z_scale: f32,
    /// Renggang Z masih otomatis (belum disentuh pengguna).
    pub z_scale_auto: bool,
    /// Lapisan skalar terpilih (nama `MeshScalars`).
    pub scalar: Option<String>,
    /// Rentang lapisan skalar; `None` = otomatis p2–p98.
    pub range: Option<[f32; 2]>,
    /// Colormap.
    pub colormap: Colormap,
    /// Hasil klik terakhir pada mesh.
    pub pick: Option<PickInfo>,
    /// Kunci mesh `(label, id objek)` yang ada di GPU.
    pub gpu_meshes: BTreeSet<MeshKey>,
    /// Pilih lapisan skalar pertama begitu tersedia (argumen `--analyze`).
    pub want_first_scalar: bool,
    /// Kubus orientasi.
    pub viewcube: ViewCube,
}

impl Default for ReconState {
    fn default() -> Self {
        ReconState {
            camera: OrbitCamera::default(),
            meshes: Vec::new(),
            needs_reload: true,
            needs_upload: false,
            center_um: [0.0; 3],
            bounds_mm: None,
            fitted: false,
            hidden: BTreeSet::new(),
            opacity: BTreeMap::new(),
            clip_enabled: false,
            clip_axis: 2,
            clip_offset: 0.5,
            show_slices: true,
            show_meshes: true,
            show_grid: false,
            slice_alpha: 0.9,
            hide_background: true,
            hidden_slices: BTreeSet::new(),
            uploaded_slices: BTreeSet::new(),
            slices: Vec::new(),
            hovered_slice: None,
            z_scale: 1.0,
            z_scale_auto: true,
            scalar: None,
            range: None,
            colormap: Colormap::Viridis,
            pick: None,
            gpu_meshes: BTreeSet::new(),
            want_first_scalar: false,
            viewcube: ViewCube::default(),
        }
    }
}

/// Pusat kotak pembatas semua verteks (µm, W); `None` bila tidak ada verteks.
pub fn meshes_center_um(meshes: &[LabelMesh]) -> Option<([f64; 3], [f64; 3], [f64; 3])> {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    let mut any = false;
    for m in meshes {
        for p in &m.positions_um {
            any = true;
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
    }
    any.then(|| {
        (
            [
                (lo[0] + hi[0]) / 2.0,
                (lo[1] + hi[1]) / 2.0,
                (lo[2] + hi[2]) / 2.0,
            ],
            lo,
            hi,
        )
    })
}

/// Kotak pembatas tumpukan irisan (µm, W): gabungan bayangan tiap irisan
/// lewat `T_i` dan rentang `z`. `None` bila tidak ada irisan.
pub fn slices_bounds_um(items: &[SliceItem]) -> Option<([f64; 3], [f64; 3])> {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for it in items {
        let (a, b) = it
            .draw
            .affine
            .transform_rect(it.draw.size_um[0], it.draw.size_um[1]);
        lo[0] = lo[0].min(a[0]);
        lo[1] = lo[1].min(a[1]);
        hi[0] = hi[0].max(b[0]);
        hi[1] = hi[1].max(b[1]);
        lo[2] = lo[2].min(it.draw.z_um);
        hi[2] = hi[2].max(it.draw.z_um);
    }
    (lo.iter().all(|v| v.is_finite()) && hi.iter().all(|v| v.is_finite())).then_some((lo, hi))
}

/// Faktor renggang Z otomatis agar tebal tumpukan ≈ [`AUTO_THICKNESS_RATIO`] ×
/// sisi XY terpanjang; irisan 4 µm selebar 20 mm tanpa ini tampak satu lembar.
pub fn auto_z_scale(lo_um: [f64; 3], hi_um: [f64; 3]) -> f32 {
    let xy = (hi_um[0] - lo_um[0]).max(hi_um[1] - lo_um[1]);
    let thickness = hi_um[2] - lo_um[2];
    if thickness <= 0.0 || xy <= 0.0 || !thickness.is_finite() || !xy.is_finite() {
        return 1.0;
    }
    let target = xy * f64::from(AUTO_THICKNESS_RATIO);
    ((target / thickness) as f32).clamp(*Z_SCALE_RANGE.start(), *Z_SCALE_RANGE.end())
}

/// Nama lapisan skalar yang tersedia di mesh (terurut, unik).
pub fn scalar_names(meshes: &[LabelMesh]) -> Vec<String> {
    let set: BTreeSet<String> = meshes
        .iter()
        .flat_map(|m| m.scalars.iter().map(|s| s.name.clone()))
        .collect();
    set.into_iter().collect()
}

/// Satuan lapisan skalar `name`.
pub fn scalar_unit(meshes: &[LabelMesh], name: &str) -> String {
    meshes
        .iter()
        .flat_map(|m| m.scalars.iter())
        .find(|s| s.name == name)
        .map(|s| s.unit.clone())
        .unwrap_or_default()
}

/// Rentang otomatis p2–p98 lapisan skalar atas semua mesh (nilai finit).
pub fn auto_range(meshes: &[LabelMesh], name: &str) -> Option<[f32; 2]> {
    let mut v: Vec<f32> = meshes
        .iter()
        .flat_map(|m| m.scalars.iter())
        .filter(|s| s.name == name)
        .flat_map(|s| s.values.iter().copied())
        .filter(|x| x.is_finite())
        .collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(f32::total_cmp);
    let at = |p: f32| v[((v.len() - 1) as f32 * p).round() as usize];
    let (lo, hi) = (at(0.02), at(0.98));
    Some(if hi > lo { [lo, hi] } else { [lo, lo + 1.0] })
}

/// Nilai skalar di titik kena: interpolasi barisentrik atas tiga verteks
/// segitiga; `None` bila ada verteks tanpa data.
pub fn scalar_at(mesh: &LabelMesh, name: &str, tri_index: usize, bary: [f32; 3]) -> Option<f32> {
    let s = mesh.scalars.iter().find(|s| s.name == name)?;
    let mut out = 0.0;
    for (k, w) in bary.iter().enumerate() {
        let vi = *mesh.indices.get(tri_index * 3 + k)? as usize;
        let val = *s.values.get(vi)?;
        if !val.is_finite() {
            return None;
        }
        out += w * val;
    }
    Some(out)
}

/// Irisan `Ok` yang `z_um`-nya paling dekat dengan `z` (seri → urutan awal).
pub fn nearest_slide(session: &Session, z: f64) -> Option<SlideId> {
    session
        .project()
        .usable_slides_ordered()
        .iter()
        .min_by(|a, b| (a.z_um - z).abs().total_cmp(&(b.z_um - z).abs()))
        .map(|s| s.id)
}

/// Nama tampil lapisan skalar.
pub fn scalar_display_name(name: &str) -> String {
    match name {
        "margin_distance" => t!("recon-scalar-margin"),
        "vessel_distance" => t!("recon-scalar-vessel"),
        other => other.strip_prefix("attr:").unwrap_or(other).to_string(),
    }
}

/// Preset kamera untuk muka ViewCube.
pub fn preset_for(action: ViewCubeAction) -> ViewPreset {
    match action {
        ViewCubeAction::Top => ViewPreset::Top,
        ViewCubeAction::Bottom => ViewPreset::Bottom,
        ViewCubeAction::Front => ViewPreset::Front,
        ViewCubeAction::Back => ViewPreset::Back,
        ViewCubeAction::Right => ViewPreset::Right,
        ViewCubeAction::Left => ViewPreset::Left,
    }
}

/// Tombol preset pandangan di laci: isometrik, atas, depan, kanan,
/// tumpukan menyamping (Z mendatar, irisan kiri → kanan), dan "Tegakkan"
/// yang menyejajarkan atas layar dengan Z lagi setelah orbit bebas.
pub fn view_preset_buttons(st: &mut ReconState, ui: &mut Ui) {
    ui.horizontal_wrapped(|ui| {
        ui.label(t!("recon-view-label"));
        for (key, preset) in [
            ("recon-view-iso", ViewPreset::Isometric),
            ("recon-view-top", ViewPreset::Top),
            ("recon-view-front", ViewPreset::Front),
            ("recon-view-right", ViewPreset::Right),
            ("recon-view-side", ViewPreset::Side),
        ] {
            if ui.small_button(t!(key)).clicked() {
                st.camera.set_preset(preset);
            }
        }
        if ui
            .small_button(t!("recon-view-upright"))
            .on_hover_text(t!("recon-view-upright-hint"))
            .clicked()
        {
            st.camera.upright();
        }
    });
}

/// Ray di ruang R **sebelum** renggang Z, dari ray layar (yang berada di
/// ruang berenggang). Parameter `t` tetap sama karena pemetaannya linear.
pub fn unscale_ray(ray: &Ray, z_scale: f32) -> Ray {
    let s = h3d_render::sanitize_z_scale(z_scale);
    Ray {
        origin: Vec3::new(ray.origin.x, ray.origin.y, ray.origin.z / s),
        dir: Vec3::new(ray.dir.x, ray.dir.y, ray.dir.z / s),
    }
}

/// Susunan irisan untuk digambar: susunan proyek, atau pratinjau usulan bila
/// ada (urutan usulan × jarak irisan, orientasi usulan relatif acuan).
pub fn slice_items(sess: &Session, preview: Option<&OrderPreview>, alpha: f32) -> Vec<SliceItem> {
    let project = sess.project();
    let reference = project.effective_reference();
    if let Some(p) = preview {
        let spacing = project.recon.default_spacing_um;
        let z0 = project
            .usable_slides_ordered()
            .iter()
            .map(|s| s.z_um)
            .fold(f64::INFINITY, f64::min);
        let z0 = if z0.is_finite() { z0 } else { 0.0 };
        // Acuan pratinjau = tengah urutan usulan (orientasinya identitas).
        let ref_id = p
            .suggestion
            .order
            .get(p.suggestion.order.len() / 2)
            .copied();
        let t_ref = ref_id
            .and_then(|r| sess.slide(r).ok())
            .map(|e| e.transform.affine())
            .unwrap_or(h3d_core::Affine2::IDENTITY);
        return p
            .suggestion
            .order
            .iter()
            .enumerate()
            .filter_map(|(k, id)| {
                let e = sess.slide(*id).ok()?;
                if e.status != SlideStatus::Ok {
                    return None;
                }
                let orient = p
                    .suggestion
                    .orient
                    .iter()
                    .find(|(s, _)| s == id)
                    .map(|(_, a)| *a)
                    .unwrap_or(h3d_core::Affine2::IDENTITY);
                Some(SliceItem {
                    id: *id,
                    number: k as u32 + 1,
                    draw: SliceDraw {
                        slice: SliceId(id.0),
                        // U irisan → U acuan (orient) → W (T acuan).
                        affine: t_ref.compose(&orient),
                        size_um: e.size_um(),
                        z_um: z0 + k as f64 * spacing,
                        alpha: alpha * 0.55,
                    },
                    flagged: p.suggestion.uncertain.contains(id),
                    unregistered: false,
                    preview: true,
                })
            })
            .collect();
    }
    project
        .usable_slides_ordered()
        .iter()
        .map(|e| SliceItem {
            id: e.id,
            number: e.order + 1,
            draw: SliceDraw {
                slice: SliceId(e.id.0),
                affine: e.transform.affine(),
                size_um: e.size_um(),
                z_um: e.z_um,
                alpha,
            },
            flagged: e.reg.as_ref().is_some_and(|r| r.flagged),
            unregistered: Some(e.id) != reference && e.reg.is_none(),
            preview: false,
        })
        .collect()
}

/// Kunci satu mesh: label dan id objek (rekonstruksi per objek).
pub type MeshKey = (LabelId, Option<u32>);

fn mesh_key(m: &LabelMesh) -> MeshKey {
    (m.label, m.instance_id)
}

/// Id mesh render: 8 bit label, sisanya id objek + 1 (0 = mesh label utuh).
fn mesh_id(key: MeshKey) -> MeshId {
    MeshId(u64::from(key.0 .0) | (u64::from(key.1.map_or(0, |i| i + 1)) << 8))
}

fn reload(app: &mut AppState, sess: &Session) {
    let st = &mut app.recon;
    st.needs_reload = false;
    match sess.meshes() {
        Ok(m) => st.meshes = m,
        Err(e) => {
            st.meshes.clear();
            app.toasts.engine_error(&e);
        }
    }
    let names = scalar_names(&st.meshes);
    if st.scalar.as_ref().is_some_and(|s| !names.contains(s)) {
        st.scalar = None;
    }
    if st.want_first_scalar && !names.is_empty() {
        st.want_first_scalar = false;
        st.scalar = names.first().cloned();
    }
    st.pick = None;
    st.needs_upload = true;
}

/// Pusat dan kotak pembatas mesh (tanpa GPU: hanya statistik).
fn mesh_bounds(st: &mut ReconState) -> Option<([f64; 3], [f64; 3], [f64; 3])> {
    let b = meshes_center_um(&st.meshes);
    if let Some((center, lo, hi)) = b {
        st.center_um = center;
        let a = Vec3::from(world_um_to_render_mm(lo, center));
        let b = Vec3::from(world_um_to_render_mm(hi, center));
        st.bounds_mm = Some((a.min(b), a.max(b)));
    } else {
        st.bounds_mm = None;
    }
    b
}

fn upload(app: &mut AppState, gpu: &Gpu) {
    let st = &mut app.recon;
    st.needs_upload = false;
    let stale: Vec<MeshKey> = st.gpu_meshes.iter().copied().collect();
    let Some((center, _, _)) = mesh_bounds(st) else {
        gpu.with(|r, _, _| {
            for l in stale {
                r.remove_mesh(mesh_id(l));
            }
        });
        st.gpu_meshes.clear();
        return;
    };
    let scalar = st.scalar.clone();
    let meshes = &st.meshes;
    let mut uploaded = BTreeSet::new();
    gpu.with(|r, device, queue| {
        for l in &stale {
            r.remove_mesh(mesh_id(*l));
        }
        for m in meshes {
            let values = scalar
                .as_ref()
                .and_then(|n| m.scalars.iter().find(|s| &s.name == n))
                .map(|s| s.values.as_slice());
            match upload_mesh(
                device,
                queue,
                &m.positions_um,
                &m.normals,
                &m.indices,
                center,
                values,
            ) {
                Ok(g) => {
                    r.insert_mesh(mesh_id(mesh_key(m)), g);
                    uploaded.insert(mesh_key(m));
                }
                Err(e) => log::warn!("mesh label {} gagal diunggah: {e}", m.label),
            }
        }
    });
    st.gpu_meshes = uploaded;
    // Pusat berubah → irisan ikut pindah; kamera dipaskan ulang.
    st.fitted = false;
}

/// Segarkan cache irisan dan unggah thumbnail yang belum ada (kuota per frame).
fn refresh_slices(app: &mut AppState, sess: &Session, gpu: Option<&Gpu>, ctx: &egui::Context) {
    let preview = app.slides.order_preview.as_ref();
    let alpha = app.recon.slice_alpha;
    let items = slice_items(sess, preview, alpha);
    let st = &mut app.recon;
    // Tanpa mesh, pusat volume = pusat tumpukan (pusat berubah → pas ulang).
    if st.meshes.is_empty() {
        if let Some((lo, hi)) = slices_bounds_um(&items) {
            let c = [
                (lo[0] + hi[0]) / 2.0,
                (lo[1] + hi[1]) / 2.0,
                (lo[2] + hi[2]) / 2.0,
            ];
            if c != st.center_um {
                st.center_um = c;
                st.fitted = false;
            }
        }
    }
    if st.z_scale_auto {
        if let Some((lo, hi)) = slices_bounds_um(&items) {
            let s = auto_z_scale(lo, hi);
            if (s - st.z_scale).abs() > 1e-3 {
                st.z_scale = s;
                st.fitted = false;
            }
        }
    }
    if let Some(g) = gpu {
        let mut budget = SLICE_UPLOADS_PER_FRAME;
        for it in &items {
            if st.uploaded_slices.contains(&it.id) {
                continue;
            }
            if budget == 0 {
                ctx.request_repaint();
                break;
            }
            budget -= 1;
            let thumb = if st.hide_background {
                sess.thumbnail_tissue(it.id, SLICE_TEX_SIDE)
            } else {
                sess.thumbnail(it.id, SLICE_TEX_SIDE)
            };
            match thumb {
                Ok(img) => {
                    let ok = g
                        .with(|r, device, queue| {
                            r.upload_slice(
                                device,
                                queue,
                                it.draw.slice,
                                img.width(),
                                img.height(),
                                img.as_raw(),
                            )
                            .is_ok()
                        })
                        .unwrap_or(false);
                    if ok {
                        st.uploaded_slices.insert(it.id);
                    }
                }
                Err(err) => log::warn!("thumbnail irisan {} gagal: {err}", it.id),
            }
        }
    }
    if items.len() != st.slices.len() {
        st.fitted = false;
    }
    st.slices = items;
}

/// Kotak pembatas seluruh isi kanvas (mm, ruang R, **setelah** renggang Z).
/// Tinggi grid lantai (mm, ruang R berenggang): sedikit di bawah dasar isi
/// kanvas supaya tidak berimpit dengan irisan terbawah; `0` bila kanvas kosong.
pub fn grid_height_mm(bounds: Option<(Vec3, Vec3)>) -> f32 {
    match bounds {
        Some((lo, hi)) => {
            let extent = (hi - lo).max_element().max(0.0);
            lo.z - GRID_GAP_RATIO * extent
        }
        None => 0.0,
    }
}

/// Jarak grid di bawah dasar isi, pecahan sisi terpanjang kotak pembatas.
const GRID_GAP_RATIO: f32 = 0.02;

fn canvas_bounds(st: &ReconState) -> Option<(Vec3, Vec3)> {
    let s = h3d_render::sanitize_z_scale(st.z_scale);
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    let mut any = false;
    if st.show_meshes {
        if let Some((a, b)) = st.bounds_mm {
            lo = lo.min(Vec3::new(a.x, a.y, a.z * s));
            hi = hi.max(Vec3::new(b.x, b.y, b.z * s));
            any = true;
        }
    }
    if st.show_slices {
        for it in st
            .slices
            .iter()
            .filter(|i| !st.hidden_slices.contains(&i.id))
        {
            let m = slice_model_matrix_scaled(
                &slice_model_matrix(&it.draw.affine, it.draw.size_um, it.draw.z_um, st.center_um),
                s,
            );
            for c in slice_corners_mm(&m) {
                lo = lo.min(c);
                hi = hi.max(c);
                any = true;
            }
        }
    }
    (any && lo.is_finite() && hi.is_finite()).then_some((lo, hi))
}

/// Proyeksikan titik R (mm, berenggang) ke layar; `None` di belakang kamera.
fn project(view_proj: &Mat4, rect: egui::Rect, p: Vec3) -> Option<egui::Pos2> {
    let clip = *view_proj * p.extend(1.0);
    if clip.w <= 1e-6 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    Some(egui::pos2(
        rect.left() + (ndc.x + 1.0) * 0.5 * rect.width(),
        rect.top() + (1.0 - ndc.y) * 0.5 * rect.height(),
    ))
}

/// Gambar kanvas 3D ke `rect`. `sess = None` saat job memegang sesi: cache
/// irisan dipakai dan tidak ada picking. Mengembalikan kejadian pilih/buka.
pub fn canvas(
    app: &mut AppState,
    sess: Option<&Session>,
    ui: &mut Ui,
    gpu: Option<&Gpu>,
    rect: egui::Rect,
) -> CanvasEvent {
    let ctx = ui.ctx().clone();
    if let Some(sess) = sess {
        if app.recon.needs_reload {
            reload(app, sess);
        }
        if app.recon.needs_upload {
            match gpu {
                Some(g) => upload(app, g),
                None => {
                    app.recon.needs_upload = false;
                    mesh_bounds(&mut app.recon);
                }
            }
        }
        refresh_slices(app, sess, gpu, &ctx);
    }

    let response = ui.interact(
        rect,
        ui.id().with("stack-canvas"),
        egui::Sense::click_and_drag(),
    );
    ui.painter_at(rect)
        .rect_filled(rect, 0.0, h3d_ui::theme::canvas_fill(ui.visuals()));
    let aspect = (rect.width() / rect.height().max(1.0)).max(0.01);
    let st = &mut app.recon;
    if !st.fitted {
        if let Some((lo, hi)) = canvas_bounds(st) {
            st.camera.fit_bounds(lo, hi, aspect);
            st.fitted = true;
        }
    }

    // Masukan kamera: seret = orbit, seret kanan/tengah atau Shift = geser,
    // roda/cubit = zoom; dua jari = geser + zoom.
    let multi = ctx.input(|i| i.multi_touch());
    if let Some(mt) = multi.filter(|m| m.num_touches >= 2) {
        st.camera.pan(
            mt.translation_delta.x,
            mt.translation_delta.y,
            rect.height(),
        );
        if (mt.zoom_delta - 1.0).abs() > 1e-4 {
            st.camera.zoom(mt.zoom_delta);
        }
    } else {
        let shift = ctx.input(|i| i.modifiers.shift);
        let d = response.drag_delta();
        let pan = response.dragged_by(egui::PointerButton::Secondary)
            || response.dragged_by(egui::PointerButton::Middle)
            || (shift && response.dragged_by(egui::PointerButton::Primary));
        if pan {
            st.camera.pan(d.x, d.y, rect.height());
        } else if response.dragged_by(egui::PointerButton::Primary) {
            st.camera.orbit(d.x, d.y);
        }
        if response.contains_pointer() {
            let (scroll, zoom) = ctx.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
            if (zoom - 1.0).abs() > 1e-4 {
                st.camera.zoom(zoom);
            } else if scroll.abs() > 0.0 {
                st.camera.zoom((scroll * 0.004).exp());
            }
        }
    }

    let z_scale = h3d_render::sanitize_z_scale(st.z_scale);
    let view_proj = st.camera.view_proj(aspect);
    let previewing = st.slices.iter().any(|s| s.preview);

    // Irisan yang digambar + matriks model berenggang (untuk picking/garis).
    let draw_slices = st.show_slices;
    let visible: Vec<(SliceItem, Mat4)> = st
        .slices
        .iter()
        .filter(|it| draw_slices && !st.hidden_slices.contains(&it.id))
        .map(|it| {
            let m =
                slice_model_matrix(&it.draw.affine, it.draw.size_um, it.draw.z_um, st.center_um);
            (it.clone(), slice_model_matrix_scaled(&m, z_scale))
        })
        .collect();

    // Picking irisan di bawah kursor (hanya bila sesi tersedia dan tidak
    // sedang menyeret kamera).
    let mut event = CanvasEvent::None;
    st.hovered_slice = None;
    let pointer = response.hover_pos().filter(|p| rect.contains(*p));
    if sess.is_some() && !response.dragged() {
        if let Some(pos) = pointer {
            let ray = ray_from_screen(&st.camera, rect, pos);
            let mut best: Option<(SlideId, f32)> = None;
            for (it, m) in &visible {
                if let Some(h) = pick_slice(&ray, m) {
                    if best.is_none_or(|(_, t)| h.t < t) {
                        best = Some((it.id, h.t));
                    }
                }
            }
            st.hovered_slice = best.map(|(id, _)| id);
            let slice_t = best.map(|(_, t)| t);

            if response.double_clicked() {
                if let Some(id) = st.hovered_slice.filter(|_| !previewing) {
                    event = CanvasEvent::OpenSlice(id);
                }
            } else if response.clicked() {
                // Mesh: hanya bila lebih dekat daripada irisan yang kena.
                let mesh_hit = match (gpu, st.show_meshes) {
                    (Some(g), true) => {
                        let unscaled = unscale_ray(&ray, z_scale);
                        let keys: Vec<MeshKey> = st
                            .meshes
                            .iter()
                            .map(mesh_key)
                            .filter(|k| st.gpu_meshes.contains(k) && !st.hidden.contains(&k.0))
                            .collect();
                        g.with(|r, _, _| {
                            let mut best: Option<(MeshKey, h3d_render::PickHit)> = None;
                            for k in keys {
                                if let Some(h) = r.mesh(mesh_id(k)).and_then(|m| m.pick(&unscaled))
                                {
                                    if best.as_ref().is_none_or(|(_, b)| h.t < b.t) {
                                        best = Some((k, h));
                                    }
                                }
                            }
                            best
                        })
                        .flatten()
                    }
                    _ => None,
                };
                match (mesh_hit, slice_t) {
                    (Some((key, h)), s) if s.is_none_or(|t| h.t < t) => {
                        let label = key.0;
                        let center = st.center_um;
                        let point_um = render_to_world_um(h.point_mm.to_array(), center);
                        let value = st.scalar.as_ref().and_then(|name| {
                            let mesh = st.meshes.iter().find(|m| mesh_key(m) == key)?;
                            let v = scalar_at(mesh, name, h.tri_index, h.bary)?;
                            Some((v, scalar_unit(&st.meshes, name)))
                        });
                        st.pick = Some(PickInfo {
                            label,
                            point_um,
                            value,
                            nearest_slide: sess.and_then(|s| nearest_slide(s, point_um[2])),
                        });
                    }
                    _ => {
                        if let Some(id) = st.hovered_slice.filter(|_| !previewing) {
                            event = CanvasEvent::Selected(id);
                        }
                    }
                }
            }
        }
    }

    // Instance mesh (disembunyikan selama pratinjau usulan: susunannya lain).
    let range = st.scalar.as_ref().map(|n| {
        st.range
            .or_else(|| auto_range(&st.meshes, n))
            .unwrap_or([0.0, 1.0])
    });
    let mut instances = Vec::new();
    if st.show_meshes && !previewing {
        for m in &st.meshes {
            if !st.gpu_meshes.contains(&mesh_key(m)) {
                continue;
            }
            let color = sess
                .and_then(|s| s.project().label(m.label).map(|l| l.color))
                .unwrap_or([200, 200, 200, 255]);
            let has_scalar = st
                .scalar
                .as_ref()
                .is_some_and(|n| m.scalars.iter().any(|s| &s.name == n));
            instances.push(MeshInstance {
                mesh: mesh_id(mesh_key(m)),
                color: [
                    f32::from(color[0]) / 255.0,
                    f32::from(color[1]) / 255.0,
                    f32::from(color[2]) / 255.0,
                    1.0,
                ],
                opacity: st.opacity.get(&m.label).copied().unwrap_or(1.0),
                visible: !st.hidden.contains(&m.label),
                scalar_mode: has_scalar,
                scalar_range: range.unwrap_or([0.0, 1.0]),
                colormap: st.colormap,
            });
        }
    }

    // Bidang potong di ruang R berenggang: fragmen dengan n·p − d > 0 dibuang.
    let clip_plane = match (st.clip_enabled, canvas_bounds(st)) {
        (true, Some((lo, hi))) => {
            let axis = st.clip_axis.min(2);
            let mut n = [0.0f32; 3];
            n[axis] = 1.0;
            let d = lo[axis] + st.clip_offset * (hi[axis] - lo[axis]);
            Some([n[0], n[1], n[2], d])
        }
        _ => None,
    };

    if gpu.is_some() {
        let slices: Vec<SliceDraw> = visible
            .iter()
            .filter(|(it, _)| st.uploaded_slices.contains(&it.id))
            .map(|(it, _)| it.draw.clone())
            .collect();
        let scene = Scene3D {
            camera: st.camera.clone(),
            aspect,
            center_um: st.center_um,
            meshes: instances,
            slices,
            clip_plane,
            lighting: Default::default(),
            z_scale,
            grid: st.show_grid.then(|| h3d_render::Grid3D {
                z_mm: grid_height_mm(canvas_bounds(st)),
                ..Default::default()
            }),
        };
        ui.painter_at(rect)
            .add(H3dPaintCallback::new(Scene::ThreeD(scene)).into_egui(rect));
    } else {
        crate::viewer::paint_background(ui, rect, false);
    }

    // Hiasan: garis tepi irisan (terpilih, hover, ragu/gagal QC, pratinjau)
    // dan nomor irisan terpilih.
    let painter = ui.painter_at(rect);
    let selected = app.current;
    for (it, m) in &visible {
        let is_sel = selected == Some(it.id);
        let is_hover = st.hovered_slice == Some(it.id);
        let stroke = if is_sel {
            Some(egui::Stroke::new(2.0, h3d_ui::theme::ACCENT))
        } else if is_hover {
            Some(egui::Stroke::new(1.5, egui::Color32::WHITE))
        } else if it.flagged {
            Some(egui::Stroke::new(1.0, h3d_ui::theme::WARNING))
        } else if it.preview {
            Some(egui::Stroke::new(
                1.0,
                h3d_ui::theme::ACCENT_LIGHT.gamma_multiply(0.6),
            ))
        } else if it.unregistered {
            Some(egui::Stroke::new(
                1.0,
                egui::Color32::from_gray(140).gamma_multiply(0.7),
            ))
        } else {
            None
        };
        let Some(stroke) = stroke else {
            continue;
        };
        let corners = slice_corners_mm(m);
        let pts: Option<Vec<egui::Pos2>> = corners
            .iter()
            .map(|c| project(&view_proj, rect, *c))
            .collect();
        if let Some(pts) = pts {
            painter.add(egui::Shape::closed_line(pts.clone(), stroke));
            if is_sel || is_hover {
                let label = format!("{}", it.number);
                let at = pts[0] + egui::vec2(6.0, -6.0);
                painter.text(
                    at,
                    egui::Align2::LEFT_BOTTOM,
                    label,
                    egui::FontId::proportional(12.0),
                    stroke.color,
                );
            }
        }
    }

    // Legenda colormap di kiri bawah (8.12.6).
    if let (Some(name), Some(range), true) = (st.scalar.clone(), range, !previewing) {
        let legend = egui::Rect::from_min_size(
            rect.left_bottom() + egui::vec2(60.0, -76.0),
            egui::vec2(240.0, 52.0),
        );
        let painter = ui.painter_at(rect);
        painter.rect_filled(
            legend.expand(6.0),
            6.0,
            egui::Color32::from_black_alpha(170),
        );
        let mut child = ui.new_child(egui::UiBuilder::new().max_rect(legend));
        child.visuals_mut().override_text_color = Some(egui::Color32::WHITE);
        child.small(scalar_display_name(&name));
        h3d_ui::widgets::colormap_legend(
            &mut child,
            st.colormap,
            range[0],
            range[1],
            &scalar_unit(&st.meshes, &name),
        );
    }

    if st.slices.is_empty() && st.meshes.is_empty() {
        let painter = ui.painter_at(rect);
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            t!("stack-empty"),
            egui::FontId::proportional(16.0),
            ui.visuals().weak_text_color(),
        );
        painter.text(
            rect.center() + egui::vec2(0.0, 24.0),
            egui::Align2::CENTER_CENTER,
            t!("stack-empty-hint"),
            egui::FontId::proportional(12.5),
            ui.visuals().weak_text_color(),
        );
    }
    if st.hovered_slice.is_some() {
        ctx.set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    event
}

/// ViewCube di pojok kanan atas `work` (area kanvas di luar laci), digambar
/// oleh cangkang setelah laci agar tidak tertutup. Klik muka → preset kamera.
pub fn viewcube(app: &mut AppState, ui: &mut Ui, work: egui::Rect) {
    let st = &mut app.recon;
    let size = st.viewcube.footprint();
    let center = work.right_top()
        + egui::vec2(
            -(size.x / 2.0 + h3d_ui::theme::FLOAT_MARGIN),
            size.y / 2.0 + h3d_ui::theme::FLOAT_MARGIN,
        );
    let area = egui::Rect::from_center_size(center, size);
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(area));
    if let Some(action) = st.viewcube.show(&mut child, center, st.camera.rotation) {
        st.camera.set_preset(preset_for(action));
    }
}

/// Isi laci "Tumor 3D": statistik mesh, parameter rekonstruksi, visibilitas
/// dan opasitas label, bidang potong, irisan di 3D, lapisan skalar, dan
/// rincian klik mesh. Mengembalikan irisan yang diminta dibuka di 2D.
pub fn tumor_drawer(app: &mut AppState, sess: &mut Session, ui: &mut Ui) -> Option<SlideId> {
    if app.recon.needs_reload {
        reload(app, sess);
    }
    let mut open = None;
    if app.recon.meshes.is_empty() {
        ui.weak(t!("recon-no-mesh"));
    } else {
        let v: usize = app.recon.meshes.iter().map(|m| m.positions_um.len()).sum();
        let tri: usize = app.recon.meshes.iter().map(|m| m.indices.len() / 3).sum();
        ui.small(t!(
            "recon-mesh-stats",
            vertices = v.to_string(),
            triangles = tri.to_string()
        ));
    }

    egui::CollapsingHeader::new(icon_label(h3d_ui::icons::TUNE, &t!("recon-params")))
        .default_open(true)
        .show(ui, |ui| {
            let mut cfg = sess.project().recon.clone();
            let mut changed = false;
            egui::Grid::new("recon-params-grid")
                .num_columns(2)
                .show(ui, |ui| {
                    ui.label(t!("recon-s-xy"));
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut cfg.s_xy_um)
                                .range(0.5..=2000.0)
                                .speed(0.5),
                        )
                        .changed();
                    ui.end_row();
                    ui.label(t!("recon-smoothing"));
                    changed |= ui
                        .add(egui::DragValue::new(&mut cfg.smooth_iterations).range(0..=200))
                        .changed();
                    ui.end_row();
                    ui.label(t!("recon-decimate"));
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut cfg.decimate_error_factor)
                                .range(0.0..=10.0)
                                .speed(0.05),
                        )
                        .changed();
                    ui.end_row();
                });
            if changed {
                if let Err(e) = sess.set_recon_config(cfg) {
                    app.toasts.engine_error(&e);
                }
                app.recon.needs_reload = true;
            }
        });

    egui::CollapsingHeader::new(icon_label(h3d_ui::icons::LAYERS, &t!("label-panel")))
        .default_open(true)
        .show(ui, |ui| {
            let st = &mut app.recon;
            let mut labels: Vec<LabelId> = st.meshes.iter().map(|m| m.label).collect();
            labels.dedup();
            if labels.is_empty() {
                ui.weak(t!("recon-no-mesh"));
            }
            for label in labels {
                let def = sess.project().label(label).cloned();
                ui.horizontal(|ui| {
                    let mut visible = !st.hidden.contains(&label);
                    let icon = if visible {
                        h3d_ui::icons::VISIBILITY_ON
                    } else {
                        h3d_ui::icons::VISIBILITY_OFF
                    };
                    if h3d_ui::widgets::icon_button(ui, icon, &t!("recon-visibility")).clicked() {
                        visible = !visible;
                        if visible {
                            st.hidden.remove(&label);
                        } else {
                            st.hidden.insert(label);
                        }
                    }
                    if let Some(d) = &def {
                        h3d_ui::widgets::color_swatch(ui, d.color);
                        ui.label(&d.name);
                    }
                    let op = st.opacity.entry(label).or_insert(1.0);
                    ui.add(
                        egui::Slider::new(op, 0.05..=1.0)
                            .show_value(false)
                            .text(t!("recon-opacity")),
                    );
                });
            }
        });

    egui::CollapsingHeader::new(icon_label(h3d_ui::icons::IMAGE, &t!("recon-slices-in-3d")))
        .default_open(false)
        .show(ui, |ui| {
            let st = &mut app.recon;
            ui.checkbox(&mut st.show_slices, t!("stack-slices-visible"));
            ui.checkbox(&mut st.show_meshes, t!("stack-mesh-visible"));
            ui.checkbox(&mut st.show_grid, t!("stack-grid-visible"));
            ui.add(egui::Slider::new(&mut st.slice_alpha, 0.1..=1.0).text(t!("recon-opacity")));
            view_preset_buttons(st, ui);
            if ui
                .checkbox(&mut st.hide_background, t!("recon-hide-background"))
                .on_hover_text(t!("recon-hide-background-hint"))
                .changed()
            {
                // Tekstur diunggah ulang dengan/ tanpa alfa masker jaringan.
                st.uploaded_slices.clear();
            }
            ui.horizontal(|ui| {
                ui.label(icon_label(h3d_ui::icons::EXPAND_Z, &t!("stack-expand")));
                if ui
                    .add(
                        egui::Slider::new(&mut st.z_scale, Z_SCALE_RANGE)
                            .logarithmic(true)
                            .suffix("x"),
                    )
                    .on_hover_text(t!("stack-expand-hint"))
                    .changed()
                {
                    st.z_scale_auto = false;
                    st.fitted = false;
                }
            });
            if !st.hidden_slices.is_empty() && ui.button(t!("stack-slice-shown")).clicked() {
                st.hidden_slices.clear();
            }
        });

    egui::CollapsingHeader::new(icon_label(h3d_ui::icons::CLIP, &t!("recon-clipping-plane")))
        .default_open(false)
        .show(ui, |ui| {
            let st = &mut app.recon;
            ui.checkbox(&mut st.clip_enabled, t!("recon-clipping-enable"));
            ui.horizontal(|ui| {
                ui.label(t!("recon-clipping-axis"));
                for (i, name) in ["X", "Y", "Z"].iter().enumerate() {
                    ui.selectable_value(&mut st.clip_axis, i, *name);
                }
            });
            ui.add(
                egui::Slider::new(&mut st.clip_offset, 0.0..=1.0).text(t!("recon-clipping-offset")),
            );
        });

    egui::CollapsingHeader::new(icon_label(
        h3d_ui::icons::PALETTE,
        &t!("recon-scalar-layer"),
    ))
    .default_open(true)
    .show(ui, |ui| {
        let st = &mut app.recon;
        let names = scalar_names(&st.meshes);
        let before = st.scalar.clone();
        let current = st
            .scalar
            .as_deref()
            .map(scalar_display_name)
            .unwrap_or_else(|| t!("recon-scalar-none"));
        egui::ComboBox::from_id_salt("scalar-layer")
            .selected_text(current)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut st.scalar, None, t!("recon-scalar-none"));
                for n in &names {
                    ui.selectable_value(&mut st.scalar, Some(n.clone()), scalar_display_name(n));
                }
            });
        if names.is_empty() {
            ui.small(t!("recon-scalar-hint"));
        }
        if st.scalar != before {
            st.range = None;
            st.needs_upload = true;
            st.pick = None;
        }
        if let Some(name) = st.scalar.clone() {
            let unit = scalar_unit(&st.meshes, &name);
            let auto = auto_range(&st.meshes, &name).unwrap_or([0.0, 1.0]);
            let mut range = st.range.unwrap_or(auto);
            ui.horizontal(|ui| {
                ui.label(t!("recon-colormap"));
                for (cm, key) in [
                    (Colormap::Viridis, "recon-colormap-viridis"),
                    (Colormap::Magma, "recon-colormap-magma"),
                    (Colormap::Coolwarm, "recon-colormap-coolwarm"),
                    (Colormap::Grey, "recon-colormap-grey"),
                ] {
                    ui.selectable_value(&mut st.colormap, cm, t!(key));
                }
            });
            let mut edited = false;
            ui.horizontal(|ui| {
                ui.label(t!("recon-range-min"));
                edited |= ui
                    .add(egui::DragValue::new(&mut range[0]).speed(1.0))
                    .changed();
                ui.label(t!("recon-range-max"));
                edited |= ui
                    .add(egui::DragValue::new(&mut range[1]).speed(1.0))
                    .changed();
            });
            if edited {
                if range[1] <= range[0] {
                    range[1] = range[0] + f32::EPSILON.max(range[0].abs() * 1e-6);
                }
                st.range = Some(range);
            }
            if ui.button(t!("recon-range-auto")).clicked() {
                st.range = None;
            }
            ui.label(t!("recon-legend"));
            h3d_ui::widgets::colormap_legend(ui, st.colormap, range[0], range[1], &unit);
        }
    });

    if let Some(pick) = app.recon.pick.clone() {
        ui.separator();
        crate::ui_util::strong(ui, t!("recon-pick-title"));
        if let Some(def) = sess.project().label(pick.label) {
            ui.label(&def.name);
        }
        match &pick.value {
            Some((v, unit)) => {
                ui.label(t!(
                    "recon-pick-value",
                    value = format!("{v:.1}"),
                    unit = unit.clone()
                ));
            }
            None => {
                ui.weak(t!("recon-pick-no-value"));
            }
        }
        ui.small(format!(
            "x {:.3}, y {:.3}, z {:.3} {}",
            pick.point_um[0] / 1000.0,
            pick.point_um[1] / 1000.0,
            pick.point_um[2] / 1000.0,
            t!("unit-mm")
        ));
        if let Some(id) = pick.nearest_slide {
            ui.label(t!(
                "recon-pick-nearest-slice",
                n = crate::ui_util::slide_title(sess, id)
            ));
            if ui.button(t!("recon-pick-open-slice")).clicked() {
                open = Some(id);
            }
        }
    }
    open
}

#[cfg(test)]
mod tests {
    use super::*;
    use h3d_volume::MeshScalars;

    #[test]
    fn grid_sits_just_below_canvas_floor() {
        assert_eq!(grid_height_mm(None), 0.0);
        let lo = Vec3::new(-10.0, -5.0, -2.0);
        let hi = Vec3::new(10.0, 5.0, 2.0);
        let z = grid_height_mm(Some((lo, hi)));
        assert!(z < lo.z);
        assert!((z - (lo.z - GRID_GAP_RATIO * 20.0)).abs() < 1e-6);
    }

    fn tri_mesh() -> LabelMesh {
        LabelMesh {
            label: LabelId(1),
            positions_um: vec![[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0], [0.0, 2000.0, 400.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            indices: vec![0, 1, 2],
            scalars: vec![MeshScalars {
                name: "margin_distance".into(),
                unit: "µm".into(),
                values: vec![100.0, 200.0, 400.0],
            }],
            instance_id: None,
        }
    }

    fn item(id: u32, z: f64, w: f64, h: f64) -> SliceItem {
        SliceItem {
            id: SlideId(id),
            number: id + 1,
            draw: SliceDraw {
                slice: SliceId(id),
                affine: h3d_core::Affine2::IDENTITY,
                size_um: [w, h],
                z_um: z,
                alpha: 1.0,
            },
            flagged: false,
            unregistered: false,
            preview: false,
        }
    }

    #[test]
    fn centre_and_scalar_helpers() {
        let m = vec![tri_mesh()];
        let (c, lo, hi) = meshes_center_um(&m).unwrap();
        assert_eq!(c, [500.0, 1000.0, 200.0]);
        assert_eq!((lo, hi), ([0.0; 3], [1000.0, 2000.0, 400.0]));
        assert!(meshes_center_um(&[]).is_none());
        assert_eq!(scalar_names(&m), ["margin_distance"]);
        assert_eq!(scalar_unit(&m, "margin_distance"), "µm");
        let r = auto_range(&m, "margin_distance").unwrap();
        assert!(r[0] >= 100.0 && r[1] <= 400.0 && r[1] > r[0]);
        assert!(auto_range(&m, "tidak-ada").is_none());
    }

    #[test]
    fn picked_value_is_barycentric_and_respects_nodata() {
        let mut m = tri_mesh();
        let v = scalar_at(&m, "margin_distance", 0, [0.5, 0.25, 0.25]).unwrap();
        assert!((v - 200.0).abs() < 1e-4);
        assert_eq!(
            scalar_at(&m, "margin_distance", 0, [1.0, 0.0, 0.0]),
            Some(100.0)
        );
        assert!(scalar_at(&m, "lain", 0, [1.0, 0.0, 0.0]).is_none());
        assert!(scalar_at(&m, "margin_distance", 5, [1.0, 0.0, 0.0]).is_none());
        m.scalars[0].values[2] = f32::NAN;
        assert!(scalar_at(&m, "margin_distance", 0, [0.3, 0.3, 0.4]).is_none());
    }

    #[test]
    fn stack_bounds_and_auto_z_scale() {
        // Tumpukan 20 mm × 15 mm setebal 36 µm (10 irisan @ 4 µm) tampak satu
        // lembar tanpa renggang; otomatis → tebal ≈ setengah lebar.
        let items: Vec<SliceItem> = (0..10)
            .map(|i| item(i, f64::from(i) * 4.0, 20_000.0, 15_000.0))
            .collect();
        let (lo, hi) = slices_bounds_um(&items).unwrap();
        assert_eq!(lo, [0.0, 0.0, 0.0]);
        assert_eq!(hi, [20_000.0, 15_000.0, 36.0]);
        let s = auto_z_scale(lo, hi);
        assert!((s - 277.78).abs() < 0.1, "{s}");
        assert!(Z_SCALE_RANGE.contains(&s));
        // Tumpukan sudah tebal → tidak direnggangkan.
        assert_eq!(auto_z_scale([0.0; 3], [100.0, 100.0, 100.0]), 1.0);
        // Satu irisan (tebal 0) → 1.
        assert_eq!(auto_z_scale([0.0; 3], [100.0, 100.0, 0.0]), 1.0);
        assert!(slices_bounds_um(&[]).is_none());
    }

    #[test]
    fn unscaled_ray_hits_same_param() {
        let ray = Ray {
            origin: Vec3::new(0.0, 0.0, 10.0),
            dir: Vec3::new(0.0, 0.0, -1.0),
        };
        let u = unscale_ray(&ray, 5.0);
        assert!((u.origin.z - 2.0).abs() < 1e-6 && (u.dir.z + 0.2).abs() < 1e-6);
        // Bidang z = 0,2 (berenggang 1,0): t sama di kedua ruang.
        let t_scaled = (ray.origin.z - 1.0) / -ray.dir.z;
        let t_unscaled = (u.origin.z - 0.2) / -u.dir.z;
        assert!((t_scaled - t_unscaled).abs() < 1e-5);
        assert_eq!(unscale_ray(&ray, f32::NAN).origin, ray.origin);
    }

    #[test]
    fn viewcube_actions_map_to_presets() {
        assert_eq!(preset_for(ViewCubeAction::Top), ViewPreset::Top);
        assert_eq!(preset_for(ViewCubeAction::Left), ViewPreset::Left);
    }
}
