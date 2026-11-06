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
use h3d_core::{Affine2, Colormap, LabelId, SlideId, SlideStatus};
use h3d_engine::{LabelMesh, Session};
use h3d_i18n::t;
use h3d_render::{
    pick_slice, ray_from_screen, render_to_world_um, slice_corners_mm, slice_model_matrix,
    slice_model_matrix_scaled, upload_mesh, world_um_to_render_mm, H3dPaintCallback, MeshId,
    MeshInstance, OrbitCamera, Ray, Scene, Scene3D, SliceDraw, SliceId, VertexAttrs, ViewPreset,
};
use h3d_ui::{ViewCube, ViewCubeAction};

use crate::features::slides::OrderPreview;
use crate::geom::manual_adjusted_transform;
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
/// Faktor opasitas objek lain saat isolasi aktif.
const ISOLATE_DIM: f32 = 0.12;
/// Kekuatan sorot objek terpilih (0..1; bentuk tetap terbaca).
const SELECT_HIGHLIGHT: f32 = 0.45;
/// Tinggi maksimum daftar objek per label di laci (point).
const OBJECT_LIST_MAX_HEIGHT: f32 = 160.0;
/// Rentang penggeser angkat objek (mm, ruang R).
pub const LIFT_RANGE: std::ops::RangeInclusive<f32> = 0.0..=50.0;
/// Rentang ukuran sel tampilan ilustratif (µm).
pub const CELL_SIZE_RANGE: std::ops::RangeInclusive<f32> = 15.0..=400.0;
/// Rentang faktor "pisahkan objek" (relatif diagonal tumpukan).
pub const EXPLODE_RANGE: std::ops::RangeInclusive<f32> = 0.0..=1.0;
/// Kecepatan putar otomatis (piksel orbit per detik).
const AUTO_ROTATE_PX_PER_S: f32 = 40.0;
/// Kekuatan kesan kaca irisan pada gaya ilustrasi.
const GLASS_STRENGTH: f32 = 0.8;
/// Jumlah irisan berkaca maksimum pada gaya ilustrasi.
const GLASS_MAX_SLICES: usize = 3;

/// Gaya tampilan kanvas 3D (hanya tampilan; tidak pernah menyentuh data).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ViewLook {
    /// Warna label polos, difus sederhana (bawaan, untuk kerja teknis).
    #[default]
    Technical,
    /// Ilustrasi jaringan: warna pulasan, kilau lembut, rim, oklusi, latar
    /// gradasi, irisan kaca. Untuk pembelajaran struktur jaringan.
    Tissue,
}

/// Hasil klik pada mesh.
#[derive(Clone, Debug, PartialEq)]
pub struct PickInfo {
    /// Label mesh yang kena.
    pub label: LabelId,
    /// Id objek bila mesh hasil rekonstruksi per objek.
    pub instance_id: Option<u32>,
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

/// Alat kanvas 3D: orbit kamera, atau koreksi geser/putar irisan terpilih
/// (8.4.5) langsung di tumpukan, supaya kesesuaiannya dengan irisan lain
/// terlihat saat disetel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum StackTool {
    /// Seret = orbit kamera (bawaan).
    #[default]
    Orbit,
    /// Seret pada irisan terpilih = putar di bidangnya.
    Rotate,
    /// Seret pada irisan terpilih = geser di bidangnya.
    Translate,
    /// Klik mesh = pilih objek 3D (lalu bisa diangkat/diisolasi, tampilan
    /// saja); seret tetap orbit kamera.
    Object,
}

impl StackTool {
    /// Alat ini mengoreksi irisan (putar/geser), bukan memilih objek.
    pub fn corrects_slice(self) -> bool {
        matches!(self, StackTool::Rotate | StackTool::Translate)
    }
}

/// Koreksi irisan terpilih yang belum diterapkan (pratinjau di kanvas).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StackEdit {
    /// Irisan yang sedang dikoreksi.
    pub slide: Option<SlideId>,
    /// Putaran (radian) di sekitar pusat irisan, ruang W.
    pub angle: f64,
    /// Geseran (µm, W).
    pub shift: [f64; 2],
    /// Cermin: `[kiri-kanan, atas-bawah]` di sekitar pusat irisan.
    pub flip: [bool; 2],
    /// Irisan sesudahnya di rantai: `T`-nya ikut berubah saat diterapkan
    /// (8.4.5), jadi dipratinjau bersama.
    pub downstream: Vec<SlideId>,
    /// Seretan yang sedang berlangsung mengedit irisan, bukan kamera.
    pub dragging: bool,
}

impl StackEdit {
    /// Ada geser/putar/balik yang belum diterapkan.
    pub fn has_change(&self) -> bool {
        self.shift != [0.0, 0.0] || self.angle != 0.0 || self.flip != [false, false]
    }

    /// Buang koreksi yang belum diterapkan (irisan dan rantai dipertahankan).
    pub fn reset(&mut self) {
        self.angle = 0.0;
        self.shift = [0.0, 0.0];
        self.flip = [false, false];
        self.dragging = false;
    }
}

/// Koreksi irisan sedang berlangsung: alat koreksi aktif, **atau** ada
/// koreksi tertunda yang belum diterapkan. Koreksi tidak dibuang saat
/// berganti alat (putar → geser → balik) supaya Terapkan cukup sekali di
/// akhir; ia hanya dibuang lewat Atur ulang, Terapkan, atau ganti irisan.
pub fn editing_active(st: &ReconState) -> bool {
    st.tool.corrects_slice() || st.edit.has_change()
}

/// Bilah pilihan alat 3D perlu tampil: koreksi berlangsung atau fokus dua
/// irisan aktif (bilah memuat pemilih patokan dan Tampilkan semua).
pub fn options_bar_active(st: &ReconState) -> bool {
    editing_active(st) || st.focus.is_some()
}

/// Transformasi irisan yang sedang dikoreksi: `(T lama, T baru, pivot W)`;
/// keduanya memetakan U irisan → W. `None` bila tidak ada irisan diedit
/// atau irisan tidak ada di cache.
pub fn edit_transforms(st: &ReconState) -> Option<(SlideId, Affine2, Affine2, [f64; 2])> {
    if !editing_active(st) {
        return None;
    }
    let id = st.edit.slide?;
    let it = st.slices.iter().find(|s| s.id == id && !s.preview)?;
    let t_old = it.draw.affine;
    let pivot = t_old.apply([it.draw.size_um[0] / 2.0, it.draw.size_um[1] / 2.0]);
    let t_new =
        manual_adjusted_transform(&t_old, st.edit.shift, st.edit.angle, st.edit.flip, pivot);
    Some((id, t_old, t_new, pivot))
}

/// Irisan `Ok` yang berada **sesudah** `slide` di rantai registrasi (lebih
/// jauh dari acuan, sisi yang sama). Koreksi `slide` menggeser semuanya.
pub fn downstream_of(sess: &Session, slide: SlideId) -> Vec<SlideId> {
    let project = sess.project();
    let Some(reference) = project.effective_reference() else {
        return Vec::new();
    };
    let usable: Vec<SlideId> = project
        .usable_slides_ordered()
        .iter()
        .map(|s| s.id)
        .collect();
    let (Some(i), Some(r)) = (
        usable.iter().position(|s| *s == slide),
        usable.iter().position(|s| *s == reference),
    ) else {
        return Vec::new();
    };
    match i.cmp(&r) {
        std::cmp::Ordering::Less => usable[..i].to_vec(),
        std::cmp::Ordering::Greater => usable[i + 1..].to_vec(),
        std::cmp::Ordering::Equal => Vec::new(),
    }
}

/// Samakan irisan yang diedit dengan irisan aktif: ganti irisan membuang
/// koreksi yang belum diterapkan dan menghitung ulang rantai sesudahnya.
pub fn sync_edit(app: &mut AppState, sess: &Session) {
    let st = &mut app.recon;
    if !editing_active(st) {
        return;
    }
    if st.edit.slide != app.current {
        st.edit.reset();
        st.edit.slide = app.current;
        st.edit.downstream = app
            .current
            .map(|id| downstream_of(sess, id))
            .unwrap_or_default();
    }
}

/// Irisan `id` ikut digambar di tumpukan: tidak disembunyikan pengguna dan,
/// bila fokus dua irisan aktif, termasuk pasangan fokus.
pub fn slice_visible(st: &ReconState, id: SlideId) -> bool {
    !st.hidden_slices.contains(&id) && st.focus.is_none_or(|(a, b)| a == id || b == id)
}

/// Nyalakan fokus dua irisan untuk `slide`: patokan bawaan adalah tetangga
/// ke arah acuan (irisan yang menjadi target registrasinya); bila tidak ada
/// (irisan acuan sendiri), dipakai irisan terdekat lain di tumpukan. `None`
/// bila tidak ada irisan lain. Kamera dipaskan ulang ke pasangan.
pub fn focus_on(app: &mut AppState, sess: &Session, slide: SlideId) -> Option<(SlideId, SlideId)> {
    let partner = crate::features::register::neighbor_of(sess, slide)
        .filter(|p| *p != slide)
        .or_else(|| nearest_other_slice(&app.recon.slices, slide))?;
    app.recon.focus = Some((slide, partner));
    app.recon.fitted = false;
    Some((slide, partner))
}

/// Matikan fokus dua irisan: semua irisan tampil lagi, kamera dipaskan ulang.
pub fn focus_clear(st: &mut ReconState) {
    if st.focus.take().is_some() {
        st.fitted = false;
    }
}

/// Irisan lain (bukan pratinjau) dengan `z` terdekat ke `slide`.
fn nearest_other_slice(items: &[SliceItem], slide: SlideId) -> Option<SlideId> {
    let z = items
        .iter()
        .find(|i| i.id == slide && !i.preview)?
        .draw
        .z_um;
    items
        .iter()
        .filter(|i| i.id != slide && !i.preview)
        .min_by(|a, b| {
            (a.draw.z_um - z)
                .abs()
                .total_cmp(&(b.draw.z_um - z).abs())
                .then(a.id.0.cmp(&b.id.0))
        })
        .map(|i| i.id)
}

/// Samakan fokus dengan keadaan tumpukan: pasangan yang sudah tidak ada di
/// tumpukan membatalkan fokus; memilih irisan ketiga (dari daftar atau
/// kanvas) menjadikannya irisan yang dikoreksi dengan patokan tetap, supaya
/// irisan yang dipilih tidak pernah tersembunyi.
pub fn sync_focus(app: &mut AppState) {
    let st = &mut app.recon;
    let Some((a, b)) = st.focus else {
        return;
    };
    let present = |id: SlideId| st.slices.iter().any(|s| s.id == id && !s.preview);
    if !present(a) || !present(b) || a == b {
        focus_clear(st);
        return;
    }
    if let Some(cur) = app.current {
        if cur != a && cur != b {
            st.focus = Some((cur, b));
            st.fitted = false;
        }
    }
}

/// Potong `ray` dengan **bidang** (tak terbatas) quad irisan bermatriks
/// `model`; hasil di ruang W (µm, XY). `None` bila sejajar atau di belakang.
fn plane_hit_w(ray: &Ray, model: &Mat4, center_um: [f64; 3]) -> Option<[f64; 2]> {
    if model.determinant().abs() < 1e-18 {
        return None;
    }
    let inv = model.inverse();
    let o = inv.project_point3(ray.origin);
    let d = inv.transform_vector3(ray.dir);
    if d.z.abs() < 1e-12 {
        return None;
    }
    let t = -o.z / d.z;
    if !(t.is_finite() && t > 0.0) {
        return None;
    }
    let p = ray.origin + ray.dir * t;
    let w = render_to_world_um(p.to_array(), center_um);
    Some([w[0], w[1]])
}

/// Sudut `p` relatif `pivot` di bidang W.
fn angle_about(p: [f64; 2], pivot: [f64; 2]) -> f64 {
    (p[1] - pivot[1]).atan2(p[0] - pivot[0])
}

/// Selisih sudut terkecil `a1 − a0` dalam `(−π, π]`.
fn wrap_angle(d: f64) -> f64 {
    let mut d = d;
    if d > std::f64::consts::PI {
        d -= std::f64::consts::TAU;
    } else if d < -std::f64::consts::PI {
        d += std::f64::consts::TAU;
    }
    d
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
    /// Fokus dua irisan (koreksi QC gagal di tengah urutan): hanya
    /// `(irisan yang dikoreksi, patokan)` yang digambar; irisan lain
    /// disembunyikan sementara (tampilan saja). `None` = tampilkan semua.
    pub focus: Option<(SlideId, SlideId)>,
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
    /// Jarak irisan (µm) yang sedang digeser di slider tetapi belum
    /// dikomit ke sesi: `z_um` bawaan digambar dengan nilai ini supaya
    /// tumpukan bereaksi langsung; satu entri undo dibuat saat dilepas.
    pub spacing_preview: Option<f64>,
    /// Lapisan skalar terpilih (nama `MeshScalars`).
    pub scalar: Option<String>,
    /// Rentang lapisan skalar; `None` = otomatis p2–p98.
    pub range: Option<[f32; 2]>,
    /// Colormap.
    pub colormap: Colormap,
    /// Hasil klik terakhir pada mesh.
    pub pick: Option<PickInfo>,
    /// Objek 3D terpilih (klik di kanvas atau daftar objek).
    pub selected: Option<MeshKey>,
    /// Angkat objek terpilih sepanjang +Z ruang R (mm, setelah renggang Z);
    /// tampilan saja, tidak pernah menyentuh data.
    pub lift_mm: f32,
    /// Isolasi: objek selain yang terpilih diredupkan (tampilan saja).
    pub isolate: bool,
    /// Permintaan memaskan kamera ke kotak `(lo, hi)` (mm, ruang R
    /// berenggang) pada frame kanvas berikutnya, saat rasio viewport
    /// diketahui.
    pub fit_request: Option<(Vec3, Vec3)>,
    /// Kunci mesh `(label, id objek)` yang ada di GPU.
    pub gpu_meshes: BTreeSet<MeshKey>,
    /// Pilih lapisan skalar pertama begitu tersedia (argumen `--analyze`).
    pub want_first_scalar: bool,
    /// Kubus orientasi.
    pub viewcube: ViewCube,
    /// Alat aktif di kanvas.
    pub tool: StackTool,
    /// Koreksi irisan terpilih yang belum diterapkan.
    pub edit: StackEdit,
    /// Gaya tampilan 3D.
    pub look: ViewLook,
    /// Gaya ilustrasi: pakai warna pulasan verteks (bila mesh membawanya).
    pub stain_colors: bool,
    /// Gaya ilustrasi: irisan tampak seperti kaca objek.
    pub glass: bool,
    /// Tonjolan sel prosedural (ILUSTRATIF, bawaan mati).
    pub cell_bumps: bool,
    /// Jarak pusat sel tampilan, µm.
    pub cell_size_um: f32,
    /// Kekuatan tonjolan sel 0..1.
    pub cell_strength: f32,
    /// Putar kamera otomatis mengelilingi target.
    pub auto_rotate: bool,
    /// Pisahkan objek menjauhi pusat tumpukan (0 = rapat; tampilan saja).
    pub explode: f32,
    /// Nama skalar densitas inti (`attr:<nama>`) untuk membobot tonjolan sel.
    pub density_scalar: Option<String>,
    /// Pusat kotak pembatas tiap mesh (mm, ruang R, sebelum renggang Z).
    pub centers_mm: BTreeMap<MeshKey, Vec3>,
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
            focus: None,
            uploaded_slices: BTreeSet::new(),
            slices: Vec::new(),
            hovered_slice: None,
            z_scale: 1.0,
            z_scale_auto: true,
            spacing_preview: None,
            scalar: None,
            range: None,
            colormap: Colormap::Viridis,
            pick: None,
            selected: None,
            lift_mm: 0.0,
            isolate: false,
            fit_request: None,
            gpu_meshes: BTreeSet::new(),
            want_first_scalar: false,
            viewcube: ViewCube::default(),
            tool: StackTool::Orbit,
            edit: StackEdit::default(),
            look: ViewLook::Technical,
            stain_colors: true,
            glass: true,
            cell_bumps: false,
            cell_size_um: 60.0,
            cell_strength: 0.55,
            auto_rotate: false,
            explode: 0.0,
            density_scalar: None,
            centers_mm: BTreeMap::new(),
        }
    }
}

impl ReconState {
    /// Terapkan gaya tampilan beserta bawaannya (preset).
    pub fn set_look(&mut self, look: ViewLook) {
        self.look = look;
        if look == ViewLook::Tissue {
            self.stain_colors = true;
            self.glass = true;
        } else {
            self.cell_bumps = false;
        }
    }
}

/// Material mesh `m` untuk gaya tampilan `st.look` (hanya tampilan).
pub fn mesh_material(st: &ReconState, m: &LabelMesh) -> h3d_render::MeshMaterial {
    match st.look {
        ViewLook::Technical => h3d_render::MeshMaterial::default(),
        ViewLook::Tissue => h3d_render::MeshMaterial {
            stain_colors: st.stain_colors && m.colors.len() == m.positions_um.len(),
            cells: st.cell_bumps.then(|| h3d_render::CellBumps {
                size_mm: st.cell_size_um.max(1.0) / 1000.0,
                strength: st.cell_strength.clamp(0.0, 1.0),
                nucleus_mix: 0.55,
            }),
            ..h3d_render::MeshMaterial::tissue()
        },
    }
}

/// Pencahayaan 3D untuk gaya tampilan.
pub fn look_lighting(look: ViewLook) -> h3d_render::Lighting {
    match look {
        ViewLook::Technical => h3d_render::Lighting::default(),
        ViewLook::Tissue => h3d_render::Lighting::studio(),
    }
}

/// Nama skalar densitas inti (atribut `NucleiDensity` pertama) di proyek.
pub fn nuclei_density_scalar(project: &h3d_core::Project) -> Option<String> {
    project
        .attrs
        .iter()
        .find(|a| matches!(a.kind, h3d_core::AttrKind::NucleiDensity))
        .map(|a| format!("attr:{}", a.name))
}

/// Indeks irisan yang diberi kesan kaca: paling banyak [`GLASS_MAX_SLICES`]
/// tersebar rata (bawah, tengah, atas). Kaca di semua irisan menumpuk
/// menjadi kabut yang menutupi mesh.
pub fn glass_slice_indices(n: usize) -> Vec<usize> {
    if n <= GLASS_MAX_SLICES {
        return (0..n).collect();
    }
    let mut v: Vec<usize> = (0..GLASS_MAX_SLICES)
        .map(|i| (i * (n - 1) + (GLASS_MAX_SLICES - 1) / 2) / (GLASS_MAX_SLICES - 1))
        .collect();
    v.dedup();
    v
}

/// Latar gradasi gelap dengan pendar lembut di tengah (gaya ilustrasi).
fn paint_tissue_background(painter: &egui::Painter, rect: egui::Rect) {
    use egui::epaint::{Mesh, Vertex, WHITE_UV};
    let top = egui::Color32::from_rgb(18, 28, 44);
    let bottom = egui::Color32::from_rgb(5, 8, 14);
    let mut m = Mesh::default();
    let v = |pos: egui::Pos2, color: egui::Color32| Vertex {
        pos,
        uv: WHITE_UV,
        color,
    };
    m.vertices.push(v(rect.left_top(), top));
    m.vertices.push(v(rect.right_top(), top));
    m.vertices.push(v(rect.right_bottom(), bottom));
    m.vertices.push(v(rect.left_bottom(), bottom));
    m.add_triangle(0, 1, 2);
    m.add_triangle(0, 2, 3);
    // Pendar: kipas dari pusat (sian redup) ke tepi transparan.
    let c = rect.center();
    let r = 0.6 * rect.width().max(rect.height());
    let glow = egui::Color32::from_rgba_unmultiplied(40, 110, 140, 70);
    let base = m.vertices.len() as u32;
    m.vertices.push(v(c, glow));
    let n = 48u32;
    for i in 0..=n {
        let a = std::f32::consts::TAU * i as f32 / n as f32;
        m.vertices.push(v(
            c + egui::vec2(a.cos() * r, a.sin() * r * 0.75),
            egui::Color32::TRANSPARENT,
        ));
    }
    for i in 0..n {
        m.add_triangle(base, base + 1 + i, base + 2 + i);
    }
    painter.add(egui::Shape::mesh(m));
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

/// Laci "Gaya tampilan": preset teknis/ilustrasi, warna pulasan, kaca,
/// tonjolan sel ilustratif, putar otomatis, pisahkan objek.
fn look_section(st: &mut ReconState, ui: &mut Ui) {
    egui::CollapsingHeader::new(icon_label(h3d_ui::icons::LIGHT_MODE, &t!("recon-look")))
        .default_open(true)
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                for (look, key) in [
                    (ViewLook::Technical, "recon-look-technical"),
                    (ViewLook::Tissue, "recon-look-tissue"),
                ] {
                    if ui.selectable_label(st.look == look, t!(key)).clicked() {
                        st.set_look(look);
                    }
                }
            });
            let tissue = st.look == ViewLook::Tissue;
            ui.add_enabled_ui(tissue, |ui| {
                let has_colors = st.meshes.iter().any(|m| !m.colors.is_empty());
                ui.add_enabled(
                    has_colors,
                    egui::Checkbox::new(&mut st.stain_colors, t!("recon-stain-colors")),
                )
                .on_hover_text(t!("recon-stain-colors-hint"))
                .on_disabled_hover_text(t!("recon-stain-colors-missing"));
                ui.checkbox(&mut st.glass, t!("recon-glass"));
                ui.checkbox(&mut st.cell_bumps, t!("recon-cell-bumps"))
                    .on_hover_text(t!("recon-illustrative-warning"));
                if st.cell_bumps {
                    ui.add(
                        egui::Slider::new(&mut st.cell_size_um, CELL_SIZE_RANGE)
                            .logarithmic(true)
                            .text(t!("recon-cell-size")),
                    );
                    ui.add(
                        egui::Slider::new(&mut st.cell_strength, 0.0..=1.0)
                            .text(t!("recon-cell-strength")),
                    );
                    ui.horizontal_wrapped(|ui| {
                        ui.label(icon_label(h3d_ui::icons::WARNING, ""));
                        ui.small(t!("recon-illustrative-warning"));
                    });
                }
            });
            ui.checkbox(
                &mut st.auto_rotate,
                icon_label(h3d_ui::icons::ROTATE, &t!("recon-auto-rotate")),
            );
            let multi = st.centers_mm.len() > 1;
            ui.add_enabled(
                multi,
                egui::Slider::new(&mut st.explode, EXPLODE_RANGE).text(t!("recon-explode")),
            )
            .on_hover_text(t!("recon-explode-hint"));
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
/// `spacing_preview` = jarak irisan (µm) yang sedang digeser di slider;
/// `z_um` yang masih bawaan dihitung dari nilai itu (tampilan saja, sesi
/// tidak disentuh); `None` = pakai `z_um` manifes.
pub fn slice_items(
    sess: &Session,
    preview: Option<&OrderPreview>,
    alpha: f32,
    spacing_preview: Option<f64>,
) -> Vec<SliceItem> {
    let project = sess.project();
    let spacing = spacing_preview.unwrap_or(project.recon.default_spacing_um);
    if let Some(p) = preview {
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
                        glass: 0.0,
                    },
                    flagged: p.suggestion.uncertain.contains(id),
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
                z_um: match spacing_preview {
                    Some(sp) if e.z_is_default => f64::from(e.order) * sp,
                    _ => e.z_um,
                },
                alpha,
                glass: 0.0,
            },
            flagged: e.reg.as_ref().is_some_and(|r| r.flagged),
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
    st.density_scalar = nuclei_density_scalar(sess.project()).filter(|n| names.contains(n));
    if st.want_first_scalar && !names.is_empty() {
        st.want_first_scalar = false;
        st.scalar = names.first().cloned();
    }
    st.pick = None;
    if st
        .selected
        .is_some_and(|k| !st.meshes.iter().any(|m| mesh_key(m) == k))
    {
        select_object(st, None);
    }
    st.needs_upload = true;
}

/// Pilih (atau lepas) objek 3D; angkat dan isolasi ikut dibuang saat
/// pilihan dilepas supaya tidak ada objek "tersangkut" di udara.
pub fn select_object(st: &mut ReconState, key: Option<MeshKey>) {
    if key != st.selected {
        st.lift_mm = 0.0;
    }
    st.selected = key;
    if key.is_none() {
        st.isolate = false;
    }
}

/// Offset tampilan mesh `key` (mm, ruang R setelah renggang Z): objek
/// terpilih diangkat `lift_mm`; bila "pisahkan objek" aktif, tiap objek
/// digeser menjauhi pusat tumpukan sebanding jaraknya × `explode`
/// (renggang Z ikut diterapkan pada arah geser). Hanya tampilan.
pub fn mesh_offset_mm(st: &ReconState, key: MeshKey) -> [f32; 3] {
    let mut off = Vec3::ZERO;
    if st.selected == Some(key) && st.lift_mm != 0.0 {
        off.z += st.lift_mm;
    }
    if st.explode > 0.0 && st.centers_mm.len() > 1 {
        if let (Some(c), Some((lo, hi))) = (st.centers_mm.get(&key), st.bounds_mm) {
            let s = h3d_render::sanitize_z_scale(st.z_scale);
            let mid = (lo + hi) * 0.5;
            let d = *c - mid;
            off += Vec3::new(d.x, d.y, d.z * s) * st.explode;
        }
    }
    off.to_array()
}

/// Kotak pembatas mesh `key` (mm, ruang R, setelah renggang Z dan offset);
/// dipakai untuk memfokuskan kamera ke objek terpilih.
pub fn mesh_bounds_mm(st: &ReconState, key: MeshKey) -> Option<(Vec3, Vec3)> {
    let m = st.meshes.iter().find(|m| mesh_key(m) == key)?;
    let (lo, hi) = m.bbox()?;
    let s = h3d_render::sanitize_z_scale(st.z_scale);
    let off = Vec3::from(mesh_offset_mm(st, key));
    let a = Vec3::from(world_um_to_render_mm(lo, st.center_um));
    let b = Vec3::from(world_um_to_render_mm(hi, st.center_um));
    let (a, b) = (a.min(b), a.max(b));
    Some((
        Vec3::new(a.x, a.y, a.z * s) + off,
        Vec3::new(b.x, b.y, b.z * s) + off,
    ))
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
    st.centers_mm = st
        .meshes
        .iter()
        .filter_map(|m| {
            let (lo, hi) = m.bbox()?;
            let a = Vec3::from(world_um_to_render_mm(lo, st.center_um));
            let c = Vec3::from(world_um_to_render_mm(hi, st.center_um));
            Some((mesh_key(m), (a + c) * 0.5))
        })
        .collect();
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
    let density = st.density_scalar.clone();
    let meshes = &st.meshes;
    let mut uploaded = BTreeSet::new();
    gpu.with(|r, device, queue| {
        for l in &stale {
            r.remove_mesh(mesh_id(*l));
        }
        for m in meshes {
            let named = |n: &Option<String>| {
                n.as_ref()
                    .and_then(|n| m.scalars.iter().find(|s| &s.name == n))
                    .map(|s| s.values.as_slice())
            };
            let nv = m.positions_um.len();
            let attrs = VertexAttrs {
                scalars: named(&scalar),
                colors: (m.colors.len() == nv).then_some(m.colors.as_slice()),
                occlusion: (m.occlusion.len() == nv).then_some(m.occlusion.as_slice()),
                density: named(&density),
            };
            match upload_mesh(
                device,
                queue,
                &m.positions_um,
                &m.normals,
                &m.indices,
                center,
                &attrs,
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
    // Slider jarak hilang di tengah geser (laci ditutup) → pratinjau basi
    // dibuang begitu tombol tetikus/jari dilepas.
    if app.recon.spacing_preview.is_some() && !ctx.input(|i| i.pointer.any_down()) {
        app.recon.spacing_preview = None;
    }
    let preview = app.slides.order_preview.as_ref();
    let alpha = app.recon.slice_alpha;
    let items = slice_items(sess, preview, alpha, app.recon.spacing_preview);
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
            // Thumbnail dibuat di thread latar; frame ini hanya mengunggah
            // yang sudah siap.
            let key = crate::thumbs::ThumbKey {
                slide: it.id,
                side: SLICE_TEX_SIDE,
                tissue: st.hide_background,
            };
            app.thumb_worker.ensure_requested(sess, key);
            if budget == 0 {
                ctx.request_repaint();
                break;
            }
            let Some(thumb) = app.thumb_worker.take(&key) else {
                ctx.request_repaint();
                continue;
            };
            budget -= 1;
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
                Err(err) => {
                    // Dicatat sekali; tidak diulang tiap frame.
                    log::warn!("thumbnail irisan {} gagal: {err}", it.id);
                    st.uploaded_slices.insert(it.id);
                }
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
        for it in st.slices.iter().filter(|i| slice_visible(st, i.id)) {
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
            if app.derived_ready() {
                reload(app, sess);
            } else {
                ctx.request_repaint_after(std::time::Duration::from_millis(
                    crate::state::DERIVED_DEBOUNCE_MS,
                ));
            }
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
        sync_focus(app);
        sync_edit(app, sess);
    }

    let response = ui.interact(
        rect,
        ui.id().with("stack-canvas"),
        egui::Sense::click_and_drag(),
    );
    if app.recon.look == ViewLook::Tissue {
        paint_tissue_background(&ui.painter_at(rect), rect);
    } else {
        ui.painter_at(rect)
            .rect_filled(rect, 0.0, h3d_ui::theme::canvas_fill(ui.visuals()));
    }
    let aspect = (rect.width() / rect.height().max(1.0)).max(0.01);
    let st = &mut app.recon;
    // Putar otomatis: berhenti selama pengguna menyeret kamera.
    if st.auto_rotate && !response.dragged() {
        let dt = ctx.input(|i| i.stable_dt).min(0.1);
        st.camera.orbit(AUTO_ROTATE_PX_PER_S * dt, 0.0);
        ctx.request_repaint();
    }
    if !st.fitted {
        if let Some((lo, hi)) = canvas_bounds(st) {
            st.camera.fit_bounds(lo, hi, aspect);
            st.fitted = true;
        }
    }
    if let Some((lo, hi)) = st.fit_request.take() {
        st.camera.fit_bounds(lo, hi, aspect);
        st.fitted = true;
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
        // Koreksi irisan: seretan yang dimulai di irisan terpilih memutar /
        // menggeser irisan itu di bidangnya; seretan di luar tetap orbit.
        let edit_ctx = sess
            .and(edit_transforms(st))
            .filter(|_| st.tool.corrects_slice());
        if let Some((_, _, t_now, pivot)) = edit_ctx {
            let z_scale = h3d_render::sanitize_z_scale(st.z_scale);
            let item = st.slices.iter().find(|s| Some(s.id) == st.edit.slide);
            let model = item.map(|it| {
                slice_model_matrix_scaled(
                    &slice_model_matrix(&t_now, it.draw.size_um, it.draw.z_um, st.center_um),
                    z_scale,
                )
            });
            if let Some(model) = model {
                if response.drag_started_by(egui::PointerButton::Primary) && !shift {
                    if let Some(pos) = response.interact_pointer_pos() {
                        let ray = ray_from_screen(&st.camera, rect, pos);
                        st.edit.dragging = pick_slice(&ray, &model).is_some();
                    }
                }
                if response.drag_stopped() {
                    st.edit.dragging = false;
                }
                if st.edit.dragging && response.dragged_by(egui::PointerButton::Primary) {
                    if let Some(pos) = response.interact_pointer_pos() {
                        let prev = pos - d;
                        let r1 = ray_from_screen(&st.camera, rect, pos);
                        let r0 = ray_from_screen(&st.camera, rect, prev);
                        let hit = (
                            plane_hit_w(&r0, &model, st.center_um),
                            plane_hit_w(&r1, &model, st.center_um),
                        );
                        if let (Some(w0), Some(w1)) = hit {
                            match st.tool {
                                StackTool::Rotate => {
                                    let p =
                                        [pivot[0] + st.edit.shift[0], pivot[1] + st.edit.shift[1]];
                                    st.edit.angle +=
                                        wrap_angle(angle_about(w1, p) - angle_about(w0, p));
                                }
                                StackTool::Translate => {
                                    st.edit.shift[0] += w1[0] - w0[0];
                                    st.edit.shift[1] += w1[1] - w0[1];
                                }
                                StackTool::Orbit | StackTool::Object => {}
                            }
                        }
                    }
                }
            }
        } else {
            st.edit.dragging = false;
        }
        if pan {
            st.camera.pan(d.x, d.y, rect.height());
        } else if response.dragged_by(egui::PointerButton::Primary) && !st.edit.dragging {
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

    if st.selected.is_some()
        && response.contains_pointer()
        && ctx.input(|i| i.key_pressed(egui::Key::Escape))
    {
        select_object(st, None);
        st.pick = None;
    }

    let z_scale = h3d_render::sanitize_z_scale(st.z_scale);
    let view_proj = st.camera.view_proj(aspect);
    let previewing = st.slices.iter().any(|s| s.preview);

    // Irisan yang digambar + matriks model berenggang (untuk picking/garis).
    // Koreksi yang belum diterapkan dipratinjau: irisan yang diedit memakai
    // `T` baru, irisan sesudahnya di rantai ikut `T_baru ∘ T_lama⁻¹` (8.4.5).
    let draw_slices = st.show_slices;
    let edit_ctx = sess.and(edit_transforms(st));
    let chain_delta =
        edit_ctx.and_then(|(_, t_old, t_new, _)| t_old.inverse().map(|inv| t_new.compose(&inv)));
    let edit_id = edit_ctx.map(|(id, ..)| id);
    let visible: Vec<(SliceItem, Mat4)> = st
        .slices
        .iter()
        .filter(|it| draw_slices && slice_visible(st, it.id))
        .map(|it| {
            let mut it = it.clone();
            if !it.preview {
                if Some(it.id) == edit_id {
                    if let Some((_, _, t_new, _)) = edit_ctx {
                        it.draw.affine = t_new;
                    }
                } else if st.edit.downstream.contains(&it.id) {
                    if let Some(delta) = chain_delta {
                        it.draw.affine = delta.compose(&it.draw.affine);
                    }
                }
            }
            let m =
                slice_model_matrix(&it.draw.affine, it.draw.size_um, it.draw.z_um, st.center_um);
            let ms = slice_model_matrix_scaled(&m, z_scale);
            (it, ms)
        })
        .collect();

    // Picking irisan di bawah kursor (hanya bila sesi tersedia dan tidak
    // sedang menyeret kamera).
    let mut event = CanvasEvent::None;
    st.hovered_slice = None;
    let pointer = response.hover_pos().filter(|p| rect.contains(*p));
    if sess.is_some() && !response.dragged() && !st.edit.dragging {
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
                        let offsets: Vec<[f32; 3]> =
                            keys.iter().map(|k| mesh_offset_mm(st, *k)).collect();
                        g.with(|r, _, _| {
                            let mut best: Option<(MeshKey, h3d_render::PickHit)> = None;
                            for (k, off) in keys.into_iter().zip(offsets) {
                                // Offset angkat diterapkan setelah renggang Z,
                                // jadi dibagi balik sebelum ray yang belum
                                // direnggangkan.
                                let off = [off[0], off[1], off[2] / z_scale];
                                if let Some(h) = r
                                    .mesh(mesh_id(k))
                                    .and_then(|m| m.pick_offset(&unscaled, off))
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
                            instance_id: key.1,
                            point_um,
                            value,
                            nearest_slide: sess.and_then(|s| nearest_slide(s, point_um[2])),
                        });
                        select_object(st, Some(key));
                    }
                    _ => {
                        if let Some(id) = st.hovered_slice.filter(|_| !previewing) {
                            event = CanvasEvent::Selected(id);
                        } else if st.tool == StackTool::Object {
                            // Klik ruang kosong dengan alat objek = lepas pilihan.
                            select_object(st, None);
                            st.pick = None;
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
            let key = mesh_key(m);
            let is_sel = st.selected == Some(key);
            let mut opacity = st.opacity.get(&m.label).copied().unwrap_or(1.0);
            if st.isolate && st.selected.is_some() && !is_sel {
                opacity *= ISOLATE_DIM;
            }
            instances.push(MeshInstance {
                mesh: mesh_id(key),
                color: [
                    f32::from(color[0]) / 255.0,
                    f32::from(color[1]) / 255.0,
                    f32::from(color[2]) / 255.0,
                    1.0,
                ],
                opacity,
                visible: !st.hidden.contains(&m.label),
                scalar_mode: has_scalar,
                scalar_range: range.unwrap_or([0.0, 1.0]),
                colormap: st.colormap,
                offset_mm: mesh_offset_mm(st, key),
                highlight: if is_sel { SELECT_HIGHLIGHT } else { 0.0 },
                material: mesh_material(st, m),
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
        let mut slices: Vec<SliceDraw> = visible
            .iter()
            .filter(|(it, _)| st.uploaded_slices.contains(&it.id))
            .map(|(it, _)| it.draw.clone())
            .collect();
        if st.look == ViewLook::Tissue && st.glass {
            for i in glass_slice_indices(slices.len()) {
                slices[i].glass = GLASS_STRENGTH;
            }
        }
        let scene = Scene3D {
            camera: st.camera.clone(),
            aspect,
            center_um: st.center_um,
            meshes: instances,
            slices,
            clip_plane,
            lighting: look_lighting(st.look),
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
        } else if edit_id.is_some() && st.edit.downstream.contains(&it.id) {
            Some(egui::Stroke::new(
                1.0,
                h3d_ui::theme::ACCENT_LIGHT.gamma_multiply(0.8),
            ))
        } else if it.flagged {
            Some(egui::Stroke::new(1.0, h3d_ui::theme::WARNING))
        } else if it.preview {
            Some(egui::Stroke::new(
                1.0,
                h3d_ui::theme::ACCENT_LIGHT.gamma_multiply(0.6),
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

    // Pegangan koreksi: pivot di pusat irisan yang diedit (sama dengan 2D).
    if let Some((id, _, _, pivot)) = edit_ctx.filter(|_| st.tool.corrects_slice()) {
        let z_um = visible
            .iter()
            .find(|(it, _)| it.id == id)
            .map(|(it, _)| it.draw.z_um);
        if let Some(z_um) = z_um {
            let p = [pivot[0] + st.edit.shift[0], pivot[1] + st.edit.shift[1]];
            let r = world_um_to_render_mm([p[0], p[1], z_um], st.center_um);
            let r = Vec3::new(r[0], r[1], r[2] * z_scale);
            if let Some(c) = project(&view_proj, rect, r) {
                let accent = h3d_ui::theme::ACCENT;
                painter.circle_filled(c, 13.0, egui::Color32::from_black_alpha(160));
                let icon = if st.tool == StackTool::Translate {
                    h3d_ui::icons::MOVE
                } else {
                    h3d_ui::icons::ROTATE
                };
                painter.text(
                    c,
                    egui::Align2::CENTER_CENTER,
                    icon.codepoint,
                    egui::FontId::new(18.0, icon.font_family()),
                    accent,
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
    if st.edit.dragging {
        ctx.set_cursor_icon(egui::CursorIcon::Grabbing);
    } else if st.tool.corrects_slice() && edit_id.is_some() && st.hovered_slice == edit_id {
        ctx.set_cursor_icon(egui::CursorIcon::Grab);
    } else if st.hovered_slice.is_some() {
        ctx.set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    event
}

/// Setel alat kanvas 3D (dari rel alat); memilih alat yang sudah aktif
/// mengembalikan ke orbit. Koreksi yang belum diterapkan **dipertahankan**
/// (tetap dipratinjau dan bilah Terapkan tetap tampil) supaya pengguna bisa
/// menggabungkan balik, putar, dan geser lalu menerapkan sekali.
pub fn set_tool(app: &mut AppState, tool: StackTool) {
    let st = &mut app.recon;
    st.tool = if st.tool == tool {
        StackTool::Orbit
    } else {
        tool
    };
    st.edit.dragging = false;
}

/// Bilah opsi alat koreksi di kanvas 3D: sudut dan geseran yang belum
/// diterapkan (bisa diketik), putar ±90°, Terapkan, dan atur ulang.
/// Penerapan memakai jalur yang sama dengan mode 2D (`apply_manual_transform`).
pub fn tool_options(app: &mut AppState, sess: &mut Session, ui: &mut Ui) {
    sync_focus(app);
    sync_edit(app, sess);
    focus_options(app, sess, ui);
    if app.recon.tool == StackTool::Object {
        object_options(app, sess, ui);
        return;
    }
    if !editing_active(&app.recon) {
        return;
    }
    let Some(slide) = app.current else {
        ui.weak(t!("stack-edit-no-selection"));
        return;
    };
    let neighbor = crate::features::register::neighbor_of(sess, slide);
    let n_down = app.recon.edit.downstream.len();
    ui.horizontal_wrapped(|ui| {
        crate::ui_util::strong(ui, crate::ui_util::slide_title(sess, slide));
        let st = &mut app.recon;
        ui.label(t!("register-rotate"));
        let mut deg = st.edit.angle.to_degrees();
        if ui
            .add(
                egui::DragValue::new(&mut deg)
                    .speed(0.5)
                    .range(-360.0..=360.0)
                    .suffix(" deg"),
            )
            .changed()
        {
            st.edit.angle = deg.to_radians();
        }
        if ui.button("-90").clicked() {
            st.edit.angle = wrap_angle(st.edit.angle - std::f64::consts::FRAC_PI_2);
        }
        if ui.button("+90").clicked() {
            st.edit.angle = wrap_angle(st.edit.angle + std::f64::consts::FRAC_PI_2);
        }
        ui.label(format!(
            "{}: {:.0}, {:.0} {}",
            t!("register-translate"),
            st.edit.shift[0],
            st.edit.shift[1],
            t!("unit-um")
        ));
        crate::features::register::flip_buttons(ui, &mut st.edit.flip);
        if crate::ui_util::icon_text_button_enabled(
            ui,
            st.edit.has_change() && neighbor.is_some(),
            h3d_ui::icons::CHECK,
            &t!("action-apply"),
        )
        .clicked()
        {
            if let Ok(entry) = sess.slide(slide) {
                let t_old = entry.transform.affine();
                let size = entry.size_um();
                let pivot = t_old.apply([size[0] / 2.0, size[1] / 2.0]);
                // `t_new` memetakan U irisan → W (arah `T_i`, 5.3).
                let t_new = manual_adjusted_transform(
                    &t_old,
                    st.edit.shift,
                    st.edit.angle,
                    st.edit.flip,
                    pivot,
                );
                match sess.apply_manual_transform(slide, t_new) {
                    Ok(()) => {
                        st.edit.reset();
                        app.toasts.info(t!("register-correction-applied"));
                        app.invalidate_derived();
                    }
                    Err(e) => app.toasts.engine_error(&e),
                }
            }
        }
        if ui.button(t!("register-reset-transform")).clicked() {
            app.recon.edit.reset();
        }
    });
    if neighbor.is_none() {
        ui.small(t!("stack-edit-reference"));
    } else {
        ui.small(t!("stack-edit-hint", n = n_down.to_string()));
    }
}

/// Nama tampilan objek `key` (`Label` atau `Label#id`).
pub fn object_title(sess: &Session, key: MeshKey) -> String {
    let name = sess
        .project()
        .label(key.0)
        .map_or_else(|| key.0.to_string(), |l| l.name.clone());
    match key.1 {
        Some(id) => format!("{name}#{id}"),
        None => name,
    }
}

/// Minta kamera difokuskan ke objek terpilih (kotak pembatasnya, termasuk
/// angkat); dipenuhi kanvas pada frame berikutnya.
fn focus_camera_on_selected(st: &mut ReconState) {
    st.fit_request = st.selected.and_then(|k| mesh_bounds_mm(st, k));
}

/// Bilah opsi alat objek: nama objek terpilih, penggeser Angkat, Isolasi,
/// Fokus kamera, Kembalikan, dan lepas pilihan. Semua hanya tampilan.
fn object_options(app: &mut AppState, sess: &Session, ui: &mut Ui) {
    let st = &mut app.recon;
    let Some(key) = st.selected else {
        ui.small(t!("recon-object-none"));
        return;
    };
    let title = object_title(sess, key);
    ui.horizontal_wrapped(|ui| {
        ui.label(icon_label(
            h3d_ui::icons::OBJECT_PICK,
            &t!("recon-object-title"),
        ));
        crate::ui_util::strong(ui, title);
        ui.label(icon_label(
            h3d_ui::icons::OBJECT_LIFT,
            &t!("recon-object-lift"),
        ));
        ui.add(
            egui::Slider::new(&mut st.lift_mm, LIFT_RANGE)
                .suffix(format!(" {}", t!("unit-mm")))
                .show_value(true),
        )
        .on_hover_text(t!("recon-object-lift-hint"));
        ui.checkbox(&mut st.isolate, t!("recon-object-isolate"))
            .on_hover_text(t!("recon-object-isolate-hint"));
        if crate::ui_util::icon_text_button(
            ui,
            h3d_ui::icons::CENTER_VIEW,
            &t!("recon-object-focus"),
        )
        .clicked()
        {
            focus_camera_on_selected(st);
        }
        if crate::ui_util::icon_text_button_enabled(
            ui,
            st.lift_mm != 0.0 || st.isolate,
            h3d_ui::icons::UNDO,
            &t!("recon-object-restore"),
        )
        .clicked()
        {
            st.lift_mm = 0.0;
            st.isolate = false;
        }
        if crate::ui_util::icon_text_button(ui, h3d_ui::icons::CLOSE, &t!("recon-object-deselect"))
            .clicked()
        {
            select_object(st, None);
            st.pick = None;
        }
    });
    ui.small(t!("stack-tool-object-hint"));
}

/// Baris fokus dua irisan di bilah pilihan alat: irisan yang dikoreksi,
/// pemilih patokan (irisan lain mana pun di tumpukan), Tukar, dan Tampilkan
/// semua. Kosong bila fokus tidak aktif.
fn focus_options(app: &mut AppState, sess: &Session, ui: &mut Ui) {
    let Some((target, partner)) = app.recon.focus else {
        return;
    };
    let others: Vec<SlideId> = app
        .recon
        .slices
        .iter()
        .filter(|s| !s.preview && s.id != target)
        .map(|s| s.id)
        .collect();
    let mut new_partner = partner;
    let mut swap = false;
    let mut show_all = false;
    ui.horizontal_wrapped(|ui| {
        ui.label(icon_label(
            h3d_ui::icons::FOCUS_PAIR,
            &t!("stack-focus-title"),
        ));
        crate::ui_util::strong(ui, crate::ui_util::slide_title(sess, target));
        ui.label(t!("stack-focus-partner"));
        egui::ComboBox::from_id_salt("stack-focus-partner")
            .selected_text(crate::ui_util::slide_title(sess, partner))
            .show_ui(ui, |ui| {
                for id in &others {
                    ui.selectable_value(
                        &mut new_partner,
                        *id,
                        crate::ui_util::slide_title(sess, *id),
                    );
                }
            });
        if ui
            .add(egui::Button::new(h3d_ui::icons::SWAP_VERT.rich_text()))
            .on_hover_text(t!("stack-focus-swap-hint"))
            .clicked()
        {
            swap = true;
        }
        if crate::ui_util::icon_text_button(
            ui,
            h3d_ui::icons::VISIBILITY_ON,
            &t!("stack-focus-show-all"),
        )
        .clicked()
        {
            show_all = true;
        }
    });
    if !editing_active(&app.recon) {
        ui.small(t!("stack-focus-hint"));
    }
    let st = &mut app.recon;
    if show_all {
        focus_clear(st);
    } else if swap {
        st.focus = Some((partner, target));
        app.current = Some(partner);
    } else if new_partner != partner {
        st.focus = Some((target, new_partner));
        st.fitted = false;
    }
}

/// Parameter rekonstruksi per objek di laci: saklar per objek, pemisah
/// objek menempel (watershed), IoU penautan, dan jumlah irisan hilang yang
/// dijembatani. Mengembalikan `true` bila ada yang berubah (konfigurasi
/// ikut kunci cache, jadi rekonstruksi dibangun ulang).
fn object_params(ui: &mut Ui, cfg: &mut h3d_core::ReconConfig) -> bool {
    let mut changed = false;
    changed |= ui
        .checkbox(&mut cfg.per_object, t!("recon-per-object"))
        .on_hover_text(t!("recon-per-object-hint"))
        .changed();
    if !cfg.per_object {
        return changed;
    }
    changed |= ui
        .checkbox(&mut cfg.track_split.enabled, t!("recon-split-enable"))
        .on_hover_text(t!("recon-split-enable-hint"))
        .changed();
    egui::Grid::new("recon-object-params-grid")
        .num_columns(2)
        .show(ui, |ui| {
            if cfg.track_split.enabled {
                ui.label(t!("recon-split-seed-distance"));
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut cfg.track_split.min_seed_distance_um)
                            .range(0.0..=5000.0)
                            .speed(1.0),
                    )
                    .changed();
                ui.end_row();
                ui.label(t!("recon-split-h"));
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut cfg.track_split.h_um)
                            .range(0.0..=1000.0)
                            .speed(0.5),
                    )
                    .changed();
                ui.end_row();
                ui.label(t!("recon-split-min-area"));
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut cfg.track_split.min_area_um2)
                            .range(0.0..=1.0e8)
                            .speed(10.0),
                    )
                    .changed();
                ui.end_row();
            }
            ui.label(t!("recon-track-min-iou"));
            changed |= ui
                .add(
                    egui::DragValue::new(&mut cfg.track_min_iou)
                        .range(0.0..=1.0)
                        .speed(0.01),
                )
                .changed();
            ui.end_row();
            ui.label(t!("recon-track-max-skip"))
                .on_hover_text(t!("recon-track-max-skip-hint"));
            changed |= ui
                .add(egui::DragValue::new(&mut cfg.track_max_skip).range(0..=5))
                .changed();
            ui.end_row();
        });
    changed
}

/// Daftar objek di laci: satu baris per mesh ber-`instance_id`, dikelompokkan
/// per label, dengan jumlah irisan dan rentang z dari `tracks.json`. Klik
/// baris = pilih (sama dengan klik di kanvas). Kosong bila tidak ada objek.
fn object_list(app: &mut AppState, sess: &Session, ui: &mut Ui) {
    let st = &mut app.recon;
    if !st.meshes.iter().any(|m| m.instance_id.is_some()) {
        return;
    }
    let tracks = sess.tracks().ok().flatten();
    let mut labels: Vec<LabelId> = st.meshes.iter().map(|m| m.label).collect();
    labels.dedup();
    egui::CollapsingHeader::new(icon_label(
        h3d_ui::icons::OBJECT_LIST,
        &t!("recon-object-list"),
    ))
    .default_open(true)
    .show(ui, |ui| {
        ui.small(t!("recon-object-list-hint"));
        let mut clicked: Option<MeshKey> = None;
        for label in labels {
            let def = sess.project().label(label).cloned();
            let set = tracks.as_ref().and_then(|t| t.set(label));
            let name = def
                .as_ref()
                .map_or_else(|| label.to_string(), |d| d.name.clone());
            ui.horizontal(|ui| {
                if let Some(d) = &def {
                    h3d_ui::widgets::color_swatch(ui, d.color);
                }
                crate::ui_util::strong(ui, name);
                if let Some(set) = set {
                    ui.weak(t!(
                        "recon-object-events",
                        n = set.event_count().to_string(),
                        b = set.bridged.to_string()
                    ));
                }
            });
            let rows: Vec<(MeshKey, String, f64)> = st
                .meshes
                .iter()
                .filter(|m| m.label == label)
                .map(|m| {
                    let key = mesh_key(m);
                    let text = match key.1 {
                        Some(id) => {
                            let track = set.and_then(|s| s.track(id));
                            let (z0, z1) = track.and_then(|t| t.z_range_um()).unwrap_or((0.0, 0.0));
                            t!(
                                "recon-object-row",
                                id = id.to_string(),
                                slices = track.map_or(0, |t| t.members.len()).to_string(),
                                z0 = format!("{:.2}", z0 / 1000.0),
                                z1 = format!("{:.2}", z1 / 1000.0)
                            )
                        }
                        None => t!("recon-object-whole"),
                    };
                    (key, text, m.signed_volume_um3())
                })
                .collect();
            egui::ScrollArea::vertical()
                .id_salt(("object-list", label.0))
                .max_height(OBJECT_LIST_MAX_HEIGHT)
                .show(ui, |ui| {
                    for (key, text, vol) in rows {
                        let sel = st.selected == Some(key);
                        let r = ui.selectable_label(sel, text).on_hover_text(t!(
                            "recon-object-volume",
                            v = format!("{:.4}", vol / 1e9)
                        ));
                        if r.clicked() {
                            clicked = Some(key);
                        }
                    }
                });
        }
        if let Some(key) = clicked {
            if st.selected == Some(key) {
                select_object(st, None);
            } else {
                select_object(st, Some(key));
                if st.tool == StackTool::Orbit {
                    st.tool = StackTool::Object;
                }
            }
        }
    });
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
        if app.derived_ready() {
            reload(app, sess);
        } else {
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(
                    crate::state::DERIVED_DEBOUNCE_MS,
                ));
        }
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
                    ui.label(t!("recon-sdf-smooth"))
                        .on_hover_text(t!("recon-sdf-smooth-hint"));
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut cfg.sdf_smooth_sigma_um)
                                .range(0.0..=200.0)
                                .speed(0.5),
                        )
                        .on_hover_text(t!("recon-sdf-smooth-hint"))
                        .changed();
                    ui.end_row();
                    ui.label(t!("recon-subdivision"));
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut cfg.subdivision_levels)
                                .range(0..=h3d_engine::MAX_SUBDIVISION_LEVELS),
                        )
                        .changed();
                    ui.end_row();
                });
            changed |= ui
                .checkbox(&mut cfg.stain_colors, t!("recon-stain-recon"))
                .on_hover_text(t!("recon-stain-colors-hint"))
                .changed();
            if ui
                .button(icon_label(h3d_ui::icons::AI, &t!("recon-organic-preset")))
                .on_hover_text(t!("recon-organic-preset-hint"))
                .clicked()
            {
                // Separuh jarak irisan: cukup menghapus tangga antar irisan.
                cfg.sdf_smooth_sigma_um = cfg.sdf_smooth_sigma_um.max(0.5 * cfg.default_spacing_um);

                cfg.subdivision_levels = cfg.subdivision_levels.max(1);
                cfg.stain_colors = true;
                changed = true;
                app.recon.set_look(ViewLook::Tissue);
            }
            changed |= object_params(ui, &mut cfg);
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

    object_list(app, sess, ui);

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
            ui.horizontal(|ui| {
                ui.label(icon_label(h3d_ui::icons::EXPAND_Z, &t!("slides-spacing")));
                crate::ui_util::spacing_slider(ui, sess, &mut app.toasts, &mut st.spacing_preview);
            });
            if !st.hidden_slices.is_empty() && ui.button(t!("stack-slice-shown")).clicked() {
                st.hidden_slices.clear();
            }
            if st.focus.is_some() && ui.button(t!("stack-focus-show-all")).clicked() {
                focus_clear(st);
            }
        });

    look_section(&mut app.recon, ui);

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
        ui.label(object_title(sess, (pick.label, pick.instance_id)));
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
            colors: Vec::new(),
            occlusion: Vec::new(),
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
                glass: 0.0,
            },
            flagged: false,
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
    fn glass_goes_to_at_most_three_spread_slices() {
        assert_eq!(glass_slice_indices(0), Vec::<usize>::new());
        assert_eq!(glass_slice_indices(2), vec![0, 1]);
        assert_eq!(glass_slice_indices(3), vec![0, 1, 2]);
        assert_eq!(glass_slice_indices(29), vec![0, 14, 28]);
        assert_eq!(glass_slice_indices(4), vec![0, 2, 3]);
    }

    #[test]
    fn look_presets_set_material_and_never_touch_geometry() {
        let mut st = ReconState::default();
        let mut m = tri_mesh();
        let tech = mesh_material(&st, &m);
        assert_eq!(tech, h3d_render::MeshMaterial::default());
        st.set_look(ViewLook::Tissue);
        assert!(st.stain_colors && st.glass);
        // Tanpa warna di mesh: mode pulasan tidak dipakai (warna label).
        assert!(!mesh_material(&st, &m).stain_colors);
        m.colors = vec![[200, 100, 150]; 3];
        let tis = mesh_material(&st, &m);
        assert!(tis.stain_colors && tis.specular > 0.0 && tis.cells.is_none());
        st.cell_bumps = true;
        st.cell_size_um = 50.0;
        let c = mesh_material(&st, &m).cells.expect("tonjolan");
        assert!((c.size_mm - 0.05).abs() < 1e-6);
        // Kembali ke teknis mematikan efek ilustratif.
        st.set_look(ViewLook::Technical);
        assert!(!st.cell_bumps);
        assert_eq!(
            look_lighting(ViewLook::Technical),
            h3d_render::Lighting::default()
        );
    }

    #[test]
    fn explode_moves_objects_away_from_stack_centre() {
        let mut st = ReconState::default();
        let a: MeshKey = (LabelId(1), Some(1));
        let b: MeshKey = (LabelId(1), Some(2));
        st.centers_mm.insert(a, Vec3::new(-1.0, 0.0, 0.5));
        st.centers_mm.insert(b, Vec3::new(1.0, 0.0, -0.5));
        st.bounds_mm = Some((Vec3::new(-2.0, -1.0, -1.0), Vec3::new(2.0, 1.0, 1.0)));
        assert_eq!(mesh_offset_mm(&st, a), [0.0; 3]);
        st.explode = 0.5;
        st.z_scale = 2.0;
        assert_eq!(mesh_offset_mm(&st, a), [-0.5, 0.0, 0.5]);
        assert_eq!(mesh_offset_mm(&st, b), [0.5, 0.0, -0.5]);
        // Angkat objek terpilih tetap ditambahkan.
        st.selected = Some(a);
        st.lift_mm = 3.0;
        assert_eq!(mesh_offset_mm(&st, a), [-0.5, 0.0, 3.5]);
        // Satu objek saja: tidak ada yang dipisahkan.
        st.centers_mm.remove(&b);
        assert_eq!(mesh_offset_mm(&st, a), [0.0, 0.0, 3.0]);
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

    #[test]
    fn wrap_angle_keeps_smallest_difference() {
        let pi = std::f64::consts::PI;
        assert!((wrap_angle(1.5 * pi) - (-0.5 * pi)).abs() < 1e-12);
        assert!((wrap_angle(-1.5 * pi) - 0.5 * pi).abs() < 1e-12);
        assert!((wrap_angle(0.3) - 0.3).abs() < 1e-12);
        assert!((angle_about([1.0, 1.0], [0.0, 0.0]) - pi / 4.0).abs() < 1e-12);
    }

    #[test]
    fn focus_pair_hides_other_slices_and_fits_camera() {
        let mut st = ReconState {
            slices: (0..5)
                .map(|i| item(i, f64::from(i) * 4.0, 1000.0, 500.0))
                .collect(),
            fitted: true,
            ..Default::default()
        };
        assert!((0..5).all(|i| slice_visible(&st, SlideId(i))));
        st.focus = Some((SlideId(2), SlideId(1)));
        assert!(slice_visible(&st, SlideId(2)) && slice_visible(&st, SlideId(1)));
        assert!(!slice_visible(&st, SlideId(0)) && !slice_visible(&st, SlideId(3)));
        // Sembunyikan manual tetap berlaku di dalam fokus.
        st.hidden_slices.insert(SlideId(1));
        assert!(!slice_visible(&st, SlideId(1)));
        // Kotak kanvas hanya memuat pasangan fokus: tebal Z = 0 (hanya z=8).
        st.hidden_slices.clear();
        st.focus = Some((SlideId(2), SlideId(2)));
        let (lo, hi) = canvas_bounds(&st).unwrap();
        assert!((hi.z - lo.z).abs() < 1e-6);
        // Tampilkan semua memaskan ulang kamera; tanpa fokus tidak berubah.
        focus_clear(&mut st);
        assert!(st.focus.is_none() && !st.fitted);
        st.fitted = true;
        focus_clear(&mut st);
        assert!(st.fitted);
    }

    #[test]
    fn nearest_other_slice_is_deterministic() {
        let items: Vec<SliceItem> = [(0, 0.0), (1, 4.0), (2, 8.0), (3, 12.0)]
            .iter()
            .map(|(i, z)| item(*i, *z, 100.0, 100.0))
            .collect();
        // Dua tetangga sama jauh → id terkecil.
        assert_eq!(nearest_other_slice(&items, SlideId(2)), Some(SlideId(1)));
        assert_eq!(nearest_other_slice(&items, SlideId(0)), Some(SlideId(1)));
        assert_eq!(nearest_other_slice(&items, SlideId(9)), None);
        assert_eq!(nearest_other_slice(&items[..1], SlideId(0)), None);
    }

    #[test]
    fn edit_transforms_rotate_about_slice_centre_and_follow_tool() {
        let mut st = ReconState {
            slices: vec![item(0, 0.0, 1000.0, 500.0), item(1, 10.0, 1000.0, 500.0)],
            edit: StackEdit {
                slide: Some(SlideId(1)),
                angle: std::f64::consts::FRAC_PI_2,
                shift: [20.0, -30.0],
                ..Default::default()
            },
            ..Default::default()
        };
        // Orbit dengan koreksi tertunda: tetap dipratinjau (koreksi tidak
        // hilang saat berganti alat); orbit tanpa koreksi: tidak ada pratinjau.
        assert!(editing_active(&st) && edit_transforms(&st).is_some());
        let quiet = ReconState {
            slices: st.slices.clone(),
            edit: StackEdit {
                slide: Some(SlideId(1)),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(!editing_active(&quiet) && edit_transforms(&quiet).is_none());
        st.tool = StackTool::Rotate;
        let (id, t_old, t_new, pivot) = edit_transforms(&st).unwrap();
        assert_eq!(id, SlideId(1));
        assert_eq!(pivot, [500.0, 250.0]);
        // Pusat irisan hanya tergeser `shift`; titik lain berputar 90°.
        let c = t_new.apply([500.0, 250.0]);
        assert!((c[0] - 520.0).abs() < 1e-9 && (c[1] - 220.0).abs() < 1e-9);
        let p = t_new.apply([600.0, 250.0]);
        assert!((p[0] - 520.0).abs() < 1e-9 && (p[1] - 320.0).abs() < 1e-9);
        // Irisan sesudahnya ikut `T_baru ∘ T_lama⁻¹`: untuk T lama identitas
        // sama dengan T baru.
        let delta = t_new.compose(&t_old.inverse().unwrap());
        assert_eq!(delta.m, t_new.m);
        // Irisan yang tidak ada di cache → tidak ada pratinjau.
        st.edit.slide = Some(SlideId(9));
        assert!(edit_transforms(&st).is_none());
        // Atur ulang membuang perubahan tetapi mempertahankan irisan.
        st.edit.reset();
        assert!(!st.edit.has_change());
        assert_eq!(st.edit.slide, Some(SlideId(9)));
    }
}
