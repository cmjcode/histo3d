//! Mesh 3D: konversi µm → mm (ruang R), unggah ke GPU, picking CPU, dan
//! urutan gambar belakang-ke-depan.
//!
//! **Konversi ruang W → R** (histo3d.md 5.2): `r = (w − pusat) / 1000`
//! dengan **y dibalik**:
//! `r.x = (w.x − c.x)/1000`, `r.y = −(w.y − c.y)/1000`, `r.z = (w.z − c.z)/1000`.
//! Ruang W memakai y ke bawah (warisan piksel gambar); pembalikan y
//! menjadikan R tangan-kanan dengan Z (arah tumpukan) ke atas. Normal ikut
//! dibalik komponen y-nya. Pencerminan membalik winding segitiga, sehingga
//! pipeline mesh tidak melakukan culling dan pencahayaan dua sisi.

use glam::Vec3;
use h3d_core::Colormap;

use crate::camera::OrbitCamera;
use crate::error::RenderError;
use crate::wgpu;

/// Nilai skalar NODATA di GPU (pengganti `NaN`, lihat `mesh.wgsl`).
pub const NODATA_SCALAR: f32 = 3.0e38;

/// Verteks mesh di GPU: posisi mm (ruang R), normal, skalar.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshVertex {
    /// Posisi (mm, ruang R).
    pub pos: [f32; 3],
    /// Normal (ruang R).
    pub normal: [f32; 3],
    /// Skalar per verteks; `NODATA_SCALAR` untuk tanpa data.
    pub scalar: f32,
}

pub(crate) const MESH_VERTEX_ATTRS: [wgpu::VertexAttribute; 3] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32];

pub(crate) fn mesh_vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<MeshVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &MESH_VERTEX_ATTRS,
    }
}

/// Pengenal mesh di `SceneRenderer` (aplikasi yang menentukan; mis. hash `LabelId`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MeshId(pub u64);

/// Ruang W (µm) → ruang R (mm), y dibalik.
pub fn world_um_to_render_mm(w_um: [f64; 3], center_um: [f64; 3]) -> [f32; 3] {
    [
        ((w_um[0] - center_um[0]) / 1000.0) as f32,
        (-(w_um[1] - center_um[1]) / 1000.0) as f32,
        ((w_um[2] - center_um[2]) / 1000.0) as f32,
    ]
}

/// Ruang R (mm) → ruang W (µm); kebalikan tepat dari [`world_um_to_render_mm`].
pub fn render_to_world_um(r_mm: [f32; 3], center_um: [f64; 3]) -> [f64; 3] {
    [
        r_mm[0] as f64 * 1000.0 + center_um[0],
        -(r_mm[1] as f64) * 1000.0 + center_um[1],
        r_mm[2] as f64 * 1000.0 + center_um[2],
    ]
}

/// Normal ruang W → ruang R (komponen y dibalik, dinormalkan).
pub fn normal_to_render(n: [f32; 3]) -> [f32; 3] {
    let v = Vec3::new(n[0], -n[1], n[2]).normalize_or_zero();
    [v.x, v.y, v.z]
}

/// Bangun verteks GPU dari data ruang W. `scalars` boleh `None` (semua NODATA);
/// `NaN`/tak hingga dipetakan ke [`NODATA_SCALAR`].
pub fn build_vertices(
    positions_um: &[[f64; 3]],
    normals: &[[f32; 3]],
    center_um: [f64; 3],
    scalars: Option<&[f32]>,
) -> Result<Vec<MeshVertex>, RenderError> {
    if normals.len() != positions_um.len() {
        return Err(RenderError::InvalidMesh(format!(
            "jumlah normal {} ≠ jumlah posisi {}",
            normals.len(),
            positions_um.len()
        )));
    }
    if let Some(s) = scalars {
        if s.len() != positions_um.len() {
            return Err(RenderError::InvalidMesh(format!(
                "jumlah skalar {} ≠ jumlah posisi {}",
                s.len(),
                positions_um.len()
            )));
        }
    }
    Ok(positions_um
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let scalar = match scalars {
                Some(s) if s[i].is_finite() => s[i],
                _ => NODATA_SCALAR,
            };
            MeshVertex {
                pos: world_um_to_render_mm(*p, center_um),
                normal: normal_to_render(normals[i]),
                scalar,
            }
        })
        .collect())
}

/// Mesh di GPU beserta salinan CPU untuk picking.
pub struct GpuMesh {
    /// Buffer verteks (`MeshVertex`).
    pub vbuf: wgpu::Buffer,
    /// Buffer indeks `u32`.
    pub ibuf: wgpu::Buffer,
    /// Jumlah indeks.
    pub index_count: u32,
    /// Posisi mm (ruang R) untuk picking.
    pub positions_mm: Vec<[f32; 3]>,
    /// Indeks segitiga.
    pub indices: Vec<u32>,
    /// Kotak pembatas min (mm).
    pub bounds_min_mm: [f32; 3],
    /// Kotak pembatas maks (mm).
    pub bounds_max_mm: [f32; 3],
    /// Pusat yang dipakai saat konversi (µm), untuk [`render_to_world_um`].
    pub center_um: [f64; 3],
}

impl GpuMesh {
    /// Pusat kotak pembatas (mm).
    pub fn bounds_center_mm(&self) -> [f32; 3] {
        [
            (self.bounds_min_mm[0] + self.bounds_max_mm[0]) * 0.5,
            (self.bounds_min_mm[1] + self.bounds_max_mm[1]) * 0.5,
            (self.bounds_min_mm[2] + self.bounds_max_mm[2]) * 0.5,
        ]
    }

    /// Picking ray–segitiga di CPU.
    pub fn pick(&self, ray: &Ray) -> Option<PickHit> {
        pick_triangle(ray.origin, ray.dir, &self.positions_mm, &self.indices).map(
            |(tri_index, t, bary)| PickHit {
                tri_index,
                t,
                bary,
                point_mm: ray.origin + ray.dir * t,
            },
        )
    }
}

/// Unggah mesh ruang W (µm, `f64`) ke GPU sebagai mm `f32` relatif `center_um`.
/// Indeks harus kelipatan 3 dan < jumlah verteks.
pub fn upload_mesh(
    device: &wgpu::Device,
    _queue: &wgpu::Queue,
    positions_um: &[[f64; 3]],
    normals: &[[f32; 3]],
    indices: &[u32],
    center_um: [f64; 3],
    scalars: Option<&[f32]>,
) -> Result<GpuMesh, RenderError> {
    use wgpu::util::DeviceExt;

    if positions_um.is_empty() {
        return Err(RenderError::InvalidMesh("tanpa verteks".into()));
    }
    if indices.len() % 3 != 0 {
        return Err(RenderError::InvalidMesh(format!(
            "jumlah indeks {} bukan kelipatan 3",
            indices.len()
        )));
    }
    if let Some(&bad) = indices.iter().find(|&&i| i as usize >= positions_um.len()) {
        return Err(RenderError::InvalidMesh(format!(
            "indeks {bad} ≥ jumlah verteks {}",
            positions_um.len()
        )));
    }
    let vertices = build_vertices(positions_um, normals, center_um, scalars)?;
    let mut bounds_min = [f32::INFINITY; 3];
    let mut bounds_max = [f32::NEG_INFINITY; 3];
    for v in &vertices {
        for k in 0..3 {
            bounds_min[k] = bounds_min[k].min(v.pos[k]);
            bounds_max[k] = bounds_max[k].max(v.pos[k]);
        }
    }
    let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("h3d-mesh-vertices"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("h3d-mesh-indices"),
        contents: bytemuck::cast_slice(indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    Ok(GpuMesh {
        vbuf,
        ibuf,
        index_count: indices.len() as u32,
        positions_mm: vertices.iter().map(|v| v.pos).collect(),
        indices: indices.to_vec(),
        bounds_min_mm: bounds_min,
        bounds_max_mm: bounds_max,
        center_um,
    })
}

/// Parameter tampilan satu mesh pada frame ini.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshInstance {
    /// Mesh yang sudah diunggah ke `SceneRenderer`.
    pub mesh: MeshId,
    /// Warna dasar RGBA (alfa diabaikan; pakai `opacity`).
    pub color: [f32; 4],
    /// Opasitas 0..1; `< 1` digambar sebagai tembus pandang.
    pub opacity: f32,
    /// Tampil atau tidak.
    pub visible: bool,
    /// Warna dari skalar verteks lewat colormap.
    pub scalar_mode: bool,
    /// `[min, max]` skalar yang dipetakan ke colormap.
    pub scalar_range: [f32; 2],
    /// Colormap untuk mode skalar.
    pub colormap: Colormap,
}

impl MeshInstance {
    /// Instance opak berwarna polos.
    pub fn solid(mesh: MeshId, color: [f32; 4]) -> Self {
        Self {
            mesh,
            color,
            opacity: 1.0,
            visible: true,
            scalar_mode: false,
            scalar_range: [0.0, 1.0],
            colormap: Colormap::default(),
        }
    }
}

/// Ray di ruang R (mm).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
    /// Titik asal.
    pub origin: Vec3,
    /// Arah (tidak harus satuan).
    pub dir: Vec3,
}

/// Hasil picking.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PickHit {
    /// Indeks segitiga (`indices[3*i..3*i+3]`).
    pub tri_index: usize,
    /// Parameter ray (`origin + dir × t`).
    pub t: f32,
    /// Koordinat barisentrik `(w0, w1, w2)` terhadap verteks segitiga.
    pub bary: [f32; 3],
    /// Titik kena (mm, ruang R).
    pub point_mm: Vec3,
}

/// Ray dunia dari posisi kursor layar (point egui) di viewport `viewport_rect`.
pub fn ray_from_screen(
    camera: &OrbitCamera,
    viewport_rect: egui::Rect,
    cursor_px: egui::Pos2,
) -> Ray {
    let aspect = viewport_rect.width() / viewport_rect.height().max(1.0);
    let inv = camera.view_proj(aspect).inverse();
    let ndc_x = ((cursor_px.x - viewport_rect.min.x) / viewport_rect.width().max(1.0)) * 2.0 - 1.0;
    let ndc_y = 1.0 - ((cursor_px.y - viewport_rect.min.y) / viewport_rect.height().max(1.0)) * 2.0;
    let p_near = inv.project_point3(Vec3::new(ndc_x, ndc_y, 0.0));
    let p_far = inv.project_point3(Vec3::new(ndc_x, ndc_y, 1.0));
    Ray {
        origin: p_near,
        dir: (p_far - p_near).normalize_or_zero(),
    }
}

/// Möller–Trumbore atas semua segitiga; mengembalikan
/// `(indeks segitiga, t, barisentrik)` dengan `t` terkecil (> 0). Seri → indeks terkecil.
pub fn pick_triangle(
    ray_origin: Vec3,
    ray_dir: Vec3,
    positions_mm: &[[f32; 3]],
    indices: &[u32],
) -> Option<(usize, f32, [f32; 3])> {
    const EPS: f32 = 1e-7;
    let mut best: Option<(usize, f32, [f32; 3])> = None;
    for (tri, idx) in indices.chunks_exact(3).enumerate() {
        let (Some(a), Some(b), Some(c)) = (
            positions_mm.get(idx[0] as usize),
            positions_mm.get(idx[1] as usize),
            positions_mm.get(idx[2] as usize),
        ) else {
            continue;
        };
        let v0 = Vec3::from(*a);
        let v1 = Vec3::from(*b);
        let v2 = Vec3::from(*c);
        let e1 = v1 - v0;
        let e2 = v2 - v0;
        let p = ray_dir.cross(e2);
        let det = e1.dot(p);
        if det.abs() < EPS {
            continue;
        }
        let inv_det = 1.0 / det;
        let s = ray_origin - v0;
        let u = s.dot(p) * inv_det;
        if !(0.0..=1.0).contains(&u) {
            continue;
        }
        let q = s.cross(e1);
        let v = ray_dir.dot(q) * inv_det;
        if v < 0.0 || u + v > 1.0 {
            continue;
        }
        let t = e2.dot(q) * inv_det;
        if t <= EPS {
            continue;
        }
        if best.is_none_or(|(_, bt, _)| t < bt) {
            best = Some((tri, t, [1.0 - u - v, u, v]));
        }
    }
    best
}

/// Urutan gambar objek tembus pandang: terjauh dari `eye` dulu. Mengembalikan
/// indeks ke `centers_mm`; seri dipecahkan dengan indeks terkecil dulu.
pub fn sort_back_to_front(centers_mm: &[[f32; 3]], eye: Vec3) -> Vec<usize> {
    let mut order: Vec<(usize, f32)> = centers_mm
        .iter()
        .enumerate()
        .map(|(i, c)| (i, Vec3::from(*c).distance_squared(eye)))
        .collect();
    order.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    order.into_iter().map(|(i, _)| i).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn um_to_mm_flips_y_and_roundtrips() {
        let c = [10_000.0, 20_000.0, 300.0];
        let w = [12_500.0, 18_000.0, 1_300.0];
        let r = world_um_to_render_mm(w, c);
        assert!((r[0] - 2.5).abs() < 1e-6);
        assert!((r[1] - 2.0).abs() < 1e-6, "y dibalik: {}", r[1]);
        assert!((r[2] - 1.0).abs() < 1e-6);
        let back = render_to_world_um(r, c);
        for k in 0..3 {
            assert!(
                (back[k] - w[k]).abs() < 1e-3,
                "sumbu {k}: {} vs {}",
                back[k],
                w[k]
            );
        }
        let n = normal_to_render([0.0, 1.0, 0.0]);
        assert!((n[1] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn build_vertices_maps_nan_to_nodata_and_checks_lengths() {
        let pos = [[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0]];
        let nrm = [[0.0, 0.0, 1.0]; 2];
        let v = build_vertices(&pos, &nrm, [0.0; 3], Some(&[1.5, f32::NAN])).unwrap();
        assert_eq!(v[0].scalar, 1.5);
        assert_eq!(v[1].scalar, NODATA_SCALAR);
        assert!(v[1].scalar > 1.0e38 && v[1].scalar.is_finite());
        let none = build_vertices(&pos, &nrm, [0.0; 3], None).unwrap();
        assert!(none.iter().all(|x| x.scalar == NODATA_SCALAR));
        assert!(build_vertices(&pos, &nrm[..1], [0.0; 3], None).is_err());
        assert!(build_vertices(&pos, &nrm, [0.0; 3], Some(&[1.0])).is_err());
        assert_eq!(std::mem::size_of::<MeshVertex>(), 28);
    }

    fn quad() -> (Vec<[f32; 3]>, Vec<u32>) {
        // Dua segitiga di bidang z = 0, kotak [0,1]².
        (
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            vec![0, 1, 2, 0, 2, 3],
        )
    }

    #[test]
    fn pick_hits_and_misses() {
        let (p, i) = quad();
        // Ray dari atas ke bawah mengenai segitiga pertama (dekat (0.7, 0.2)).
        let hit = pick_triangle(Vec3::new(0.7, 0.2, 5.0), Vec3::new(0.0, 0.0, -1.0), &p, &i)
            .expect("harus kena");
        assert_eq!(hit.0, 0);
        assert!((hit.1 - 5.0).abs() < 1e-5);
        let b = hit.2;
        assert!((b[0] + b[1] + b[2] - 1.0).abs() < 1e-5);
        // Rekonstruksi titik dari barisentrik.
        let pt = Vec3::from(p[0]) * b[0] + Vec3::from(p[1]) * b[1] + Vec3::from(p[2]) * b[2];
        assert!((pt - Vec3::new(0.7, 0.2, 0.0)).length() < 1e-5);
        // Segitiga kedua.
        let hit2 =
            pick_triangle(Vec3::new(0.2, 0.7, 5.0), Vec3::new(0.0, 0.0, -1.0), &p, &i).unwrap();
        assert_eq!(hit2.0, 1);
        // Meleset di luar kotak.
        assert!(
            pick_triangle(Vec3::new(1.5, 0.5, 5.0), Vec3::new(0.0, 0.0, -1.0), &p, &i).is_none()
        );
        // Arah menjauh (t negatif) → tidak kena.
        assert!(
            pick_triangle(Vec3::new(0.5, 0.5, 5.0), Vec3::new(0.0, 0.0, 1.0), &p, &i).is_none()
        );
        // Ray sejajar bidang → tidak kena.
        assert!(
            pick_triangle(Vec3::new(0.5, 0.5, 0.0), Vec3::new(1.0, 0.0, 0.0), &p, &i).is_none()
        );
    }

    #[test]
    fn pick_prefers_nearest_triangle() {
        let mut p = quad().0;
        let mut i = quad().1;
        // Kotak kedua di z = 2 (lebih dekat ke ray dari z = 5).
        let base = p.len() as u32;
        p.extend([
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [1.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
        ]);
        i.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        let hit =
            pick_triangle(Vec3::new(0.7, 0.2, 5.0), Vec3::new(0.0, 0.0, -1.0), &p, &i).unwrap();
        assert_eq!(hit.0, 2);
        assert!((hit.1 - 3.0).abs() < 1e-5);
    }

    #[test]
    fn back_to_front_order() {
        let centers = [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 10.0],
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 10.0],
        ];
        let eye = Vec3::new(0.0, 0.0, 20.0);
        // Terjauh dulu: indeks 0 (jarak 19), 2 (15), lalu 1 dan 3 (10, seri → indeks kecil dulu).
        assert_eq!(sort_back_to_front(&centers, eye), vec![0, 2, 1, 3]);
        assert!(sort_back_to_front(&[], eye).is_empty());
    }

    #[test]
    fn ray_from_screen_center_points_at_target() {
        let cam = OrbitCamera::default();
        let rect = egui::Rect::from_min_size(egui::pos2(10.0, 20.0), egui::vec2(640.0, 480.0));
        let ray = ray_from_screen(&cam, rect, rect.center());
        // Ray dari pusat layar melewati target kamera.
        let to_target = cam.target - ray.origin;
        let along = ray.dir * to_target.dot(ray.dir);
        assert!(
            (to_target - along).length() < 1e-2,
            "ray tidak lewat target"
        );
        assert!(to_target.dot(ray.dir) > 0.0);
    }
}
