//! Irisan di 3D: quad bertekstur thumbnail slide pada `z_i` dengan `T_i`.
//!
//! Matriks model memetakan kotak satuan `(u, v) ∈ [0,1]²` → ruang R (mm):
//! `p_U = (u·w_um, v·h_um)` → `w = T_i(p_U)`, `z = z_um` →
//! `r = ((w.x − c.x)/1000, −(w.y − c.y)/1000, (z − c.z)/1000)`
//! (konversi yang sama dengan [`crate::mesh::world_um_to_render_mm`]).

use glam::{DMat4, DVec4, Mat4};
use h3d_core::Affine2;

use crate::error::RenderError;
use crate::wgpu;

/// Pengenal tekstur irisan (aplikasi memetakan slide → `u32`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SliceId(pub u32);

/// Satu irisan untuk digambar di 3D.
#[derive(Clone, Debug, PartialEq)]
pub struct SliceDraw {
    /// Tekstur thumbnail yang sudah diunggah.
    pub slice: SliceId,
    /// `T_i`: ruang U irisan (µm) → ruang W (µm).
    pub affine: Affine2,
    /// Ukuran slide `(lebar, tinggi)` dalam µm ruang U (`piksel_level0 × mpp`).
    pub size_um: [f64; 2],
    /// Posisi Z irisan (µm ruang W).
    pub z_um: f64,
    /// Alfa 0..1.
    pub alpha: f32,
    /// Kesan kaca objek 0..1: bingkai terang di tepi quad dan kaca tipis
    /// kebiruan di latar transparan (0 = mati). Hanya tampilan.
    pub glass: f32,
}

/// Bangun matriks model (lihat dokumentasi modul). `center_um` = pusat volume
/// yang juga dipakai saat mengunggah mesh.
pub fn slice_model_matrix(
    affine: &Affine2,
    size_um: [f64; 2],
    z_um: f64,
    center_um: [f64; 3],
) -> Mat4 {
    let [a, b, tx, c, d, ty] = affine.m;
    // Kolom-mayor: setiap DVec4 adalah satu kolom.
    let scale = DMat4::from_cols(
        DVec4::new(size_um[0], 0.0, 0.0, 0.0),
        DVec4::new(0.0, size_um[1], 0.0, 0.0),
        DVec4::new(0.0, 0.0, 1.0, 0.0),
        DVec4::new(0.0, 0.0, 0.0, 1.0),
    );
    let t4 = DMat4::from_cols(
        DVec4::new(a, c, 0.0, 0.0),
        DVec4::new(b, d, 0.0, 0.0),
        DVec4::new(0.0, 0.0, 1.0, 0.0),
        DVec4::new(tx, ty, z_um, 1.0),
    );
    let k = 1.0 / 1000.0;
    let conv = DMat4::from_cols(
        DVec4::new(k, 0.0, 0.0, 0.0),
        DVec4::new(0.0, -k, 0.0, 0.0),
        DVec4::new(0.0, 0.0, k, 0.0),
        DVec4::new(-center_um[0] * k, center_um[1] * k, -center_um[2] * k, 1.0),
    );
    (conv * t4 * scale).as_mat4()
}

/// Matriks model irisan setelah renggang Z tampilan ([`crate::Scene3D::z_scale`]):
/// hasil `slice_model_matrix` dikalikan skala `(1, 1, z_scale)` di ruang R.
/// Dipakai CPU (picking, garis tepi) agar sama persis dengan shader.
pub fn slice_model_matrix_scaled(model: &Mat4, z_scale: f32) -> Mat4 {
    let z = crate::scene::sanitize_z_scale(z_scale);
    Mat4::from_scale(glam::Vec3::new(1.0, 1.0, z)) * *model
}

/// Keempat sudut quad irisan di ruang R (mm), urutan `(0,0) (1,0) (1,1) (0,1)`.
pub fn slice_corners_mm(model: &Mat4) -> [glam::Vec3; 4] {
    [
        model.project_point3(glam::Vec3::new(0.0, 0.0, 0.0)),
        model.project_point3(glam::Vec3::new(1.0, 0.0, 0.0)),
        model.project_point3(glam::Vec3::new(1.0, 1.0, 0.0)),
        model.project_point3(glam::Vec3::new(0.0, 1.0, 0.0)),
    ]
}

/// Hasil klik pada quad irisan.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlicePick {
    /// Parameter ray (`origin + dir × t`), > 0.
    pub t: f32,
    /// Koordinat kotak satuan `(u, v) ∈ [0,1]²` pada irisan; `u` searah
    /// lebar slide, `v` searah tinggi (ruang U).
    pub uv: [f32; 2],
    /// Titik kena (mm, ruang R, sudah termasuk renggang Z bila ada).
    pub point_mm: glam::Vec3,
}

/// Potong `ray` (ruang R, mm) dengan quad irisan bermatriks `model`
/// (sudah termasuk renggang Z bila dipakai). `None` bila ray sejajar bidang,
/// perpotongan di belakang asal ray, atau di luar kotak satuan.
pub fn pick_slice(ray: &crate::mesh::Ray, model: &Mat4) -> Option<SlicePick> {
    if model.determinant().abs() < 1e-18 {
        return None;
    }
    let inv = model.inverse();
    // Ray ke ruang kotak satuan irisan; bidang irisan adalah z = 0.
    let o = inv.project_point3(ray.origin);
    let d = inv.transform_vector3(ray.dir);
    if d.z.abs() < 1e-12 {
        return None;
    }
    let t = -o.z / d.z;
    if !(t.is_finite() && t > 0.0) {
        return None;
    }
    let p = o + d * t;
    if !(0.0..=1.0).contains(&p.x) || !(0.0..=1.0).contains(&p.y) {
        return None;
    }
    Some(SlicePick {
        t,
        uv: [p.x, p.y],
        point_mm: ray.origin + ray.dir * t,
    })
}

/// Tekstur thumbnail irisan di GPU.
pub struct SliceTexture {
    /// Tekstur `Rgba8UnormSrgb`.
    pub texture: wgpu::Texture,
    /// View.
    pub view: wgpu::TextureView,
    /// Bind group grup 2 (tekstur + sampler).
    pub bind_group: wgpu::BindGroup,
    /// Lebar piksel.
    pub width: u32,
    /// Tinggi piksel.
    pub height: u32,
}

/// Layout bind group tekstur irisan.
pub(crate) fn slice_texture_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("h3d-slice-tex"),
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
    })
}

impl SliceTexture {
    #[allow(clippy::too_many_arguments)]
    /// Unggah thumbnail RGBA8 `width × height`. Gagal bila melebihi `max_dim`
    /// (`device.limits().max_texture_dimension_2d`); aplikasi harus
    /// mengecilkan thumbnail terlebih dahulu.
    pub fn upload(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        width: u32,
        height: u32,
        rgba: &[u8],
        max_dim: u32,
    ) -> Result<Self, RenderError> {
        if width == 0 || height == 0 || width > max_dim || height > max_dim {
            return Err(RenderError::TextureTooLarge {
                width,
                height,
                max: max_dim,
            });
        }
        let expected = width as usize * height as usize * 4;
        if rgba.len() != expected {
            return Err(RenderError::DataSizeMismatch {
                expected,
                actual: rgba.len(),
            });
        }
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("h3d-slice"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
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
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            size,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("h3d-slice"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        Ok(Self {
            texture,
            view,
            bind_group,
            width,
            height,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::world_um_to_render_mm;
    use glam::Vec3;

    #[test]
    fn model_matrix_matches_mesh_conversion() {
        // T: rotasi 90° + translasi; slide 8000 × 6000 µm di z = 2500.
        let t = Affine2::rotation(std::f64::consts::FRAC_PI_2)
            .compose(&Affine2::translation(500.0, -200.0));
        let size = [8000.0, 6000.0];
        let z = 2500.0;
        let center = [1000.0, 2000.0, 1500.0];
        let m = slice_model_matrix(&t, size, z, center);
        for (u, v) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.3, 0.7)] {
            let p_u = [u * size[0], v * size[1]];
            let w = t.apply(p_u);
            let expected = world_um_to_render_mm([w[0], w[1], z], center);
            let got = m.project_point3(Vec3::new(u as f32, v as f32, 0.0));
            assert!(
                (got - Vec3::from(expected)).length() < 1e-3,
                "({u},{v}): {got:?} vs {expected:?}"
            );
        }
    }

    #[test]
    fn identity_affine_maps_corners() {
        let m = slice_model_matrix(&Affine2::IDENTITY, [2000.0, 1000.0], 0.0, [0.0; 3]);
        let p = m.project_point3(Vec3::new(1.0, 1.0, 0.0));
        // (2000 µm, 1000 µm) → (2 mm, −1 mm) karena y dibalik.
        assert!((p - Vec3::new(2.0, -1.0, 0.0)).length() < 1e-6);
        let c = slice_corners_mm(&m);
        assert!((c[0] - Vec3::ZERO).length() < 1e-6);
        assert!((c[2] - p).length() < 1e-6);
    }

    #[test]
    fn pick_hits_inside_quad_and_respects_z_scale() {
        use crate::mesh::Ray;
        // Irisan 2 × 1 mm di z = 4 µm; pusat volume di z = 0.
        let m = slice_model_matrix(&Affine2::IDENTITY, [2000.0, 1000.0], 4.0, [0.0; 3]);
        let down = |x: f32, y: f32| Ray {
            origin: Vec3::new(x, y, 10.0),
            dir: Vec3::new(0.0, 0.0, -1.0),
        };
        let hit = pick_slice(&down(0.5, -0.25), &m).expect("tengah quad kena");
        assert!((hit.uv[0] - 0.25).abs() < 1e-5 && (hit.uv[1] - 0.25).abs() < 1e-5);
        assert!((hit.point_mm.z - 0.004).abs() < 1e-6);
        assert!(pick_slice(&down(2.5, -0.5), &m).is_none(), "di luar lebar");
        assert!(pick_slice(&down(0.5, 0.5), &m).is_none(), "di luar tinggi");
        // Ray sejajar bidang dan ray yang menjauh tidak kena.
        let along = Ray {
            origin: Vec3::new(-1.0, -0.5, 0.004),
            dir: Vec3::X,
        };
        assert!(pick_slice(&along, &m).is_none());
        let away = Ray {
            origin: Vec3::new(0.5, -0.5, 10.0),
            dir: Vec3::Z,
        };
        assert!(pick_slice(&away, &m).is_none());
        // Renggang Z 100× memindahkan bidang ke z = 0,4 mm.
        let ms = slice_model_matrix_scaled(&m, 100.0);
        let hit = pick_slice(&down(0.5, -0.25), &ms).unwrap();
        assert!((hit.point_mm.z - 0.4).abs() < 1e-5);
        assert_eq!(slice_model_matrix_scaled(&m, f32::NAN), m);
    }
}
