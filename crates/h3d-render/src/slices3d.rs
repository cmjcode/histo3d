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
    }
}
