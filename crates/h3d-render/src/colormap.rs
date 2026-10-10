//! Tekstur LUT 256 × 1 RGBA: colormap skalar (dari `h3d_core::Colormap`)
//! dan palet label overlay. Keduanya memakai bentuk tekstur yang sama.

use crate::wgpu;
use h3d_core::Colormap;

/// Palet 256 entri RGBA8 (indeks label → warna). Entri 0 lazimnya transparan.
pub type Palette256 = [[u8; 4]; 256];

/// Byte RGBA8 untuk colormap `cm` (1024 byte).
pub fn colormap_bytes(cm: Colormap) -> Vec<u8> {
    cm.lut256().into_iter().flatten().collect()
}

/// Tekstur LUT 256 × 1 beserta bind group-nya (grup: `@binding(0) texture_2d<f32>`).
pub struct LutTexture {
    /// Tekstur `Rgba8Unorm` 256 × 1.
    pub texture: wgpu::Texture,
    /// View tekstur.
    pub view: wgpu::TextureView,
    /// Bind group dengan layout [`lut_bind_group_layout`].
    pub bind_group: wgpu::BindGroup,
}

/// Layout bind group LUT: satu `texture_2d<f32>` (dibaca dengan `textureLoad`).
pub fn lut_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("h3d-lut"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }],
    })
}

impl LutTexture {
    /// Buat tekstur LUT dan isi dengan `rgba` (tepat 1024 byte; kelebihan
    /// dipotong, kekurangan diisi nol).
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        rgba: &[u8],
        label: &str,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: 256,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        });
        let lut = Self {
            texture,
            view,
            bind_group,
        };
        lut.write(queue, rgba);
        lut
    }

    /// Buat dari `h3d_core::Colormap`.
    pub fn from_colormap(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        cm: Colormap,
    ) -> Self {
        Self::new(device, queue, layout, &colormap_bytes(cm), "h3d-colormap")
    }

    /// Buat dari palet label.
    pub fn from_palette(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        palette: &Palette256,
    ) -> Self {
        Self::new(device, queue, layout, palette.as_flattened(), "h3d-palette")
    }

    /// Tulis ulang isi LUT (mis. palet label berubah).
    pub fn write(&self, queue: &wgpu::Queue, rgba: &[u8]) {
        let mut data = [0u8; 1024];
        let n = rgba.len().min(1024);
        data[..n].copy_from_slice(&rgba[..n]);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(1024),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 256,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Tulis palet label.
    pub fn write_palette(&self, queue: &wgpu::Queue, palette: &Palette256) {
        self.write(queue, palette.as_flattened());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colormap_bytes_is_1024_and_opaque() {
        for cm in Colormap::ALL {
            let b = colormap_bytes(cm);
            assert_eq!(b.len(), 1024);
            assert!(b.chunks(4).all(|px| px[3] == 255));
        }
        let grey = colormap_bytes(Colormap::Grey);
        assert_eq!(&grey[0..3], &[0, 0, 0]);
        assert_eq!(&grey[1020..1023], &[255, 255, 255]);
    }
}
