//! Overlay label: tile `R8Uint` + palet 256 × 1 RGBA. Warna diambil dari
//! palet di shader; opasitas global; mode garis tepi (bandingkan dengan 4
//! tetangga, hanya tepi yang digambar).

use crate::colormap::Palette256;
use crate::tiles::TileDraw;
use crate::wgpu;
use h3d_core::Affine2;

/// Satu lapisan overlay label untuk digambar.
#[derive(Clone, Debug, PartialEq)]
pub struct LabelOverlayDraw {
    /// `T_i`: ruang U irisan (µm) → ruang W (µm). Identitas bila tile sudah
    /// berada di ruang pandangan.
    pub affine: Affine2,
    /// Tile label (`R8Uint`) dengan rect dalam µm ruang U.
    pub tiles: Vec<TileDraw>,
    /// Opasitas global 0..1.
    pub opacity: f32,
    /// `true` = hanya garis tepi label yang digambar.
    pub outline: bool,
}

/// Palet bawaan: indeks 0 transparan, indeks lain warna dengan sudut emas HSV
/// supaya label bertetangga kontras.
pub fn default_palette() -> Palette256 {
    let mut p = [[0u8; 4]; 256];
    for (i, entry) in p.iter_mut().enumerate().skip(1) {
        let hue = (i as f32 * 137.508) % 360.0;
        let (r, g, b) = hsv_to_rgb(hue, 0.75, 0.95);
        *entry = [r, g, b, 255];
    }
    p
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let c = v * s;
    let hp = h / 60.0;
    let x = c * (1.0 - (hp % 2.0 - 1.0).abs());
    let (r, g, b) = match hp as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    let to8 = |f: f32| ((f + m) * 255.0).round().clamp(0.0, 255.0) as u8;
    (to8(r), to8(g), to8(b))
}

/// Buat pipeline overlay label (`depth_stencil` harus cocok dengan render
/// pass egui: format sama, tanpa tulis/uji depth). Grup 0 = `Layer` uniform, grup 1 = tile
/// `texture_2d<u32>`, grup 2 = palet LUT.
pub(crate) fn create_overlay_pipeline(
    device: &wgpu::Device,
    color_format: wgpu::TextureFormat,
    depth_stencil: Option<wgpu::DepthStencilState>,
    layer_layout: &wgpu::BindGroupLayout,
    label_tile_layout: &wgpu::BindGroupLayout,
    lut_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("h3d-overlay"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/overlay.wgsl").into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("h3d-overlay"),
        bind_group_layouts: &[
            Some(layer_layout),
            Some(label_tile_layout),
            Some(lut_layout),
        ],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("h3d-overlay"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_overlay"),
            compilation_options: Default::default(),
            buffers: &[Some(crate::tiles::tile_instance_layout())],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_overlay"),
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
        depth_stencil,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_palette_has_transparent_zero_and_opaque_rest() {
        let p = default_palette();
        assert_eq!(p[0], [0, 0, 0, 0]);
        assert!(p[1..].iter().all(|c| c[3] == 255));
        // Label bertetangga tidak sama warna.
        assert_ne!(p[1], p[2]);
        assert_ne!(p[2], p[3]);
    }

    #[test]
    fn hsv_primary_colours() {
        assert_eq!(hsv_to_rgb(0.0, 1.0, 1.0), (255, 0, 0));
        assert_eq!(hsv_to_rgb(120.0, 1.0, 1.0), (0, 255, 0));
        assert_eq!(hsv_to_rgb(240.0, 1.0, 1.0), (0, 0, 255));
    }
}
