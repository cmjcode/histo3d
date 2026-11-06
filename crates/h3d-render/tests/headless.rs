//! Uji headless: bila ada adapter wgpu, buat `SceneRenderer`, unggah mesh,
//! tile, label, irisan, lalu render adegan 2D dan 3D ke tekstur offscreen
//! tanpa error validasi. Tanpa adapter, uji dilewati dengan catatan log.

use std::future::Future;
use std::pin::pin;
use std::task::{Context, Poll, Waker};

use h3d_core::Affine2;
use h3d_render::wgpu;
use h3d_render::{
    upload_mesh, Camera2D, CompareMode, CompareParams, H3dPaintCallback, LabelOverlayDraw, MeshId,
    MeshInstance, OrbitCamera, RenderBudgets, Scene, Scene2D, Scene3D, SceneRenderer, SliceDraw,
    SliceId, TileDraw, TileKey, TileLayerDraw, UploadStatus, VertexAttrs,
};

/// Eksekutor minimal untuk future wgpu (selesai setelah `device.poll`).
fn block_on<F: Future>(fut: F, device: Option<&wgpu::Device>) -> F::Output {
    let mut fut = pin!(fut);
    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    loop {
        if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
            return v;
        }
        if let Some(d) = device {
            let _ = d.poll(wgpu::PollType::wait_indefinitely());
        }
        std::thread::yield_now();
    }
}

fn try_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = match block_on(
        instance.request_adapter(&wgpu::RequestAdapterOptions::default()),
        None,
    ) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("tidak ada adapter wgpu, uji headless dilewati: {e}");
            return None;
        }
    };
    match block_on(
        adapter.request_device(&wgpu::DeviceDescriptor::default()),
        None,
    ) {
        Ok(dq) => Some(dq),
        Err(e) => {
            eprintln!("gagal membuat device, uji headless dilewati: {e}");
            None
        }
    }
}

const W: u32 = 64;
const H: u32 = 48;
const COLOR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

fn make_texture(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    samples: u32,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen"),
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: samples,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

fn make_target(device: &wgpu::Device, format: wgpu::TextureFormat) -> wgpu::TextureView {
    make_texture(device, format, 1, wgpu::TextureUsages::RENDER_ATTACHMENT)
        .create_view(&Default::default())
}

fn render(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &mut SceneRenderer,
    slot: usize,
    scene: &Scene,
) {
    let screen = egui_wgpu::ScreenDescriptor {
        size_in_pixels: [W, H],
        pixels_per_point: 1.0,
    };
    renderer.prepare(device, queue, &screen, slot, scene);
    let color = make_target(device, COLOR);
    let depth = make_target(device, DEPTH);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("h3d-test"),
    });
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("h3d-test"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &color,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        renderer.paint(slot, &mut rpass);
    }
    queue.submit([encoder.finish()]);
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
}

#[test]
fn headless_renders_2d_and_3d_without_validation_errors() {
    let Some((device, queue)) = try_device() else {
        return;
    };
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

    let mut renderer = SceneRenderer::new(
        &device,
        &queue,
        COLOR,
        Some(DEPTH),
        1,
        RenderBudgets::new(8 * 1024 * 1024, 2 * 1024 * 1024),
    );
    assert!(renderer.max_texture_dimension() >= 2048);

    // --- mesh: tetrahedron di ruang W (µm) dengan skalar (satu NaN) ---
    let center = [10_000.0, 10_000.0, 500.0];
    let pos = [
        [9_000.0, 9_000.0, 0.0],
        [11_000.0, 9_000.0, 0.0],
        [10_000.0, 11_000.0, 0.0],
        [10_000.0, 10_000.0, 1_000.0],
    ];
    let nrm = [[0.0, 0.0, 1.0]; 4];
    let idx = [0u32, 1, 2, 0, 1, 3, 1, 2, 3, 2, 0, 3];
    let mesh = upload_mesh(
        &device,
        &queue,
        &pos,
        &nrm,
        &idx,
        center,
        &VertexAttrs {
            scalars: Some(&[0.0, 1.0, f32::NAN, 0.5]),
            ..Default::default()
        },
    )
    .expect("mesh valid");
    assert_eq!(mesh.index_count, 12);
    renderer.insert_mesh(MeshId(1), mesh);
    let mesh2 = upload_mesh(
        &device,
        &queue,
        &pos,
        &nrm,
        &idx,
        center,
        &VertexAttrs::default(),
    )
    .expect("mesh valid");
    renderer.insert_mesh(MeshId(2), mesh2);

    // --- irisan ---
    let thumb = vec![200u8; 16 * 12 * 4];
    renderer
        .upload_slice(&device, &queue, SliceId(0), 16, 12, &thumb)
        .expect("thumbnail");
    assert!(renderer
        .upload_slice(&device, &queue, SliceId(1), 16, 12, &thumb[..10])
        .is_err());

    let mut inst_scalar = MeshInstance::solid(MeshId(1), [0.8, 0.2, 0.2, 1.0]);
    inst_scalar.scalar_mode = true;
    inst_scalar.scalar_range = [0.0, 1.0];
    let mut inst_alpha = MeshInstance::solid(MeshId(2), [0.2, 0.8, 0.2, 1.0]);
    inst_alpha.opacity = 0.4;
    // Objek "diangkat" dan tersorot (tampilan saja).
    let mut inst_lifted = MeshInstance::solid(MeshId(1), [0.3, 0.3, 0.9, 1.0]);
    inst_lifted.offset_mm = [0.0, 0.0, 1.5];
    inst_lifted.highlight = 1.0;
    let mut inst_lifted_alpha = inst_lifted;
    inst_lifted_alpha.mesh = MeshId(2);
    inst_lifted_alpha.opacity = 0.5;
    inst_lifted_alpha.highlight = 0.5;
    let mut cam = OrbitCamera::default();
    cam.fit_bounds(
        glam::Vec3::splat(-2.0),
        glam::Vec3::splat(2.0),
        W as f32 / H as f32,
    );
    let scene3d = Scene::ThreeD(Scene3D {
        camera: cam,
        aspect: W as f32 / H as f32,
        center_um: center,
        meshes: vec![
            inst_scalar,
            inst_alpha,
            inst_lifted,
            inst_lifted_alpha,
            MeshInstance::solid(MeshId(99), [1.0; 4]),
        ],
        slices: vec![SliceDraw {
            slice: SliceId(0),
            affine: Affine2::IDENTITY,
            size_um: [20_000.0, 20_000.0],
            z_um: 0.0,
            alpha: 0.6,
            glass: 0.8,
        }],
        clip_plane: Some([0.0, 0.0, 1.0, 0.3]),
        lighting: Default::default(),
        z_scale: 2.5,
        grid: Some(h3d_render::Grid3D {
            z_mm: -2.0,
            ..Default::default()
        }),
    });
    render(&device, &queue, &mut renderer, 0, &scene3d);

    // --- 2D: tile RGBA + label + pembanding ---
    renderer.begin_frame();
    let key = |slide: u32, tx: u32| TileKey {
        slide,
        level: 0,
        tx,
        ty: 0,
    };
    let rgba = vec![128u8; 32 * 32 * 4];
    assert_eq!(
        renderer
            .upload_tile(&device, &queue, key(0, 0), 32, 32, &rgba)
            .unwrap(),
        UploadStatus::Uploaded
    );
    assert_eq!(
        renderer
            .upload_tile(&device, &queue, key(0, 0), 32, 32, &rgba)
            .unwrap(),
        UploadStatus::AlreadyPresent
    );
    assert_eq!(
        renderer
            .upload_tile(&device, &queue, key(1, 0), 32, 32, &rgba)
            .unwrap(),
        UploadStatus::Uploaded
    );
    let labels: Vec<u8> = (0..32 * 32)
        .map(|i| if i % 7 == 0 { 3 } else { 0 })
        .collect();
    assert_eq!(
        renderer
            .upload_label_tile(&device, &queue, key(0, 0), 32, 32, &labels)
            .unwrap(),
        UploadStatus::Uploaded
    );
    assert!(renderer
        .upload_tile(&device, &queue, key(2, 0), 32, 32, &rgba[..8])
        .is_err());

    let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(W as f32, H as f32));
    let mut cam2 = Camera2D::default();
    cam2.fit_rect_um([0.0, 0.0], [1000.0, 1000.0], rect);
    let tile_a = TileDraw::new(key(0, 0), [0.0, 0.0, 1000.0, 1000.0]);
    let tile_b = TileDraw::new(key(1, 0), [0.0, 0.0, 1000.0, 1000.0]);
    let missing = TileDraw::new(key(5, 5), [0.0, 0.0, 1000.0, 1000.0]);
    let label_layer = LabelOverlayDraw {
        affine: Affine2::IDENTITY,
        tiles: vec![tile_a],
        opacity: 0.7,
        outline: true,
    };
    for mode in [
        CompareMode::Blend { alpha: 0.5 },
        CompareMode::Checkerboard { size_px: 8.0 },
        CompareMode::Difference,
    ] {
        let scene2d = Scene::TwoD(Scene2D {
            view: cam2.view(rect),
            base: Some(TileLayerDraw::new(vec![tile_a, missing])),
            label_layers: vec![label_layer.clone()],
            compare: Some(CompareParams {
                fixed: TileLayerDraw::new(vec![tile_a]),
                moving: TileLayerDraw {
                    affine: Affine2::translation(50.0, -20.0),
                    tiles: vec![tile_b],
                    opacity: 1.0,
                    mask: None,
                },
                mode,
            }),
        });
        render(&device, &queue, &mut renderer, 1, &scene2d);
    }

    // Paint callback: prepare lewat CallbackResources.
    let mut resources = egui_wgpu::CallbackResources::default();
    resources.insert(renderer);
    let cb = H3dPaintCallback::with_slot(scene3d, 2);
    let screen = egui_wgpu::ScreenDescriptor {
        size_in_pixels: [W, H],
        pixels_per_point: 1.0,
    };
    let mut encoder = device.create_command_encoder(&Default::default());
    let bufs = egui_wgpu::CallbackTrait::prepare(
        &cb,
        &device,
        &queue,
        &screen,
        &mut encoder,
        &mut resources,
    );
    assert!(bufs.is_empty());
    let renderer = resources
        .get::<SceneRenderer>()
        .expect("renderer tersimpan");
    assert_eq!(renderer.mesh_ids(), vec![MeshId(1), MeshId(2)]);
    // Picking mesh yang "diangkat": ray di atas posisi geseran kena, di
    // posisi lama meleset; titik kena dilaporkan di ruang yang terlihat.
    {
        let m = renderer.mesh(MeshId(1)).expect("mesh 1");
        let c = m.bounds_center_mm();
        let off = [0.0, 0.0, 1.5];
        let down = glam::Vec3::new(0.0, 0.0, -1.0);
        let ray_new = h3d_render::Ray {
            origin: glam::Vec3::new(c[0], c[1], c[2] + off[2] + 10.0),
            dir: down,
        };
        let hit = m.pick_offset(&ray_new, off).expect("kena mesh terangkat");
        let hit_old = m.pick(&ray_new).expect("kena mesh asli di bawahnya");
        assert!((hit.point_mm.z - (hit_old.point_mm.z + off[2])).abs() < 1e-4);
        assert!(hit.t < hit_old.t);
    }
    assert!(renderer.has_tile(&key(0, 0)) && renderer.has_label_tile(&key(0, 0)));

    let err = block_on(scope.pop(), Some(&device));
    assert!(err.is_none(), "error validasi wgpu: {err:?}");
}

/// Render adegan 3D ke target MSAA `samples`× yang di-resolve ke tekstur
/// tunggal, lalu baca balik RGBA (baris dipadatkan).
fn render_readback(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &mut SceneRenderer,
    samples: u32,
    scene: &Scene,
) -> Vec<u8> {
    let screen = egui_wgpu::ScreenDescriptor {
        size_in_pixels: [W, H],
        pixels_per_point: 1.0,
    };
    renderer.prepare(device, queue, &screen, 0, scene);
    let resolve = make_texture(
        device,
        COLOR,
        1,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
    );
    let resolve_view = resolve.create_view(&Default::default());
    let msaa = make_texture(
        device,
        COLOR,
        samples,
        wgpu::TextureUsages::RENDER_ATTACHMENT,
    )
    .create_view(&Default::default());
    let depth = make_texture(
        device,
        DEPTH,
        samples,
        wgpu::TextureUsages::RENDER_ATTACHMENT,
    )
    .create_view(&Default::default());
    let row = (W * 4).div_ceil(256) * 256;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: u64::from(row * H),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("h3d-test-msaa"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &msaa,
                depth_slice: None,
                resolve_target: Some(&resolve_view),
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Discard,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        renderer.paint(0, &mut rpass);
    }
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &resolve,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(H),
            },
        },
        wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
    let slice = buf.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let data = slice.get_mapped_range().expect("buffer terpetakan");

    let mut out = Vec::with_capacity((W * H * 4) as usize);
    for y in 0..H {
        let s = (y * row) as usize;
        out.extend_from_slice(&data[s..s + (W * 4) as usize]);
    }
    out
}

/// Bola lintang-bujur di ruang W, jari-jari `r` µm (normal = arah radial).
fn sphere(r: f64, center: [f64; 3]) -> (Vec<[f64; 3]>, Vec<[f32; 3]>, Vec<u32>) {
    let (nu, nv) = (48u32, 24u32);
    let mut pos = Vec::new();
    let mut nrm = Vec::new();
    for j in 0..=nv {
        let th = std::f64::consts::PI * f64::from(j) / f64::from(nv);
        for i in 0..=nu {
            let ph = std::f64::consts::TAU * f64::from(i) / f64::from(nu);
            let d = [th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()];
            pos.push([
                center[0] + r * d[0],
                center[1] + r * d[1],
                center[2] + r * d[2],
            ]);
            nrm.push([d[0] as f32, d[1] as f32, d[2] as f32]);
        }
    }
    let mut idx = Vec::new();
    for j in 0..nv {
        for i in 0..nu {
            let a = j * (nu + 1) + i;
            let b = a + nu + 1;
            idx.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    (pos, nrm, idx)
}

#[test]
fn tissue_material_with_stain_colours_renders_pink_under_msaa() {
    let Some((device, queue)) = try_device() else {
        return;
    };
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut renderer = SceneRenderer::new(
        &device,
        &queue,
        COLOR,
        Some(DEPTH),
        4,
        RenderBudgets::new(8 * 1024 * 1024, 2 * 1024 * 1024),
    );
    assert_eq!(renderer.sample_count(), 4);
    let center = [5_000.0, 5_000.0, 5_000.0];
    let (pos, nrm, idx) = sphere(1_000.0, center);
    let n = pos.len();
    // Pulasan eosin merah muda, oklusi bervariasi, densitas bertingkat.
    let colors = vec![[230u8, 120, 190]; n];
    let occ: Vec<f32> = (0..n).map(|i| 0.5 + 0.5 * (i % 2) as f32).collect();
    let dens: Vec<f32> = (0..n).map(|i| (i % 5) as f32 * 100.0).collect();
    let mesh = upload_mesh(
        &device,
        &queue,
        &pos,
        &nrm,
        &idx,
        center,
        &VertexAttrs {
            colors: Some(&colors),
            occlusion: Some(&occ),
            density: Some(&dens),
            ..Default::default()
        },
    )
    .expect("mesh bola");
    renderer.insert_mesh(MeshId(7), mesh);
    renderer
        .upload_slice(&device, &queue, SliceId(3), 8, 8, &vec![0u8; 8 * 8 * 4])
        .expect("irisan kosong");

    let mut inst = MeshInstance::solid(MeshId(7), [0.2, 0.9, 0.2, 1.0]);
    inst.material = h3d_render::MeshMaterial {
        cells: Some(h3d_render::CellBumps {
            size_mm: 0.25,
            strength: 0.6,
            nucleus_mix: 0.5,
        }),
        ..h3d_render::MeshMaterial::tissue()
    };
    let mut cam = OrbitCamera::default();
    cam.fit_bounds(glam::Vec3::splat(-1.2), glam::Vec3::splat(1.2), 1.0);
    let scene = Scene::ThreeD(Scene3D {
        camera: cam,
        aspect: W as f32 / H as f32,
        center_um: center,
        meshes: vec![inst],
        slices: vec![SliceDraw {
            slice: SliceId(3),
            affine: Affine2::translation(3_000.0, 3_000.0),
            size_um: [4_000.0, 4_000.0],
            z_um: 5_000.0,
            alpha: 0.6,
            glass: 1.0,
        }],
        clip_plane: None,
        lighting: h3d_render::Lighting::studio(),
        z_scale: 1.0,
        grid: None,
    });
    let px = render_readback(&device, &queue, &mut renderer, 4, &scene);
    let err = block_on(scope.pop(), Some(&device));
    assert!(err.is_none(), "error validasi wgpu: {err:?}");

    // Piksel tengah: bola berwarna pulasan (merah > hijau), bukan warna
    // label hijau dan bukan latar hitam.
    let c = ((H / 2 * W + W / 2) * 4) as usize;
    let (r, g, b) = (px[c], px[c + 1], px[c + 2]);
    assert!(r > 60 && r > g && b > g, "piksel tengah {r} {g} {b}");
    // Sudut: hanya kaca tipis irisan atau latar → gelap.
    assert!(px[0] < 120 && px[1] < 140, "sudut {} {}", px[0], px[1]);
}
