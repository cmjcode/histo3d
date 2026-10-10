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
    SliceId, TileDraw, TileKey, TileLayerDraw, UploadStatus,
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

fn make_target(device: &wgpu::Device, format: wgpu::TextureFormat) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen"),
            size: wgpu::Extent3d {
                width: W,
                height: H,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
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
        Some(&[0.0, 1.0, f32::NAN, 0.5]),
    )
    .expect("mesh valid");
    assert_eq!(mesh.index_count, 12);
    renderer.insert_mesh(MeshId(1), mesh);
    let mesh2 = upload_mesh(&device, &queue, &pos, &nrm, &idx, center, None).expect("mesh valid");
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
            MeshInstance::solid(MeshId(99), [1.0; 4]),
        ],
        slices: vec![SliceDraw {
            slice: SliceId(0),
            affine: Affine2::IDENTITY,
            size_um: [20_000.0, 20_000.0],
            z_um: 0.0,
            alpha: 0.6,
        }],
        clip_plane: Some([0.0, 0.0, 1.0, 0.3]),
        lighting: Default::default(),
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
    assert!(renderer.has_tile(&key(0, 0)) && renderer.has_label_tile(&key(0, 0)));

    let err = block_on(scope.pop(), Some(&device));
    assert!(err.is_none(), "error validasi wgpu: {err:?}");
}
