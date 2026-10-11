//! Mode `Reconstruct` (8.11, 8.12.10): viewport 3D (orbit/geser/zoom),
//! job "Rekonstruksi" dan "Analisis", parameter, visibilitas dan opasitas
//! per label, bidang potong, irisan di 3D, pemilih lapisan skalar dengan
//! legenda dan rentang, serta klik pada mesh → nilai dan irisan terdekat.

use std::collections::{BTreeMap, BTreeSet};

use egui::Ui;
use glam::Vec3;
use h3d_core::{Colormap, LabelId, SlideId};
use h3d_engine::{LabelMesh, Session};
use h3d_i18n::t;
use h3d_render::{
    ray_from_screen, render_to_world_um, upload_mesh, world_um_to_render_mm, H3dPaintCallback,
    MeshId, MeshInstance, OrbitCamera, Scene, Scene3D, SliceDraw, SliceId, ViewPreset,
};

use crate::gpu::Gpu;
use crate::state::{AppState, JobOutput};
use crate::ui_util::{icon_label, icon_text_button, icon_text_button_enabled};
use crate::viewer::paint_background;

/// Sisi terpanjang thumbnail irisan di 3D (piksel).
pub const SLICE_TEX_SIDE: u32 = 512;

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

/// Keadaan mode Rekonstruksi.
pub struct ReconState {
    /// Kamera orbit (mm, ruang R).
    pub camera: OrbitCamera,
    /// Mesh terakhir (salinan CPU untuk nilai skalar dan statistik).
    pub meshes: Vec<LabelMesh>,
    /// Mesh perlu dimuat ulang dari sesi.
    pub needs_reload: bool,
    /// Mesh perlu diunggah ulang ke GPU (lapisan skalar diganti).
    pub needs_upload: bool,
    /// Pusat volume (µm) yang dipakai saat mengunggah.
    pub center_um: [f64; 3],
    /// Kotak pembatas semua mesh (mm, ruang R).
    pub bounds_mm: Option<(Vec3, Vec3)>,
    /// Kamera sudah dipaskan ke mesh.
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
    /// Alfa irisan di 3D.
    pub slice_alpha: f32,
    /// Irisan yang thumbnailnya sudah diunggah.
    pub uploaded_slices: BTreeSet<SlideId>,
    /// Lapisan skalar terpilih (nama `MeshScalars`).
    pub scalar: Option<String>,
    /// Rentang lapisan skalar; `None` = otomatis p2–p98.
    pub range: Option<[f32; 2]>,
    /// Colormap.
    pub colormap: Colormap,
    /// Hasil klik terakhir.
    pub pick: Option<PickInfo>,
    /// Id mesh yang ada di GPU.
    pub gpu_meshes: BTreeSet<LabelId>,
    /// Pilih lapisan skalar pertama begitu tersedia (argumen `--analyze`).
    pub want_first_scalar: bool,
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
            show_slices: false,
            slice_alpha: 0.85,
            uploaded_slices: BTreeSet::new(),
            scalar: None,
            range: None,
            colormap: Colormap::Viridis,
            pick: None,
            gpu_meshes: BTreeSet::new(),
            want_first_scalar: false,
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

fn mesh_id(label: LabelId) -> MeshId {
    MeshId(u64::from(label.0))
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

fn upload(app: &mut AppState, gpu: &Gpu) {
    let st = &mut app.recon;
    st.needs_upload = false;
    let Some((center, lo, hi)) = meshes_center_um(&st.meshes) else {
        let stale: Vec<LabelId> = st.gpu_meshes.iter().copied().collect();
        gpu.with(|r, _, _| {
            for l in stale {
                r.remove_mesh(mesh_id(l));
            }
        });
        st.gpu_meshes.clear();
        st.bounds_mm = None;
        return;
    };
    st.center_um = center;
    let a = world_um_to_render_mm(lo, center);
    let b = world_um_to_render_mm(hi, center);
    let (a, b) = (Vec3::from(a), Vec3::from(b));
    st.bounds_mm = Some((a.min(b), a.max(b)));
    let scalar = st.scalar.clone();
    let meshes = &st.meshes;
    let mut uploaded = BTreeSet::new();
    let stale: Vec<LabelId> = st.gpu_meshes.iter().copied().collect();
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
                    r.insert_mesh(mesh_id(m.label), g);
                    uploaded.insert(m.label);
                }
                Err(e) => log::warn!("mesh label {} gagal diunggah: {e}", m.label),
            }
        }
    });
    st.gpu_meshes = uploaded;
}

/// Gambar mode Rekonstruksi.
pub fn ui(app: &mut AppState, sess: &mut Session, ui: &mut Ui, gpu: Option<&Gpu>) {
    if app.recon.needs_reload {
        reload(app, sess);
    }
    if app.recon.needs_upload {
        match gpu {
            Some(g) => upload(app, g),
            None => {
                app.recon.needs_upload = false;
                if let Some((center, lo, hi)) = meshes_center_um(&app.recon.meshes) {
                    app.recon.center_um = center;
                    let a = Vec3::from(world_um_to_render_mm(lo, center));
                    let b = Vec3::from(world_um_to_render_mm(hi, center));
                    app.recon.bounds_mm = Some((a.min(b), a.max(b)));
                }
            }
        }
    }

    egui::Panel::left("recon-left")
        .default_size(320.0)
        .show(ui, |ui| {
            egui::ScrollArea::vertical()
                .id_salt("recon-left-scroll")
                .show(ui, |ui| side_panel(app, sess, ui));
        });

    egui::CentralPanel::default().show(ui, |ui| viewport(app, sess, ui, gpu));
}

fn side_panel(app: &mut AppState, sess: &mut Session, ui: &mut Ui) {
    ui.horizontal_wrapped(|ui| {
        if icon_text_button(ui, h3d_ui::icons::PLAY, &t!("recon-run")).clicked() {
            app.start_job(t!("recon-run"), |s, p| {
                s.reconstruct(p).map(|m| JobOutput::Reconstructed(m.len()))
            });
        }
        if icon_text_button_enabled(
            ui,
            !app.recon.meshes.is_empty(),
            h3d_ui::icons::ANALYZE,
            &t!("recon-analyze"),
        )
        .clicked()
        {
            app.start_job(t!("recon-analyze"), |s, p| {
                s.analyze_spatial(p).map(|_| JobOutput::Analyzed)
            });
        }
    });
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
            let labels: Vec<LabelId> = st.meshes.iter().map(|m| m.label).collect();
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

    egui::CollapsingHeader::new(icon_label(h3d_ui::icons::IMAGE, &t!("recon-slices-in-3d")))
        .default_open(false)
        .show(ui, |ui| {
            let st = &mut app.recon;
            ui.checkbox(&mut st.show_slices, t!("recon-show-slice"));
            ui.add(egui::Slider::new(&mut st.slice_alpha, 0.1..=1.0).text(t!("recon-opacity")));
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

    if let Some(pick) = &app.recon.pick {
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
                app.current = Some(id);
                app.set_mode(h3d_ui::AppMode::Segment);
            }
        }
    }
}

fn viewport(app: &mut AppState, sess: &mut Session, ui: &mut Ui, gpu: Option<&Gpu>) {
    // Bilah kamera.
    ui.horizontal(|ui| {
        if icon_text_button(ui, h3d_ui::icons::CENTER_VIEW, &t!("recon-reset-camera")).clicked() {
            app.recon.fitted = false;
            app.recon.camera = OrbitCamera::default();
        }
        for (preset, key) in [
            (ViewPreset::Top, "recon-view-top"),
            (ViewPreset::Front, "recon-view-front"),
            (ViewPreset::Right, "recon-view-right"),
            (ViewPreset::Isometric, "recon-view-iso"),
        ] {
            if ui.button(t!(key)).clicked() {
                app.recon.camera.set_preset(preset);
            }
        }
        ui.weak(t!("recon-pick-hint"));
    });

    let size = ui.available_size().max(egui::vec2(64.0, 64.0));
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());
    paint_background(ui, rect, gpu.is_some());
    let aspect = (rect.width() / rect.height().max(1.0)).max(0.01);
    let st = &mut app.recon;
    if !st.fitted {
        if let Some((lo, hi)) = st.bounds_mm {
            st.camera.fit_bounds(lo, hi, aspect);
            st.fitted = true;
        }
    }

    // Masukan kamera: seret = orbit, seret kanan/tengah atau Shift = geser,
    // roda/cubit = zoom; dua jari = geser + zoom.
    let ctx = ui.ctx().clone();
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

    // Instance mesh.
    let range = st.scalar.as_ref().map(|n| {
        st.range
            .or_else(|| auto_range(&st.meshes, n))
            .unwrap_or([0.0, 1.0])
    });
    let mut instances = Vec::new();
    for m in &st.meshes {
        if !st.gpu_meshes.contains(&m.label) {
            continue;
        }
        let color = sess
            .project()
            .label(m.label)
            .map_or([200, 200, 200, 255], |l| l.color);
        let has_scalar = st
            .scalar
            .as_ref()
            .is_some_and(|n| m.scalars.iter().any(|s| &s.name == n));
        instances.push(MeshInstance {
            mesh: mesh_id(m.label),
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

    // Irisan di 3D: thumbnail pada z_i dengan T_i.
    let mut slices = Vec::new();
    if st.show_slices {
        if let Some(g) = gpu {
            let entries: Vec<_> = sess
                .project()
                .usable_slides_ordered()
                .into_iter()
                .cloned()
                .collect();
            let mut budget = 2;
            for e in &entries {
                if !st.uploaded_slices.contains(&e.id) {
                    if budget == 0 {
                        ctx.request_repaint();
                        continue;
                    }
                    budget -= 1;
                    match sess.thumbnail(e.id, SLICE_TEX_SIDE) {
                        Ok(img) => {
                            let ok = g
                                .with(|r, device, queue| {
                                    r.upload_slice(
                                        device,
                                        queue,
                                        SliceId(e.id.0),
                                        img.width(),
                                        img.height(),
                                        img.as_raw(),
                                    )
                                    .is_ok()
                                })
                                .unwrap_or(false);
                            if ok {
                                st.uploaded_slices.insert(e.id);
                            }
                        }
                        Err(err) => log::warn!("thumbnail irisan {} gagal: {err}", e.id),
                    }
                }
                if st.uploaded_slices.contains(&e.id) {
                    slices.push(SliceDraw {
                        slice: SliceId(e.id.0),
                        affine: e.transform.affine(),
                        size_um: e.size_um(),
                        z_um: e.z_um,
                        alpha: st.slice_alpha,
                    });
                }
            }
        }
    }

    // Bidang potong di ruang R: fragmen dengan n·p − d > 0 dibuang.
    let clip_plane = match (st.clip_enabled, st.bounds_mm) {
        (true, Some((lo, hi))) => {
            let axis = st.clip_axis.min(2);
            let mut n = [0.0f32; 3];
            n[axis] = 1.0;
            let d = lo[axis] + st.clip_offset * (hi[axis] - lo[axis]);
            Some([n[0], n[1], n[2], d])
        }
        _ => None,
    };

    // Klik pada mesh → nilai di titik itu dan irisan terdekat (picking CPU).
    if response.clicked() {
        if let (Some(g), Some(pos)) = (gpu, response.interact_pointer_pos()) {
            let ray = ray_from_screen(&st.camera, rect, pos);
            let visible: Vec<LabelId> = st
                .meshes
                .iter()
                .map(|m| m.label)
                .filter(|l| st.gpu_meshes.contains(l) && !st.hidden.contains(l))
                .collect();
            let center = st.center_um;
            let hit = g
                .with(|r, _, _| {
                    let mut best: Option<(LabelId, h3d_render::PickHit)> = None;
                    for l in visible {
                        if let Some(h) = r.mesh(mesh_id(l)).and_then(|m| m.pick(&ray)) {
                            if best.as_ref().is_none_or(|(_, b)| h.t < b.t) {
                                best = Some((l, h));
                            }
                        }
                    }
                    best
                })
                .flatten();
            st.pick = hit.map(|(label, h)| {
                let point_um = render_to_world_um(h.point_mm.to_array(), center);
                let value = st.scalar.as_ref().and_then(|name| {
                    let mesh = st.meshes.iter().find(|m| m.label == label)?;
                    let v = scalar_at(mesh, name, h.tri_index, h.bary)?;
                    Some((v, scalar_unit(&st.meshes, name)))
                });
                PickInfo {
                    label,
                    point_um,
                    value,
                    nearest_slide: nearest_slide(sess, point_um[2]),
                }
            });
        }
    }

    if gpu.is_some() {
        let scene = Scene3D {
            camera: st.camera.clone(),
            aspect,
            center_um: st.center_um,
            meshes: instances,
            slices,
            clip_plane,
            lighting: Default::default(),
        };
        ui.painter_at(rect)
            .add(H3dPaintCallback::new(Scene::ThreeD(scene)).into_egui(rect));
    }

    // Legenda colormap di atas viewport (8.12.6).
    if let (Some(name), Some(range)) = (st.scalar.clone(), range) {
        let legend = egui::Rect::from_min_size(
            rect.left_bottom() + egui::vec2(12.0, -64.0),
            egui::vec2(260.0, 52.0),
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
    if st.meshes.is_empty() {
        ui.painter_at(rect).text(
            rect.center() + egui::vec2(0.0, 24.0),
            egui::Align2::CENTER_CENTER,
            t!("recon-no-mesh"),
            egui::FontId::proportional(14.0),
            ui.visuals().weak_text_color(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use h3d_volume::MeshScalars;

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
}
