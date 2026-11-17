//! Fitur Penandaan (8.11, 8.5.2, 8.12.10) untuk tampilan 2D: viewer slide +
//! overlay label ([`viewer`]), pilihan alat di bilah atas kanvas
//! ([`tool_options`]), dan isi laci "Label dan usulan" ([`labels_drawer`]):
//! panel label (jenis berikon, grup), usulan klasik/AI/interpolasi dengan
//! Terima/Tolak, overlay inti positif/negatif untuk kalibrasi `t_D`, dan
//! "Tandai sudah ditinjau". Pemilihan alat (kuas/penghapus/laso/poligon)
//! ada di rel alat kiri (lihat `shell::rail`).
//!
//! Posisi penunjuk selalu dikonversi layar → W → U sebelum masuk ke
//! `LabelStore` (8.5.2). Satu goresan = satu langkah undo, lalu label
//! irisan langsung disimpan (9.4). Usulan tidak pernah menulis label tanpa
//! "Terima" (aturan keras 8).

use std::hash::{Hash, Hasher};

use egui::Ui;
use h3d_core::{
    AttrDef, AttrKind, LabelDef, LabelId, LabelKind, SlideId, SlideTransform, Stain, VesselKind,
};
use h3d_engine::{
    AttrParams, CellCount2D, CellMethod, ClassicNucleiDetector, ClassicParams, ColorPreset,
    CountParams, EngineError, EpitheliumSource, GlandParams, GlandSummary, ObjectCount2D,
    PaintMode, Palette, Proposal, Session, WatershedNucleiDetector, WholeRegionDetector,
};
use h3d_i18n::t;
use h3d_render::{Camera2D, H3dPaintCallback, LabelOverlayDraw, Scene, Scene2D};
use h3d_segment::StainChannel;

use crate::geom::{brush_radius_um, navigate, screen_to_slide_um};
use crate::gpu::Gpu;
use crate::panels::sample::combo;
use crate::state::{AppState, Drawer, JobOutput};
use crate::tiles::{label_tiles_touched, palette_from_labels};
use crate::ui_util::{icon_label, icon_text_button, icon_text_button_enabled};
use crate::viewer::{
    hide_background_checkbox, label_layer, paint_background, slide_layer, tissue_mask_if_enabled,
    Viewer2D,
};

/// Alat aktif penandaan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Tool {
    /// Geser/zoom pandangan.
    Pan,
    /// Kuas.
    #[default]
    Brush,
    /// Penghapus.
    Eraser,
    /// Laso.
    Lasso,
    /// Poligon titik demi titik.
    Polygon,
}

impl Tool {
    /// Semua alat, urutan di rel alat.
    pub const ALL: [Tool; 5] = [
        Tool::Pan,
        Tool::Brush,
        Tool::Eraser,
        Tool::Lasso,
        Tool::Polygon,
    ];

    /// Ikon alat.
    pub fn icon(self) -> h3d_ui::icons::MaterialIcon {
        use h3d_ui::icons;
        match self {
            Tool::Pan => icons::PAN,
            Tool::Brush => icons::BRUSH,
            Tool::Eraser => icons::ERASER,
            Tool::Lasso => icons::LASSO,
            Tool::Polygon => icons::POLYGON,
        }
    }

    /// Kunci terjemahan nama alat.
    pub fn i18n_key(self) -> &'static str {
        match self {
            Tool::Pan => "tool-pan",
            Tool::Brush => "segment-tool-brush",
            Tool::Eraser => "segment-tool-eraser",
            Tool::Lasso => "segment-tool-lasso",
            Tool::Polygon => "segment-tool-polygon",
        }
    }
}

/// Inti terdeteksi pada wilayah yang tampil, untuk kalibrasi `t_D`.
#[derive(Clone, Debug, PartialEq)]
pub struct NucleiOverlay {
    /// Irisan.
    pub slide: SlideId,
    /// `(posisi di U µm, rata-rata OD kromogen)` per inti.
    pub points: Vec<([f64; 2], f32)>,
    /// Ukuran piksel kerja (µm).
    pub mpp: f64,
}

impl NucleiOverlay {
    /// `(positif, negatif)` untuk ambang `t_d` (inti positif bila `D > t_d`).
    pub fn counts(&self, t_d: f32) -> (usize, usize) {
        let pos = self.points.iter().filter(|(_, d)| *d > t_d).count();
        (pos, self.points.len() - pos)
    }
}

struct ProposalTex {
    key: u64,
    items: Vec<(egui::TextureHandle, [f64; 2], [f64; 2])>,
}

/// Tekstur overlay usulan yang di-cache (opak bagi modul lain).
pub struct ProposalTexCache(ProposalTex);

/// Keadaan mode Segmentasi.
pub struct SegmentState {
    /// Viewer.
    pub viewer: Viewer2D,
    /// Alat aktif.
    pub tool: Tool,
    /// Label aktif.
    pub active_label: Option<LabelId>,
    /// Jari-jari dasar kuas (µm).
    pub brush_radius_um: f64,
    /// Opasitas overlay label.
    pub opacity: f32,
    /// Hanya garis tepi label.
    pub outline: bool,
    /// Goresan berjalan: titik U (µm) dan jari-jari (µm) tiap titik.
    pub stroke: Vec<([f64; 2], f64)>,
    /// Titik laso/poligon berjalan (U, µm).
    pub polygon: Vec<[f64; 2]>,
    /// Usulan klasik yang menunggu keputusan.
    pub classic: Option<(SlideId, Proposal)>,
    /// Parameter segmentasi klasik.
    pub classic_params: ClassicParams,
    /// Ambang klasik otomatis (Otsu).
    pub classic_otsu: bool,
    /// Parameter segmentasi objek 2D kelenjar (ADR 0004; bagian "Lanjutan").
    pub gland_params: GlandParams,
    /// Mode warna (palet + toleransi) aktif; selain itu ambang hematoksilin.
    pub gland_color_mode: bool,
    /// Palet warna irisan aktif (diambil lewat job).
    pub palette: Option<(SlideId, Palette)>,
    /// Indeks warna palet yang dipilih.
    pub palette_picks: Vec<usize>,
    /// Toleransi warna ΔE.
    pub palette_tolerance: f32,
    /// Preset warna.
    pub gland_preset: ColorPreset,
    /// Ukuran objek: 0 kecil, 1 sedang, 2 besar.
    pub gland_size: usize,
    /// Ringkasan usulan kelenjar terakhir (usulannya ada di `classic`).
    pub glands: Option<GlandSummary>,
    /// Overlay inti untuk kalibrasi.
    pub nuclei: Option<NucleiOverlay>,
    /// Hasil penghitungan objek 2D terakhir (overlay nomor di viewer).
    pub count: Option<ObjectCount2D>,
    /// Parameter penghitungan objek.
    pub count_params: CountParams,
    /// Batasi penghitungan ke wilayah yang tampil.
    pub count_in_view: bool,
    /// Hasil penghitungan sel otomatis terakhir (overlay poligon inti).
    pub cells: Option<CellCount2D>,
    /// Metode sel: 0 = watershed klasik, 1 = LoG, 2 = objek utuh,
    /// `i + 3` = model inti ke-`i`.
    pub cell_method: usize,
    /// Ambang OD hematoksilin pendeteksi klasik.
    pub cell_h_threshold: f32,
    /// Soliditas penyatuan keping cembung watershed (0 = mati).
    pub cell_merge_solidity: f64,
    /// Objek utuh: radius penutupan celah, µm.
    pub cell_close_um: f64,
    /// Objek utuh: luas objek minimum, µm².
    pub cell_min_area_um2: f64,
    /// Batasi deteksi sel ke wilayah yang tampil (bawaan benar: seluruh
    /// slide bisa memuat ratusan ribu inti).
    pub cells_in_view: bool,
    /// Isi kontur inti dengan warna transparan pada overlay sel.
    pub cells_fill: bool,
    /// Ambang OD kromogen `t_D`.
    pub t_d: f32,
    /// Cache tekstur usulan.
    pub proposal_tex: Option<ProposalTexCache>,
    /// Nama label baru.
    pub new_label_name: String,
    /// Jenis label baru.
    pub new_label_kind: LabelKind,
    /// Penyangga ganti nama label.
    pub name_edit: Option<(LabelId, String)>,
    /// Kotak pandang terakhir di ruang U irisan aktif (wilayah kalibrasi).
    pub last_view_u: Option<([f64; 2], [f64; 2])>,
}

impl Default for SegmentState {
    fn default() -> Self {
        SegmentState {
            viewer: Viewer2D::default(),
            tool: Tool::Brush,
            active_label: None,
            brush_radius_um: 80.0,
            opacity: 0.5,
            outline: false,
            stroke: Vec::new(),
            polygon: Vec::new(),
            classic: None,
            classic_params: ClassicParams::default(),
            classic_otsu: true,
            gland_params: GlandParams::default(),
            gland_color_mode: true,
            palette: None,
            palette_picks: Vec::new(),
            palette_tolerance: 12.0,
            gland_preset: ColorPreset::Gland,
            gland_size: 1,
            glands: None,
            nuclei: None,
            count: None,
            count_params: CountParams::default(),
            count_in_view: false,
            cells: None,
            cell_method: 0,
            cell_h_threshold: 0.15,
            cell_merge_solidity: 0.93,
            cell_close_um: WholeRegionDetector::default().close_radius_um,
            cell_min_area_um2: WholeRegionDetector::default().min_area_um2,
            cells_in_view: true,
            cells_fill: false,
            t_d: h3d_imgproc::ihc::IhcThresholds::default().t_d,
            proposal_tex: None,
            new_label_name: String::new(),
            new_label_kind: LabelKind::Tumor,
            name_edit: None,
            last_view_u: None,
        }
    }
}

impl SegmentState {
    /// Simpan usulan klasik baru (menggantikan yang lama).
    pub fn set_classic(&mut self, slide: SlideId, proposal: Proposal) {
        self.classic = Some((slide, proposal));
        self.proposal_tex = None;
        self.glands = None;
    }

    /// Simpan palet irisan dan terapkan preset (pilihan warna + toleransi).
    pub fn set_palette(&mut self, slide: SlideId, palette: Palette) {
        self.palette_picks = palette.preset_indices(self.gland_preset);
        self.palette_tolerance = palette.default_tolerance(&self.palette_picks);
        self.palette = Some((slide, palette));
    }

    /// Parameter kelenjar efektif dari pilihan sederhana (mode, preset,
    /// ukuran) di atas `gland_params`; `None` bila mode warna belum punya
    /// warna terpilih.
    pub fn effective_gland_params(&self, slide: SlideId) -> Option<GlandParams> {
        let (min_area, min_lumen) = match self.gland_size {
            0 => (150.0, 30.0),
            1 => (300.0, 50.0),
            _ => (1000.0, 100.0),
        };
        let mut p = GlandParams {
            min_area_um2: min_area,
            min_lumen_area_um2: min_lumen,
            ..self.gland_params.clone()
        };
        if !self.gland_color_mode {
            p.epithelium = EpitheliumSource::Hematoxylin;
            return Some(p);
        }
        let (s, pal) = self.palette.as_ref()?;
        if *s != slide {
            return None;
        }
        let picks = pal.picks(&self.palette_picks);
        if picks.is_empty() {
            return None;
        }
        p.epithelium = EpitheliumSource::Color {
            picks,
            tolerance: self.palette_tolerance,
        };
        if self.gland_preset == ColorPreset::SolidNest {
            p.solid_seed_distance_um = p.solid_seed_distance_um.min(25.0);
        }
        Some(p)
    }

    /// Buang goresan/poligon yang sedang berjalan.
    pub fn cancel_drawing(&mut self) {
        self.stroke.clear();
        self.polygon.clear();
    }
}

/// Ikon jenis label (8.12.10: pemilih `kind` berikon).
pub fn kind_icon(kind: LabelKind) -> h3d_ui::icons::MaterialIcon {
    use h3d_ui::icons;
    match kind {
        LabelKind::Tumor => icons::KIND_TUMOR,
        LabelKind::Necrosis => icons::KIND_NECROSIS,
        LabelKind::Stroma => icons::KIND_STROMA,
        LabelKind::Normal => icons::KIND_NORMAL,
        LabelKind::Vessel(_) => icons::KIND_VESSEL,
        LabelKind::Nerve => icons::KIND_NERVE,
        LabelKind::Capsule => icons::KIND_CAPSULE,
        LabelKind::Margin => icons::KIND_MARGIN,
        LabelKind::Surface => icons::KIND_SURFACE,
        LabelKind::Reference => icons::MEASURE,
        LabelKind::Other => icons::KIND_OTHER,
    }
}

/// Kunci terjemahan jenis label; pembuluh dibedakan menurut jenisnya.
pub fn kind_key(kind: LabelKind) -> &'static str {
    match kind {
        LabelKind::Vessel(VesselKind::Blood) => "label-kind-vessel-blood",
        LabelKind::Vessel(VesselKind::Lymph) => "label-kind-vessel-lymph",
        LabelKind::Vessel(VesselKind::Unknown) => "label-kind-vessel-unknown",
        other => other.i18n_key(),
    }
}

/// Combo jenis label berikon; `true` bila berubah.
pub fn kind_combo(ui: &mut Ui, id: &str, kind: &mut LabelKind) -> bool {
    let mut changed = false;
    egui::ComboBox::from_id_salt(id)
        .selected_text(icon_label(kind_icon(*kind), &t!(kind_key(*kind))))
        .show_ui(ui, |ui| {
            for k in LabelKind::ALL {
                if ui
                    .selectable_label(*kind == k, icon_label(kind_icon(k), &t!(kind_key(k))))
                    .clicked()
                    && *kind != k
                {
                    *kind = k;
                    changed = true;
                }
            }
        });
    changed
}

/// Goreskan `points` (U µm, jari-jari per titik) ke label irisan sebagai
/// **satu** langkah undo. Jari-jari tiap ruas = rata-rata kedua ujungnya,
/// sehingga tekanan Pencil mengubah lebar sepanjang goresan.
pub fn commit_stroke(
    sess: &mut Session,
    slide: SlideId,
    label: LabelId,
    points: &[([f64; 2], f64)],
    erase: bool,
) -> Result<bool, EngineError> {
    if points.is_empty() {
        return Ok(false);
    }
    let pts = points.to_vec();
    // Penghapus menghapus label apa pun (label latar pada mode `Erase`).
    let (target, mode, touched) = if erase {
        (LabelId::BACKGROUND, PaintMode::Erase, None)
    } else {
        (label, PaintMode::Replace, Some(label))
    };
    let name = if erase { "penghapus" } else { "kuas" };
    sess.edit_labels(slide, touched, name, move |store| {
        if pts.len() == 1 {
            store.paint_disc(pts[0].0, pts[0].1, target, mode);
        }
        for w in pts.windows(2) {
            let r = (w[0].1 + w[1].1) / 2.0;
            store.paint_stroke(&[w[0].0, w[1].0], r, target, mode);
        }
    })
}

/// Deteksi inti pada wilayah U `[lo, hi]` irisan dan ukur OD kromogennya
/// (untuk overlay kalibrasi `t_D`). Level dipilih supaya wilayah paling
/// besar 2048 piksel per sisi.
pub fn detect_nuclei_region(
    sess: &Session,
    slide: SlideId,
    lo: [f64; 2],
    hi: [f64; 2],
) -> Result<NucleiOverlay, EngineError> {
    use h3d_imgproc::color::{deconvolve, StainMatrix};
    use h3d_imgproc::nuclei::{detect_nuclei, measure_chromogen, NucleiParams};

    let entry = sess.slide(slide)?.clone();
    let size = entry.size_um();
    let lo = [lo[0].max(0.0), lo[1].max(0.0)];
    let hi = [hi[0].min(size[0]), hi[1].min(size[1])];
    if !(hi[0] > lo[0] && hi[1] > lo[1]) {
        return Err(EngineError::invalid("wilayah kalibrasi di luar irisan"));
    }
    let src = sess.open_source(slide)?;
    let info = src.info().clone();
    let mut level = info.levels.len().saturating_sub(1) as u32;
    for l in 0..info.levels.len() as u32 {
        if let Some(s) = crate::tiles::level_scale(&info, l) {
            let w = (hi[0] - lo[0]) / entry.mpp[0] / s[0];
            let h = (hi[1] - lo[1]) / entry.mpp[1] / s[1];
            if w.max(h) <= 2048.0 {
                level = l;
                break;
            }
        }
    }
    let s = crate::tiles::level_scale(&info, level)
        .ok_or_else(|| EngineError::invalid("slide tanpa level"))?;
    let px = [entry.mpp[0] * s[0], entry.mpp[1] * s[1]];
    let x0 = (lo[0] / px[0]).floor();
    let y0 = (lo[1] / px[1]).floor();
    let w = (((hi[0] - lo[0]) / px[0]).ceil().clamp(1.0, 2048.0)) as u32;
    let h = (((hi[1] - lo[1]) / px[1]).ceil().clamp(1.0, 2048.0)) as u32;
    let img = src.read_region(level, x0 as i64, y0 as i64, w, h)?;
    let matrix = match entry.ihc.as_ref().and_then(|i| i.od_vectors) {
        Some(v) => StainMatrix::new(v)?,
        None => StainMatrix::h_dab(),
    };
    let [hch, dch, _] = deconvolve(img.as_raw(), w as usize, h as usize, &matrix)?;
    let mpp = (px[0] + px[1]) / 2.0;
    let mut nuclei = detect_nuclei(&hch, mpp, None, &NucleiParams::default());
    measure_chromogen(&mut nuclei, &dch, mpp);
    let points = nuclei
        .iter()
        .map(|n| {
            (
                [(x0 + n.x_um / mpp) * px[0], (y0 + n.y_um / mpp) * px[1]],
                n.d_mean,
            )
        })
        .collect();
    Ok(NucleiOverlay { slide, points, mpp })
}

fn palette_hash(labels: &[LabelDef]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for l in labels {
        (l.id.0, l.color, l.visible).hash(&mut h);
    }
    // Tidak pernah 0 agar berbeda dari keadaan "belum dikirim".
    h.finish() | 1
}

/// Urung label irisan aktif.
pub fn undo(app: &mut AppState, sess: &mut Session, redo: bool) {
    let Some(slide) = app.current else {
        return;
    };
    let res = if redo {
        sess.redo(slide)
    } else {
        sess.undo(slide)
    };
    match res {
        Ok(Some(_)) => {
            app.labels_changed(slide);
            if let Err(e) = sess.save_labels(slide) {
                app.toasts.engine_error(&e);
            }
        }
        Ok(None) => {}
        Err(e) => app.toasts.engine_error(&e),
    }
}

/// Pastikan label aktif sah dan palet label sudah di GPU. Dipanggil sebelum
/// viewer dan laci label digambar.
pub fn sync_labels(app: &mut AppState, sess: &Session, gpu: Option<&Gpu>) {
    let labels = sess.project().labels.clone();
    if !app
        .segment
        .active_label
        .is_some_and(|a| labels.iter().any(|l| l.id == a))
    {
        app.segment.active_label = labels.first().map(|l| l.id);
    }
    let key = palette_hash(&labels);
    if key != app.palette_key {
        if let Some(g) = gpu {
            let palette = palette_from_labels(&labels);
            g.with(|r, _, queue| r.set_palette(queue, &palette));
            app.palette_key = key;
        }
    }
}

/// Setel alat aktif (dari rel alat); goresan berjalan dibatalkan.
pub fn set_tool(app: &mut AppState, tool: Tool) {
    if app.segment.tool != tool {
        app.segment.tool = tool;
        app.segment.cancel_drawing();
    }
}

/// Isi laci "Label dan usulan".
pub fn labels_drawer(app: &mut AppState, sess: &mut Session, ui: &mut Ui) {
    sync_labels(app, sess, None);
    label_panel(app, sess, ui);
    ui.separator();
    proposals_panel(app, sess, ui);
    ui.separator();
    count_panel(app, ui);
    ui.separator();
    cells_panel(app, sess, ui);
    ui.separator();
    calibration_panel(app, sess, ui);
}

/// Pilihan alat penandaan di bilah atas kanvas 2D: ukuran kuas, isi lubang,
/// salin ke irisan berikut, tandai ditinjau, status provenance, opasitas dan
/// garis tepi overlay, jari menggambar.
pub fn tool_options(app: &mut AppState, sess: &mut Session, ui: &mut Ui) {
    ui.horizontal_wrapped(|ui| {
        if matches!(app.segment.tool, Tool::Brush | Tool::Eraser) {
            ui.add(
                egui::Slider::new(&mut app.segment.brush_radius_um, 4.0..=2000.0)
                    .logarithmic(true)
                    .suffix(format!(" {}", t!("unit-um")))
                    .text(t!("segment-brush-size")),
            );
            ui.separator();
        }
        let Some(slide) = app.current else {
            return;
        };
        let active = app.segment.active_label;
        if icon_text_button_enabled(
            ui,
            active.is_some(),
            h3d_ui::icons::FILL,
            &t!("segment-fill-holes"),
        )
        .clicked()
        {
            if let Some(label) = active {
                match sess.fill_label_holes(slide, label) {
                    Ok(_) => after_edit(app, sess, slide),
                    Err(e) => app.toasts.engine_error(&e),
                }
            }
        }
        if icon_text_button(ui, h3d_ui::icons::COPY, &t!("segment-copy-to-next")).clicked() {
            match sess.copy_labels_to_next(slide) {
                Ok(dst) => {
                    after_edit(app, sess, dst);
                    app.current = Some(dst);
                }
                Err(e) => app.toasts.engine_error(&e),
            }
        }
        if icon_text_button_enabled(
            ui,
            active.is_some(),
            h3d_ui::icons::REVIEWED,
            &t!("segment-mark-reviewed"),
        )
        .clicked()
        {
            if let Some(label) = active {
                let by = app.settings.user_initials.trim().to_owned();
                if by.is_empty() {
                    app.toasts.error(t!("settings-initials-required"));
                    app.show_settings = true;
                } else {
                    match sess.mark_reviewed(slide, label, &by) {
                        Ok(()) => app.report.needs_refresh = true,
                        Err(e) => app.toasts.engine_error(&e),
                    }
                }
            }
        }
        // Status provenance label aktif pada irisan ini.
        if let Some(rec) = active.and_then(|l| sess.project().annotation(slide, l)) {
            let source = match rec.source.short_name() {
                "manual" => t!("segment-source-manual"),
                "classic" => t!("segment-source-classic"),
                "ai" => t!("segment-source-ai"),
                "interpolated" => t!("segment-source-interpolated"),
                "imported" => t!("segment-source-imported"),
                other => other.to_string(),
            };
            let (text, color) = if rec.review.is_reviewed() {
                (t!("segment-review-reviewed"), h3d_ui::theme::SUCCESS)
            } else {
                (t!("segment-review-draft"), h3d_ui::theme::WARNING)
            };
            ui.label(format!("{source} /"));
            ui.colored_label(color, text);
        }
        ui.separator();
        ui.add(
            egui::Slider::new(&mut app.segment.opacity, 0.0..=1.0)
                .text(t!("segment-overlay-opacity")),
        );
        ui.toggle_value(
            &mut app.segment.outline,
            icon_label(h3d_ui::icons::OUTLINE, &t!("segment-outline")),
        );
        hide_background_checkbox(app, ui);
        if crate::platform::is_mobile() || app.settings.finger_draw {
            let mut on = app.settings.finger_draw;
            if ui
                .toggle_value(
                    &mut on,
                    icon_label(h3d_ui::icons::TOUCH, &t!("segment-finger-draw")),
                )
                .changed()
            {
                app.settings.finger_draw = on;
                app.persist_settings();
            }
        }
    });
}

/// Setelah suntingan label non-goresan: segarkan tile, simpan (9.4).
pub fn after_edit(app: &mut AppState, sess: &mut Session, slide: SlideId) {
    // Label ditulis ke disk oleh simpan otomatis (satu `project.json` per
    // selang, bukan satu per goresan); provenance tetap tercatat di sesi.
    let _ = sess;
    app.labels_changed(slide);
}

fn label_panel(app: &mut AppState, sess: &mut Session, ui: &mut Ui) {
    ui.heading(t!("label-panel"));
    let labels = sess.project().labels.clone();
    let mut update: Option<LabelDef> = None;
    let mut remove: Option<LabelId> = None;
    for def in &labels {
        let active = app.segment.active_label == Some(def.id);
        let mut d = def.clone();
        let mut changed = false;
        ui.horizontal(|ui| {
            if ui.radio(active, "").clicked() {
                app.segment.active_label = Some(def.id);
            }
            changed |= ui
                .color_edit_button_srgba_unmultiplied(&mut d.color)
                .changed();
            ui.label(kind_icon(def.kind).rich_text())
                .on_hover_text(t!(kind_key(def.kind)));
            // Nama: disunting di penyangga, diterapkan saat fokus lepas.
            let editing = app
                .segment
                .name_edit
                .as_ref()
                .is_some_and(|(id, _)| *id == def.id);
            let mut buf = if editing {
                app.segment
                    .name_edit
                    .as_ref()
                    .map(|(_, s)| s.clone())
                    .unwrap_or_default()
            } else {
                def.name.clone()
            };
            let resp = ui.add(egui::TextEdit::singleline(&mut buf).desired_width(110.0));
            if resp.changed() {
                app.segment.name_edit = Some((def.id, buf.clone()));
            }
            if resp.lost_focus() && editing {
                app.segment.name_edit = None;
                if buf.trim() != def.name {
                    d.name = buf.trim().to_owned();
                    changed = true;
                }
            }
            let vis_icon = if d.visible {
                h3d_ui::icons::VISIBILITY_ON
            } else {
                h3d_ui::icons::VISIBILITY_OFF
            };
            if h3d_ui::widgets::icon_button(ui, vis_icon, &t!("label-visible")).clicked() {
                d.visible = !d.visible;
                changed = true;
            }
            changed |= h3d_ui::widgets::toggle_icon_button(
                ui,
                &mut d.reconstruct,
                h3d_ui::icons::CUBE,
                &t!("label-reconstruct"),
            )
            .clicked();
        });
        if active {
            ui.indent(("label-detail", def.id.0), |ui| {
                ui.horizontal(|ui| {
                    ui.label(t!("label-kind"));
                    changed |= kind_combo(ui, "label-kind-combo", &mut d.kind);
                });
                ui.horizontal(|ui| {
                    ui.label(t!("label-group"));
                    changed |= crate::ui_util::opt_text(ui, &mut d.group, 120.0);
                    if h3d_ui::widgets::icon_button(ui, h3d_ui::icons::DELETE, &t!("action-delete"))
                        .clicked()
                    {
                        remove = Some(def.id);
                    }
                });
            });
        }
        if changed {
            update = Some(d);
        }
    }
    if let Some(d) = update {
        if let Err(e) = sess.update_label(d) {
            app.toasts.engine_error(&e);
        }
        app.recon.needs_reload = true;
    }
    if let Some(id) = remove {
        if let Err(e) = sess.remove_label(id) {
            app.toasts.engine_error(&e);
        }
    }
    ui.separator();
    ui.horizontal(|ui| {
        ui.add(
            egui::TextEdit::singleline(&mut app.segment.new_label_name)
                .hint_text(t!("label-name"))
                .desired_width(100.0),
        );
        kind_combo(ui, "new-label-kind", &mut app.segment.new_label_kind);
        if h3d_ui::widgets::icon_button(ui, h3d_ui::icons::ADD, &t!("action-add")).clicked() {
            let n = sess.project().labels.len() as f32;
            let c = h3d_core::Colormap::Viridis.sample((n * 0.37).fract());
            match sess.add_label(
                &app.segment.new_label_name,
                [c[0], c[1], c[2], 255],
                app.segment.new_label_kind,
            ) {
                Ok(id) => {
                    app.segment.active_label = Some(id);
                    app.segment.new_label_name.clear();
                }
                Err(e) => app.toasts.engine_error(&e),
            }
        }
    });
}

fn proposals_panel(app: &mut AppState, sess: &mut Session, ui: &mut Ui) {
    ui.heading(t!("segment-proposals"));
    ui.small(t!("segment-proposal-hint"));
    let Some(slide) = app.current else {
        return;
    };
    let active = app.segment.active_label;

    // --- Keputusan atas usulan yang menunggu (lapisan terpisah di viewer) ---
    let n_pending = sess.pending_proposals(slide).len();
    let has_classic = app
        .segment
        .classic
        .as_ref()
        .is_some_and(|(s, _)| *s == slide);
    if n_pending > 0 || has_classic {
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.label(crate::ui_util::icon_label_colored(
                h3d_ui::icons::LAYERS,
                &t!(
                    "segment-proposals-waiting",
                    n = (n_pending + usize::from(has_classic)).to_string()
                ),
                h3d_ui::theme::WARNING,
            ));
            for p in sess.pending_proposals(slide) {
                let name = sess
                    .project()
                    .label(p.proposal.label)
                    .map(|l| l.name.clone())
                    .unwrap_or_default();
                let conf = p
                    .confidence
                    .map(|c| format!(", {:.2}", c))
                    .unwrap_or_default();
                ui.small(format!(
                    "{name}: {:.3} {}{conf}",
                    p.proposal.area_um2() / 1.0e6,
                    t!("unit-mm2")
                ));
            }
            ui.horizontal(|ui| {
                if icon_text_button(ui, h3d_ui::icons::CHECK, &t!("segment-proposal-accept"))
                    .clicked()
                {
                    let mut ok = true;
                    if let Some((s, p)) = app.segment.classic.take() {
                        if s == slide {
                            if let Err(e) =
                                sess.accept_proposal(slide, &p, PaintMode::OnlyBackground)
                            {
                                app.toasts.engine_error(&e);
                                ok = false;
                            }
                        } else {
                            app.segment.classic = Some((s, p));
                        }
                    }
                    if ok && n_pending > 0 {
                        if let Err(e) = sess.accept_pending(slide, PaintMode::OnlyBackground) {
                            app.toasts.engine_error(&e);
                        }
                    }
                    app.segment.proposal_tex = None;
                    after_edit(app, sess, slide);
                }
                if icon_text_button(ui, h3d_ui::icons::CLOSE, &t!("segment-proposal-reject"))
                    .clicked()
                {
                    if has_classic {
                        app.segment.classic = None;
                    }
                    sess.reject_pending(slide);
                    app.segment.proposal_tex = None;
                }
            });
        });
    }
    let others = sess
        .pending_slides()
        .into_iter()
        .filter(|s| *s != slide)
        .count();
    if others > 0 {
        ui.weak(t!("segment-proposals-other-slides", n = others.to_string()));
    }

    // --- Klasik ---
    egui::CollapsingHeader::new(icon_label(
        h3d_ui::icons::CLASSIC,
        &t!("segment-proposal-classic"),
    ))
    .default_open(true)
    .show(ui, |ui| {
        let st = &mut app.segment;
        ui.horizontal(|ui| {
            ui.label(t!("segment-classic-channel"));
            combo(
                ui,
                "classic-channel",
                &mut st.classic_params.channel,
                &[
                    (StainChannel::Hematoxylin, t!("segment-channel-hematoxylin")),
                    (StainChannel::Eosin, t!("segment-channel-eosin")),
                    (StainChannel::Dab, t!("segment-channel-dab")),
                ],
            );
        });
        ui.checkbox(&mut st.classic_otsu, t!("segment-classic-otsu"));
        if !st.classic_otsu {
            let mut thr = st.classic_params.threshold.unwrap_or(0.5);
            ui.add(egui::Slider::new(&mut thr, 0.0..=1.0).text(t!("segment-threshold")));
            st.classic_params.threshold = Some(thr);
        } else {
            st.classic_params.threshold = None;
        }
        ui.checkbox(&mut st.classic_params.invert, t!("segment-classic-invert"));
        ui.checkbox(&mut st.classic_params.fill_holes, t!("segment-fill-holes"));
        let params = st.classic_params.clone();
        if icon_text_button_enabled(ui, active.is_some(), h3d_ui::icons::PLAY, &t!("action-run"))
            .clicked()
        {
            if let Some(label) = active {
                app.start_job(t!("segment-proposal-classic"), move |s, _p| {
                    s.segment_classic(slide, label, &params)
                        .map(|p| JobOutput::ClassicProposal(slide, Box::new(p)))
                });
            }
        }
    });

    // --- Objek 2D kelenjar (ADR 0004): mode warna sederhana + Lanjutan ---
    egui::CollapsingHeader::new(icon_label(h3d_ui::icons::GLANDS, &t!("glands-title")))
        .default_open(true)
        .show(ui, |ui| {
            ui.small(t!("glands-hint"));
            // Preset.
            {
                let st = &mut app.segment;
                let before = st.gland_preset;
                ui.horizontal(|ui| {
                    ui.label(t!("glands-preset"));
                    combo(
                        ui,
                        "glands-preset",
                        &mut st.gland_preset,
                        &[
                            (ColorPreset::Gland, t!("glands-preset-gland")),
                            (ColorPreset::SolidNest, t!("glands-preset-solid")),
                            (ColorPreset::Vessel, t!("glands-preset-vessel")),
                        ],
                    );
                });
                if st.gland_preset != before {
                    if let Some((s, pal)) = st.palette.clone() {
                        if s == slide {
                            st.set_palette(s, pal);
                        }
                    }
                }
            }
            // Palet warna irisan aktif.
            let has_palette = app
                .segment
                .palette
                .as_ref()
                .is_some_and(|(s, _)| *s == slide);
            if !has_palette {
                ui.small(t!("glands-palette-hint"));
                let mpp = app.segment.gland_params.mpp_um;
                if icon_text_button(ui, h3d_ui::icons::PALETTE, &t!("glands-palette-get")).clicked()
                {
                    app.start_job(t!("glands-palette-get"), move |s, p| {
                        s.slide_palette(slide, mpp, h3d_engine::PALETTE_K, p)
                            .map(|pal| JobOutput::Palette(slide, Box::new(pal)))
                    });
                }
            } else {
                let st = &mut app.segment;
                ui.label(t!("glands-pick-hint"));
                let colors: Vec<(usize, [u8; 3], f64, bool)> = st
                    .palette
                    .as_ref()
                    .map(|(_, pal)| {
                        pal.colors
                            .iter()
                            .enumerate()
                            .map(|(i, c)| (i, c.rgb, c.fraction, c.is_white()))
                            .collect()
                    })
                    .unwrap_or_default();
                ui.horizontal_wrapped(|ui| {
                    for (i, rgb, frac, white) in colors {
                        let picked = st.palette_picks.contains(&i);
                        let fill = egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
                        let stroke = if picked {
                            egui::Stroke::new(3.0, ui.visuals().selection.stroke.color)
                        } else {
                            egui::Stroke::new(1.0, ui.visuals().widgets.inactive.bg_stroke.color)
                        };
                        let btn = egui::Button::new("")
                            .fill(fill)
                            .stroke(stroke)
                            .min_size(egui::vec2(30.0, 30.0));
                        let resp = ui.add(btn).on_hover_text(format!(
                            "{}: {:.0} %{}",
                            t!("glands-color-share"),
                            frac * 100.0,
                            if white {
                                format!(" ({})", t!("glands-color-white"))
                            } else {
                                String::new()
                            }
                        ));
                        if resp.clicked() {
                            if picked {
                                st.palette_picks.retain(|&p| p != i);
                            } else {
                                st.palette_picks.push(i);
                                st.palette_picks.sort_unstable();
                            }
                            if let Some((_, pal)) = st.palette.as_ref() {
                                st.palette_tolerance = pal.default_tolerance(&st.palette_picks);
                            }
                        }
                    }
                });
                ui.add(
                    egui::Slider::new(&mut st.palette_tolerance, 3.0..=40.0)
                        .step_by(1.0)
                        .text(t!("glands-tolerance")),
                );
                ui.horizontal(|ui| {
                    ui.label(t!("glands-size"));
                    combo(
                        ui,
                        "glands-size",
                        &mut st.gland_size,
                        &[
                            (0usize, t!("glands-size-small")),
                            (1usize, t!("glands-size-medium")),
                            (2usize, t!("glands-size-large")),
                        ],
                    );
                });
                if ui.small_button(t!("glands-palette-refresh")).clicked() {
                    st.palette = None;
                }
            }
            // Lanjutan: mode hematoksilin dan parameter mentah.
            egui::CollapsingHeader::new(t!("glands-advanced"))
                .default_open(false)
                .show(ui, |ui| {
                    let gp = &mut app.segment.gland_params;
                    ui.checkbox(&mut app.segment.gland_color_mode, t!("glands-mode-color"));
                    ui.add(
                        egui::Slider::new(&mut gp.mpp_um, 1.0..=4.0)
                            .step_by(0.5)
                            .text(t!("glands-mpp")),
                    );
                    ui.add(
                        egui::Slider::new(&mut gp.epithelium_sigma_um, 1.0..=10.0)
                            .step_by(0.5)
                            .text(t!("glands-sigma")),
                    );
                    ui.add(
                        egui::Slider::new(&mut gp.min_ring_fraction, 0.0..=1.0)
                            .step_by(0.05)
                            .text(t!("glands-min-ring")),
                    );
                    ui.checkbox(&mut gp.star_hull, t!("glands-star-hull"));
                    if gp.star_hull {
                        ui.add(
                            egui::Slider::new(&mut gp.smooth_sigma_deg, 0.0..=20.0)
                                .step_by(1.0)
                                .text(t!("glands-smooth")),
                        );
                    }
                });
            let params = app.segment.effective_gland_params(slide);
            let can_run = active.is_some() && params.is_some();
            if icon_text_button_enabled(ui, can_run, h3d_ui::icons::PLAY, &t!("action-run"))
                .clicked()
            {
                if let (Some(label), Some(params)) = (active, params) {
                    app.start_job(t!("glands-title"), move |s, p| {
                        s.segment_glands(slide, label, &params, p)
                            .map(|g| JobOutput::GlandProposal(slide, Box::new(g)))
                    });
                }
            }
            if let Some(g) = app.segment.glands.as_ref().filter(|g| g.slide == slide) {
                ui.strong(t!(
                    "glands-result",
                    n = g.n.to_string(),
                    lo = g.n_min.to_string(),
                    hi = g.n_max.to_string(),
                    label = g.label_name.clone()
                ));
                ui.label(format!(
                    "{}: {} · {}: {:.0} {}",
                    t!("glands-with-lumen"),
                    g.n_with_lumen,
                    t!("count-median-area"),
                    g.median_area_um2,
                    t!("unit-um2")
                ));
                ui.weak(format!(
                    "{}: {} · {}: {:016x}",
                    t!("glands-stain"),
                    if g.stain_estimated {
                        t!("glands-stain-macenko")
                    } else {
                        t!("glands-stain-ruifrok")
                    },
                    t!("count-params-hash"),
                    g.params_hash
                ));
            }
        });

    // --- AI: alur lengkap ada di laci Deteksi AI ---
    egui::CollapsingHeader::new(icon_label(h3d_ui::icons::AI, &t!("segment-proposal-ai")))
        .default_open(false)
        .show(ui, |ui| {
            ui.small(t!("segment-ai-see-detect"));
            if icon_text_button(ui, h3d_ui::icons::AI, &t!("drawer-detect")).clicked() {
                app.toggle_drawer(Drawer::Detect);
            }
        });

    // --- Interpolasi ---
    egui::CollapsingHeader::new(icon_label(
        h3d_ui::icons::INTERPOLATE,
        &t!("segment-proposal-interpolate"),
    ))
    .default_open(false)
    .show(ui, |ui| {
        ui.small(t!("segment-interpolate-hint"));
        if icon_text_button_enabled(ui, active.is_some(), h3d_ui::icons::PLAY, &t!("action-run"))
            .clicked()
        {
            if let Some(label) = active {
                app.start_job(t!("segment-proposal-interpolate"), move |s, p| {
                    s.interpolate_labels(label, p)?;
                    Ok(JobOutput::PendingProposals)
                });
            }
        }
    });
}

/// Panel "Hitung objek": jumlah objek tertutup label aktif pada irisan
/// aktif (kelenjar, pulau tumor, dst.) dari label tersimpan atau dari usulan
/// klasik yang belum diterima. Tidak mengubah label; hasil dilaporkan
/// dengan rentang ketidakpastian (aturan keras 18).
fn count_panel(app: &mut AppState, ui: &mut Ui) {
    egui::CollapsingHeader::new(icon_label(h3d_ui::icons::COUNT, &t!("count-title")))
        .default_open(false)
        .show(ui, |ui| {
            let Some(slide) = app.current else {
                return;
            };
            let active = app.segment.active_label;
            ui.small(t!("count-hint"));
            {
                let p = &mut app.segment.count_params;
                ui.checkbox(&mut p.fill_holes, t!("count-fill-holes"));
                ui.checkbox(&mut p.split_enabled, t!("count-split"));
                if p.split_enabled {
                    ui.add(
                        egui::Slider::new(&mut p.min_seed_distance_um, 5.0..=500.0)
                            .logarithmic(true)
                            .suffix(format!(" {}", t!("unit-um")))
                            .text(t!("count-min-seed")),
                    );
                    ui.add(
                        egui::Slider::new(&mut p.h_um, 1.0..=100.0)
                            .logarithmic(true)
                            .suffix(format!(" {}", t!("unit-um")))
                            .text(t!("count-h")),
                    );
                }
                ui.add(
                    egui::Slider::new(&mut p.min_area_um2, 10.0..=1.0e6)
                        .logarithmic(true)
                        .suffix(format!(" {}", t!("unit-um2")))
                        .text(t!("count-min-area")),
                );
            }
            ui.checkbox(&mut app.segment.count_in_view, t!("count-in-view"));
            let params = app.segment.count_params.clone();
            let roi = if app.segment.count_in_view {
                app.segment.last_view_u
            } else {
                None
            };
            ui.horizontal(|ui| {
                if icon_text_button_enabled(
                    ui,
                    active.is_some(),
                    h3d_ui::icons::PLAY,
                    &t!("count-run-labels"),
                )
                .clicked()
                {
                    if let Some(label) = active {
                        let params = params.clone();
                        app.start_job(t!("count-title"), move |s, _p| {
                            s.count_objects_2d(slide, label, roi, &params)
                                .map(|c| JobOutput::ObjectCount(Box::new(c)))
                        });
                    }
                }
                let proposal = app
                    .segment
                    .classic
                    .as_ref()
                    .filter(|(s, _)| *s == slide)
                    .map(|(_, p)| p.clone());
                if icon_text_button_enabled(
                    ui,
                    proposal.is_some(),
                    h3d_ui::icons::CLASSIC,
                    &t!("count-run-proposal"),
                )
                .clicked()
                {
                    if let Some(p) = proposal {
                        app.start_job(t!("count-title"), move |s, _p| {
                            s.count_proposal_objects(slide, &p, &params)
                                .map(|c| JobOutput::ObjectCount(Box::new(c)))
                        });
                    }
                }
            });
            if let Some(c) = app.segment.count.as_ref().filter(|c| c.slide == slide) {
                ui.strong(t!(
                    "count-result",
                    n = c.n().to_string(),
                    lo = c.estimate.n_min.to_string(),
                    hi = c.estimate.n_max.to_string(),
                    label = c.label_name.clone()
                ));
                if c.estimate.count.n_excluded > 0 {
                    ui.weak(t!(
                        "count-excluded",
                        n = c.estimate.count.n_excluded.to_string()
                    ));
                }
                if c.n() > 0 {
                    ui.label(format!(
                        "{}: {:.0} {}",
                        t!("count-median-area"),
                        c.estimate.count.median_area_um2(),
                        t!("unit-um2")
                    ));
                }
                if let Some(d) = c.density_per_mm2() {
                    ui.label(format!(
                        "{}: {d:.1} /{}",
                        t!("count-density"),
                        t!("unit-mm2")
                    ));
                }
                ui.weak(format!(
                    "{}: {:016x}",
                    t!("count-params-hash"),
                    c.estimate.count.params_hash
                ));
                if ui.button(t!("count-clear")).clicked() {
                    app.segment.count = None;
                }
            }
        });
}

/// Panel "Hitung sel": deteksi inti otomatis langsung dari citra (StarDist
/// ONNX bila ada model bertugas `nuclei`, atau LoG klasik), tanpa label.
/// Hasil: jumlah dengan rentang, densitas/mm², dan overlay poligon inti.
fn cells_panel(app: &mut AppState, sess: &mut Session, ui: &mut Ui) {
    egui::CollapsingHeader::new(icon_label(h3d_ui::icons::CELLS, &t!("cells-title")))
        .default_open(false)
        .show(ui, |ui| {
            let Some(slide) = app.current else {
                return;
            };
            app.detect.ensure_scanned(sess);
            ui.small(t!("cells-hint"));
            let names: Vec<String> = [
                t!("cells-method-watershed"),
                t!("cells-method-classic"),
                t!("cells-method-whole"),
            ]
            .into_iter()
            .chain(app.detect.nuclei_models.iter().map(|m| m.display_name()))
            .collect();
            let st = &mut app.segment;
            st.cell_method = st.cell_method.min(names.len() - 1);
            egui::ComboBox::from_id_salt("cells-method")
                .width(ui.available_width() - 8.0)
                .selected_text(names[st.cell_method].clone())
                .show_ui(ui, |ui| {
                    for (i, n) in names.iter().enumerate() {
                        ui.selectable_value(&mut st.cell_method, i, n);
                    }
                });
            if app.detect.nuclei_models.is_empty() {
                ui.small(t!("cells-no-model"));
            }
            if app.segment.cell_method < 3 {
                ui.add(
                    egui::Slider::new(&mut app.segment.cell_h_threshold, 0.05..=0.5)
                        .text(t!("cells-h-threshold")),
                );
            }
            if app.segment.cell_method == 2 {
                ui.small(t!("cells-whole-hint"));
                ui.add(
                    egui::Slider::new(&mut app.segment.cell_close_um, 0.0..=5.0)
                        .text(t!("cells-close-gap")),
                );
                ui.add(
                    egui::Slider::new(&mut app.segment.cell_min_area_um2, 1.0..=2000.0)
                        .logarithmic(true)
                        .text(t!("cells-min-area")),
                );
            }
            if app.segment.cell_method == 0 {
                ui.add(
                    egui::Slider::new(&mut app.segment.cell_merge_solidity, 0.0..=1.0)
                        .text(t!("cells-merge-solidity")),
                );
            }
            ui.checkbox(&mut app.segment.cells_in_view, t!("cells-in-view"));
            ui.checkbox(&mut app.segment.cells_fill, t!("cells-fill"));
            let h_threshold = app.segment.cell_h_threshold;
            let method = match app.segment.cell_method {
                0 => CellMethod::Watershed(WatershedNucleiDetector {
                    h_threshold,
                    merge_solidity: app.segment.cell_merge_solidity,
                    ..WatershedNucleiDetector::default()
                }),
                1 => CellMethod::Classic(ClassicNucleiDetector {
                    min_h_od: h_threshold,
                    ..ClassicNucleiDetector::default()
                }),
                2 => CellMethod::WholeRegions(WholeRegionDetector {
                    h_threshold,
                    close_radius_um: app.segment.cell_close_um,
                    min_area_um2: app.segment.cell_min_area_um2,
                    ..WholeRegionDetector::default()
                }),
                i => CellMethod::StarDist(app.detect.nuclei_models[i - 3].dir.clone()),
            };
            let roi = if app.segment.cells_in_view {
                app.segment.last_view_u
            } else {
                None
            };
            let keep_polygons = app.segment.cells_in_view;
            if icon_text_button_enabled(
                ui,
                !app.segment.cells_in_view || roi.is_some(),
                h3d_ui::icons::PLAY,
                &t!("cells-run"),
            )
            .clicked()
            {
                app.start_job(t!("cells-title"), move |s, p| {
                    s.count_cells(slide, roi, &method, keep_polygons, p)
                        .map(|c| JobOutput::CellCount(Box::new(c)))
                });
            }
            if let Some(c) = app.segment.cells.as_ref().filter(|c| c.slide == slide) {
                let whole = c.is_whole_regions();
                ui.strong(t!(
                    if whole {
                        "cells-result-objects"
                    } else {
                        "cells-result"
                    },
                    n = c.n().to_string(),
                    lo = c.detection.n_min.to_string(),
                    hi = c.detection.n_max.to_string()
                ));
                if let Some(d) = c.density_per_mm2() {
                    ui.label(format!(
                        "{}: {d:.0} /{}",
                        t!(if whole {
                            "cells-density-objects"
                        } else {
                            "cells-density"
                        }),
                        t!("unit-mm2")
                    ));
                }
                if c.n() > 0 {
                    ui.label(format!(
                        "{}: {:.1} {}",
                        t!(if whole {
                            "cells-median-area-objects"
                        } else {
                            "cells-median-area"
                        }),
                        c.median_area_um2(),
                        t!("unit-um2")
                    ));
                }
                if whole && c.detection.mpp > WholeRegionDetector::default().mpp {
                    ui.weak(t!(
                        "cells-whole-coarse",
                        mpp = format!("{:.2}", c.detection.mpp)
                    ));
                }
                ui.weak(format!(
                    "{} ({}: {:016x})",
                    c.detection.method,
                    t!("count-params-hash"),
                    c.detection.params_hash
                ));
                if ui.button(t!("count-clear")).clicked() {
                    app.segment.cells = None;
                }
            }
        });
}

fn calibration_panel(app: &mut AppState, sess: &mut Session, ui: &mut Ui) {
    egui::CollapsingHeader::new(icon_label(
        h3d_ui::icons::CALIBRATE,
        &t!("segment-calibrate-threshold"),
    ))
    .default_open(false)
    .show(ui, |ui| {
        let Some(slide) = app.current else {
            return;
        };
        ui.small(t!("segment-calibrate-hint"));
        let stain = sess.slide(slide).map(|s| s.stain).ok();
        if stain != Some(Stain::Ihc) {
            ui.weak(t!("segment-calibrate-not-ihc"));
        }
        if icon_text_button(ui, h3d_ui::icons::PLAY, &t!("segment-calibrate-detect")).clicked() {
            if let Some((lo, hi)) = app.segment.last_view_u {
                app.start_job(t!("segment-calibrate-threshold"), move |s, _p| {
                    detect_nuclei_region(s, slide, lo, hi).map(|n| JobOutput::Nuclei(Box::new(n)))
                });
            }
        }
        ui.add(
            egui::Slider::new(&mut app.segment.t_d, 0.01..=1.0).text(t!("segment-threshold-td")),
        );
        if let Some(n) = app.segment.nuclei.as_ref().filter(|n| n.slide == slide) {
            let (pos, neg) = n.counts(app.segment.t_d);
            ui.horizontal(|ui| {
                ui.colored_label(
                    POSITIVE_COLOR,
                    format!("{}: {pos}", t!("segment-positive-nuclei")),
                );
                ui.colored_label(
                    NEGATIVE_COLOR,
                    format!("{}: {neg}", t!("segment-negative-nuclei")),
                );
            });
            if pos + neg > 0 {
                ui.label(format!(
                    "{:.1} {}",
                    100.0 * pos as f64 / (pos + neg) as f64,
                    t!("unit-percent")
                ));
            }
            if ui.button(t!("segment-calibrate-clear")).clicked() {
                app.segment.nuclei = None;
            }
        }
        // Hitung atribut skor IHC irisan ini dengan ambang terkalibrasi.
        let scoring = sess
            .slide(slide)
            .ok()
            .and_then(|s| s.ihc.as_ref().map(|i| i.scoring));
        if icon_text_button_enabled(
            ui,
            scoring.is_some(),
            h3d_ui::icons::ANALYZE,
            &t!("attr-compute-ihc"),
        )
        .clicked()
        {
            if let Some(scoring) = scoring {
                let t_d = app.segment.t_d;
                app.start_job(t!("attr-compute-ihc"), move |s, p| {
                    let mut params = AttrParams::default();
                    params.thresholds.t_d = t_d;
                    let def = AttrDef {
                        kind: AttrKind::IhcScore { slide, scoring },
                        ..Default::default()
                    };
                    s.compute_attr_with(def, &[slide], &params, p)
                        .map(JobOutput::AttrComputed)
                });
            }
        }
    });
}

const POSITIVE_COLOR: egui::Color32 = egui::Color32::from_rgb(240, 90, 60);
const NEGATIVE_COLOR: egui::Color32 = egui::Color32::from_rgb(70, 150, 250);
/// Warna penanda hasil hitung objek.
const COUNT_COLOR: egui::Color32 = egui::Color32::from_rgb(255, 200, 40);
/// Warna poligon inti hasil deteksi sel.
/// Isi poligon berlubang dengan aturan genap-ganjil sebagai pita
/// horizontal setinggi `step` px layar, hanya di dalam `clip`. Benar untuk
/// kontur cekung dan lubang (lumen) yang tidak bisa diisi kipas atau
/// `convex_polygon`. `rings` = kontur luar lalu lubang, koordinat layar.
fn fill_even_odd(
    rings: &[&[egui::Pos2]],
    color: egui::Color32,
    clip: egui::Rect,
    step: f32,
) -> egui::Mesh {
    let mut mesh = egui::Mesh::default();
    let step = step.max(0.5);
    let (mut y0, mut y1) = (f32::INFINITY, f32::NEG_INFINITY);
    for r in rings {
        for q in *r {
            y0 = y0.min(q.y);
            y1 = y1.max(q.y);
        }
    }
    y0 = y0.max(clip.top());
    y1 = y1.min(clip.bottom());
    // Juga menangkap NaN (kontur kosong → batas tak hingga).
    if y0.partial_cmp(&y1) != Some(std::cmp::Ordering::Less) {
        return mesh;
    }
    let mut xs: Vec<f32> = Vec::new();
    let mut y = y0;
    while y < y1 {
        let yc = y + 0.5 * step;
        xs.clear();
        for r in rings {
            let k = r.len();
            for j in 0..k {
                let (a, b) = (r[j], r[(j + 1) % k]);
                if (a.y <= yc) != (b.y <= yc) {
                    xs.push(a.x + (yc - a.y) / (b.y - a.y) * (b.x - a.x));
                }
            }
        }
        xs.sort_by(f32::total_cmp);
        for pair in xs.chunks_exact(2) {
            let (xa, xb) = (pair[0].max(clip.left()), pair[1].min(clip.right()));
            if xa < xb {
                let rr =
                    egui::Rect::from_min_max(egui::pos2(xa, y), egui::pos2(xb, (y + step).min(y1)));
                mesh.add_colored_rect(rr, color);
            }
        }
        y += step;
    }
    mesh
}

const CELL_COLOR: egui::Color32 = egui::Color32::from_rgb(60, 220, 90);

fn proposal_key(slide: SlideId, props: &[&Proposal]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    slide.hash(&mut h);
    for p in props {
        (
            p.label.0,
            p.mask.dim(),
            p.origin_um[0].to_bits(),
            p.mpp.to_bits(),
        )
            .hash(&mut h);
        p.area_um2().to_bits().hash(&mut h);
    }
    h.finish()
}

/// Gambar masker usulan sebagai citra bergaris diagonal berwarna label,
/// paling besar 1024 piksel per sisi.
pub fn proposal_image(p: &Proposal, color: [u8; 4]) -> egui::ColorImage {
    let (h, w) = p.mask.dim();
    let step = (h.max(w)).div_ceil(1024).max(1);
    let (ow, oh) = (w.div_ceil(step).max(1), h.div_ceil(step).max(1));
    let mut img = egui::ColorImage::filled([ow, oh], egui::Color32::TRANSPARENT);
    for y in 0..oh {
        for x in 0..ow {
            let (sx, sy) = (
                (x * step).min(w.saturating_sub(1)),
                (y * step).min(h.saturating_sub(1)),
            );
            if h > 0 && w > 0 && p.mask[[sy, sx]] != 0 {
                let alpha = if ((x + y) / 4) % 2 == 0 { 190 } else { 70 };
                img.pixels[y * ow + x] =
                    egui::Color32::from_rgba_unmultiplied(color[0], color[1], color[2], alpha);
            }
        }
    }
    img
}

/// Viewer penandaan: gambar irisan aktif + overlay label, usulan, inti
/// kalibrasi, dan masukan alat, mengisi sisa `ui`.
pub fn viewer(app: &mut AppState, sess: &mut Session, ui: &mut Ui, gpu: Option<&Gpu>) {
    app.fix_current(sess);
    sync_labels(app, sess, gpu);
    let Some(slide) = app.current else {
        ui.centered_and_justified(|ui| ui.weak(t!("slides-empty")));
        return;
    };
    let Ok(entry) = sess.slide(slide).cloned() else {
        return;
    };
    let t_affine = entry.transform.affine();
    let transform = entry.transform.clone();
    let size = entry.size_um();
    let (bmin, bmax) = t_affine.transform_rect(size[0], size[1]);

    let can_draw = app.segment.tool != Tool::Pan && app.touch.may_draw(app.settings.finger_draw);
    let frame = app
        .segment
        .viewer
        .begin(ui, u64::from(slide.0), Some((bmin, bmax)), !can_draw);
    if let Some(dir) = frame.nav {
        let order = AppState::ordered_ids(sess);
        app.current = navigate(&order, app.current, dir);
        app.segment.cancel_drawing();
    }
    let rect = frame.rect;
    let camera = app.segment.viewer.camera;
    app.segment.last_view_u = crate::tiles::view_bbox_in_slide(&t_affine, &frame.view);
    paint_background(ui, rect, gpu.is_some());

    if frame.multi_touch {
        // Gestur dua jari membatalkan goresan yang sedang berjalan.
        app.segment.cancel_drawing();
    } else if can_draw {
        handle_drawing(
            app,
            sess,
            slide,
            &transform,
            &camera,
            &frame.response,
            rect,
            ui,
        );
    }

    // Adegan: gambar (latar kaca disembunyikan bila diminta) + overlay label.
    let mut base = slide_layer(&mut app.tiles, sess, gpu, slide, &t_affine, &frame.view);
    base.mask = tissue_mask_if_enabled(app, sess, gpu, ui.ctx(), slide);
    let mut label_layers = Vec::new();
    match sess.labels(slide) {
        Ok(_) => {}
        Err(e) => log::warn!("label irisan {slide} gagal dimuat: {e}"),
    }
    if let Some(store) = sess.labels_loaded(slide) {
        let tiles = label_layer(&mut app.tiles, gpu, slide, store, &t_affine, &frame.view);
        if !tiles.is_empty() {
            label_layers.push(LabelOverlayDraw {
                affine: t_affine,
                tiles,
                opacity: app.segment.opacity,
                outline: app.segment.outline,
            });
        }
    }
    if gpu.is_some() {
        let scene = Scene2D {
            view: camera.view(rect),
            base: Some(base),
            label_layers,
            compare: None,
        };
        ui.painter_at(rect)
            .add(H3dPaintCallback::new(Scene::TwoD(scene)).into_egui(rect));
    }

    let painter = ui.painter_at(rect);
    let to_screen =
        |u: [f64; 2]| -> egui::Pos2 { camera.world_um_to_screen(transform.apply(u), rect) };

    // Lapisan usulan (terpisah dari label; menunggu Terima/Tolak).
    {
        let mut props: Vec<&Proposal> = sess
            .pending_proposals(slide)
            .iter()
            .map(|p| &p.proposal)
            .collect();
        if let Some((s, p)) = &app.segment.classic {
            if *s == slide {
                props.push(p);
            }
        }
        if props.is_empty() {
            app.segment.proposal_tex = None;
        } else {
            let key = proposal_key(slide, &props);
            let stale = app
                .segment
                .proposal_tex
                .as_ref()
                .is_none_or(|c| c.0.key != key);
            if stale {
                let items = props
                    .iter()
                    .enumerate()
                    .map(|(i, p)| {
                        let color = sess
                            .project()
                            .label(p.label)
                            .map_or([255, 200, 0, 255], |l| l.color);
                        let (h, w) = p.mask.dim();
                        let tex = ui.ctx().load_texture(
                            format!("proposal-{i}"),
                            proposal_image(p, color),
                            egui::TextureOptions::NEAREST,
                        );
                        (tex, p.origin_um, [w as f64 * p.mpp, h as f64 * p.mpp])
                    })
                    .collect();
                app.segment.proposal_tex = Some(ProposalTexCache(ProposalTex { key, items }));
            }
            if let Some(cache) = &app.segment.proposal_tex {
                for (tex, origin, size) in &cache.0.items {
                    let corners = [
                        [origin[0], origin[1]],
                        [origin[0] + size[0], origin[1]],
                        [origin[0] + size[0], origin[1] + size[1]],
                        [origin[0], origin[1] + size[1]],
                    ];
                    let uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
                    let mut mesh = egui::Mesh::with_texture(tex.id());
                    for (c, uv) in corners.iter().zip(uvs) {
                        mesh.vertices.push(egui::epaint::Vertex {
                            pos: to_screen(*c),
                            uv: egui::pos2(uv.0, uv.1),
                            color: egui::Color32::WHITE,
                        });
                    }
                    mesh.indices.extend([0, 1, 2, 0, 2, 3]);
                    painter.add(egui::Shape::mesh(mesh));
                    let outline: Vec<egui::Pos2> = corners.iter().map(|c| to_screen(*c)).collect();
                    painter.add(egui::Shape::closed_line(
                        outline,
                        egui::Stroke::new(1.0, h3d_ui::theme::WARNING),
                    ));
                }
            }
        }
    }

    // Overlay inti kalibrasi.
    if let Some(n) = app.segment.nuclei.as_ref().filter(|n| n.slide == slide) {
        let t_d = app.segment.t_d;
        for (u, d) in n.points.iter().take(20_000) {
            let p = to_screen(*u);
            if rect.contains(p) {
                let color = if *d > t_d {
                    POSITIVE_COLOR
                } else {
                    NEGATIVE_COLOR
                };
                painter.circle_stroke(p, 4.0, egui::Stroke::new(1.5, color));
            }
        }
    }

    // Overlay hasil hitung objek: lingkaran + nomor di sentroid; objek yang
    // dikecualikan aturan tepi diredupkan.
    if let Some(c) = app.segment.count.as_ref().filter(|c| c.slide == slide) {
        let font = egui::FontId::proportional(11.0);
        for o in c.estimate.count.objects.iter().take(20_000) {
            let p = to_screen(c.centroid_u(o));
            if !rect.contains(p) {
                continue;
            }
            let color = if o.excluded {
                egui::Color32::GRAY
            } else {
                COUNT_COLOR
            };
            let r = ((o.equiv_diameter_um / 2.0) / camera.um_per_px) as f32;
            painter.circle_stroke(p, r.clamp(3.0, 400.0), egui::Stroke::new(1.5, color));
            if !o.excluded {
                painter.text(
                    p,
                    egui::Align2::CENTER_CENTER,
                    o.id.to_string(),
                    font.clone(),
                    color,
                );
            }
        }
    }

    // Overlay deteksi sel: kontur/lingkaran hijau per inti + jumlah di
    // sudut kanvas, seperti tampilan metode deteksi sel.
    //
    // Anggaran vertex: egui menumpuk semua bentuk satu frame ke satu buffer
    // GPU (batas Metal 256 MB; puluhan ribu kontur × ~100 titik pernah
    // melewatinya dan membuat aplikasi panik). Karena itu titik kontur
    // ditipiskan menurut zoom (jarak layar < 1 px dilewati), inti yang di
    // layar lebih kecil dari beberapa piksel digambar sebagai lingkaran
    // kecil, dan total titik per frame dibatasi; sisanya tidak digambar
    // tetapi jumlahnya tetap tampil di teks.
    if let Some(c) = app.segment.cells.as_ref().filter(|c| c.slide == slide) {
        const POINT_BUDGET: usize = 200_000;
        const MIN_SCREEN_RADIUS_PX: f64 = 3.0;
        let stroke = egui::Stroke::new(1.6, CELL_COLOR);
        let fill = app
            .segment
            .cells_fill
            .then(|| egui::Color32::from_rgba_unmultiplied(60, 220, 90, 60));
        let mut points_used = 0usize;
        let mut pts: Vec<egui::Pos2> = Vec::with_capacity(128);
        let mut hole_pts: Vec<Vec<egui::Pos2>> = Vec::new();
        // Kontur ke layar dengan penipisan (titik < 1 px dari titik
        // terakhir dilewati).
        let project = |center: [f64; 2], poly: &[[f32; 2]], out: &mut Vec<egui::Pos2>| {
            out.clear();
            for v in poly {
                let q = to_screen([center[0] + f64::from(v[0]), center[1] + f64::from(v[1])]);
                if out.last().is_none_or(|l| l.distance_sq(q) >= 1.0) {
                    out.push(q);
                }
            }
        };
        for n in &c.detection.nuclei {
            let p = to_screen(n.center_um);
            // Objek besar (cincin epitel) bisa tampil walau pusatnya di luar
            // layar: uji terhadap jangkauan konturnya.
            let reach_um = n
                .polygon_um
                .iter()
                .map(|v| f64::from(v[0].hypot(v[1])))
                .fold(n.radius_um(), f64::max);
            let reach_px = (reach_um / camera.um_per_px) as f32;
            if !rect.expand(reach_px).contains(p) {
                continue;
            }
            if points_used >= POINT_BUDGET {
                break;
            }
            let r_px = n.radius_um() / camera.um_per_px;
            if n.polygon_um.len() >= 3 && r_px >= MIN_SCREEN_RADIUS_PX {
                project(n.center_um, &n.polygon_um, &mut pts);
                hole_pts.resize_with(n.holes_um.len(), Vec::new);
                for (h, out) in n.holes_um.iter().zip(hole_pts.iter_mut()) {
                    project(n.center_um, h, out);
                }
                hole_pts.retain(|h| h.len() >= 3);
                if pts.len() >= 3 {
                    points_used += pts.len() + hole_pts.iter().map(Vec::len).sum::<usize>();
                    if let Some(fill) = fill {
                        if hole_pts.is_empty() && !c.is_whole_regions() {
                            // Kipas dari sentroid: tepat untuk kontur
                            // star-convex terhadap sentroid (hampir semua
                            // inti), murah untuk ribuan inti;
                            // `convex_polygon` salah pada kontur cekung.
                            let mut mesh = egui::Mesh::default();
                            mesh.colored_vertex(p, fill);
                            for q in &pts {
                                mesh.colored_vertex(*q, fill);
                            }
                            let k = pts.len() as u32;
                            for j in 0..k {
                                mesh.add_triangle(0, 1 + j, 1 + (j + 1) % k);
                            }
                            painter.add(egui::Shape::mesh(mesh));
                        } else {
                            // Objek utuh bisa cekung dan berlubang (lumen):
                            // isi genap-ganjil per pita pindai.
                            let mut rings: Vec<&[egui::Pos2]> = vec![&pts];
                            rings.extend(hole_pts.iter().map(Vec::as_slice));
                            let mesh = fill_even_odd(&rings, fill, rect, 1.0);
                            points_used += mesh.vertices.len() / 4;
                            painter.add(egui::Shape::mesh(mesh));
                        }
                    }
                    painter.add(egui::Shape::closed_line(pts.clone(), stroke));
                    for h in &hole_pts {
                        painter.add(egui::Shape::closed_line(h.clone(), stroke));
                    }
                    continue;
                }
            }
            // Inti kecil di layar atau tanpa kontur: lingkaran kecil.
            points_used += 8;
            painter.circle_stroke(p, (r_px as f32).clamp(1.5, 200.0), stroke);
        }
        let text = t!(
            "cells-overlay-count",
            n = c.n().to_string(),
            lo = c.detection.n_min.to_string(),
            hi = c.detection.n_max.to_string()
        );
        let font = egui::FontId::proportional(14.0);
        let galley = painter.layout_no_wrap(text, font, egui::Color32::WHITE);
        let pos = rect.left_bottom() + egui::vec2(8.0, -8.0 - galley.size().y);
        let bg = egui::Rect::from_min_size(pos, galley.size()).expand(4.0);
        painter.rect_filled(bg, 4.0, egui::Color32::from_black_alpha(160));
        painter.galley(pos, galley, egui::Color32::WHITE);
    }

    // Pratinjau goresan/poligon berjalan.
    let color = app
        .segment
        .active_label
        .and_then(|l| sess.project().label(l))
        .map_or(egui::Color32::YELLOW, |l| {
            egui::Color32::from_rgb(l.color[0], l.color[1], l.color[2])
        });
    let st = &app.segment;
    for w in st.stroke.windows(2) {
        let r = ((w[0].1 + w[1].1) / camera.um_per_px) as f32;
        painter.line_segment(
            [to_screen(w[0].0), to_screen(w[1].0)],
            egui::Stroke::new(r.max(1.0), color.gamma_multiply(0.6)),
        );
    }
    if let [only] = st.stroke.as_slice() {
        painter.circle_filled(
            to_screen(only.0),
            (only.1 / camera.um_per_px) as f32,
            color.gamma_multiply(0.6),
        );
    }
    if !st.polygon.is_empty() {
        let pts: Vec<egui::Pos2> = st.polygon.iter().map(|u| to_screen(*u)).collect();
        painter.add(egui::Shape::line(
            pts.clone(),
            egui::Stroke::new(2.0, color),
        ));
        if st.tool == Tool::Polygon {
            for p in &pts {
                painter.circle_filled(*p, 3.0, color);
            }
        }
    }
    // Kursor kuas.
    if matches!(st.tool, Tool::Brush | Tool::Eraser) {
        if let Some(pos) = frame.response.hover_pos() {
            let r = (brush_radius_um(st.brush_radius_um, app.touch.pressure()) / camera.um_per_px)
                as f32;
            painter.circle_stroke(pos, r, egui::Stroke::new(1.0, egui::Color32::WHITE));
        }
    }
    if st.tool == Tool::Polygon && !st.polygon.is_empty() {
        let pos = rect.left_top() + egui::vec2(8.0, 8.0);
        painter.text(
            pos,
            egui::Align2::LEFT_TOP,
            t!("segment-polygon-hint"),
            egui::FontId::proportional(12.0),
            egui::Color32::WHITE,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_drawing(
    app: &mut AppState,
    sess: &mut Session,
    slide: SlideId,
    transform: &SlideTransform,
    camera: &Camera2D,
    response: &egui::Response,
    rect: egui::Rect,
    ui: &Ui,
) {
    let Some(label) = app.segment.active_label else {
        return;
    };
    let tool = app.segment.tool;
    let pressure = app.touch.pressure();
    // Layar → W → U (8.5.2).
    let to_u = |pos: egui::Pos2| screen_to_slide_um(camera, rect, pos, transform);
    let primary = egui::PointerButton::Primary;
    match tool {
        Tool::Brush | Tool::Eraser => {
            let radius = brush_radius_um(app.segment.brush_radius_um, pressure);
            let pos = response.interact_pointer_pos();
            if response.drag_started_by(primary) {
                app.segment.stroke.clear();
            }
            if response.dragged_by(primary) {
                if let Some(u) = pos.and_then(to_u) {
                    let far = app.segment.stroke.last().is_none_or(|(p, _)| {
                        ((p[0] - u[0]).powi(2) + (p[1] - u[1]).powi(2)).sqrt()
                            >= camera.um_per_px * 1.5
                    });
                    if far {
                        app.segment.stroke.push((u, radius));
                    }
                }
            }
            let mut finish = response.drag_stopped_by(primary);
            if response.clicked_by(primary) {
                if let Some(u) = pos.and_then(to_u) {
                    app.segment.stroke = vec![(u, radius)];
                    finish = true;
                }
            }
            if finish && !app.segment.stroke.is_empty() {
                let pts = std::mem::take(&mut app.segment.stroke);
                let erase = tool == Tool::Eraser;
                match commit_stroke(sess, slide, label, &pts, erase) {
                    Ok(_) => {
                        // Hanya tile yang tersentuh yang diunggah ulang.
                        let max_r = pts.iter().map(|p| p.1).fold(0.0, f64::max);
                        let only: Vec<[f64; 2]> = pts.iter().map(|p| p.0).collect();
                        let range = sess
                            .labels_loaded(slide)
                            .and_then(|store| label_tiles_touched(store, &only, max_r));
                        match range {
                            Some(r) => app.tiles.invalidate_label_tiles(slide, r),
                            None => app.tiles.invalidate_labels(slide),
                        }
                        app.recon.needs_reload = true;
                        app.report.needs_refresh = true;
                        // Simpan otomatis per goresan (9.4).
                        if let Err(e) = sess.save_labels(slide) {
                            app.toasts.engine_error(&e);
                        }
                    }
                    Err(e) => app.toasts.engine_error(&e),
                }
            }
        }
        Tool::Lasso => {
            if response.drag_started_by(primary) {
                app.segment.polygon.clear();
            }
            if response.dragged_by(primary) {
                if let Some(u) = response.interact_pointer_pos().and_then(to_u) {
                    app.segment.polygon.push(u);
                }
            }
            if response.drag_stopped_by(primary) {
                fill_polygon(app, sess, slide, label);
            }
        }
        Tool::Polygon => {
            let (enter, esc) = ui.input(|i| {
                (
                    i.key_pressed(egui::Key::Enter),
                    i.key_pressed(egui::Key::Escape),
                )
            });
            if esc {
                app.segment.polygon.clear();
            } else if response.double_clicked_by(primary) || enter {
                fill_polygon(app, sess, slide, label);
            } else if response.clicked_by(primary) {
                if let Some(u) = response.interact_pointer_pos().and_then(to_u) {
                    app.segment.polygon.push(u);
                }
            }
        }
        Tool::Pan => {}
    }
}

fn fill_polygon(app: &mut AppState, sess: &mut Session, slide: SlideId, label: LabelId) {
    let pts = std::mem::take(&mut app.segment.polygon);
    if pts.len() < 3 {
        return;
    }
    match sess.fill_polygon(slide, label, &pts, PaintMode::Replace) {
        Ok(_) => after_edit(app, sess, slide),
        Err(e) => app.toasts.engine_error(&e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_label_kind_has_icon_and_translation() {
        for k in LabelKind::ALL {
            assert!(!kind_icon(k).codepoint.is_empty());
            for lang in [h3d_i18n::Language::Id, h3d_i18n::Language::En] {
                assert!(h3d_i18n::has_key(lang, kind_key(k)), "{}", kind_key(k));
            }
        }
    }

    #[test]
    fn nuclei_counts_follow_threshold() {
        let n = NucleiOverlay {
            slide: SlideId(0),
            points: vec![([0.0, 0.0], 0.05), ([1.0, 0.0], 0.2), ([2.0, 0.0], 0.6)],
            mpp: 1.0,
        };
        assert_eq!(n.counts(0.15), (2, 1));
        assert_eq!(n.counts(0.5), (1, 2));
        assert_eq!(n.counts(0.9), (0, 3));
    }
}
