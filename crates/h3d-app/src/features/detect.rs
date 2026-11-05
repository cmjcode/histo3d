//! Laci "Deteksi AI": satu alur bertahap untuk menjalankan model segmentasi
//! ML pada irisan tanpa harus memahami folder bundel atau antrean usulan.
//!
//! Langkah di layar, dari atas ke bawah:
//! 1. **Model** — pilih model terpasang, atau "Impor model" (folder berisi
//!    `model.json` + `model.onnx`) yang disalin ke `models/` bundel.
//! 2. **Cakupan** — irisan yang terbuka saja, atau semua irisan berstatus Ok.
//! 3. **Jalankan** — satu tombol; label untuk kelas model dibuat otomatis bila
//!    belum ada, lalu inferensi berjalan sebagai job latar dengan progres.
//! 4. **Hasil** — ringkasan usulan per irisan; Terima semua / Tolak semua /
//!    Tinjau di 2D. Label tidak pernah berubah sebelum Terima (aturan 8).
//!
//! Semua teks lewat `t!`; ikon dari `h3d_ui::icons` (aturan 13).

use std::path::PathBuf;

use egui::Ui;
use h3d_core::SlideId;
use h3d_engine::{ModelEntry, PaintMode, Session};
use h3d_i18n::t;

use crate::features::segment;
use crate::state::{AppState, Drawer, JobOutput};
use crate::ui_util::{icon_label, icon_text_button, icon_text_button_enabled, slide_title};

/// Irisan mana yang dideteksi.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DetectScope {
    /// Irisan yang sedang terbuka/terpilih.
    #[default]
    Current,
    /// Semua irisan berstatus Ok.
    All,
}

/// Ringkasan satu deteksi yang selesai (hasil job).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DetectSummary {
    /// Jumlah irisan yang diproses.
    pub slides: usize,
    /// Label yang dibuat otomatis untuk kelas model.
    pub labels_created: Vec<String>,
}

/// Keadaan laci Deteksi AI.
#[derive(Default)]
pub struct DetectState {
    /// Model terpasang di bundel (dipindai malas).
    pub models: Vec<ModelEntry>,
    /// Daftar model sudah dipindai sejak proyek dibuka/diubah.
    pub scanned: bool,
    /// Indeks model terpilih.
    pub model_index: usize,
    /// Cakupan deteksi.
    pub scope: DetectScope,
    /// Deteksi terakhir yang selesai (untuk ringkasan).
    pub last: Option<DetectSummary>,
}

impl DetectState {
    /// Paksa pindai ulang daftar model pada frame berikutnya.
    pub fn invalidate(&mut self) {
        self.scanned = false;
    }

    fn selected(&self) -> Option<&ModelEntry> {
        self.models.get(self.model_index)
    }
}

/// Isi laci. Mengembalikan irisan yang diminta dibuka di 2D ("Tinjau").
pub fn detect_drawer(app: &mut AppState, sess: &mut Session, ui: &mut Ui) -> Option<SlideId> {
    app.fix_current(sess);
    if !app.detect.scanned {
        app.detect.models = sess.list_models();
        app.detect.scanned = true;
        app.detect.model_index = app
            .detect
            .model_index
            .min(app.detect.models.len().saturating_sub(1));
    }
    let busy = app.job.is_some();

    ui.label(t!("detect-intro"));
    ui.add_space(4.0);

    // ---- 1. Model ----
    step_title(ui, 1, &t!("detect-step-model"));
    model_section(app, sess, ui, busy);
    ui.add_space(6.0);

    // ---- 2. Cakupan ----
    step_title(ui, 2, &t!("detect-step-scope"));
    let targets = scope_section(app, sess, ui);
    ui.add_space(6.0);

    // ---- 3. Jalankan ----
    step_title(ui, 3, &t!("detect-step-run"));
    run_section(app, sess, ui, busy, &targets);
    ui.add_space(6.0);

    // ---- 4. Hasil ----
    step_title(ui, 4, &t!("detect-step-review"));
    let open = result_section(app, sess, ui, busy);

    ui.add_space(8.0);
    h3d_ui::widgets::research_notice(ui);
    open
}

fn step_title(ui: &mut Ui, n: u8, text: &str) {
    ui.add_space(2.0);
    ui.horizontal(|ui| {
        h3d_ui::widgets::count_badge(ui, usize::from(n), h3d_ui::theme::ACCENT);
        ui.label(egui::RichText::new(text).strong());
    });
}

fn model_section(app: &mut AppState, sess: &mut Session, ui: &mut Ui, busy: bool) {
    let st = &mut app.detect;
    if st.models.is_empty() {
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.label(t!("detect-no-model"));
            ui.small(t!("detect-no-model-hint"));
        });
    } else {
        egui::ComboBox::from_id_salt("detect-model")
            .width(ui.available_width() - 8.0)
            .selected_text(
                st.selected()
                    .map(ModelEntry::display_name)
                    .unwrap_or_default(),
            )
            .show_ui(ui, |ui| {
                for (i, m) in st.models.iter().enumerate() {
                    ui.selectable_value(&mut st.model_index, i, m.display_name());
                }
            });
        if let Some(m) = st.selected() {
            match &m.spec {
                Ok(spec) => {
                    ui.horizontal_wrapped(|ui| {
                        ui.small(t!("detect-model-classes"));
                        for name in m.class_names() {
                            ui.small(egui::RichText::new(name).strong());
                        }
                    });
                    ui.small(t!(
                        "detect-model-resolution",
                        mpp = format!("{:.2}", spec.mpp),
                        px = spec.input_size.to_string()
                    ));
                }
                Err(e) => {
                    h3d_ui::widgets::warning_banner(ui, &t!("detect-model-broken", reason = e));
                }
            }
        }
    }
    ui.horizontal(|ui| {
        if icon_text_button_enabled(
            ui,
            !busy && !crate::platform::is_mobile(),
            h3d_ui::icons::IMPORT,
            &t!("detect-import-model"),
        )
        .on_hover_text(t!("detect-import-model-hint"))
        .clicked()
        {
            if let Some(dir) = crate::platform::pick_model_dir() {
                install_model(app, sess, dir);
            }
        }
        if h3d_ui::widgets::icon_button(ui, h3d_ui::icons::REFRESH, &t!("action-refresh")).clicked()
        {
            app.detect.invalidate();
        }
    });
    if crate::platform::is_mobile() {
        ui.small(t!("detect-import-model-mobile"));
    }
}

/// Salin folder model ke bundel, lalu pilih model itu.
fn install_model(app: &mut AppState, sess: &mut Session, dir: PathBuf) {
    match sess.install_model(&dir) {
        Ok(dest) => {
            app.detect.models = sess.list_models();
            app.detect.scanned = true;
            if let Some(i) = app.detect.models.iter().position(|m| m.dir == dest) {
                app.detect.model_index = i;
            }
            let name = app
                .detect
                .selected()
                .map(ModelEntry::display_name)
                .unwrap_or_default();
            app.toasts.info(t!("detect-model-installed", name = name));
        }
        Err(e) => app.toasts.engine_error(&e),
    }
}

/// Pilihan cakupan; mengembalikan irisan target.
fn scope_section(app: &mut AppState, sess: &Session, ui: &mut Ui) -> Vec<SlideId> {
    let all: Vec<SlideId> = sess
        .project()
        .usable_slides_ordered()
        .iter()
        .map(|s| s.id)
        .collect();
    let current = app.current.filter(|c| sess.slide(*c).is_ok());
    let st = &mut app.detect;
    if current.is_none() {
        st.scope = DetectScope::All;
    }
    let current_text = match current {
        Some(id) => t!("detect-scope-current", slide = slide_title(sess, id)),
        None => t!("detect-scope-current-none"),
    };
    ui.add_enabled_ui(current.is_some(), |ui| {
        ui.radio_value(&mut st.scope, DetectScope::Current, current_text);
    });
    ui.radio_value(
        &mut st.scope,
        DetectScope::All,
        t!("detect-scope-all", n = all.len().to_string()),
    );
    match (st.scope, current) {
        (DetectScope::Current, Some(id)) => vec![id],
        _ => all,
    }
}

fn run_section(
    app: &mut AppState,
    sess: &mut Session,
    ui: &mut Ui,
    busy: bool,
    targets: &[SlideId],
) {
    let model = app
        .detect
        .selected()
        .and_then(|m| m.spec.as_ref().ok().map(|s| (m.dir.clone(), s.clone())));
    let reason = if model.is_none() {
        Some(t!("detect-blocked-no-model"))
    } else if targets.is_empty() {
        Some(t!("detect-blocked-no-slides"))
    } else if busy {
        Some(t!("job-busy"))
    } else {
        None
    };
    let n = targets.len();
    let button = ui.add_enabled(
        reason.is_none(),
        egui::Button::new(icon_label(
            h3d_ui::icons::AI,
            &t!("detect-run", n = n.to_string()),
        ))
        .min_size(egui::vec2(ui.available_width() - 8.0, 32.0)),
    );
    match &reason {
        Some(r) => {
            ui.small(egui::RichText::new(r).color(ui.visuals().weak_text_color()));
        }
        None => {
            ui.small(t!("detect-run-hint"));
        }
    }
    if !button.clicked() {
        return;
    }
    let Some((dir, spec)) = model else {
        return;
    };
    // Label kelas dibuat di thread UI agar panel label langsung
    // memperlihatkannya; inferensi berjalan di job latar.
    let created = match sess.ensure_model_labels(&spec) {
        Ok(setup) => setup.created,
        Err(e) => {
            app.toasts.engine_error(&e);
            return;
        }
    };
    app.invalidate_derived();
    let ids = targets.to_vec();
    app.start_job(t!("detect-job-title"), move |s, p| {
        s.segment_ai(&dir, &ids, p)?;
        Ok(JobOutput::Detected(DetectSummary {
            slides: ids.len(),
            labels_created: created,
        }))
    });
}

/// Ringkasan usulan tertunda dan keputusan massal. Mengembalikan irisan yang
/// ingin ditinjau di 2D.
fn result_section(
    app: &mut AppState,
    sess: &mut Session,
    ui: &mut Ui,
    busy: bool,
) -> Option<SlideId> {
    let pending = sess.pending_slides();
    if pending.is_empty() {
        if busy {
            ui.weak(t!("common-working"));
        } else if app.detect.last.is_some() {
            ui.weak(t!("detect-result-none"));
        } else {
            ui.weak(t!("detect-result-empty"));
        }
        return None;
    }
    if let Some(s) = &app.detect.last {
        if !s.labels_created.is_empty() {
            ui.small(t!(
                "detect-labels-created",
                names = s.labels_created.join(", ")
            ));
        }
    }
    let total: usize = pending
        .iter()
        .map(|s| sess.pending_proposals(*s).len())
        .sum();
    ui.label(crate::ui_util::icon_label_colored(
        h3d_ui::icons::LAYERS,
        &t!(
            "detect-result-summary",
            slides = pending.len().to_string(),
            proposals = total.to_string()
        ),
        h3d_ui::theme::WARNING,
    ));
    let mut open = None;
    egui::ScrollArea::vertical()
        .id_salt("detect-pending")
        .max_height(160.0)
        .show(ui, |ui| {
            for &slide in &pending {
                ui.horizontal(|ui| {
                    if ui
                        .link(slide_title(sess, slide))
                        .on_hover_text(t!("detect-review-open"))
                        .clicked()
                    {
                        open = Some(slide);
                    }
                    let parts: Vec<String> = sess
                        .pending_proposals(slide)
                        .iter()
                        .map(|p| {
                            let name = sess
                                .project()
                                .label(p.proposal.label)
                                .map(|l| l.name.clone())
                                .unwrap_or_default();
                            format!(
                                "{name} {:.3} {}",
                                p.proposal.area_um2() / 1.0e6,
                                t!("unit-mm2")
                            )
                        })
                        .collect();
                    ui.small(parts.join(", "));
                });
            }
        });
    ui.small(t!("segment-proposal-hint"));
    ui.horizontal(|ui| {
        if icon_text_button_enabled(ui, !busy, h3d_ui::icons::CHECK, &t!("detect-accept-all"))
            .clicked()
        {
            accept_all(app, sess);
        }
        if icon_text_button_enabled(ui, !busy, h3d_ui::icons::CLOSE, &t!("detect-reject-all"))
            .clicked()
        {
            sess.reject_all_pending();
            app.segment.proposal_tex = None;
            app.toasts.info(t!("detect-rejected"));
        }
    });
    if let Some(first) = pending.first() {
        if icon_text_button(ui, h3d_ui::icons::DRAW, &t!("detect-review-2d")).clicked() {
            open = Some(*first);
        }
    }
    if open.is_some() {
        app.drawer = Some(Drawer::Labels);
    }
    open
}

/// Terima semua usulan di semua irisan (hanya mengisi piksel latar), simpan
/// label tiap irisan, lalu buka laci Tumor 3D sebagai langkah berikutnya.
fn accept_all(app: &mut AppState, sess: &mut Session) {
    let slides = sess.pending_slides();
    match sess.accept_all_pending(PaintMode::OnlyBackground) {
        Ok(n) => {
            for slide in slides {
                segment::after_edit(app, sess, slide);
            }
            app.segment.proposal_tex = None;
            app.toasts.info(t!("detect-accepted", n = n.to_string()));
            app.toggle_drawer(Drawer::Tumor3D);
        }
        Err(e) => app.toasts.engine_error(&e),
    }
}
