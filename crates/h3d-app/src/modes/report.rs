//! Mode `Report` (8.11, 8.12.8): peringatan di atas, delapan bagian laporan
//! sebagai tab yang dirender dari `Report`/`SpatialAnalysis`, tab
//! provenance dengan irisan draf, dan tombol ekspor.
//!
//! Hanya ukuran yang ditampilkan — setiap jarak dengan ± ketidakpastian dan
//! sumbu mentah di sebelah nama anatomis; tidak ada teks rekomendasi
//! (aturan keras 18). Kalimat "Hanya untuk riset" selalu tampil (aturan 16).
//! Kotak "sertakan data klinis" mati secara bawaan (aturan 17).

use egui::Ui;
use h3d_core::{SignedAxis, SlideId};
use h3d_engine::{ExportKind, Report, Session};
use h3d_i18n::t;
use h3d_volume::{MarginStatus, Uncertainty, Warning};

use crate::geom::fmt_um_pm;
use crate::panels::sample::direction_label;
use crate::platform;
use crate::state::{AppState, JobOutput};
use crate::ui_util::{icon_label_colored, icon_text_button};

/// Kunci judul tab bagian 1–8 (urutan tetap, 8.12.8).
pub const SECTION_KEYS: [&str; 8] = [
    "report-section-1-context",
    "report-section-2-components",
    "report-section-3-margin",
    "report-section-4-invasion",
    "report-section-5-staging",
    "report-section-6-attributes",
    "report-section-7-provenance",
    "report-section-8-params",
];

/// Jenis ekspor yang ditawarkan (satu per `ExportKind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExportChoice {
    /// `mesh.glb`.
    Glb,
    /// STL per label.
    Stl,
    /// `mesh_mm.obj`.
    Obj,
    /// NIfTI.
    Nifti,
    /// GeoJSON per irisan.
    GeoJson,
    /// `report.md` + `report.csv`.
    Report,
    /// `analysis.json`.
    Analysis,
}

impl ExportChoice {
    /// Semua pilihan, urutan tampil.
    pub const ALL: [ExportChoice; 7] = [
        ExportChoice::Report,
        ExportChoice::Analysis,
        ExportChoice::Glb,
        ExportChoice::Stl,
        ExportChoice::Obj,
        ExportChoice::Nifti,
        ExportChoice::GeoJson,
    ];

    /// Kunci terjemahan tombol.
    pub fn i18n_key(self) -> &'static str {
        match self {
            ExportChoice::Glb => "export-glb",
            ExportChoice::Stl => "export-stl",
            ExportChoice::Obj => "export-obj",
            ExportChoice::Nifti => "export-nifti",
            ExportChoice::GeoJson => "export-geojson",
            ExportChoice::Report => "export-report",
            ExportChoice::Analysis => "export-analysis",
        }
    }

    /// `ExportKind` mesin. `with_clinical` hanya berlaku untuk laporan;
    /// `scalar` hanya untuk GLB.
    pub fn kind(self, with_clinical: bool, scalar: Option<String>) -> ExportKind {
        match self {
            ExportChoice::Glb => ExportKind::Glb { scalar },
            ExportChoice::Stl => ExportKind::Stl,
            ExportChoice::Obj => ExportKind::Obj,
            ExportChoice::Nifti => ExportKind::Nifti,
            ExportChoice::GeoJson => ExportKind::GeoJson,
            ExportChoice::Report => ExportKind::Report { with_clinical },
            ExportChoice::Analysis => ExportKind::Analysis,
        }
    }
}

/// Keadaan mode Laporan.
pub struct ReportState {
    /// Laporan terakhir.
    pub report: Option<Report>,
    /// Laporan perlu dihitung ulang.
    pub needs_refresh: bool,
    /// Tab aktif (0..8).
    pub tab: usize,
    /// Sertakan data klinis saat mengekspor laporan (bawaan: tidak).
    pub with_clinical: bool,
    /// Folder ekspor terakhir.
    pub last_export: Option<String>,
}

impl Default for ReportState {
    fn default() -> Self {
        ReportState {
            report: None,
            needs_refresh: true,
            tab: 0,
            with_clinical: false,
            last_export: None,
        }
    }
}

/// Teks peringatan dalam bahasa aktif (kunci `warn-*` + argumennya).
pub fn warning_message(w: &Warning) -> String {
    let mut args = h3d_i18n::fluent_bundle::FluentArgs::new();
    for (name, value) in w.args() {
        args.set(name, value);
    }
    h3d_i18n::translate(w.i18n_key(), Some(&args))
}

/// Kunci terjemahan status margin (status, bukan anjuran).
pub fn margin_status_key(s: MarginStatus) -> &'static str {
    match s {
        MarginStatus::Involved => "margin-status-involved",
        MarginStatus::Close => "margin-status-close",
        MarginStatus::Clear => "margin-status-clear",
    }
}

/// Jarak beserta ketidakpastiannya; menandai bila dibatasi resolusi Z.
pub fn fmt_unc(u: &Uncertainty) -> String {
    let base = fmt_um_pm(u.value_um, u.plus_minus_um);
    if u.z_limited {
        format!("{base} ({})", t!("report-z-limited"))
    } else {
        base
    }
}

fn mm3(um3: f64) -> String {
    format!("{:.4}", um3 / 1.0e9)
}

fn mm2(um2: f64) -> String {
    format!("{:.4}", um2 / 1.0e6)
}

fn point_mm(p: [f64; 3]) -> String {
    format!(
        "{:.3}, {:.3}, {:.3}",
        p[0] / 1000.0,
        p[1] / 1000.0,
        p[2] / 1000.0
    )
}

fn yes_no(v: bool) -> String {
    if v {
        t!("common-yes")
    } else {
        t!("common-no")
    }
}

fn opt<T>(v: Option<T>, f: impl FnOnce(T) -> String) -> String {
    v.map(f).unwrap_or_else(|| "-".to_string())
}

/// Tabel bergaris dari kepala dan baris teks.
pub fn table(ui: &mut Ui, id: &str, headers: &[String], rows: &[Vec<String>]) {
    egui::ScrollArea::horizontal()
        .id_salt((id, "scroll"))
        .show(ui, |ui| {
            egui::Grid::new(id)
                .num_columns(headers.len())
                .striped(true)
                .spacing(egui::vec2(14.0, 4.0))
                .show(ui, |ui| {
                    for h in headers {
                        crate::ui_util::strong(ui, h);
                    }
                    ui.end_row();
                    for row in rows {
                        for cell in row {
                            ui.label(cell);
                        }
                        ui.end_row();
                    }
                });
        });
}

fn key_values(ui: &mut Ui, id: &str, rows: &[(String, String)]) {
    egui::Grid::new(id)
        .num_columns(2)
        .striped(true)
        .spacing(egui::vec2(14.0, 4.0))
        .show(ui, |ui| {
            for (k, v) in rows {
                ui.label(k);
                ui.label(v);
                ui.end_row();
            }
        });
}

/// Gambar mode Laporan.
pub fn ui(app: &mut AppState, sess: &mut Session, ui: &mut Ui) {
    if app.report.needs_refresh {
        app.report.needs_refresh = false;
        match sess.measure() {
            Ok(r) => app.report.report = Some(r),
            Err(e) => {
                app.report.report = None;
                app.toasts.engine_error(&e);
            }
        }
    }

    egui::Panel::right("report-right")
        .default_size(280.0)
        .show(ui, |ui| export_panel(app, sess, ui));

    egui::Panel::bottom("report-notice").show(ui, |ui| {
        // Aturan keras 16: selalu tampil di mode Laporan.
        h3d_ui::widgets::research_notice(ui);
    });

    egui::CentralPanel::default().show(ui, |ui| {
        let Some(report) = app.report.report.clone() else {
            ui.weak(t!("report-unavailable"));
            return;
        };
        // Peringatan di atas.
        ui.horizontal(|ui| {
            ui.heading(t!("report-warnings"));
            if h3d_ui::widgets::icon_button(ui, h3d_ui::icons::REFRESH, &t!("action-refresh"))
                .clicked()
            {
                app.report.needs_refresh = true;
            }
        });
        let warnings = report.all_warnings();
        if warnings.is_empty() {
            ui.weak(t!("report-no-warnings"));
        } else {
            egui::ScrollArea::vertical()
                .id_salt("report-warnings")
                .max_height(150.0)
                .show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        for w in &warnings {
                            h3d_ui::widgets::warning_banner(ui, &warning_message(w));
                        }
                    });
                });
        }
        ui.separator();
        ui.horizontal_wrapped(|ui| {
            for (i, key) in SECTION_KEYS.iter().enumerate() {
                if ui.selectable_label(app.report.tab == i, t!(key)).clicked() {
                    app.report.tab = i;
                }
            }
        });
        ui.separator();
        let mut open_slide = None;
        egui::ScrollArea::vertical()
            .id_salt("report-body")
            .auto_shrink([false, false])
            .show(ui, |ui| match app.report.tab {
                0 => section_context(ui, &report, sess),
                1 => section_components(ui, &report),
                2 => section_margin(ui, &report),
                3 => section_invasion(ui, &report),
                4 => section_staging(ui, &report),
                5 => section_attributes(ui, &report, sess),
                6 => open_slide = section_provenance(ui, &report, sess),
                _ => section_params(ui, &report),
            });
        if let Some(id) = open_slide {
            app.current = Some(id);
            app.set_mode(h3d_ui::AppMode::Segment);
        }
    });
}

fn export_panel(app: &mut AppState, sess: &mut Session, ui: &mut Ui) {
    ui.heading(t!("report-export"));
    ui.checkbox(&mut app.report.with_clinical, t!("clinical-export-with"));
    ui.small(t!("export-clinical-hint"));
    ui.separator();
    for choice in ExportChoice::ALL {
        if icon_text_button(ui, h3d_ui::icons::EXPORT, &t!(choice.i18n_key())).clicked() {
            let out = if platform::is_mobile() {
                Some(sess.bundle().out_dir())
            } else {
                platform::pick_export_dir()
            };
            if let Some(out) = out {
                let with_clinical = app.report.with_clinical;
                let scalar = app.recon.scalar.clone();
                app.start_job(t!(choice.i18n_key()), move |s, _p| {
                    std::fs::create_dir_all(&out)
                        .map_err(|e| h3d_engine::EngineError::invalid(e.to_string()))?;
                    s.export_files(choice.kind(with_clinical, scalar), &out)
                        .map(JobOutput::Exported)
                });
            }
        }
    }
    if let Some(dir) = &app.report.last_export {
        ui.separator();
        ui.small(t!("export-last", dir = dir.clone()));
    }
}

fn section_context(ui: &mut Ui, r: &Report, sess: &Session) {
    let mut rows = vec![
        (t!("slides-sample-name"), r.sample_name.clone()),
        (t!("slides-sample-notes"), r.sample_notes.clone()),
        (
            t!("slides-shrinkage"),
            opt(r.shrinkage_linear, |f| format!("{f:.3}")),
        ),
    ];
    // Orientasi: sumbu mentah selalu tampil, nama anatomis di sebelahnya.
    for axis in [SignedAxis::PosX, SignedAxis::PosY, SignedAxis::PosZ] {
        let name = r.orientation.as_ref().and_then(|o| o.name_of(axis));
        rows.push((
            format!("{} {}", t!("measure-margin-direction"), axis.as_str()),
            direction_label(axis, name),
        ));
    }
    key_values(ui, "report-context", &rows);
    if r.orientation.is_none() {
        ui.weak(t!("report-orientation-unset"));
    }
    ui.separator();
    crate::ui_util::strong(ui, t!("clinical-panel"));
    // Di dalam aplikasi hanya konteks ringkas yang tampil (8.12.9).
    match &sess.project().sample.clinical {
        Some(c) => key_values(
            ui,
            "report-clinical",
            &[
                (
                    t!("clinical-subject-pseudo-id"),
                    c.subject_pseudo_id.clone(),
                ),
                (
                    t!("clinical-topography-icdo"),
                    c.topography_icdo.clone().unwrap_or_default(),
                ),
                (
                    t!("clinical-morphology-icdo"),
                    c.morphology_icdo.clone().unwrap_or_default(),
                ),
            ],
        ),
        None => {
            ui.weak(t!("common-none"));
        }
    }
    ui.small(t!("clinical-hint"));
}

fn section_components(ui: &mut Ui, r: &Report) {
    crate::ui_util::strong(ui, t!("report-measurements"));
    let unc = r.dimension_uncertainty_um;
    let headers = vec![
        t!("label-name"),
        t!("label-kind"),
        t!("label-group"),
        format!("{} ({})", t!("measure-volume-cavalieri"), t!("unit-mm3")),
        format!(
            "{} ({})",
            t!("provenance-reviewed-only-volume"),
            t!("unit-mm3")
        ),
        format!("{} ({})", t!("measure-volume-corrected"), t!("unit-mm3")),
        format!("{} ({})", t!("measure-volume-mesh"), t!("unit-mm3")),
        format!("{} ({})", t!("measure-surface-area"), t!("unit-mm2")),
        t!("measure-max-diameter"),
        t!("measure-components"),
        t!("report-slices-annotated"),
    ];
    let rows: Vec<Vec<String>> = r
        .labels
        .iter()
        .map(|l| {
            vec![
                l.name.clone(),
                t!(crate::modes::segment::kind_key(l.kind)),
                l.group.clone(),
                mm3(l.volume_cavalieri_um3),
                opt(l.volume_reviewed_um3, mm3),
                opt(l.volume_corrected_um3, mm3),
                opt(l.volume_mesh_um3, mm3),
                opt(l.surface_area_um2, mm2),
                opt(l.max_diameter_um, |d| fmt_um_pm(d, unc)),
                opt(l.n_components, |n| n.to_string()),
                format!(
                    "{} / {} / {}",
                    l.n_slices_annotated, l.n_slices_interpolated, l.n_slices_reviewed
                ),
            ]
        })
        .collect();
    table(ui, "report-labels", &headers, &rows);
    ui.small(t!("report-slices-annotated-hint"));
    ui.small(t!(
        "report-dimension-uncertainty",
        value = format!("{:.3}", unc / 1000.0)
    ));

    ui.separator();
    crate::ui_util::strong(ui, t!("label-group"));
    let headers = vec![
        t!("label-group"),
        t!("label-name"),
        format!("{} ({})", t!("measure-volume-cavalieri"), t!("unit-mm3")),
        t!("measure-fraction"),
    ];
    let mut rows = Vec::new();
    for g in &r.groups {
        for m in &g.members {
            rows.push(vec![
                g.group.clone(),
                m.name.clone(),
                mm3(m.volume_um3),
                format!("{:.1} {}", m.fraction * 100.0, t!("unit-percent")),
            ]);
        }
    }
    table(ui, "report-groups", &headers, &rows);

    ui.separator();
    crate::ui_util::strong(ui, t!("measure-foci"));
    match r.analysis.as_ref().and_then(|a| a.foci.as_ref()) {
        Some(f) => {
            ui.label(format!("{}: {}", t!("measure-foci"), f.n));
            let headers = vec![
                "#".to_string(),
                format!("{} ({})", t!("measure-volume-mesh"), t!("unit-mm3")),
                t!("measure-max-diameter"),
                format!("{} ({})", t!("report-centroid"), t!("unit-mm")),
            ];
            let rows: Vec<Vec<String>> = f
                .foci
                .iter()
                .enumerate()
                .map(|(i, x)| {
                    vec![
                        (i + 1).to_string(),
                        mm3(x.volume_um3),
                        fmt_um_pm(x.max_diameter_um, unc),
                        point_mm(x.centroid_um),
                    ]
                })
                .collect();
            table(ui, "report-foci", &headers, &rows);
            if f.distances_um.len() > 1 {
                ui.label(t!("report-foci-distance"));
                let mut rows = Vec::new();
                for (i, row) in f.distances_um.iter().enumerate() {
                    for (j, d) in row.iter().enumerate().skip(i + 1) {
                        rows.push(vec![format!("{} - {}", i + 1, j + 1), fmt_unc(d)]);
                    }
                }
                table(
                    ui,
                    "report-foci-dist",
                    &[t!("measure-foci"), t!("report-distance")],
                    &rows,
                );
            }
        }
        None => {
            ui.weak(t!("report-analysis-missing"));
        }
    }
    if let Some(p) = r
        .analysis
        .as_ref()
        .and_then(|a| a.fractions.necrosis_percent)
    {
        ui.label(format!(
            "{}: {:.1} {}",
            t!("label-kind-necrosis"),
            p,
            t!("unit-percent")
        ));
    }
}

fn section_margin(ui: &mut Ui, r: &Report) {
    let Some(a) = &r.analysis else {
        ui.weak(t!("report-analysis-missing"));
        return;
    };
    let Some(m) = &a.margin else {
        ui.weak(t!("report-margin-missing"));
        return;
    };
    key_values(
        ui,
        "report-margin",
        &[
            (t!("measure-margin-dmin"), fmt_unc(&m.d_min)),
            (
                t!("measure-margin-direction"),
                direction_label(m.direction, m.direction_name),
            ),
            (t!("measure-margin-status"), t!(margin_status_key(m.status))),
            (
                format!("{} ({})", t!("report-nearest-tumor"), t!("unit-mm")),
                point_mm(m.nearest_tumor_um),
            ),
            (
                format!("{} ({})", t!("measure-margin-nearest"), t!("unit-mm")),
                point_mm(m.nearest_margin_um),
            ),
        ],
    );
    ui.separator();
    let rows: Vec<Vec<String>> = SignedAxis::ALL
        .iter()
        .zip(m.per_direction.iter())
        .map(|(axis, d)| {
            let name = r.orientation.as_ref().and_then(|o| o.name_of(*axis));
            vec![
                direction_label(*axis, name),
                match d {
                    Some(u) => fmt_unc(u),
                    None => t!("margin-status-unknown"),
                },
            ]
        })
        .collect();
    table(
        ui,
        "report-margin-dir",
        &[t!("measure-margin-direction"), t!("measure-margin-dmin")],
        &rows,
    );
}

fn section_invasion(ui: &mut Ui, r: &Report) {
    let Some(a) = &r.analysis else {
        ui.weak(t!("report-analysis-missing"));
        return;
    };
    if a.contacts.is_empty() {
        ui.weak(t!("report-invasion-none"));
    } else {
        let headers = vec![
            t!("label-name"),
            t!("label-kind"),
            t!("report-contact-present"),
            format!("{} ({})", t!("measure-contact-area"), t!("unit-mm2")),
            t!("measure-patches"),
        ];
        let rows: Vec<Vec<String>> = a
            .contacts
            .iter()
            .map(|c| {
                vec![
                    c.name.clone(),
                    t!(crate::modes::segment::kind_key(c.kind)),
                    yes_no(c.result.any),
                    format!("{:.4}", c.result.area_mm2),
                    c.result.patches.len().to_string(),
                ]
            })
            .collect();
        table(ui, "report-contacts", &headers, &rows);
        let mut rows = Vec::new();
        for c in &a.contacts {
            for (i, p) in c.result.patches.iter().enumerate() {
                rows.push(vec![
                    c.name.clone(),
                    (i + 1).to_string(),
                    format!("{:.4}", p.area_mm2),
                    point_mm(p.centroid_um),
                ]);
            }
        }
        if !rows.is_empty() {
            ui.separator();
            table(
                ui,
                "report-patches",
                &[
                    t!("label-name"),
                    t!("measure-patches"),
                    format!("{} ({})", t!("measure-contact-area"), t!("unit-mm2")),
                    format!("{} ({})", t!("report-centroid"), t!("unit-mm")),
                ],
                &rows,
            );
        }
    }
    if let Some(c) = &a.capsule {
        ui.separator();
        key_values(
            ui,
            "report-capsule",
            &[
                (t!("report-capsule-breach"), yes_no(c.breached)),
                (
                    format!("{} ({})", t!("report-capsule-outside"), t!("unit-mm3")),
                    mm3(c.volume_outside_um3),
                ),
                (
                    format!("{} ({})", t!("report-capsule-inside"), t!("unit-mm3")),
                    mm3(c.volume_inside_um3),
                ),
            ],
        );
    }
    if let Some(h) = &a.hypoxia {
        ui.separator();
        ui.label(t!(
            "report-hypoxia",
            threshold = format!("{:.0}", h.threshold_um),
            percent = format!("{:.1}", h.fraction_beyond * 100.0)
        ));
    }
}

fn section_staging(ui: &mut Ui, r: &Report) {
    let Some(a) = &r.analysis else {
        ui.weak(t!("report-analysis-missing"));
        return;
    };
    let mut rows = Vec::new();
    if let Some(s) = &a.size {
        rows.push((t!("measure-max-diameter"), fmt_unc(&s.max_diameter_um)));
        for (i, axis) in s.pca_axes_um.iter().enumerate() {
            let d = s.pca_directions[i];
            let dominant = SignedAxis::dominant(d);
            let name = r.orientation.as_ref().and_then(|o| o.name_of(dominant));
            rows.push((
                format!(
                    "{} {} [{}]",
                    t!("measure-pca-axes"),
                    i + 1,
                    direction_label(dominant, name)
                ),
                fmt_unc(axis),
            ));
        }
    }
    if let Some(d) = &a.depth {
        rows.push((t!("measure-depth"), fmt_unc(&d.depth)));
    }
    if let Some(b) = &a.breslow {
        rows.push((t!("measure-breslow"), fmt_unc(&b.depth)));
    }
    rows.push((
        format!("{} ({})", t!("report-tumor-volume"), t!("unit-mm3")),
        mm3(a.tumor_volume_um3),
    ));
    key_values(ui, "report-staging", &rows);
    if a.size.is_none() {
        ui.weak(t!("report-size-missing"));
    }
}

fn section_attributes(ui: &mut Ui, r: &Report, sess: &Session) {
    if r.attrs.is_empty() {
        ui.weak(t!("report-attr-none"));
        return;
    }
    for (i, attr) in r.attrs.iter().enumerate() {
        crate::ui_util::strong(ui, format!("{} ({})", attr.name, attr.unit));
        key_values(
            ui,
            &format!("report-attr-meta-{i}"),
            &[
                (t!("register-method"), attr.method.clone()),
                (
                    t!("measure-score-params"),
                    opt(attr.params_hash, |h| {
                        t!("report-params-hash", hash = format!("{h:016x}"))
                    }),
                ),
                (
                    t!("measure-heterogeneity"),
                    opt(attr.range_between_slides, |v| format!("{v:.2}")),
                ),
            ],
        );
        ui.label(t!("measure-per-slice"));
        let rows: Vec<Vec<String>> = attr
            .per_slide
            .iter()
            .map(|s| {
                vec![
                    crate::ui_util::slide_title(sess, s.slide),
                    format!("{:.2}", s.mean),
                    opt(s.intratumoral, |v| format!("{v:.2}")),
                    opt(s.peritumoral, |v| format!("{v:.2}")),
                ]
            })
            .collect();
        table(
            ui,
            &format!("report-attr-slides-{i}"),
            &[
                t!("mode-slides"),
                t!("report-attr-mean"),
                t!("report-attr-intratumoral"),
                t!("report-attr-peritumoral"),
            ],
            &rows,
        );
        if let Some(st) = r
            .analysis
            .as_ref()
            .and_then(|a| a.attr_stats.iter().find(|s| s.name == attr.name))
        {
            ui.label(t!("measure-per-region"));
            key_values(
                ui,
                &format!("report-attr-stats-{i}"),
                &[
                    (t!("report-attr-n"), st.stats.n.to_string()),
                    (t!("report-attr-mean"), format!("{:.2}", st.stats.mean)),
                    (t!("report-attr-median"), format!("{:.2}", st.stats.median)),
                    (
                        "p10 / p90".to_string(),
                        format!("{:.2} / {:.2}", st.stats.p10, st.stats.p90),
                    ),
                    (
                        t!("measure-heterogeneity"),
                        opt(st.stats.heterogeneity_cv, |v| format!("{v:.3}")),
                    ),
                ],
            );
        }
        ui.separator();
    }
}

fn section_provenance(ui: &mut Ui, r: &Report, sess: &Session) -> Option<SlideId> {
    let p = &r.provenance;
    crate::ui_util::strong(ui, t!("provenance-source-fraction"));
    let rows: Vec<Vec<String>> = p
        .by_source
        .iter()
        .map(|(source, n)| {
            let name = match source.as_str() {
                "manual" => t!("segment-source-manual"),
                "classic" => t!("segment-source-classic"),
                "ai" => t!("segment-source-ai"),
                "interpolated" => t!("segment-source-interpolated"),
                "imported" => t!("segment-source-imported"),
                other => other.to_string(),
            };
            let frac = if p.total > 0 {
                100.0 * *n as f64 / p.total as f64
            } else {
                0.0
            };
            vec![
                name,
                n.to_string(),
                format!("{frac:.1} {}", t!("unit-percent")),
            ]
        })
        .collect();
    table(
        ui,
        "report-prov",
        &[
            t!("report-prov-source"),
            t!("report-prov-count"),
            t!("measure-fraction"),
        ],
        &rows,
    );
    ui.label(format!(
        "{}: {} / {}",
        t!("segment-review-reviewed"),
        p.reviewed,
        p.total
    ));
    if r.slides.order_suggested {
        ui.label(t!("report-order-suggested"));
    }
    ui.separator();
    crate::ui_util::strong(ui, t!("provenance-unreviewed"));
    let mut open = None;
    if p.drafts.is_empty() {
        ui.weak(t!("common-none"));
    }
    for (slide, label) in &p.drafts {
        ui.horizontal(|ui| {
            ui.label(icon_label_colored(
                h3d_ui::icons::DRAW,
                &t!("segment-review-draft"),
                h3d_ui::theme::WARNING,
            ));
            ui.label(format!(
                "{} / {label}",
                crate::ui_util::slide_title(sess, *slide)
            ));
            if ui.small_button(t!("recon-pick-open-slice")).clicked() {
                open = Some(*slide);
            }
        });
    }
    open
}

fn section_params(ui: &mut Ui, r: &Report) {
    key_values(
        ui,
        "report-params",
        &[
            (
                t!("report-app-version", version = r.app_version.clone()),
                String::new(),
            ),
            (t!("recon-s-xy"), format!("{}", r.recon.s_xy_um)),
            (t!("recon-smoothing"), r.recon.smooth_iterations.to_string()),
            (
                t!("recon-decimate"),
                format!("{}", r.recon.decimate_error_factor),
            ),
            (
                t!("slides-spacing"),
                format!("{}", r.recon.default_spacing_um),
            ),
            (
                t!("report-margin-close"),
                format!("{} {}", r.spatial.margin_close_um, t!("unit-um")),
            ),
            (
                t!("report-peri-band"),
                format!("{} {}", r.spatial.peri_band_um, t!("unit-um")),
            ),
            (
                t!("report-hypoxia-threshold"),
                format!("{} {}", r.spatial.hypoxia_um, t!("unit-um")),
            ),
            (
                t!("report-grid"),
                opt(r.grid.as_ref(), |g| {
                    format!(
                        "{} x {} x {}, s_xy {} {}, s_z {} {}",
                        g.size[0],
                        g.size[1],
                        g.nz,
                        g.s_xy_um,
                        t!("unit-um"),
                        g.s_z_um,
                        t!("unit-um")
                    )
                }),
            ),
            (
                t!("report-slides-summary"),
                t!(
                    "report-slides-summary-value",
                    total = r.slides.total.to_string(),
                    usable = r.slides.usable.to_string(),
                    registered = r.slides.registered.to_string(),
                    flagged = r.slides.flagged.len().to_string()
                ),
            ),
            (
                t!("measure-uncertainty"),
                format!(
                    "+/- {:.3} {}",
                    r.dimension_uncertainty_um / 1000.0,
                    t!("unit-mm")
                ),
            ),
        ],
    );
    ui.small(t!("report-uncertainty-note"));
    let notes = r.all_notes();
    if !notes.is_empty() {
        ui.separator();
        crate::ui_util::strong(ui, t!("slides-sample-notes"));
        for n in &notes {
            ui.label(n);
        }
    }
    ui.separator();
    crate::ui_util::strong(ui, t!("report-warnings"));
    let warnings = r.all_warnings();
    if warnings.is_empty() {
        ui.weak(t!("report-no-warnings"));
    }
    for w in &warnings {
        ui.label(warning_message(w));
    }
    ui.separator();
    h3d_ui::widgets::research_notice(ui);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn section_and_export_keys_exist() {
        for lang in [h3d_i18n::Language::Id, h3d_i18n::Language::En] {
            for k in SECTION_KEYS {
                assert!(h3d_i18n::has_key(lang, k), "{k}");
            }
            for c in ExportChoice::ALL {
                assert!(h3d_i18n::has_key(lang, c.i18n_key()), "{}", c.i18n_key());
            }
            for s in [
                MarginStatus::Involved,
                MarginStatus::Close,
                MarginStatus::Clear,
            ] {
                assert!(h3d_i18n::has_key(lang, margin_status_key(s)));
            }
        }
    }

    #[test]
    fn clinical_export_is_off_by_default_and_only_affects_report() {
        let st = ReportState::default();
        assert!(!st.with_clinical);
        assert!(matches!(
            ExportChoice::Report.kind(false, None),
            ExportKind::Report {
                with_clinical: false
            }
        ));
        assert!(matches!(
            ExportChoice::Report.kind(true, None),
            ExportKind::Report {
                with_clinical: true
            }
        ));
        assert!(matches!(
            ExportChoice::Stl.kind(true, None),
            ExportKind::Stl
        ));
        assert!(matches!(
            ExportChoice::Glb.kind(true, Some("margin_distance".into())),
            ExportKind::Glb { scalar: Some(_) }
        ));
    }

    #[test]
    fn distances_always_carry_uncertainty() {
        let _g = crate::test_util::lang_lock();
        h3d_i18n::set_language(h3d_i18n::Language::Id);
        let u = Uncertainty {
            value_um: 1500.0,
            plus_minus_um: 58.0,
            z_limited: false,
        };
        let s = fmt_unc(&u);
        assert!(
            s.contains("1.500") && s.contains("0.058") && s.contains("+/-"),
            "{s}"
        );
        let z = fmt_unc(&Uncertainty {
            z_limited: true,
            ..u
        });
        assert!(z.len() > s.len());
    }
}
