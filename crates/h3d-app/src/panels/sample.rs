//! Panel sampel (mode Irisan, 8.12.10): nama/catatan/susut, form klinis
//! pseudonim (`ClinicalMeta`), dan orientasi anatomis dengan gambar tiga
//! sumbu.
//!
//! Data klinis hanya ditulis lewat `Session::set_clinical` ke
//! `clinical.json` (aturan keras 17) dan tidak pernah menjadi masukan
//! perhitungan (aturan keras 18).

use egui::Ui;
use h3d_core::{
    AnatDir, ClinicalMeta, Grade, Laterality, MolecularResult, Procedure, Ptnm, Sex, SignedAxis,
    SpecimenOrientation, Therapy,
};
use h3d_engine::Session;
use h3d_i18n::t;

use crate::ui_util::{icon_text_button, opt_number, opt_text, Toasts};

/// Kunci terjemahan arah anatomis.
pub fn anat_key(dir: AnatDir) -> String {
    format!("anat-{}", dir.as_str().to_ascii_lowercase())
}

/// Nama arah anatomis dalam bahasa aktif.
pub fn anat_name(dir: AnatDir) -> String {
    t!(&anat_key(dir))
}

/// Teks arah untuk tabel laporan: sumbu mentah selalu tampil, nama anatomis
/// di sebelahnya bila diketahui, mis. `+X (Anterior)` (8.12.8).
pub fn direction_label(axis: SignedAxis, name: Option<AnatDir>) -> String {
    match name {
        Some(d) if d != AnatDir::Unknown => format!("{} ({})", axis.as_str(), anat_name(d)),
        _ => axis.as_str().to_string(),
    }
}

/// Combo pilihan dari daftar `(nilai, teks)`; `true` bila berubah.
pub fn combo<T: PartialEq + Clone>(
    ui: &mut Ui,
    id: &str,
    current: &mut T,
    options: &[(T, String)],
) -> bool {
    let selected = options
        .iter()
        .find(|(v, _)| v == current)
        .map(|(_, s)| s.clone())
        .unwrap_or_default();
    let mut changed = false;
    egui::ComboBox::from_id_salt(id)
        .selected_text(selected)
        .show_ui(ui, |ui| {
            for (value, text) in options {
                if ui
                    .selectable_label(value == current, text.as_str())
                    .clicked()
                    && value != current
                {
                    *current = value.clone();
                    changed = true;
                }
            }
        });
    changed
}

/// Bagian nama, catatan, faktor susut, dan jarak irisan.
pub fn sample_ui(ui: &mut Ui, session: &mut Session, toasts: &mut Toasts) {
    let sample = session.project().sample.clone();
    let mut name = sample.name.clone();
    let mut notes = sample.notes.clone();
    let mut changed = false;
    egui::Grid::new("sample-grid")
        .num_columns(2)
        .show(ui, |ui| {
            ui.label(t!("slides-sample-name"));
            changed |= ui
                .add(egui::TextEdit::singleline(&mut name).desired_width(170.0))
                .changed();
            ui.end_row();
            ui.label(t!("slides-sample-notes"));
            changed |= ui
                .add(
                    egui::TextEdit::multiline(&mut notes)
                        .desired_width(170.0)
                        .desired_rows(2),
                )
                .changed();
            ui.end_row();

            ui.label(t!("slides-shrinkage"));
            let mut shrink = sample.shrinkage_linear;
            if opt_number(ui, &mut shrink, 1.0, 0.1..=3.0, 0.005) {
                if let Err(e) = session.set_shrinkage(shrink) {
                    toasts.engine_error(&e);
                }
            }
            ui.end_row();

            ui.label(t!("slides-spacing"));
            let mut spacing = session.project().recon.default_spacing_um;
            if ui
                .add(
                    egui::DragValue::new(&mut spacing)
                        .range(0.1..=100_000.0)
                        .speed(0.5),
                )
                .changed()
            {
                if let Err(e) = session.set_spacing(spacing) {
                    toasts.engine_error(&e);
                }
            }
            ui.end_row();
        });
    if changed {
        session.set_sample_info(&name, &notes);
    }
    ui.small(t!("slides-shrinkage-hint"));
}

/// Keadaan form klinis: penyangga suntingan sampai "Simpan" ditekan.
#[derive(Clone, Debug, Default)]
pub struct ClinicalForm {
    /// Penyangga.
    pub buf: ClinicalMeta,
    /// Penyangga sudah diisi dari sesi.
    pub loaded: bool,
    /// Ada perubahan yang belum disimpan.
    pub dirty: bool,
}

fn opt_choice<T: PartialEq + Clone>(
    ui: &mut Ui,
    id: &str,
    current: &mut Option<T>,
    options: &[(T, String)],
) -> bool {
    let mut all: Vec<(Option<T>, String)> = vec![(None, t!("common-none"))];
    all.extend(options.iter().map(|(v, s)| (Some(v.clone()), s.clone())));
    combo(ui, id, current, &all)
}

impl ClinicalForm {
    /// Gambar form semua field `ClinicalMeta`. Petunjuk bahwa data bersifat
    /// pseudonim dan disimpan terpisah selalu tampil.
    pub fn ui(&mut self, ui: &mut Ui, session: &mut Session, toasts: &mut Toasts) {
        if !self.loaded {
            self.buf = session
                .project()
                .sample
                .clinical
                .clone()
                .unwrap_or_default();
            self.loaded = true;
            self.dirty = false;
        }
        ui.horizontal_wrapped(|ui| {
            ui.label(h3d_ui::icons::INFO.rich_text().color(h3d_ui::theme::ACCENT));
            ui.label(egui::RichText::new(t!("clinical-hint")).small());
        });
        let c = &mut self.buf;
        let mut ch = false;
        egui::Grid::new("clinical-grid")
            .num_columns(2)
            .show(ui, |ui| {
                ui.label(t!("clinical-subject-pseudo-id"));
                ch |= ui
                    .add(egui::TextEdit::singleline(&mut c.subject_pseudo_id).desired_width(150.0))
                    .changed();
                ui.end_row();

                ui.label(t!("clinical-age-years"));
                let mut age = c.age_years.map(f64::from);
                if opt_number(ui, &mut age, 50.0, 0.0..=130.0, 1.0) {
                    c.age_years = age.map(|a| a.round().clamp(0.0, 255.0) as u8);
                    ch = true;
                }
                ui.end_row();

                ui.label(t!("clinical-sex"));
                ch |= opt_choice(
                    ui,
                    "clin-sex",
                    &mut c.sex,
                    &[
                        (Sex::Female, t!("clinical-sex-female")),
                        (Sex::Male, t!("clinical-sex-male")),
                        (Sex::Other, t!("clinical-sex-other")),
                        (Sex::Unknown, t!("clinical-sex-unknown")),
                    ],
                );
                ui.end_row();

                ui.label(t!("clinical-topography-icdo"));
                ch |= opt_text(ui, &mut c.topography_icdo, 150.0);
                ui.end_row();
                ui.label(t!("clinical-morphology-icdo"));
                ch |= opt_text(ui, &mut c.morphology_icdo, 150.0);
                ui.end_row();

                ui.label(t!("clinical-laterality"));
                ch |= opt_choice(
                    ui,
                    "clin-lat",
                    &mut c.laterality,
                    &[
                        (Laterality::Left, t!("clinical-laterality-left")),
                        (Laterality::Right, t!("clinical-laterality-right")),
                        (Laterality::Bilateral, t!("clinical-laterality-bilateral")),
                        (
                            Laterality::NotApplicable,
                            t!("clinical-laterality-not-applicable"),
                        ),
                    ],
                );
                ui.end_row();

                ui.label(t!("clinical-procedure"));
                ui.vertical(|ui| {
                    // `Other` membawa teks bebas; combo memilih jenisnya saja.
                    let mut kind: Option<u8> = c.procedure.as_ref().map(|p| match p {
                        Procedure::Biopsy => 0,
                        Procedure::Excision => 1,
                        Procedure::Resection => 2,
                        Procedure::Other(_) => 3,
                    });
                    if opt_choice(
                        ui,
                        "clin-proc",
                        &mut kind,
                        &[
                            (0u8, t!("clinical-procedure-biopsy")),
                            (1u8, t!("clinical-procedure-excision")),
                            (2u8, t!("clinical-procedure-resection")),
                            (3u8, t!("clinical-procedure-other")),
                        ],
                    ) {
                        c.procedure = kind.map(|k| match k {
                            0 => Procedure::Biopsy,
                            1 => Procedure::Excision,
                            2 => Procedure::Resection,
                            _ => Procedure::Other(String::new()),
                        });
                        ch = true;
                    }
                    if let Some(Procedure::Other(text)) = &mut c.procedure {
                        ch |= ui
                            .add(egui::TextEdit::singleline(text).desired_width(150.0))
                            .changed();
                    }
                });
                ui.end_row();

                ui.label(t!("clinical-specimen-month"));
                ch |= opt_text(ui, &mut c.specimen_month, 90.0);
                ui.end_row();
            });
        if !c.month_format_ok() {
            ui.colored_label(h3d_ui::theme::ERROR, t!("clinical-month-invalid"));
        }

        // Grade.
        let mut has = c.grade.is_some();
        if ui.checkbox(&mut has, t!("clinical-grade")).changed() {
            c.grade = has.then(Grade::default);
            ch = true;
        }
        if let Some(g) = &mut c.grade {
            ui.horizontal(|ui| {
                ui.label(t!("clinical-grade-system"));
                ch |= ui
                    .add(egui::TextEdit::singleline(&mut g.system).desired_width(90.0))
                    .changed();
                ui.label(t!("clinical-grade-value"));
                ch |= ui
                    .add(egui::TextEdit::singleline(&mut g.value).desired_width(50.0))
                    .changed();
            });
        }

        // pTNM.
        let mut has = c.ptnm.is_some();
        if ui.checkbox(&mut has, t!("clinical-ptnm")).changed() {
            c.ptnm = has.then(|| Ptnm {
                edition: 8,
                prefix: "p".to_string(),
                ..Default::default()
            });
            ch = true;
        }
        if let Some(p) = &mut c.ptnm {
            ui.horizontal_wrapped(|ui| {
                for (key, field) in [
                    ("clinical-ptnm-prefix", &mut p.prefix),
                    ("clinical-ptnm-t", &mut p.t),
                    ("clinical-ptnm-n", &mut p.n),
                    ("clinical-ptnm-m", &mut p.m),
                ] {
                    ui.label(t!(key));
                    ch |= ui
                        .add(egui::TextEdit::singleline(field).desired_width(36.0))
                        .changed();
                }
                ui.label(t!("clinical-ptnm-edition"));
                ch |= ui
                    .add(egui::DragValue::new(&mut p.edition).range(1..=20))
                    .changed();
            });
        }

        // Neoadjuvan.
        let mut has = c.neoadjuvant.is_some();
        if ui.checkbox(&mut has, t!("clinical-neoadjuvant")).changed() {
            c.neoadjuvant = has.then(Therapy::default);
            ch = true;
        }
        if let Some(n) = &mut c.neoadjuvant {
            ui.horizontal(|ui| {
                ch |= ui
                    .checkbox(&mut n.given, t!("clinical-neoadjuvant-given"))
                    .changed();
                ui.label(t!("clinical-neoadjuvant-regimen"));
                ch |= opt_text(ui, &mut n.regimen, 110.0);
            });
        }

        // Hasil molekuler.
        ui.label(t!("clinical-molecular"));
        let mut remove = None;
        for (i, m) in c.molecular.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                ch |= ui
                    .add(
                        egui::TextEdit::singleline(&mut m.gene)
                            .hint_text(t!("clinical-molecular-gene"))
                            .desired_width(56.0),
                    )
                    .changed();
                ch |= ui
                    .add(
                        egui::TextEdit::singleline(&mut m.result)
                            .hint_text(t!("clinical-molecular-result"))
                            .desired_width(70.0),
                    )
                    .changed();
                let mut method = m.method.clone().unwrap_or_default();
                if ui
                    .add(
                        egui::TextEdit::singleline(&mut method)
                            .hint_text(t!("clinical-molecular-method"))
                            .desired_width(56.0),
                    )
                    .changed()
                {
                    m.method = (!method.trim().is_empty()).then_some(method);
                    ch = true;
                }
                if h3d_ui::widgets::icon_button(ui, h3d_ui::icons::DELETE, &t!("action-delete"))
                    .clicked()
                {
                    remove = Some(i);
                }
            });
        }
        if let Some(i) = remove {
            c.molecular.remove(i);
            ch = true;
        }
        if icon_text_button(ui, h3d_ui::icons::ADD, &t!("action-add")).clicked() {
            c.molecular.push(MolecularResult::default());
            ch = true;
        }
        self.dirty |= ch;

        ui.separator();
        ui.horizontal(|ui| {
            let can_save = self.dirty && self.buf.month_format_ok();
            if ui
                .add_enabled(
                    can_save,
                    egui::Button::new(crate::ui_util::icon_label(
                        h3d_ui::icons::SAVE,
                        &t!("action-save"),
                    )),
                )
                .clicked()
            {
                match session.set_clinical(Some(self.buf.clone())) {
                    Ok(()) => {
                        self.dirty = false;
                        toasts.info(t!("clinical-saved"));
                    }
                    Err(e) => toasts.engine_error(&e),
                }
            }
            if icon_text_button(ui, h3d_ui::icons::DELETE, &t!("clinical-strip")).clicked() {
                match session.set_clinical(None) {
                    Ok(()) => {
                        self.buf = ClinicalMeta::default();
                        self.dirty = false;
                    }
                    Err(e) => toasts.engine_error(&e),
                }
            }
        });
    }
}

/// Keadaan panel orientasi anatomis.
#[derive(Clone, Debug, Default)]
pub struct OrientationForm {
    /// Penyangga.
    pub buf: SpecimenOrientation,
    /// Penyangga sudah diisi dari sesi.
    pub loaded: bool,
    /// Pesan galat validasi terakhir (sudah diterjemahkan).
    pub error: Option<String>,
}

impl OrientationForm {
    /// Terapkan penyangga ke sesi. Orientasi tidak sah (dua sumbu dari
    /// pasangan anatomis yang sama) ditolak dan pesannya ditampilkan.
    pub fn apply(&mut self, session: &mut Session) -> bool {
        let o = self.buf;
        let all_unknown = [o.pos_x, o.pos_y, o.pos_z]
            .iter()
            .all(|d| *d == AnatDir::Unknown);
        let value = (!all_unknown).then_some(o);
        match session.set_orientation(value) {
            Ok(()) => {
                self.error = None;
                true
            }
            Err(e) => {
                self.error = Some(t!("anat-invalid", message = e.to_string()));
                false
            }
        }
    }

    /// Gambar tiga combo arah, gambar tiga sumbu, dan galat validasi.
    pub fn ui(&mut self, ui: &mut Ui, session: &mut Session) {
        if !self.loaded {
            self.buf = session.project().sample.orientation.unwrap_or_default();
            self.loaded = true;
            self.error = None;
        }
        let options: Vec<(AnatDir, String)> =
            AnatDir::ALL.iter().map(|d| (*d, anat_name(*d))).collect();
        let mut changed = false;
        egui::Grid::new("anat-grid").num_columns(2).show(ui, |ui| {
            ui.label(t!("anat-pos-x"));
            changed |= combo(ui, "anat-x", &mut self.buf.pos_x, &options);
            ui.end_row();
            ui.label(t!("anat-pos-y"));
            changed |= combo(ui, "anat-y", &mut self.buf.pos_y, &options);
            ui.end_row();
            ui.label(t!("anat-pos-z"));
            changed |= combo(ui, "anat-z", &mut self.buf.pos_z, &options);
            ui.end_row();
        });
        if changed {
            self.apply(session);
        }
        if let Some(err) = &self.error {
            ui.colored_label(h3d_ui::theme::ERROR, err);
        }
        axes_drawing(ui, &self.buf);
    }
}

/// Gambar tiga sumbu ruang W dengan nama anatomisnya: +X ke kanan, +Y ke
/// bawah (sumbu gambar), +Z arah tumpukan (digambar miring keluar layar).
pub fn axes_drawing(ui: &mut Ui, o: &SpecimenOrientation) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(230.0, 150.0), egui::Sense::hover());
    if !ui.is_rect_visible(rect) {
        return;
    }
    let painter = ui.painter_at(rect);
    painter.rect_stroke(
        rect,
        4.0,
        egui::Stroke::new(1.0, ui.visuals().widgets.noninteractive.bg_stroke.color),
        egui::StrokeKind::Inside,
    );
    let origin = rect.center() + egui::vec2(-20.0, -8.0);
    let text_color = ui.visuals().text_color();
    let font = egui::FontId::proportional(11.0);
    let axes: [(egui::Vec2, &str, AnatDir, egui::Color32); 3] = [
        (
            egui::vec2(62.0, 0.0),
            "+X",
            o.pos_x,
            egui::Color32::from_rgb(230, 80, 80),
        ),
        (
            egui::vec2(0.0, 50.0),
            "+Y",
            o.pos_y,
            egui::Color32::from_rgb(80, 190, 90),
        ),
        (
            egui::vec2(-42.0, -40.0),
            "+Z",
            o.pos_z,
            egui::Color32::from_rgb(90, 140, 240),
        ),
    ];
    for (vec, raw, dir, color) in axes {
        painter.arrow(origin, vec, egui::Stroke::new(2.0, color));
        // Arah negatif: garis tipis ke arah lawan dengan nama lawannya.
        painter.line_segment(
            [origin, origin - vec * 0.45],
            egui::Stroke::new(1.0, color.gamma_multiply(0.5)),
        );
        let label = if dir == AnatDir::Unknown {
            raw.to_string()
        } else {
            format!("{raw} {}", anat_name(dir))
        };
        let tip = origin + vec;
        let align = if vec.x > 1.0 {
            egui::Align2::LEFT_CENTER
        } else if vec.y > 1.0 {
            egui::Align2::CENTER_TOP
        } else {
            egui::Align2::CENTER_BOTTOM
        };
        let offset = if vec.x > 1.0 {
            egui::vec2(4.0, 0.0)
        } else if vec.y > 1.0 {
            egui::vec2(0.0, 3.0)
        } else {
            egui::vec2(0.0, -3.0)
        };
        painter.text(tip + offset, align, label, font.clone(), text_color);
    }
    painter.circle_filled(origin, 3.0, text_color);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_anatomical_direction_has_a_translation() {
        for lang in [h3d_i18n::Language::Id, h3d_i18n::Language::En] {
            for d in AnatDir::ALL {
                assert!(
                    h3d_i18n::has_key(lang, &anat_key(d)),
                    "kunci {} hilang",
                    anat_key(d)
                );
            }
        }
    }

    #[test]
    fn direction_label_always_shows_raw_axis() {
        let _g = crate::test_util::lang_lock();
        h3d_i18n::set_language(h3d_i18n::Language::Id);
        assert_eq!(direction_label(SignedAxis::PosX, None), "+X");
        assert_eq!(
            direction_label(SignedAxis::NegZ, Some(AnatDir::Unknown)),
            SignedAxis::NegZ.as_str()
        );
        let s = direction_label(SignedAxis::PosX, Some(AnatDir::Anterior));
        assert!(s.starts_with("+X (") && s.contains("Anterior"));
    }
}
