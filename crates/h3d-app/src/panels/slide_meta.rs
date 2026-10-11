//! Panel per irisan (mode Irisan): mpp, posisi Z, status, acuan,
//! pewarnaan, panel IHC, dan pra-analitik (8.12.3, 8.12.10).

use egui::Ui;
use h3d_core::{
    Chromogen, IhcPanel, IhcScoring, OrderSource, SlideId, SlideStatus, Stain, StainPattern,
};
use h3d_engine::Session;
use h3d_i18n::t;

use crate::panels::sample::combo;
use crate::ui_util::{icon_selectable, icon_text_button, opt_number, opt_text, Toasts};

/// Kunci terjemahan status irisan.
pub fn status_key(status: SlideStatus) -> &'static str {
    match status {
        SlideStatus::Ok => "slides-status-ok",
        SlideStatus::Excluded => "slides-status-excluded",
        SlideStatus::Damaged => "slides-status-damaged",
    }
}

/// Kunci terjemahan asal urutan.
pub fn order_source_key(source: OrderSource) -> &'static str {
    match source {
        OrderSource::FileName => "slides-order-source-filename",
        OrderSource::Manual => "slides-order-source-manual",
        OrderSource::Suggested => "slides-order-source-suggested",
    }
}

/// Kunci terjemahan skema skor IHC.
pub fn scoring_key(s: IhcScoring) -> &'static str {
    match s {
        IhcScoring::PercentPositive => "ihc-scoring-percent-positive",
        IhcScoring::HScore => "ihc-scoring-h-score",
        IhcScoring::Allred => "ihc-scoring-allred",
        IhcScoring::Tps => "ihc-scoring-tps",
        IhcScoring::Cps => "ihc-scoring-cps",
        IhcScoring::Intensity03 => "ihc-scoring-intensity03",
    }
}

/// Semua skema skor, urutan tampil.
pub const ALL_SCORINGS: [IhcScoring; 6] = [
    IhcScoring::PercentPositive,
    IhcScoring::HScore,
    IhcScoring::Allred,
    IhcScoring::Tps,
    IhcScoring::Cps,
    IhcScoring::Intensity03,
];

/// Penyangga isian mpp (diterapkan dengan tombol, karena mengganti mpp
/// membuang registrasi irisan itu).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SlideMetaState {
    /// Irisan yang penyangganya sedang dimuat.
    pub loaded_for: Option<SlideId>,
    /// mpp isian.
    pub mpp: [f64; 2],
}

/// Hasil panel: irisan diminta dihapus dari proyek.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlideMetaAction {
    /// Tidak ada.
    None,
    /// mpp irisan diganti (sumber tile harus dibuka ulang).
    MppChanged,
    /// Irisan dihapus dari proyek.
    Removed,
}

/// Gambar panel untuk irisan `id`.
pub fn slide_meta_ui(
    ui: &mut Ui,
    session: &mut Session,
    toasts: &mut Toasts,
    state: &mut SlideMetaState,
    id: SlideId,
) -> SlideMetaAction {
    let Ok(entry) = session.slide(id).cloned() else {
        return SlideMetaAction::None;
    };
    if state.loaded_for != Some(id) {
        state.loaded_for = Some(id);
        state.mpp = entry.mpp;
    }
    let mut action = SlideMetaAction::None;
    let is_reference = session.project().effective_reference() == Some(id);
    let size_um = entry.size_um();

    crate::ui_util::strong(ui, entry.source.file_name());
    egui::Grid::new("slide-meta-grid")
        .num_columns(2)
        .show(ui, |ui| {
            ui.label(t!("slides-order"));
            ui.label(format!(
                "{} ({})",
                entry.order + 1,
                t!(order_source_key(entry.order_source))
            ));
            ui.end_row();
            ui.label(t!("slides-size-px"));
            ui.label(format!("{} x {}", entry.size_l0[0], entry.size_l0[1]));
            ui.end_row();
            ui.label(t!("slides-size-um"));
            ui.label(format!("{:.0} x {:.0}", size_um[0], size_um[1]));
            ui.end_row();

            ui.label(t!("slides-mpp-x"));
            ui.add(
                egui::DragValue::new(&mut state.mpp[0])
                    .range(0.001..=1000.0)
                    .speed(0.01)
                    .max_decimals(4),
            );
            ui.end_row();
            ui.label(t!("slides-mpp-y"));
            ui.add(
                egui::DragValue::new(&mut state.mpp[1])
                    .range(0.001..=1000.0)
                    .speed(0.01)
                    .max_decimals(4),
            );
            ui.end_row();

            ui.label(t!("slides-z-position"));
            let mut z = entry.z_um;
            if ui
                .add(egui::DragValue::new(&mut z).speed(1.0).max_decimals(1))
                .changed()
            {
                if let Err(e) = session.set_z(id, z) {
                    toasts.engine_error(&e);
                }
            }
            ui.end_row();
        });
    if state.mpp != entry.mpp {
        ui.horizontal(|ui| {
            if icon_text_button(ui, h3d_ui::icons::CHECK, &t!("slides-mpp-apply")).clicked() {
                match session.set_mpp(id, state.mpp) {
                    Ok(()) => action = SlideMetaAction::MppChanged,
                    Err(e) => {
                        toasts.engine_error(&e);
                        state.mpp = entry.mpp;
                    }
                }
            }
            if ui.button(t!("action-reset")).clicked() {
                state.mpp = entry.mpp;
            }
        });
        ui.small(t!("slides-mpp-change-hint"));
    }
    if entry.z_is_default {
        ui.small(t!("slides-z-default"));
    }

    ui.separator();
    ui.horizontal_wrapped(|ui| {
        for (status, icon) in [
            (SlideStatus::Ok, h3d_ui::icons::CHECK),
            (SlideStatus::Excluded, h3d_ui::icons::EXCLUDE),
            (SlideStatus::Damaged, h3d_ui::icons::DAMAGED),
        ] {
            if icon_selectable(ui, entry.status == status, icon, &t!(status_key(status))).clicked()
                && entry.status != status
            {
                if let Err(e) = session.set_status(id, status) {
                    toasts.engine_error(&e);
                }
            }
        }
    });
    ui.horizontal(|ui| {
        if is_reference {
            ui.label(crate::ui_util::icon_label(
                h3d_ui::icons::REFERENCE,
                &t!("slides-is-reference"),
            ));
        } else if ui
            .add_enabled(
                entry.status == SlideStatus::Ok,
                egui::Button::new(crate::ui_util::icon_label(
                    h3d_ui::icons::REFERENCE,
                    &t!("slides-set-reference"),
                )),
            )
            .clicked()
        {
            if let Err(e) = session.set_reference(Some(id)) {
                toasts.engine_error(&e);
            }
        }
    });

    ui.separator();
    ui.collapsing(t!("slides-stain"), |ui| stain_ui(ui, session, toasts, id));
    ui.collapsing(t!("preanalytic-panel"), |ui| {
        preanalytic_ui(ui, session, toasts, id)
    });

    ui.separator();
    if icon_text_button(ui, h3d_ui::icons::DELETE, &t!("slides-remove")).clicked() {
        match session.remove_slide(id) {
            Ok(()) => action = SlideMetaAction::Removed,
            Err(e) => toasts.engine_error(&e),
        }
    }
    action
}

fn stain_ui(ui: &mut Ui, session: &mut Session, toasts: &mut Toasts, id: SlideId) {
    let Ok(entry) = session.slide(id).cloned() else {
        return;
    };
    let mut stain = entry.stain;
    let mut ihc = entry.ihc.clone();
    let mut changed = combo(
        ui,
        "slide-stain",
        &mut stain,
        &[
            (Stain::He, t!("slides-stain-he")),
            (Stain::Ihc, t!("slides-stain-ihc")),
            (Stain::Other, t!("slides-stain-other")),
        ],
    );
    if stain == Stain::Ihc {
        let panel = ihc.get_or_insert_with(IhcPanel::default);
        crate::ui_util::strong(ui, t!("ihc-panel"));
        if panel.antibody.trim().is_empty() {
            ui.colored_label(h3d_ui::theme::WARNING, t!("ihc-required"));
        }
        egui::Grid::new("ihc-grid").num_columns(2).show(ui, |ui| {
            ui.label(t!("ihc-antibody"));
            changed |= ui
                .add(egui::TextEdit::singleline(&mut panel.antibody).desired_width(120.0))
                .changed();
            ui.end_row();
            ui.label(t!("ihc-clone"));
            changed |= opt_text(ui, &mut panel.clone, 120.0);
            ui.end_row();
            ui.label(t!("ihc-chromogen"));
            changed |= combo(
                ui,
                "ihc-chromogen",
                &mut panel.chromogen,
                &[
                    (Chromogen::Dab, t!("ihc-chromogen-dab")),
                    (Chromogen::Aec, t!("ihc-chromogen-aec")),
                    (Chromogen::FastRed, t!("ihc-chromogen-fast-red")),
                    (Chromogen::Other, t!("ihc-chromogen-other")),
                ],
            );
            ui.end_row();
            ui.label(t!("ihc-pattern"));
            changed |= combo(
                ui,
                "ihc-pattern",
                &mut panel.pattern,
                &[
                    (StainPattern::Nuclear, t!("ihc-pattern-nuclear")),
                    (StainPattern::Membranous, t!("ihc-pattern-membranous")),
                    (StainPattern::Cytoplasmic, t!("ihc-pattern-cytoplasmic")),
                ],
            );
            ui.end_row();
            ui.label(t!("ihc-scoring"));
            let options: Vec<(IhcScoring, String)> = ALL_SCORINGS
                .iter()
                .map(|s| (*s, t!(scoring_key(*s))))
                .collect();
            changed |= combo(ui, "ihc-scoring", &mut panel.scoring, &options);
            ui.end_row();
            ui.label(t!("ihc-od-vectors"));
            ui.label(if panel.od_vectors.is_some() {
                t!("ihc-od-calibrated")
            } else {
                t!("ihc-od-default")
            });
            ui.end_row();
        });
    }
    if changed {
        let ihc = if stain == Stain::Ihc { ihc } else { None };
        if let Err(e) = session.set_stain(id, stain, ihc) {
            toasts.engine_error(&e);
        }
    }
}

fn preanalytic_ui(ui: &mut Ui, session: &mut Session, toasts: &mut Toasts, id: SlideId) {
    let Ok(entry) = session.slide(id).cloned() else {
        return;
    };
    let mut p = entry.preanalytic.clone();
    let mut changed = false;
    egui::Grid::new("preanalytic-grid")
        .num_columns(2)
        .show(ui, |ui| {
            ui.label(t!("preanalytic-block-id"));
            changed |= opt_text(ui, &mut p.block_id, 110.0);
            ui.end_row();
            ui.label(t!("preanalytic-section-thickness-um"));
            changed |= opt_number(ui, &mut p.section_thickness_um, 4.0, 0.1..=1000.0, 0.1);
            ui.end_row();
            ui.label(t!("preanalytic-fixation-hours"));
            changed |= opt_number(ui, &mut p.fixation_hours, 24.0, 0.0..=10_000.0, 1.0);
            ui.end_row();
            ui.label(t!("preanalytic-fixative"));
            changed |= opt_text(ui, &mut p.fixative, 110.0);
            ui.end_row();
            ui.label(t!("preanalytic-stained-month"));
            changed |= opt_text(ui, &mut p.stained_month, 80.0);
            ui.end_row();
            ui.label(t!("preanalytic-scanner"));
            changed |= opt_text(ui, &mut p.scanner, 110.0);
            ui.end_row();
            ui.label(t!("preanalytic-objective"));
            changed |= opt_text(ui, &mut p.objective, 60.0);
            ui.end_row();
        });
    if changed {
        if let Err(e) = session.set_preanalytic(id, p) {
            toasts.engine_error(&e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enum_keys_exist_in_both_locales() {
        let mut keys: Vec<&str> = Vec::new();
        keys.extend([SlideStatus::Ok, SlideStatus::Excluded, SlideStatus::Damaged].map(status_key));
        keys.extend(
            [
                OrderSource::FileName,
                OrderSource::Manual,
                OrderSource::Suggested,
            ]
            .map(order_source_key),
        );
        keys.extend(ALL_SCORINGS.map(scoring_key));
        for lang in [h3d_i18n::Language::Id, h3d_i18n::Language::En] {
            for k in &keys {
                assert!(h3d_i18n::has_key(lang, k), "kunci {k} hilang");
            }
        }
    }
}
