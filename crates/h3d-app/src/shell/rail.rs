//! Rel alat kiri: isinya mengikuti tampilan.
//!
//! - Tumpukan 3D: Susun otomatis, Sejajarkan semua, Bangun 3D, Analisis
//!   spasial | Pas ke layar, Irisan di 3D, Mesh di 3D, Bidang potong.
//!   (Dari mode Irisan, Registrasi, dan Rekonstruksi.)
//! - Irisan 2D, Tandai: Geser, Kuas, Penghapus, Laso, Poligon | Isi lubang,
//!   Salin ke irisan berikut, Tandai ditinjau. (Dari mode Segmentasi.)
//! - Irisan 2D, Sejajarkan: Geser, Landmark, Geser irisan, Putar irisan |
//!   Tandai rusak. (Dari mode Registrasi.)
//!
//! Pemilih mode Tandai/Sejajarkan ada di paling atas rel saat 2D.

use egui::{Rect, Ui};
use h3d_core::SlideStatus;
use h3d_engine::Session;
use h3d_i18n::t;
use h3d_ui::RailItem;

use crate::features::{register, segment};
use crate::state::{AppState, JobOutput, SliceMode, View};

/// Aksi rel alat.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RailAction {
    AutoStack,
    RegisterAll,
    Reconstruct,
    Analyze,
    FitView,
    ToggleSlices,
    ToggleMeshes,
    ToggleClip,
    Mode(SliceMode),
    Annotate(segment::Tool),
    Align(register::RegTool),
    FillHoles,
    CopyToNext,
    MarkReviewed,
    MarkDamaged,
}

/// Gambar rel alat di sisi kiri `work` dan jalankan aksinya.
pub fn show(app: &mut AppState, sess: Option<&mut Session>, ui: &mut Ui, work: Rect) {
    let busy = app.job.is_some();
    let items = match app.view {
        View::Stack3D => stack_items(app, busy),
        View::Slice2D => slice_items(app),
    };
    let Some(action) = h3d_ui::show_rail(ui, work, &items) else {
        return;
    };
    match action {
        RailAction::AutoStack => app.start_auto_stack(false),
        RailAction::RegisterAll => app.start_register_all(),
        RailAction::Reconstruct => {
            app.start_job(t!("recon-run"), |s, p| {
                s.reconstruct(p).map(|m| JobOutput::Reconstructed(m.len()))
            });
        }
        RailAction::Analyze => {
            app.start_job(t!("recon-analyze"), |s, p| {
                s.analyze_spatial(p).map(|_| JobOutput::Analyzed)
            });
        }
        RailAction::FitView => app.recon.fitted = false,
        RailAction::ToggleSlices => {
            app.recon.show_slices = !app.recon.show_slices;
            app.recon.fitted = false;
        }
        RailAction::ToggleMeshes => {
            app.recon.show_meshes = !app.recon.show_meshes;
            app.recon.fitted = false;
        }
        RailAction::ToggleClip => app.recon.clip_enabled = !app.recon.clip_enabled,
        RailAction::Mode(m) => {
            app.slice_mode = m;
            app.segment.cancel_drawing();
            app.register.reset_edit();
        }
        RailAction::Annotate(tool) => segment::set_tool(app, tool),
        RailAction::Align(tool) => register::set_tool(app, tool),
        RailAction::FillHoles => {
            if let (Some(sess), Some(slide), Some(label)) =
                (sess, app.current, app.segment.active_label)
            {
                match sess.fill_label_holes(slide, label) {
                    Ok(_) => segment::after_edit(app, sess, slide),
                    Err(e) => app.toasts.engine_error(&e),
                }
            }
        }
        RailAction::CopyToNext => {
            if let (Some(sess), Some(slide)) = (sess, app.current) {
                match sess.copy_labels_to_next(slide) {
                    Ok(dst) => {
                        segment::after_edit(app, sess, dst);
                        app.current = Some(dst);
                    }
                    Err(e) => app.toasts.engine_error(&e),
                }
            }
        }
        RailAction::MarkReviewed => {
            if let (Some(sess), Some(slide), Some(label)) =
                (sess, app.current, app.segment.active_label)
            {
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
        RailAction::MarkDamaged => {
            if let (Some(sess), Some(slide)) = (sess, app.current) {
                let status = sess.slide(slide).map(|s| s.status).ok();
                let target = if status == Some(SlideStatus::Damaged) {
                    SlideStatus::Ok
                } else {
                    SlideStatus::Damaged
                };
                if let Err(e) = sess.set_status(slide, target) {
                    app.toasts.engine_error(&e);
                }
                app.invalidate_derived();
            }
        }
    }
}

fn stack_items(app: &AppState, busy: bool) -> Vec<RailItem<RailAction>> {
    let n = app.recon.slices.len();
    let st = &app.recon;
    vec![
        RailItem::tool(
            RailAction::AutoStack,
            h3d_ui::icons::AUTO_STACK,
            t!("stack-auto"),
        )
        .hint(t!("stack-auto-hint"))
        .enabled(!busy && n >= crate::state::AUTO_STACK_MIN_SLIDES),
        RailItem::tool(
            RailAction::RegisterAll,
            h3d_ui::icons::COMPARE,
            t!("register-all"),
        )
        .hint(t!("stack-align-hint"))
        .enabled(!busy && n >= 2),
        RailItem::tool(
            RailAction::Reconstruct,
            h3d_ui::icons::CUBE,
            t!("recon-run"),
        )
        .enabled(!busy && n >= 1),
        RailItem::tool(
            RailAction::Analyze,
            h3d_ui::icons::ANALYZE,
            t!("recon-analyze"),
        )
        .enabled(!busy && !st.meshes.is_empty()),
        RailItem::Separator,
        RailItem::tool(
            RailAction::FitView,
            h3d_ui::icons::FIT_SCREEN,
            t!("stack-fit"),
        ),
        RailItem::tool(
            RailAction::ToggleSlices,
            h3d_ui::icons::IMAGE,
            t!("stack-slices-visible"),
        )
        .selected(st.show_slices),
        RailItem::tool(
            RailAction::ToggleMeshes,
            h3d_ui::icons::LAYERS,
            t!("stack-mesh-visible"),
        )
        .selected(st.show_meshes)
        .enabled(!st.meshes.is_empty()),
        RailItem::tool(
            RailAction::ToggleClip,
            h3d_ui::icons::CLIP,
            t!("recon-clipping-plane"),
        )
        .selected(st.clip_enabled),
    ]
}

fn slice_items(app: &AppState) -> Vec<RailItem<RailAction>> {
    let mut items = vec![
        RailItem::tool(
            RailAction::Mode(SliceMode::Annotate),
            h3d_ui::icons::DRAW,
            t!("slice-mode-annotate"),
        )
        .hint(t!("slice-mode-annotate-hint"))
        .selected(app.slice_mode == SliceMode::Annotate),
        RailItem::tool(
            RailAction::Mode(SliceMode::Align),
            h3d_ui::icons::COMPARE,
            t!("slice-mode-align"),
        )
        .hint(t!("slice-mode-align-hint"))
        .selected(app.slice_mode == SliceMode::Align),
        RailItem::Separator,
    ];
    match app.slice_mode {
        SliceMode::Annotate => {
            for tool in segment::Tool::ALL {
                items.push(
                    RailItem::tool(RailAction::Annotate(tool), tool.icon(), t!(tool.i18n_key()))
                        .selected(app.segment.tool == tool),
                );
            }
            let has_label = app.segment.active_label.is_some();
            items.push(RailItem::Separator);
            items.push(
                RailItem::tool(
                    RailAction::FillHoles,
                    h3d_ui::icons::FILL,
                    t!("segment-fill-holes"),
                )
                .enabled(has_label),
            );
            items.push(RailItem::tool(
                RailAction::CopyToNext,
                h3d_ui::icons::COPY,
                t!("segment-copy-to-next"),
            ));
            items.push(
                RailItem::tool(
                    RailAction::MarkReviewed,
                    h3d_ui::icons::REVIEWED,
                    t!("segment-mark-reviewed"),
                )
                .enabled(has_label),
            );
        }
        SliceMode::Align => {
            for tool in register::RegTool::ALL {
                items.push(
                    RailItem::tool(RailAction::Align(tool), tool.icon(), t!(tool.i18n_key()))
                        .selected(app.register.tool == tool),
                );
            }
            items.push(RailItem::Separator);
            items.push(RailItem::tool(
                RailAction::MarkDamaged,
                h3d_ui::icons::DAMAGED,
                t!("register-mark-damaged"),
            ));
        }
    }
    items
}
