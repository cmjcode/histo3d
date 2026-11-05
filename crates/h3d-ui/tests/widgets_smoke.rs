//! Uji asap: widget h3d-ui dirender di dalam `egui::Context::run` tanpa panik,
//! dalam kedua bahasa dan kedua tema.

use h3d_core::Colormap;
use h3d_ui::widgets;
use h3d_ui::{show_rail, RailItem, ThemeMode, ViewCube};

#[derive(Clone, Copy, Debug, PartialEq)]
enum Tool {
    Pan,
    Brush,
}

#[test]
fn widgets_render_without_panic() {
    let ctx = egui::Context::default();
    for (lang, theme) in [
        (h3d_i18n::Language::Id, ThemeMode::Dark),
        (h3d_i18n::Language::En, ThemeMode::Light),
    ] {
        h3d_i18n::set_language(lang);
        h3d_ui::theme::apply(&ctx, theme);

        let mut on = false;
        let mut spacing = 50.0_f64;
        let rail = vec![
            RailItem::tool(Tool::Pan, h3d_ui::icons::PAN, h3d_i18n::t!("tool-pan")),
            RailItem::Separator,
            RailItem::tool(
                Tool::Brush,
                h3d_ui::icons::BRUSH,
                h3d_i18n::t!("segment-tool-brush"),
            )
            .selected(true),
        ];

        let mut out = ctx.run_ui(Default::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let area = ui.max_rect();
                assert_eq!(show_rail(ui, area, &rail), None);
                widgets::research_notice(ui);
                widgets::warning_banner(ui, &h3d_i18n::t!("warn-has-draft"));
                widgets::icon_button(ui, h3d_ui::icons::SAVE, &h3d_i18n::t!("action-save"));
                widgets::tool_button(
                    ui,
                    h3d_ui::icons::SELECT,
                    false,
                    &h3d_i18n::t!("tool-select"),
                    None,
                );
                widgets::drawer_header(ui, h3d_ui::icons::LAYERS, &h3d_i18n::t!("drawer-layers"));
                widgets::toggle_icon_button(
                    ui,
                    &mut on,
                    h3d_ui::icons::VISIBILITY_ON,
                    &h3d_i18n::t!("label-visible"),
                );
                widgets::labeled_slider_f64(
                    ui,
                    &h3d_i18n::t!("slides-spacing"),
                    &mut spacing,
                    1.0..=500.0,
                    &h3d_i18n::t!("unit-um"),
                );
                for cm in Colormap::ALL {
                    widgets::colormap_legend(ui, cm, 0.0, 100.0, "%");
                }
                widgets::color_swatch(ui, [220, 40, 40, 255]);
                let cube = ViewCube::default();
                assert!(cube
                    .show(ui, egui::pos2(600.0, 80.0), -0.785, 0.6)
                    .is_none());
            });
        });
        out.textures_delta.clear();
    }
    h3d_i18n::set_language(h3d_i18n::Language::default());
}
