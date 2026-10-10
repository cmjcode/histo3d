//! Uji asap: widget h3d-ui dirender di dalam `egui::Context::run` tanpa panik,
//! dalam kedua bahasa dan kedua tema.

use h3d_core::Colormap;
use h3d_ui::widgets;
use h3d_ui::{AppMode, ThemeMode};

#[test]
fn widgets_render_without_panic() {
    let ctx = egui::Context::default();
    for (lang, theme) in [
        (h3d_i18n::Language::Id, ThemeMode::Dark),
        (h3d_i18n::Language::En, ThemeMode::Light),
    ] {
        h3d_i18n::set_language(lang);
        h3d_ui::theme::apply(&ctx, theme);

        let mut mode = AppMode::Slides;
        let mut on = false;
        let mut spacing = 50.0_f64;

        let mut out = ctx.run_ui(Default::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let changed = widgets::mode_bar(ui, &mut mode);
                assert!(!changed, "tanpa input mode tidak boleh berubah");
                widgets::research_notice(ui);
                widgets::warning_banner(ui, &h3d_i18n::t!("warn-has-draft"));
                widgets::icon_button(ui, h3d_ui::icons::SAVE, &h3d_i18n::t!("action-save"));
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
            });
        });
        out.textures_delta.clear();
    }
    h3d_i18n::set_language(h3d_i18n::Language::default());
}
