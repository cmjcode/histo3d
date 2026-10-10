//! Tema (terang/gelap) dan pemasangan font HISTO3D.
//!
//! Diterapkan sekali lewat [`apply`] saat startup, lalu lagi tiap kali
//! [`ThemeMode`] berubah. [`apply`] juga memasang font ikon `egui_icons`
//! (Material Symbols + MDI) sehingga `egui_icons::icons::ICON_*` terender;
//! pemanggilan berulang aman (egui mengganti font bernama sama).
//!
//! Versi minimal dari `ducad-ui/src/theme.rs` tanpa material kaca.

use egui::{Color32, CornerRadius, Stroke, Style, Vec2, Visuals};

/// Tinggi minimum widget interaktif untuk desktop.
pub const MIN_TOUCH_TARGET: f32 = 28.0;

/// Tinggi target sentuh Apple HIG untuk iPad (44pt).
pub const TOUCH_TARGET_IPAD: f32 = 44.0;

/// Ukuran ikon standar pada tombol bilah alat.
pub const ICON_SIZE_DEFAULT: f32 = 18.0;

/// Warna aksen utama (tombol aktif, seleksi).
pub const ACCENT: Color32 = Color32::from_rgb(10, 132, 255);
/// Warna peringatan (spanduk, QC gagal).
pub const WARNING: Color32 = Color32::from_rgb(255, 149, 0);
/// Warna sukses (QC lolos, sudah ditinjau).
pub const SUCCESS: Color32 = Color32::from_rgb(48, 209, 88);
/// Warna galat.
pub const ERROR: Color32 = Color32::from_rgb(255, 69, 58);

/// Mode tema aplikasi.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThemeMode {
    /// Latar terang.
    Light,
    /// Latar gelap (bawaan; nyaman untuk viewer slide).
    #[default]
    Dark,
}

impl ThemeMode {
    /// Mode lawan.
    pub fn toggled(self) -> Self {
        match self {
            ThemeMode::Light => ThemeMode::Dark,
            ThemeMode::Dark => ThemeMode::Light,
        }
    }

    /// Kunci terjemahan nama mode (`theme-light` / `theme-dark`).
    pub fn i18n_key(self) -> &'static str {
        match self {
            ThemeMode::Light => "theme-light",
            ThemeMode::Dark => "theme-dark",
        }
    }

    fn egui_theme(self) -> egui::Theme {
        match self {
            ThemeMode::Dark => egui::Theme::Dark,
            ThemeMode::Light => egui::Theme::Light,
        }
    }

    fn visuals(self) -> Visuals {
        let radius = CornerRadius::same(6);
        match self {
            ThemeMode::Dark => {
                let mut v = Visuals::dark();
                v.panel_fill = Color32::from_rgb(24, 26, 30);
                v.window_fill = Color32::from_rgb(30, 32, 38);
                v.extreme_bg_color = Color32::from_rgb(12, 13, 15);
                v.window_corner_radius = CornerRadius::same(10);
                v.menu_corner_radius = CornerRadius::same(8);
                v.widgets.inactive.corner_radius = radius;
                v.widgets.hovered.corner_radius = radius;
                v.widgets.hovered.bg_stroke = Stroke::new(1.0, ACCENT);
                v.widgets.active.corner_radius = radius;
                v.widgets.active.bg_fill = ACCENT;
                v.widgets.active.fg_stroke = Stroke::new(1.0, Color32::WHITE);
                v.widgets.open.corner_radius = radius;
                // Unmultiplied agar tidak di-blend aditif; teks terpilih putih.
                v.selection.bg_fill = Color32::from_rgba_unmultiplied(10, 132, 255, 90);
                v.selection.stroke = Stroke::new(1.0, Color32::WHITE);
                v
            }
            ThemeMode::Light => {
                let mut v = Visuals::light();
                v.window_corner_radius = CornerRadius::same(10);
                v.menu_corner_radius = CornerRadius::same(8);
                v.widgets.inactive.corner_radius = radius;
                v.widgets.hovered.corner_radius = radius;
                v.widgets.hovered.bg_stroke = Stroke::new(1.0, ACCENT);
                v.widgets.active.corner_radius = radius;
                v.widgets.active.bg_fill = ACCENT;
                v.widgets.active.fg_stroke = Stroke::new(1.0, Color32::WHITE);
                v.widgets.open.corner_radius = radius;
                v.selection.bg_fill = Color32::from_rgba_unmultiplied(10, 132, 255, 70);
                v.selection.stroke = Stroke::new(1.0, Color32::from_rgb(10, 40, 80));
                v
            }
        }
    }
}

/// Pasang font ikon `egui_icons` (Material Symbols Rounded + Pictogrammers MDI).
///
/// Dipanggil otomatis oleh [`apply`]; tersedia terpisah untuk konteks yang
/// hanya butuh ikon tanpa mengubah gaya (mis. tes).
pub fn install_fonts(ctx: &egui::Context) {
    egui_icons::initialize(ctx);
}

/// Terapkan tema, font ikon, dan ukuran target desktop ke context egui.
pub fn apply(ctx: &egui::Context, mode: ThemeMode) {
    apply_with_touch(ctx, mode, MIN_TOUCH_TARGET);
}

/// Terapkan tema dengan tinggi target sentuh kustom (mis. [`TOUCH_TARGET_IPAD`]).
pub fn apply_with_touch(ctx: &egui::Context, mode: ThemeMode, touch_target_y: f32) {
    install_fonts(ctx);

    let theme = mode.egui_theme();
    ctx.set_theme(theme);

    let mut style = Style {
        visuals: mode.visuals(),
        ..Default::default()
    };
    style.spacing.interact_size.y = touch_target_y.max(MIN_TOUCH_TARGET);
    style.spacing.button_padding = Vec2::new(8.0, 4.0);
    style.spacing.item_spacing = Vec2::new(6.0, 4.0);
    ctx.set_style_of(theme, style);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_both_modes_and_render_icons() {
        let ctx = egui::Context::default();
        for mode in [ThemeMode::Dark, ThemeMode::Light] {
            apply(&ctx, mode);
            let mut out = ctx.run_ui(Default::default(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    ui.label(crate::icons::CLOSE.rich_text());
                    ui.label(crate::icons::CUBE.rich_text());
                });
            });
            out.textures_delta.clear();
        }
        assert_eq!(ThemeMode::Dark.toggled(), ThemeMode::Light);
    }
}
