//! Tema (terang/gelap) dan pemasangan font HISTO3D.
//!
//! Diterapkan sekali lewat [`apply`] saat startup, lalu lagi tiap kali
//! [`ThemeMode`] berubah. [`apply`] juga memasang font ikon `egui_icons`
//! (Material Symbols + MDI) sehingga `egui_icons::icons::ICON_*` terender;
//! pemanggilan berulang aman (egui mengganti font bernama sama).
//!
//! Disalin dari `ducad-ui/src/theme.rs` tanpa material kaca GPU: panel
//! mengambang memakai isian datar setengah tembus (versi "fallback" DUCAD),
//! sehingga tampil sama di tes headless, macOS, dan iPad.

use egui::{Color32, CornerRadius, Frame, Margin, Shadow, Stroke, Style, Vec2, Visuals};

/// Tinggi minimum widget interaktif untuk desktop.
pub const MIN_TOUCH_TARGET: f32 = 28.0;

/// Tinggi target sentuh Apple HIG untuk iPad (44pt).
pub const TOUCH_TARGET_IPAD: f32 = 44.0;

/// Ukuran ikon standar pada tombol bilah alat.
pub const ICON_SIZE_DEFAULT: f32 = 18.0;

/// Sisi tombol alat persegi di rel alat dan bilah atas.
pub const TOOL_BUTTON_SIDE: f32 = 34.0;

/// Lebar laci kanan (Lapisan, Sampel, Tumor 3D, Label).
pub const DRAWER_WIDTH: f32 = 320.0;

/// Jarak panel mengambang dari tepi kanvas.
pub const FLOAT_MARGIN: f32 = 10.0;

/// Warna aksen utama (tombol aktif, seleksi).
pub const ACCENT: Color32 = Color32::from_rgb(10, 132, 255);
/// Aksen terang untuk bingkai tombol aktif.
pub const ACCENT_LIGHT: Color32 = Color32::from_rgb(100, 180, 255);
/// Warna peringatan (spanduk, QC gagal).
pub const WARNING: Color32 = Color32::from_rgb(255, 149, 0);
/// Warna sukses (QC lolos, sudah ditinjau).
pub const SUCCESS: Color32 = Color32::from_rgb(48, 209, 88);
/// Warna galat.
pub const ERROR: Color32 = Color32::from_rgb(255, 69, 58);
/// Latar kanvas 3D gelap (arang).
pub const BG_CANVAS_DARK: Color32 = Color32::from_rgb(18, 19, 22);
/// Latar kanvas terang.
pub const BG_CANVAS_LIGHT: Color32 = Color32::from_rgb(228, 230, 235);
/// Latar panel mengambang gelap (~84 % pekat; isi kanvas hanya samar).
pub const BG_PANEL_DARK: Color32 = Color32::from_rgba_premultiplied(16, 18, 22, 215);
/// Latar panel mengambang terang.
pub const BG_PANEL_LIGHT: Color32 = Color32::from_rgba_premultiplied(242, 243, 246, 236);
/// Latar kartu di dalam laci (gelap).
pub const BG_CARD_DARK: Color32 = Color32::from_rgba_premultiplied(26, 30, 38, 160);
/// Latar kartu di dalam laci (terang).
pub const BG_CARD_LIGHT: Color32 = Color32::from_rgba_premultiplied(255, 255, 255, 200);
/// Bingkai tipis panel (gelap).
pub const BORDER_SUBTLE_DARK: Color32 = Color32::from_rgba_premultiplied(50, 56, 68, 130);
/// Bingkai tipis panel (terang).
pub const BORDER_SUBTLE_LIGHT: Color32 = Color32::from_rgba_premultiplied(160, 165, 175, 150);

/// Bayangan lembut panel mengambang.
const PANEL_SHADOW: Shadow = Shadow {
    offset: [0, 3],
    blur: 12,
    spread: 0,
    color: Color32::from_black_alpha(70),
};

/// Palet panel menurut tema aktif (dibaca dari `Visuals::dark_mode`).
fn panel_colors(v: &Visuals) -> (Color32, Color32, Color32) {
    if v.dark_mode {
        (BG_PANEL_DARK, BG_CARD_DARK, BORDER_SUBTLE_DARK)
    } else {
        (BG_PANEL_LIGHT, BG_CARD_LIGHT, BORDER_SUBTLE_LIGHT)
    }
}

/// Warna latar kanvas menurut tema.
pub fn canvas_fill(v: &Visuals) -> Color32 {
    if v.dark_mode {
        BG_CANVAS_DARK
    } else {
        BG_CANVAS_LIGHT
    }
}

/// Frame panel mengambang (laci, bilah atas, kartu usulan).
pub fn panel_frame(v: &Visuals) -> Frame {
    let (fill, _, border) = panel_colors(v);
    Frame {
        inner_margin: Margin::symmetric(10, 6),
        outer_margin: Margin::ZERO,
        corner_radius: CornerRadius::same(10),
        shadow: PANEL_SHADOW,
        fill,
        stroke: Stroke::new(1.0, border),
    }
}

/// Frame rel alat berisi tombol persegi (margin kecil, sudut 8).
pub fn toolbar_frame(v: &Visuals) -> Frame {
    Frame {
        inner_margin: Margin::same(4),
        corner_radius: CornerRadius::same(8),
        ..panel_frame(v)
    }
}

/// Frame kapsul (bilah konteks di bawah kanvas, kartu progres).
pub fn pill_frame(v: &Visuals) -> Frame {
    Frame {
        inner_margin: Margin::symmetric(12, 6),
        corner_radius: CornerRadius::same(16),
        ..panel_frame(v)
    }
}

/// Frame kartu di dalam laci (baris irisan, label).
pub fn card_frame(v: &Visuals) -> Frame {
    let (_, card, border) = panel_colors(v);
    Frame {
        inner_margin: Margin::same(6),
        outer_margin: Margin::symmetric(0, 2),
        corner_radius: CornerRadius::same(6),
        shadow: Shadow::NONE,
        fill: card,
        stroke: Stroke::new(0.5, border),
    }
}

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

    /// Panel di atas kanvas harus cukup pekat agar teks terbaca (pelajaran
    /// DUCAD: ~57 % membuat grid tembus di balik teks).
    #[test]
    fn floating_panels_are_opaque_enough_in_both_themes() {
        for mode in [ThemeMode::Dark, ThemeMode::Light] {
            let v = mode.visuals();
            let f = panel_frame(&v);
            assert!(f.fill.a() >= 200, "{mode:?}: alpha panel {}", f.fill.a());
            assert!(f.stroke.width > 0.0);
            assert_eq!(toolbar_frame(&v).fill, f.fill);
            assert_eq!(pill_frame(&v).fill, f.fill);
            assert!(card_frame(&v).fill.a() > 0);
            assert_eq!(canvas_fill(&v).a(), 255);
        }
    }
}
