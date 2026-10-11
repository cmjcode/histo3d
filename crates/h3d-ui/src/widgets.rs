//! Widget generik HISTO3D: tombol ikon, slider berlabel, bilah mode,
//! pemberitahuan riset, spanduk peringatan, legenda colormap, swatch warna.
//!
//! Semua teks lewat `h3d_i18n::t!`; semua ikon lewat [`crate::icons`].

use egui::{Color32, Rect, Response, RichText, Sense, Stroke, Ui, Vec2};
use h3d_core::Colormap;
use h3d_i18n::t;

use crate::icons::{self, MaterialIcon};
use crate::theme;

/// Mode aplikasi (histo3d.md 8.11).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum AppMode {
    /// Impor, urutan, mpp, metadata.
    #[default]
    Slides,
    /// Registrasi antar irisan.
    Register,
    /// Pelabelan.
    Segment,
    /// Viewport 3D.
    Reconstruct,
    /// Pengukuran dan ekspor.
    Report,
}

impl AppMode {
    /// Semua mode, urutan alur kerja.
    pub const ALL: [AppMode; 5] = [
        AppMode::Slides,
        AppMode::Register,
        AppMode::Segment,
        AppMode::Reconstruct,
        AppMode::Report,
    ];

    /// Kunci terjemahan nama mode (`mode-*`).
    pub fn i18n_key(self) -> &'static str {
        match self {
            AppMode::Slides => "mode-slides",
            AppMode::Register => "mode-register",
            AppMode::Segment => "mode-segment",
            AppMode::Reconstruct => "mode-reconstruct",
            AppMode::Report => "mode-report",
        }
    }

    /// Ikon mode.
    pub fn icon(self) -> MaterialIcon {
        match self {
            AppMode::Slides => icons::GRID,
            AppMode::Register => icons::COMPARE,
            AppMode::Segment => icons::DRAW,
            AppMode::Reconstruct => icons::CUBE,
            AppMode::Report => icons::REPORT,
        }
    }
}

/// Tombol ikon tanpa bingkai dengan tooltip.
pub fn icon_button(ui: &mut Ui, icon: MaterialIcon, tooltip: &str) -> Response {
    let resp = ui.add(
        egui::Button::new(icon.rich_text().size(theme::ICON_SIZE_DEFAULT))
            .frame(false)
            .min_size(Vec2::splat(ui.spacing().interact_size.y)),
    );
    if tooltip.is_empty() {
        resp
    } else {
        resp.on_hover_text(tooltip)
    }
}

/// Tombol ikon yang bisa dinyalakan; mengembalikan respons dan membalik
/// `on` saat diklik.
pub fn toggle_icon_button(
    ui: &mut Ui,
    on: &mut bool,
    icon: MaterialIcon,
    tooltip: &str,
) -> Response {
    let resp = ui.add(
        egui::Button::new(icon.rich_text().size(theme::ICON_SIZE_DEFAULT))
            .selected(*on)
            .min_size(Vec2::splat(ui.spacing().interact_size.y)),
    );
    if resp.clicked() {
        *on = !*on;
    }
    if tooltip.is_empty() {
        resp
    } else {
        resp.on_hover_text(tooltip)
    }
}

/// Slider `f64` dengan label di kiri dan satuan di kanan; `true` bila berubah.
pub fn labeled_slider_f64(
    ui: &mut Ui,
    label: &str,
    value: &mut f64,
    range: std::ops::RangeInclusive<f64>,
    suffix: &str,
) -> bool {
    ui.horizontal(|ui| {
        ui.label(label);
        let mut slider = egui::Slider::new(value, range);
        if !suffix.is_empty() {
            slider = slider.suffix(format!(" {suffix}"));
        }
        ui.add(slider).changed()
    })
    .inner
}

/// Ukuran teks di sebelah ikon (sama dengan gaya tombol bawaan egui).
pub const ICON_TEXT_SIZE: f32 = 12.5;

/// Teks "ikon + label" sebagai dua ruas tata letak: ikon memakai font ikon,
/// label memakai font proporsional. Label **tidak boleh** ikut memakai font
/// ikon: Material Symbols berligatur, sehingga potongan kata seperti "men",
/// "lan", atau "3d" berubah menjadi ikon. Warna mengikuti widget.
pub fn icon_text(icon: MaterialIcon, text: &str) -> egui::WidgetText {
    icon_text_colored(icon, text, Color32::PLACEHOLDER)
}

/// Seperti [`icon_text`] dengan warna tetap.
pub fn icon_text_colored(icon: MaterialIcon, text: &str, color: Color32) -> egui::WidgetText {
    let mut job = egui::text::LayoutJob::default();
    job.append(
        icon.codepoint,
        0.0,
        egui::TextFormat {
            font_id: egui::FontId::new(theme::ICON_SIZE_DEFAULT - 3.0, icon.font_family()),
            color,
            valign: egui::Align::Center,
            ..Default::default()
        },
    );
    job.append(
        text,
        6.0,
        egui::TextFormat {
            font_id: egui::FontId::proportional(ICON_TEXT_SIZE),
            color,
            valign: egui::Align::Center,
            ..Default::default()
        },
    );
    egui::WidgetText::LayoutJob(std::sync::Arc::new(job))
}

/// Bilah pemilih mode (ikon + nama); `true` bila mode berubah.
pub fn mode_bar(ui: &mut Ui, mode: &mut AppMode) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        for m in AppMode::ALL {
            let selected = *mode == m;
            let label = icon_text(m.icon(), &t!(m.i18n_key()));
            if ui.selectable_label(selected, label).clicked() && !selected {
                *mode = m;
                changed = true;
            }
        }
    });
    changed
}

/// Pemberitahuan "Hanya untuk riset, bukan untuk diagnosis" (aturan keras 16).
pub fn research_notice(ui: &mut Ui) -> Response {
    ui.horizontal(|ui| {
        ui.label(icons::INFO.rich_text().color(theme::ACCENT));
        ui.label(RichText::new(h3d_core::RESEARCH_ONLY_NOTICE).italics())
    })
    .inner
}

/// Spanduk peringatan berlatar oranye lembut.
pub fn warning_banner(ui: &mut Ui, text: &str) -> Response {
    egui::Frame::new()
        .fill(theme::WARNING.gamma_multiply(0.18))
        .stroke(Stroke::new(1.0, theme::WARNING))
        .corner_radius(6)
        .inner_margin(egui::Margin::symmetric(8, 4))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(icons::WARNING.rich_text().color(theme::WARNING));
                ui.label(text)
            })
            .inner
        })
        .inner
}

/// Legenda colormap: bilah gradien 256 langkah dengan label min/max + satuan.
pub fn colormap_legend(ui: &mut Ui, colormap: Colormap, min: f32, max: f32, unit: &str) {
    let bar_h = 12.0;
    let width = ui.available_width().clamp(96.0, 320.0);
    let (rect, _) = ui.allocate_exact_size(Vec2::new(width, bar_h), Sense::hover());
    if ui.is_rect_visible(rect) {
        let painter = ui.painter();
        let step = rect.width() / 256.0;
        for i in 0..256usize {
            let c = colormap.sample(i as f32 / 255.0);
            let x0 = rect.left() + i as f32 * step;
            // +0,5 px agar tidak ada celah antar langkah akibat pembulatan.
            let seg = Rect::from_min_max(
                egui::pos2(x0, rect.top()),
                egui::pos2(x0 + step + 0.5, rect.bottom()),
            );
            painter.rect_filled(seg, 0.0, Color32::from_rgb(c[0], c[1], c[2]));
        }
        painter.rect_stroke(
            rect,
            0.0,
            Stroke::new(1.0, ui.visuals().widgets.inactive.bg_stroke.color),
            egui::StrokeKind::Inside,
        );
    }
    ui.horizontal(|ui| {
        ui.set_width(width);
        ui.small(format_value(min, unit));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.small(format_value(max, unit));
        });
    });
}

fn format_value(v: f32, unit: &str) -> String {
    let num = if v.abs() >= 100.0 || v == v.trunc() {
        format!("{v:.0}")
    } else if v.abs() >= 10.0 {
        format!("{v:.1}")
    } else {
        format!("{v:.2}")
    };
    if unit.is_empty() {
        num
    } else {
        format!("{num} {unit}")
    }
}

/// Kotak kecil berwarna (warna label) dengan bingkai tipis.
pub fn color_swatch(ui: &mut Ui, rgba: [u8; 4]) -> Response {
    let size = Vec2::splat(ui.spacing().interact_size.y * 0.6);
    let (rect, resp) = ui.allocate_exact_size(size, Sense::hover());
    if ui.is_rect_visible(rect) {
        let color = Color32::from_rgba_unmultiplied(rgba[0], rgba[1], rgba[2], rgba[3]);
        ui.painter().rect(
            rect,
            3.0,
            color,
            Stroke::new(1.0, ui.visuals().widgets.inactive.fg_stroke.color),
            egui::StrokeKind::Inside,
        );
    }
    resp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn app_mode_keys_exist_in_both_languages() {
        for m in AppMode::ALL {
            for lang in h3d_i18n::Language::all() {
                assert!(h3d_i18n::has_key(*lang, m.i18n_key()), "{:?}", m);
            }
        }
    }

    #[test]
    fn format_value_shapes() {
        assert_eq!(format_value(0.0, "mm"), "0 mm");
        assert_eq!(format_value(12.34, ""), "12.3");
        assert_eq!(format_value(1.234, "%"), "1.23 %");
        assert_eq!(format_value(1234.5, "um"), "1234 um");
    }
}
