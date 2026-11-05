//! Widget generik HISTO3D: tombol ikon, tombol alat persegi (rel alat ala
//! DUCAD), kepala laci, slider berlabel, pemberitahuan riset, spanduk
//! peringatan, legenda colormap, swatch warna.
//!
//! Semua teks lewat `h3d_i18n::t!`; semua ikon lewat [`crate::icons`].

use egui::{Color32, Rect, Response, RichText, Sense, Stroke, Ui, Vec2};
use h3d_core::Colormap;
use h3d_i18n::t;

use crate::icons::{self, MaterialIcon};
use crate::theme;

/// Warna teks tebal yang terbaca di kedua tema. `RichText::strong` memakai
/// `widgets.active.fg_stroke` yang oleh tema HISTO3D dibuat putih (teks
/// tombol aktif) sehingga tak terlihat di tema terang; pakai ini sebagai ganti.
pub fn strong_color(ui: &Ui) -> Color32 {
    ui.visuals().widgets.inactive.fg_stroke.color
}

/// Tombol alat persegi berikon (rel alat kiri, bilah atas): latar aksen bila
/// `selected`, sorot saat hover, tooltip berjudul `title` dengan keterangan
/// opsional `subtitle`. Sisi tombol = [`theme::TOOL_BUTTON_SIDE`].
pub fn tool_button(
    ui: &mut Ui,
    icon: MaterialIcon,
    selected: bool,
    title: &str,
    subtitle: Option<&str>,
) -> Response {
    let side = theme::TOOL_BUTTON_SIDE.max(ui.spacing().interact_size.y);
    let (rect, response) = ui.allocate_exact_size(Vec2::splat(side), Sense::click());
    if ui.is_rect_visible(rect) {
        let v = ui.visuals();
        let (fill, stroke, fg) = if !ui.is_enabled() {
            (
                Color32::TRANSPARENT,
                Stroke::NONE,
                v.weak_text_color().gamma_multiply(0.6),
            )
        } else if selected {
            (
                theme::ACCENT,
                Stroke::new(1.0, theme::ACCENT_LIGHT),
                Color32::WHITE,
            )
        } else if response.hovered() {
            (
                v.widgets.hovered.bg_fill,
                Stroke::new(1.0, theme::ACCENT.gamma_multiply(0.7)),
                v.widgets.hovered.fg_stroke.color,
            )
        } else {
            (
                Color32::TRANSPARENT,
                Stroke::NONE,
                v.widgets.inactive.fg_stroke.color,
            )
        };
        ui.painter()
            .rect(rect, 6.0, fill, stroke, egui::StrokeKind::Inside);
        ui.painter().text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            icon.codepoint,
            egui::FontId::new(theme::ICON_SIZE_DEFAULT, icon.font_family()),
            fg,
        );
    }
    if subtitle.is_some() || !title.is_empty() {
        let title = title.to_owned();
        let subtitle = subtitle.map(str::to_owned);
        response.on_hover_ui(|ui| {
            ui.set_max_width(260.0);
            ui.label(RichText::new(title).color(strong_color(ui)));
            if let Some(s) = subtitle {
                ui.small(s);
            }
        })
    } else {
        response
    }
}

/// Kepala laci: ikon + judul tebal dan tombol tutup di kanan.
/// Mengembalikan `true` bila tombol tutup diklik.
pub fn drawer_header(ui: &mut Ui, icon: MaterialIcon, title: &str) -> bool {
    let mut close = false;
    ui.horizontal(|ui| {
        ui.label(
            icon.rich_text()
                .size(theme::ICON_SIZE_DEFAULT)
                .color(theme::ACCENT),
        );
        ui.label(RichText::new(title).color(strong_color(ui)).size(14.0));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            close = icon_button(ui, icons::CLOSE, &t!("action-close")).clicked();
        });
    });
    ui.add_space(2.0);
    close
}

/// Judul bagian kecil di dalam laci (huruf kapital kecil berwarna redup).
pub fn section_title(ui: &mut Ui, text: &str) {
    ui.add_space(4.0);
    ui.label(
        RichText::new(text.to_uppercase())
            .size(10.5)
            .color(ui.visuals().weak_text_color()),
    );
}

/// Lencana angka bulat kecil (jumlah irisan, usulan menunggu).
pub fn count_badge(ui: &mut Ui, n: usize, color: Color32) -> Response {
    let text = n.to_string();
    let font = egui::FontId::proportional(10.5);
    let galley = ui.painter().layout_no_wrap(text, font, Color32::WHITE);
    let h = 16.0;
    let w = (galley.size().x + 10.0).max(h);
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(w, h), Sense::hover());
    if ui.is_rect_visible(rect) {
        ui.painter().rect_filled(rect, h / 2.0, color);
        ui.painter()
            .galley(rect.center() - galley.size() / 2.0, galley, Color32::WHITE);
    }
    resp
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
    fn tool_button_and_header_render_in_both_states() {
        let ctx = egui::Context::default();
        theme::apply(&ctx, theme::ThemeMode::Dark);
        let mut out = ctx.run_ui(Default::default(), |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                let a = tool_button(ui, icons::BRUSH, true, "Kuas", Some("B"));
                let b = tool_button(ui, icons::ERASER, false, "Penghapus", None);
                assert!(!a.clicked() && !b.clicked());
                assert_eq!(a.rect.size(), b.rect.size());
                assert!(!drawer_header(ui, icons::LAYERS, "Lapisan"));
                section_title(ui, "irisan");
                count_badge(ui, 12, theme::WARNING);
            });
        });
        out.textures_delta.clear();
    }

    #[test]
    fn format_value_shapes() {
        assert_eq!(format_value(0.0, "mm"), "0 mm");
        assert_eq!(format_value(12.34, ""), "12.3");
        assert_eq!(format_value(1.234, "%"), "1.23 %");
        assert_eq!(format_value(1234.5, "um"), "1234 um");
    }
}
