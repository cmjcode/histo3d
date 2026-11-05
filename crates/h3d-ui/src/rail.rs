//! Rel alat vertikal mengambang (pola `ducad-ui/src/left_toolbar.rs`):
//! kolom tombol persegi berikon, dipusatkan vertikal di sisi kiri kanvas.
//! Generik atas pengenal alat `T`; aplikasi menentukan isinya per tampilan.

use egui::{Pos2, Rect, Ui, Vec2};

use crate::icons::MaterialIcon;
use crate::theme;
use crate::widgets::tool_button;

/// Satu butir rel alat.
#[derive(Clone, Debug)]
pub enum RailItem<T> {
    /// Tombol alat.
    Tool {
        /// Pengenal alat (dikembalikan saat diklik).
        id: T,
        /// Ikon.
        icon: MaterialIcon,
        /// Judul tooltip (sudah diterjemahkan).
        title: String,
        /// Keterangan tooltip.
        subtitle: Option<String>,
        /// Tampil aktif (alat terpilih / sakelar menyala).
        selected: bool,
        /// Bisa diklik.
        enabled: bool,
    },
    /// Garis pemisah.
    Separator,
}

impl<T> RailItem<T> {
    /// Tombol alat aktif dan bisa diklik.
    pub fn tool(id: T, icon: MaterialIcon, title: impl Into<String>) -> Self {
        RailItem::Tool {
            id,
            icon,
            title: title.into(),
            subtitle: None,
            selected: false,
            enabled: true,
        }
    }

    /// Tambah keterangan tooltip.
    pub fn hint(mut self, text: impl Into<String>) -> Self {
        if let RailItem::Tool { subtitle, .. } = &mut self {
            *subtitle = Some(text.into());
        }
        self
    }

    /// Setel keadaan terpilih.
    pub fn selected(mut self, on: bool) -> Self {
        if let RailItem::Tool { selected, .. } = &mut self {
            *selected = on;
        }
        self
    }

    /// Setel bisa diklik.
    pub fn enabled(mut self, on: bool) -> Self {
        if let RailItem::Tool { enabled, .. } = &mut self {
            *enabled = on;
        }
        self
    }
}

/// Tinggi yang dibutuhkan rel untuk `items` (untuk memusatkan vertikal).
pub fn rail_height<T>(items: &[RailItem<T>]) -> f32 {
    let mut h = 8.0;
    for it in items {
        h += match it {
            RailItem::Tool { .. } => theme::TOOL_BUTTON_SIDE + 2.5,
            RailItem::Separator => 9.0,
        };
    }
    h
}

/// Gambar rel alat di dalam `area` (tepi kiri kanvas, dipusatkan vertikal).
/// Mengembalikan pengenal alat yang diklik.
pub fn show_rail<T: Copy>(ui: &mut Ui, area: Rect, items: &[RailItem<T>]) -> Option<T> {
    if items.is_empty() {
        return None;
    }
    let height = rail_height(items).min(area.height());
    let top = area.center().y - height / 2.0;
    let origin = Pos2::new(area.left() + theme::FLOAT_MARGIN, top.max(area.top()));
    let max_rect = Rect::from_min_size(origin, Vec2::new(theme::TOOL_BUTTON_SIDE + 8.0, height));
    let mut clicked = None;
    let mut child = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(max_rect)
            .layout(egui::Layout::top_down(egui::Align::Center)),
    );
    theme::toolbar_frame(child.visuals()).show(&mut child, |ui| {
        ui.spacing_mut().item_spacing = Vec2::new(0.0, 2.5);
        for it in items {
            match it {
                RailItem::Tool {
                    id,
                    icon,
                    title,
                    subtitle,
                    selected,
                    enabled,
                } => {
                    let resp = ui.add_enabled_ui(*enabled, |ui| {
                        tool_button(ui, *icon, *selected, title, subtitle.as_deref())
                    });
                    if resp.inner.clicked() {
                        clicked = Some(*id);
                    }
                }
                RailItem::Separator => {
                    ui.add_space(1.0);
                    ui.separator();
                    ui.add_space(1.0);
                }
            }
        }
    });
    clicked
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::icons;

    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Tool {
        A,
        B,
    }

    #[test]
    fn rail_renders_and_measures() {
        let items = vec![
            RailItem::tool(Tool::A, icons::BRUSH, "A").selected(true),
            RailItem::Separator,
            RailItem::tool(Tool::B, icons::ERASER, "B")
                .hint("keterangan")
                .enabled(false),
        ];
        let h = rail_height(&items);
        assert!(h > 2.0 * theme::TOOL_BUTTON_SIDE);
        let ctx = egui::Context::default();
        theme::apply(&ctx, theme::ThemeMode::Light);
        let mut out = ctx.run_ui(Default::default(), |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                let area = ui.max_rect();
                assert_eq!(show_rail(ui, area, &items), None);
                assert_eq!(show_rail::<Tool>(ui, area, &[]), None);
            });
        });
        out.textures_delta.clear();
    }
}
