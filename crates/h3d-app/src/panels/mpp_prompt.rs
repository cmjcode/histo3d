//! Dialog isian mpp saat impor melaporkan `EngineError::MppMissing`.
//!
//! mpp tidak pernah ditebak (aturan keras 7): pengguna mengetiknya, atau
//! mengukurnya dengan alat skala batang — tarik garis di atas skala batang
//! pada gambar lalu ketik panjang aslinya.

use std::path::PathBuf;

use h3d_i18n::t;

use crate::geom::scale_bar_mpp_from_thumb;
use crate::state::ImportRequest;
use crate::ui_util::icon_text_button_enabled;

/// Sisi terpanjang gambar pratinjau untuk alat skala batang (piksel).
pub const PREVIEW_SIDE: u32 = 1536;

/// Tindakan pengguna pada dialog.
#[derive(Clone, Debug, PartialEq)]
pub enum MppPromptAction {
    /// Belum ada keputusan.
    None,
    /// Impor dibatalkan.
    Cancel,
    /// Ulangi impor dengan mpp ini `[x, y]`.
    Apply([f64; 2]),
}

struct Preview {
    texture: egui::TextureHandle,
    size: [u32; 2],
    /// Piksel level 0 per piksel pratinjau.
    scale: f64,
}

/// Keadaan dialog isian mpp.
pub struct MppPrompt {
    /// Impor yang akan diulang.
    pub request: ImportRequest,
    /// Nama berkas yang tidak memuat mpp.
    pub file: String,
    /// mpp isian `[x, y]`; 0 = belum diisi.
    pub mpp: [f64; 2],
    /// Panjang asli skala batang (µm).
    pub length_um: f64,
    /// Garis di atas pratinjau (piksel pratinjau).
    pub line: Option<([f64; 2], [f64; 2])>,
    /// Perbesaran pratinjau.
    pub zoom: f32,
    preview: Option<Preview>,
    preview_tried: bool,
}

impl MppPrompt {
    /// Dialog baru untuk `file` yang tidak memuat mpp.
    pub fn new(request: ImportRequest, file: String) -> Self {
        MppPrompt {
            request,
            file,
            mpp: [0.0, 0.0],
            length_um: 100.0,
            line: None,
            zoom: 1.0,
            preview: None,
            preview_tried: false,
        }
    }

    fn file_path(&self) -> Option<PathBuf> {
        self.request
            .paths
            .iter()
            .find(|p| {
                p.file_name()
                    .is_some_and(|n| n.to_string_lossy() == self.file)
            })
            .or_else(|| self.request.paths.first())
            .cloned()
    }

    fn ensure_preview(&mut self, ctx: &egui::Context) {
        if self.preview_tried {
            return;
        }
        self.preview_tried = true;
        let Some(path) = self.file_path() else {
            return;
        };
        let loaded = h3d_slide::open_slide(&path).and_then(|src| {
            let l0 = src.info().level0_size();
            h3d_slide::thumbnail(src.as_ref(), PREVIEW_SIDE).map(|img| (img, l0))
        });
        match loaded {
            Ok((img, l0)) if img.width() > 0 && img.height() > 0 => {
                let size = [img.width(), img.height()];
                let color = egui::ColorImage::from_rgba_unmultiplied(
                    [size[0] as usize, size[1] as usize],
                    img.as_raw(),
                );
                let texture =
                    ctx.load_texture("mpp-prompt-preview", color, egui::TextureOptions::LINEAR);
                self.preview = Some(Preview {
                    texture,
                    size,
                    scale: f64::from(l0[0]) / f64::from(size[0]),
                });
            }
            Ok(_) => {}
            Err(e) => log::warn!("pratinjau {path:?} gagal: {e}"),
        }
    }

    /// Piksel level 0 per piksel pratinjau (bila pratinjau termuat).
    pub fn preview_scale(&self) -> Option<f64> {
        self.preview.as_ref().map(|p| p.scale)
    }

    /// mpp hasil pengukuran skala batang, bila garis dan panjangnya sah.
    pub fn measured_mpp(&self, preview_scale: f64) -> Option<f64> {
        let (a, b) = self.line?;
        scale_bar_mpp_from_thumb(a, b, preview_scale, self.length_um)
    }

    /// mpp isian sah (keduanya positif).
    pub fn mpp_valid(&self) -> bool {
        self.mpp.iter().all(|m| m.is_finite() && *m > 0.0)
    }

    /// Gambar dialog; mengembalikan keputusan pengguna.
    pub fn ui(&mut self, ctx: &egui::Context) -> MppPromptAction {
        self.ensure_preview(ctx);
        let mut action = MppPromptAction::None;
        egui::Window::new(t!("mpp-prompt-title"))
            .id(egui::Id::new("h3d-mpp-prompt"))
            .collapsible(false)
            .default_width(560.0)
            .show(ctx, |ui| {
                h3d_ui::widgets::warning_banner(ui, &t!("mpp-required"));
                ui.label(format!("{}: {}", t!("slides-file-name"), self.file));
                ui.separator();
                egui::Grid::new("mpp-prompt-grid")
                    .num_columns(2)
                    .show(ui, |ui| {
                        ui.label(t!("slides-mpp-x"));
                        ui.add(
                            egui::DragValue::new(&mut self.mpp[0])
                                .range(0.0..=1000.0)
                                .speed(0.01)
                                .max_decimals(4),
                        );
                        ui.end_row();
                        ui.label(t!("slides-mpp-y"));
                        ui.add(
                            egui::DragValue::new(&mut self.mpp[1])
                                .range(0.0..=1000.0)
                                .speed(0.01)
                                .max_decimals(4),
                        );
                        ui.end_row();
                    });

                ui.separator();
                crate::ui_util::strong(ui, t!("mpp-scale-bar-title"));
                ui.label(t!("mpp-scale-bar-hint"));
                ui.horizontal(|ui| {
                    ui.label(t!("mpp-scale-bar-length"));
                    ui.add(
                        egui::DragValue::new(&mut self.length_um)
                            .range(0.0..=1.0e6)
                            .speed(1.0)
                            .suffix(format!(" {}", t!("unit-um"))),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.zoom, 0.25..=8.0).text(t!("action-zoom-in")),
                    );
                });

                let scale = self.preview_scale();
                if let Some(p) = &self.preview {
                    let tex = p.texture.id();
                    let size = egui::vec2(p.size[0] as f32, p.size[1] as f32) * self.zoom;
                    let zoom = f64::from(self.zoom);
                    let mut new_line = self.line;
                    egui::ScrollArea::both()
                        .max_height(360.0)
                        .auto_shrink([false, true])
                        .show(ui, |ui| {
                            let (rect, resp) =
                                ui.allocate_exact_size(size, egui::Sense::click_and_drag());
                            let painter = ui.painter_at(rect);
                            painter.image(
                                tex,
                                rect,
                                egui::Rect::from_min_max(
                                    egui::pos2(0.0, 0.0),
                                    egui::pos2(1.0, 1.0),
                                ),
                                egui::Color32::WHITE,
                            );
                            let to_img = |p: egui::Pos2| {
                                [
                                    f64::from(p.x - rect.min.x) / zoom,
                                    f64::from(p.y - rect.min.y) / zoom,
                                ]
                            };
                            if resp.drag_started() {
                                if let Some(p) = resp.interact_pointer_pos() {
                                    new_line = Some((to_img(p), to_img(p)));
                                }
                            } else if resp.dragged() {
                                if let (Some(p), Some((a, _))) =
                                    (resp.interact_pointer_pos(), new_line)
                                {
                                    new_line = Some((a, to_img(p)));
                                }
                            }
                            if let Some((a, b)) = new_line {
                                let to_screen = |q: [f64; 2]| {
                                    egui::pos2(
                                        rect.min.x + (q[0] * zoom) as f32,
                                        rect.min.y + (q[1] * zoom) as f32,
                                    )
                                };
                                let stroke = egui::Stroke::new(2.0, h3d_ui::theme::ACCENT);
                                painter.line_segment([to_screen(a), to_screen(b)], stroke);
                                painter.circle_filled(to_screen(a), 3.0, h3d_ui::theme::ACCENT);
                                painter.circle_filled(to_screen(b), 3.0, h3d_ui::theme::ACCENT);
                            }
                        });
                    self.line = new_line;
                } else {
                    ui.weak(t!("mpp-preview-unavailable"));
                }

                let measured = scale.and_then(|s| self.measured_mpp(s));
                ui.horizontal(|ui| {
                    match measured {
                        Some(m) => {
                            ui.label(t!("mpp-measured", value = format!("{m:.4}")));
                        }
                        None => {
                            ui.weak(t!("mpp-measured-none"));
                        }
                    }
                    if icon_text_button_enabled(
                        ui,
                        measured.is_some(),
                        h3d_ui::icons::MEASURE,
                        &t!("mpp-use-measured"),
                    )
                    .clicked()
                    {
                        if let Some(m) = measured {
                            self.mpp = [m, m];
                        }
                    }
                });

                ui.separator();
                ui.horizontal(|ui| {
                    if icon_text_button_enabled(
                        ui,
                        self.mpp_valid(),
                        h3d_ui::icons::CHECK,
                        &t!("action-apply"),
                    )
                    .clicked()
                    {
                        action = MppPromptAction::Apply(self.mpp);
                    }
                    if ui.button(t!("action-cancel")).clicked() {
                        action = MppPromptAction::Cancel;
                    }
                });
            });
        action
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measured_mpp_needs_line_and_length() {
        let mut p = MppPrompt::new(
            ImportRequest {
                paths: vec![PathBuf::from("/x/a.png")],
                embed: false,
            },
            "a.png".into(),
        );
        assert!(!p.mpp_valid());
        assert_eq!(p.measured_mpp(4.0), None);
        // Garis 50 px pratinjau × skala 4 = 200 px level 0; 100 µm → 0,5.
        p.line = Some(([10.0, 20.0], [60.0, 20.0]));
        p.length_um = 100.0;
        assert!((p.measured_mpp(4.0).unwrap() - 0.5).abs() < 1e-12);
        p.length_um = 0.0;
        assert_eq!(p.measured_mpp(4.0), None);
        p.mpp = [0.5, 0.5];
        assert!(p.mpp_valid());
        assert_eq!(p.file_path(), Some(PathBuf::from("/x/a.png")));
    }
}
