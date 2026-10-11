//! Pembantu UI kecil yang dipakai semua mode: tombol ikon + teks, isian
//! opsional, teks galat terjemahan, dan cache thumbnail.

use std::collections::{HashMap, HashSet};

use egui::{Response, Ui};
use h3d_core::SlideId;
use h3d_engine::{EngineError, Session};
use h3d_i18n::t;
use h3d_ui::icons::MaterialIcon;

/// Teks "ikon + label": ikon dari font ikon, label dari font proporsional
/// (lihat `h3d_ui::widgets::icon_text` tentang ligatur font ikon).
pub fn icon_label(icon: MaterialIcon, text: &str) -> egui::WidgetText {
    h3d_ui::widgets::icon_text(icon, text)
}

/// Seperti [`icon_label`] dengan warna tetap.
pub fn icon_label_colored(
    icon: MaterialIcon,
    text: &str,
    color: egui::Color32,
) -> egui::WidgetText {
    h3d_ui::widgets::icon_text_colored(icon, text, color)
}

/// Teks tebal yang terbaca di kedua tema. `Ui::strong` memakai warna
/// `widgets.active.fg_stroke`, yang oleh tema HISTO3D dibuat putih (teks
/// tombol aktif) sehingga tak terlihat di tema terang.
pub fn strong(ui: &mut Ui, text: impl Into<String>) -> Response {
    let color = ui.visuals().widgets.inactive.fg_stroke.color;
    ui.label(egui::RichText::new(text).color(color).size(13.0))
}

/// Tombol berikon dan berteks.
pub fn icon_text_button(ui: &mut Ui, icon: MaterialIcon, text: &str) -> Response {
    ui.button(icon_label(icon, text))
}

/// Tombol berikon dan berteks yang bisa dimatikan.
pub fn icon_text_button_enabled(
    ui: &mut Ui,
    enabled: bool,
    icon: MaterialIcon,
    text: &str,
) -> Response {
    ui.add_enabled(enabled, egui::Button::new(icon_label(icon, text)))
}

/// Tombol pilih (toggle) berikon dan berteks.
pub fn icon_selectable(ui: &mut Ui, selected: bool, icon: MaterialIcon, text: &str) -> Response {
    ui.selectable_label(selected, icon_label(icon, text))
}

/// Isian teks satu baris untuk `Option<String>`; kosong → `None`.
/// Mengembalikan `true` bila nilainya berubah.
pub fn opt_text(ui: &mut Ui, value: &mut Option<String>, width: f32) -> bool {
    let mut buf = value.clone().unwrap_or_default();
    let changed = ui
        .add(egui::TextEdit::singleline(&mut buf).desired_width(width))
        .changed();
    if changed {
        *value = if buf.trim().is_empty() {
            None
        } else {
            Some(buf)
        };
    }
    changed
}

/// Isian angka untuk `Option<f64>` dengan kotak centang "diisi".
pub fn opt_number(
    ui: &mut Ui,
    value: &mut Option<f64>,
    default: f64,
    range: std::ops::RangeInclusive<f64>,
    speed: f64,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        let mut on = value.is_some();
        if ui.checkbox(&mut on, "").changed() {
            *value = on.then_some(default);
            changed = true;
        }
        if let Some(v) = value.as_mut() {
            changed |= ui
                .add(egui::DragValue::new(v).range(range).speed(speed))
                .changed();
        }
    });
    changed
}

/// Pesan galat untuk pengguna dalam bahasa aktif.
pub fn error_text(e: &EngineError) -> String {
    match e {
        EngineError::MppMissing { file } => t!("err-mpp-missing", name = file.clone()),
        EngineError::Cancelled => t!("job-cancelled"),
        EngineError::Io(inner) => t!("err-io", message = inner.to_string()),
        other => t!("err-generic", message = other.to_string()),
    }
}

/// Nama tampil sebuah irisan: nomor urut (1-basis) dan nama berkas.
pub fn slide_title(session: &Session, id: SlideId) -> String {
    match session.slide(id) {
        Ok(s) => format!("{}. {}", s.order + 1, s.source.file_name()),
        Err(_) => id.to_string(),
    }
}

/// Sisi terpanjang thumbnail kisi (piksel).
pub const THUMB_SIDE: u32 = 256;
/// Thumbnail yang dibuat per frame paling banyak (agar frame tidak tersendat).
pub const THUMBS_PER_FRAME: usize = 2;

/// Cache tekstur thumbnail irisan (egui). Dibuat malas, beberapa per frame.
#[derive(Default)]
pub struct ThumbCache {
    map: HashMap<SlideId, (egui::TextureHandle, [u32; 2])>,
    failed: HashSet<SlideId>,
    loaded_this_frame: usize,
}

impl ThumbCache {
    /// Awal frame: reset kuota.
    pub fn begin_frame(&mut self) {
        self.loaded_this_frame = 0;
    }

    /// Thumbnail irisan bila sudah (atau baru saja) dibuat. `None` berarti
    /// belum tersedia pada frame ini; pemanggil menggambar tempat kosong.
    pub fn get(
        &mut self,
        ctx: &egui::Context,
        session: &Session,
        id: SlideId,
    ) -> Option<(egui::TextureId, [u32; 2])> {
        if !self.map.contains_key(&id) {
            if self.failed.contains(&id) || self.loaded_this_frame >= THUMBS_PER_FRAME {
                if !self.failed.contains(&id) {
                    ctx.request_repaint();
                }
                return None;
            }
            self.loaded_this_frame += 1;
            match session.thumbnail(id, THUMB_SIDE) {
                Ok(img) => {
                    let size = [img.width(), img.height()];
                    let color = egui::ColorImage::from_rgba_unmultiplied(
                        [size[0] as usize, size[1] as usize],
                        img.as_raw(),
                    );
                    let tex = ctx.load_texture(
                        format!("thumb-{}", id.0),
                        color,
                        egui::TextureOptions::LINEAR,
                    );
                    self.map.insert(id, (tex, size));
                }
                Err(e) => {
                    log::warn!("thumbnail irisan {id} gagal: {e}");
                    self.failed.insert(id);
                    return None;
                }
            }
        }
        self.map.get(&id).map(|(t, s)| (t.id(), *s))
    }

    /// Buang satu thumbnail (mis. irisan dihapus).
    pub fn forget(&mut self, id: SlideId) {
        self.map.remove(&id);
        self.failed.remove(&id);
    }

    /// Kosongkan (proyek diganti, peringatan memori).
    pub fn clear(&mut self) {
        self.map.clear();
        self.failed.clear();
    }

    /// Jumlah thumbnail tersimpan.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Benar bila kosong.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

/// Pemberitahuan sementara di pojok layar.
#[derive(Clone, Debug, PartialEq)]
pub struct Toast {
    /// Teks (sudah diterjemahkan).
    pub text: String,
    /// Galat (merah) atau informasi.
    pub is_error: bool,
    /// Waktu kedaluwarsa (detik waktu egui).
    pub expires_at: f64,
}

/// Lama tampil pemberitahuan informasi (detik).
pub const TOAST_INFO_SECS: f64 = 4.0;
/// Lama tampil pemberitahuan galat (detik).
pub const TOAST_ERROR_SECS: f64 = 9.0;

/// Daftar pemberitahuan.
#[derive(Default)]
pub struct Toasts {
    items: Vec<Toast>,
    now: f64,
}

impl Toasts {
    /// Setel waktu sekarang dan buang yang kedaluwarsa.
    pub fn begin_frame(&mut self, now: f64) {
        self.now = now;
        self.items.retain(|t| t.expires_at > now);
    }

    /// Tambah informasi.
    pub fn info(&mut self, text: impl Into<String>) {
        self.items.push(Toast {
            text: text.into(),
            is_error: false,
            expires_at: self.now + TOAST_INFO_SECS,
        });
    }

    /// Tambah galat (teks sudah diterjemahkan).
    pub fn error(&mut self, text: impl Into<String>) {
        let text = text.into();
        log::warn!("{text}");
        self.items.push(Toast {
            text,
            is_error: true,
            expires_at: self.now + TOAST_ERROR_SECS,
        });
    }

    /// Tambah galat mesin dengan pesan terjemahan.
    pub fn engine_error(&mut self, e: &EngineError) {
        self.error(error_text(e));
    }

    /// Pemberitahuan aktif.
    pub fn items(&self) -> &[Toast] {
        &self.items
    }

    /// Gambar tumpukan pemberitahuan di pojok kanan bawah.
    pub fn show(&mut self, ctx: &egui::Context) {
        if self.items.is_empty() {
            return;
        }
        let mut dismiss = None;
        egui::Area::new(egui::Id::new("h3d-toasts"))
            .anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(-12.0, -12.0))
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                ui.set_max_width(420.0);
                for (i, toast) in self.items.iter().enumerate() {
                    let color = if toast.is_error {
                        h3d_ui::theme::ERROR
                    } else {
                        h3d_ui::theme::ACCENT
                    };
                    egui::Frame::popup(ui.style())
                        .stroke(egui::Stroke::new(1.0, color))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                let icon = if toast.is_error {
                                    h3d_ui::icons::ERROR
                                } else {
                                    h3d_ui::icons::INFO
                                };
                                ui.label(icon.rich_text().color(color));
                                ui.add(egui::Label::new(&toast.text).wrap());
                                if h3d_ui::widgets::icon_button(
                                    ui,
                                    h3d_ui::icons::CLOSE,
                                    &t!("action-close"),
                                )
                                .clicked()
                                {
                                    dismiss = Some(i);
                                }
                            });
                        });
                }
            });
        if let Some(i) = dismiss {
            self.items.remove(i);
        }
        ctx.request_repaint_after(std::time::Duration::from_millis(500));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toasts_expire() {
        let mut t = Toasts::default();
        t.begin_frame(10.0);
        t.info("a");
        t.error("b");
        assert_eq!(t.items().len(), 2);
        t.begin_frame(10.0 + TOAST_INFO_SECS + 0.1);
        assert_eq!(t.items().len(), 1);
        assert!(t.items()[0].is_error);
        t.begin_frame(10.0 + TOAST_ERROR_SECS + 0.1);
        assert!(t.items().is_empty());
    }

    #[test]
    fn engine_errors_are_translated() {
        let _g = crate::test_util::lang_lock();
        h3d_i18n::set_language(h3d_i18n::Language::En);
        let e = EngineError::MppMissing {
            file: "s1.png".into(),
        };
        let en = error_text(&e);
        h3d_i18n::set_language(h3d_i18n::Language::Id);
        let id = error_text(&e);
        assert!(en.contains("s1.png") && id.contains("s1.png"));
        assert_ne!(en, id);
        assert!(!error_text(&EngineError::Cancelled).contains("job-cancelled"));
    }
}
