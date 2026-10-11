//! Peramban Documents dalam-aplikasi (9.3). Di iPad tidak ada dialog berkas
//! native (`rfd` tidak mendukung iOS): proyek `.h3d` dan berkas gambar
//! diambil dari folder Documents tersandbox, yang terbuka ke aplikasi Files.
//! Impor dari sini selalu menyalin slide ke dalam bundel; ukuran yang
//! dibutuhkan ditampilkan sebelum menyalin, dan mesin memeriksa ruang kosong.

use std::collections::BTreeSet;
use std::path::PathBuf;

use h3d_i18n::t;

use crate::platform::{self, DocListing};
use crate::ui_util::{icon_label, icon_text_button, icon_text_button_enabled};

/// Untuk apa peramban dibuka.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrowserPurpose {
    /// Membuka atau membuat proyek.
    Projects,
    /// Memilih berkas slide untuk diimpor.
    ImportSlides,
}

/// Tindakan pengguna.
#[derive(Clone, Debug, PartialEq)]
pub enum BrowserAction {
    /// Belum ada keputusan.
    None,
    /// Tutup peramban.
    Close,
    /// Buka bundel ini.
    Open(PathBuf),
    /// Buat bundel baru di path ini.
    Create(PathBuf),
    /// Impor berkas-berkas ini (disalin ke bundel).
    Import(Vec<PathBuf>),
}

/// Keadaan peramban.
#[derive(Clone, Debug)]
pub struct BrowserState {
    /// Tujuan.
    pub purpose: BrowserPurpose,
    /// Folder yang ditampilkan.
    pub dir: PathBuf,
    /// Isi folder.
    pub listing: DocListing,
    /// Berkas slide terpilih.
    pub selected: BTreeSet<PathBuf>,
    /// Nama proyek baru.
    pub new_name: String,
}

impl BrowserState {
    /// Peramban folder Documents aplikasi.
    pub fn new(purpose: BrowserPurpose) -> Self {
        Self::in_dir(purpose, platform::documents_dir())
    }

    /// Peramban folder tertentu (tes).
    pub fn in_dir(purpose: BrowserPurpose, dir: PathBuf) -> Self {
        let listing = platform::list_documents(&dir);
        BrowserState {
            purpose,
            dir,
            listing,
            selected: BTreeSet::new(),
            new_name: String::new(),
        }
    }

    /// Baca ulang isi folder.
    pub fn refresh(&mut self) {
        self.listing = platform::list_documents(&self.dir);
        let existing: BTreeSet<PathBuf> =
            self.listing.slides.iter().map(|e| e.path.clone()).collect();
        self.selected.retain(|p| existing.contains(p));
    }

    /// Jumlah byte berkas terpilih (ruang yang dibutuhkan untuk menyalin).
    pub fn selected_bytes(&self) -> u64 {
        self.listing
            .slides
            .iter()
            .filter(|e| self.selected.contains(&e.path))
            .map(|e| e.size_bytes)
            .sum()
    }

    /// Path bundel untuk nama proyek baru.
    pub fn new_project_path(&self) -> PathBuf {
        self.dir.join(platform::bundle_dir_name(&self.new_name))
    }

    /// Gambar peramban; mengembalikan keputusan pengguna.
    pub fn ui(&mut self, ctx: &egui::Context) -> BrowserAction {
        let mut action = BrowserAction::None;
        let title = match self.purpose {
            BrowserPurpose::Projects => t!("browser-title-projects"),
            BrowserPurpose::ImportSlides => t!("browser-title-import"),
        };
        egui::Window::new(title)
            .id(egui::Id::new("h3d-doc-browser"))
            .collapsible(false)
            .default_width(480.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(h3d_ui::icons::FOLDER.rich_text());
                    ui.weak(self.dir.display().to_string());
                    if h3d_ui::widgets::icon_button(
                        ui,
                        h3d_ui::icons::REFRESH,
                        &t!("action-refresh"),
                    )
                    .clicked()
                    {
                        self.refresh();
                    }
                });
                ui.small(t!("browser-files-hint"));
                ui.separator();
                match self.purpose {
                    BrowserPurpose::Projects => self.projects_ui(ui, &mut action),
                    BrowserPurpose::ImportSlides => self.import_ui(ui, &mut action),
                }
                ui.separator();
                if ui.button(t!("action-close")).clicked() {
                    action = BrowserAction::Close;
                }
            });
        action
    }

    fn projects_ui(&mut self, ui: &mut egui::Ui, action: &mut BrowserAction) {
        crate::ui_util::strong(ui, t!("browser-projects"));
        if self.listing.bundles.is_empty() {
            ui.weak(t!("browser-no-projects"));
        }
        egui::ScrollArea::vertical()
            .id_salt("browser-projects")
            .max_height(260.0)
            .show(ui, |ui| {
                for entry in &self.listing.bundles {
                    if ui
                        .button(icon_label(h3d_ui::icons::OPEN_FOLDER, &entry.name))
                        .clicked()
                    {
                        *action = BrowserAction::Open(entry.path.clone());
                    }
                }
            });
        ui.separator();
        crate::ui_util::strong(ui, t!("action-new"));
        ui.horizontal(|ui| {
            ui.label(t!("slides-sample-name"));
            ui.add(egui::TextEdit::singleline(&mut self.new_name).desired_width(200.0));
            let ok = !self.new_name.trim().is_empty();
            if icon_text_button_enabled(ui, ok, h3d_ui::icons::NEW, &t!("action-new")).clicked() {
                *action = BrowserAction::Create(self.new_project_path());
            }
        });
    }

    fn import_ui(&mut self, ui: &mut egui::Ui, action: &mut BrowserAction) {
        crate::ui_util::strong(ui, t!("browser-slides"));
        if self.listing.slides.is_empty() {
            ui.weak(t!("browser-no-slides"));
        }
        ui.horizontal(|ui| {
            if ui.button(t!("browser-select-all")).clicked() {
                self.selected = self.listing.slides.iter().map(|e| e.path.clone()).collect();
            }
            if ui.button(t!("browser-select-none")).clicked() {
                self.selected.clear();
            }
        });
        egui::ScrollArea::vertical()
            .id_salt("browser-slides")
            .max_height(300.0)
            .show(ui, |ui| {
                for entry in &self.listing.slides {
                    let mut on = self.selected.contains(&entry.path);
                    let text = format!(
                        "{}  ({})",
                        entry.name,
                        platform::fmt_bytes(entry.size_bytes)
                    );
                    if ui
                        .checkbox(&mut on, icon_label(h3d_ui::icons::IMAGE, &text))
                        .changed()
                    {
                        if on {
                            self.selected.insert(entry.path.clone());
                        } else {
                            self.selected.remove(&entry.path);
                        }
                    }
                }
            });
        ui.label(t!(
            "browser-space-needed",
            size = platform::fmt_bytes(self.selected_bytes())
        ));
        ui.small(t!("browser-copy-note"));
        if icon_text_button(ui, h3d_ui::icons::IMPORT, &t!("slides-import")).clicked()
            && !self.selected.is_empty()
        {
            *action = BrowserAction::Import(self.selected.iter().cloned().collect());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_and_sizes() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.png"), [0u8; 100]).unwrap();
        std::fs::write(dir.path().join("b.png"), [0u8; 50]).unwrap();
        let mut b = BrowserState::in_dir(BrowserPurpose::ImportSlides, dir.path().to_path_buf());
        assert_eq!(b.listing.slides.len(), 2);
        assert_eq!(b.selected_bytes(), 0);
        b.selected.insert(dir.path().join("a.png"));
        b.selected.insert(dir.path().join("b.png"));
        assert_eq!(b.selected_bytes(), 150);
        std::fs::remove_file(dir.path().join("b.png")).unwrap();
        b.refresh();
        assert_eq!(b.selected.len(), 1);
        assert_eq!(b.selected_bytes(), 100);
        b.new_name = "Kasus 7".into();
        assert_eq!(b.new_project_path(), dir.path().join("Kasus_7.h3d"));
    }
}
