//! Riwayat undo/redo tingkat proyek (8.8): perubahan manifes yang dipicu
//! pengguna (koreksi manual, urutan, status, acuan, z, jarak) direkam
//! sebagai potret `Project` sebelum/sesudah sehingga bisa dibatalkan persis.
//! Suntingan label punya riwayat sendiri per irisan (`labels.rs`).

use h3d_core::{Command, Project, UndoStack};

use super::Session;
use crate::error::EngineError;

/// Batas entri riwayat proyek; manifes kecil (puluhan KB) jadi 100 potret
/// masih ringan.
pub(crate) const PROJECT_HISTORY_LIMIT: usize = 100;

/// Satu perubahan manifes: potret sebelum dan sesudah. Nama adalah kunci
/// i18n (`undo-*`) agar GUI bisa menampilkan "Urungkan: …".
struct ProjectSnapshot {
    name: &'static str,
    before: Project,
    after: Project,
}

impl Command<Project> for ProjectSnapshot {
    fn apply(&mut self, state: &mut Project) {
        *state = self.after.clone();
    }
    fn revert(&mut self, state: &mut Project) {
        *state = self.before.clone();
    }
    fn name(&self) -> &str {
        self.name
    }
}

/// Riwayat proyek: tumpukan perintah atas `Project`.
pub(crate) struct ProjectHistory {
    stack: UndoStack<Project>,
}

impl Default for ProjectHistory {
    fn default() -> Self {
        Self {
            stack: UndoStack::new(PROJECT_HISTORY_LIMIT),
        }
    }
}

impl Session {
    /// Jalankan `f` sebagai satu langkah ber-undo bernama `name` (kunci
    /// i18n `undo-*`). Bila `f` gagal, manifes dikembalikan ke keadaan
    /// semula dan tidak ada entri riwayat. Bila manifes tidak berubah,
    /// tidak ada entri juga (mis. status yang sama disetel ulang).
    pub(crate) fn recorded<T>(
        &mut self,
        name: &'static str,
        f: impl FnOnce(&mut Self) -> Result<T, EngineError>,
    ) -> Result<T, EngineError> {
        let before = self.project.clone();
        let dirty_before = self.project_dirty;
        match f(self) {
            Ok(v) => {
                if self.project != before {
                    let after = self.project.clone();
                    let cmd = ProjectSnapshot {
                        name,
                        before,
                        after,
                    };
                    // `push` memanggil `apply` yang menyalin `after` ke
                    // manifes: identik dengan keadaan sekarang.
                    self.history.stack.push(Box::new(cmd), &mut self.project);
                }
                Ok(v)
            }
            Err(e) => {
                self.project = before;
                self.project_dirty = dirty_before;
                Err(e)
            }
        }
    }

    /// Ada langkah proyek yang bisa dibatalkan.
    pub fn can_undo_project(&self) -> bool {
        self.history.stack.can_undo()
    }

    /// Ada langkah proyek yang bisa diulang.
    pub fn can_redo_project(&self) -> bool {
        self.history.stack.can_redo()
    }

    /// Kunci i18n langkah yang akan dibatalkan berikutnya.
    pub fn project_undo_name(&self) -> Option<&str> {
        self.history.stack.undo_name()
    }

    /// Batalkan langkah proyek terakhir; `Some(kunci)` bila ada. Manifes
    /// langsung disimpan (atomik) karena langkah aslinya pun tersimpan.
    pub fn undo_project(&mut self) -> Result<Option<String>, EngineError> {
        let name = self.history.stack.undo(&mut self.project);
        self.after_history_step(name)
    }

    /// Ulangi langkah proyek yang dibatalkan; `Some(kunci)` bila ada.
    pub fn redo_project(&mut self) -> Result<Option<String>, EngineError> {
        let name = self.history.stack.redo(&mut self.project);
        self.after_history_step(name)
    }

    /// Kosongkan riwayat proyek (mis. setelah operasi yang tidak bisa
    /// dibatalkan seperti hapus irisan atau ganti mpp).
    pub(crate) fn clear_project_history(&mut self) {
        self.history.stack.clear();
    }

    fn after_history_step(&mut self, name: Option<String>) -> Result<Option<String>, EngineError> {
        if name.is_some() {
            self.project_dirty = true;
            self.save_project()?;
        }
        Ok(name)
    }
}
