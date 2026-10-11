//! Pengaturan pengguna (inisial untuk provenance, mode jari-menggambar,
//! bahasa, tema), disimpan sebagai JSON di folder data aplikasi.

use std::path::{Path, PathBuf};

use h3d_i18n::Language;
use h3d_ui::ThemeMode;
use serde::{Deserialize, Serialize};

/// Nama berkas pengaturan di dalam folder data.
pub const SETTINGS_FILE: &str = "settings.json";

/// Pengaturan pengguna.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// Inisial pengguna; dicatat sebagai `by` pada anotasi manual dan
    /// tinjauan (8.12.7).
    pub user_initials: String,
    /// "Jari menggambar" (8.11): tanpa Pencil, satu jari menggambar dan dua
    /// jari menggeser/zoom.
    pub finger_draw: bool,
    /// Kode bahasa: `"id"` atau `"en"`.
    pub language: String,
    /// Tema gelap.
    pub dark: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            user_initials: String::new(),
            finger_draw: false,
            language: "id".to_string(),
            dark: true,
        }
    }
}

impl Settings {
    /// Bahasa aktif; kode tak dikenal → Indonesia (bawaan).
    pub fn language(&self) -> Language {
        match self.language.as_str() {
            "en" => Language::En,
            _ => Language::Id,
        }
    }

    /// Setel bahasa.
    pub fn set_language(&mut self, lang: Language) {
        self.language = match lang {
            Language::En => "en",
            Language::Id => "id",
        }
        .to_string();
    }

    /// Tema aktif.
    pub fn theme(&self) -> ThemeMode {
        if self.dark {
            ThemeMode::Dark
        } else {
            ThemeMode::Light
        }
    }

    /// Setel tema.
    pub fn set_theme(&mut self, theme: ThemeMode) {
        self.dark = theme == ThemeMode::Dark;
    }

    /// Path berkas pengaturan bawaan.
    pub fn default_path() -> PathBuf {
        crate::platform::data_dir().join(SETTINGS_FILE)
    }

    /// Muat dari `path`. Berkas tidak ada atau rusak → pengaturan bawaan
    /// (berkas pengaturan adalah masukan luar: tidak pernah panik).
    pub fn load(path: &Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(text) => serde_json::from_str(&text).unwrap_or_else(|e| {
                log::warn!("pengaturan {path:?} rusak, memakai bawaan: {e}");
                Settings::default()
            }),
            Err(_) => Settings::default(),
        }
    }

    /// Simpan ke `path` secara atomik (`.tmp` lalu `rename`).
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let text = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, text)?;
        std::fs::rename(&tmp, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_and_corrupt_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sub").join(SETTINGS_FILE);
        assert_eq!(Settings::load(&path), Settings::default());

        let mut s = Settings {
            user_initials: "YJ".into(),
            finger_draw: true,
            ..Default::default()
        };
        s.set_language(Language::En);
        s.set_theme(ThemeMode::Light);
        s.save(&path).unwrap();
        let back = Settings::load(&path);
        assert_eq!(back, s);
        assert_eq!(back.language(), Language::En);
        assert_eq!(back.theme(), ThemeMode::Light);

        std::fs::write(&path, "{ bukan json").unwrap();
        assert_eq!(Settings::load(&path), Settings::default());
        // Field yang hilang memakai bawaan.
        std::fs::write(&path, r#"{"user_initials":"AB"}"#).unwrap();
        let partial = Settings::load(&path);
        assert_eq!(partial.user_initials, "AB");
        assert!(partial.dark);
    }
}
