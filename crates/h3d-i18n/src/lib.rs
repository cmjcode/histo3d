//! Internasionalisasi (i18n) HISTO3D memakai Fluent (`fluent-bundle`).
//!
//! Bahasa bawaan aplikasi adalah **Bahasa Indonesia (`id-ID`)** (histo3d.md
//! 8.11); **English (`en-US`)** tersedia sebagai bahasa kedua. Kunci yang tidak
//! ditemukan di bahasa aktif dicari di bahasa Indonesia, lalu di Inggris; bila
//! tetap tidak ada, kuncinya sendiri yang dikembalikan agar mudah terlihat.
//!
//! Disalin dari `ducad-i18n` (histo3d.md 7.4) dengan bahasa bawaan dibalik.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::OnceLock;

use fluent_bundle::concurrent::FluentBundle;
use fluent_bundle::{FluentArgs, FluentResource, FluentValue};
use serde::{Deserialize, Serialize};
use unic_langid::{langid, LanguageIdentifier};

/// Bahasa yang didukung.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Language {
    /// Bahasa Indonesia (bawaan).
    #[default]
    Id,
    /// English.
    En,
}

impl Language {
    /// Kode BCP-47.
    pub const fn code(&self) -> &'static str {
        match self {
            Language::Id => "id-ID",
            Language::En => "en-US",
        }
    }

    /// Nama bahasa untuk pemilih di UI.
    pub const fn display_name(&self) -> &'static str {
        match self {
            Language::Id => "Bahasa Indonesia",
            Language::En => "English",
        }
    }

    /// Semua bahasa, urutan tetap (bawaan dahulu).
    pub const fn all() -> &'static [Language] {
        &[Language::Id, Language::En]
    }

    /// Dari indeks u8 (nilai lain -> bawaan).
    pub const fn from_u8(v: u8) -> Self {
        match v {
            1 => Language::En,
            _ => Language::Id,
        }
    }

    /// Ke indeks u8.
    pub const fn to_u8(&self) -> u8 {
        match self {
            Language::Id => 0,
            Language::En => 1,
        }
    }
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

static ACTIVE_LANGUAGE: AtomicU8 = AtomicU8::new(0);

/// Set bahasa aktif aplikasi (global, aman antar-thread).
pub fn set_language(lang: Language) {
    ACTIVE_LANGUAGE.store(lang.to_u8(), Ordering::SeqCst);
}

/// Bahasa aktif saat ini.
pub fn current_language() -> Language {
    Language::from_u8(ACTIVE_LANGUAGE.load(Ordering::SeqCst))
}

const ID_RESOURCE: &str = include_str!("../locales/id-ID/main.ftl");
const EN_RESOURCE: &str = include_str!("../locales/en-US/main.ftl");

struct LocaleStore {
    id_bundle: FluentBundle<FluentResource>,
    en_bundle: FluentBundle<FluentResource>,
}

static STORE: OnceLock<LocaleStore> = OnceLock::new();

/// Bangun satu bundle. Sumber `.ftl` tertanam saat kompilasi, jadi kegagalan
/// parse adalah bug program (bukan masukan luar) dan boleh panik.
fn build_bundle(lang: LanguageIdentifier, src: &str, what: &str) -> FluentBundle<FluentResource> {
    let mut bundle = FluentBundle::new_concurrent(vec![lang]);
    bundle.set_use_isolating(false);
    let res = match FluentResource::try_new(src.to_string()) {
        Ok(r) => r,
        Err((_, errs)) => panic!("berkas Fluent {what} tidak valid: {errs:?}"),
    };
    if let Err(errs) = bundle.add_resource(res) {
        panic!("gagal menambah sumber Fluent {what}: {errs:?}");
    }
    bundle
}

fn get_store() -> &'static LocaleStore {
    STORE.get_or_init(|| LocaleStore {
        id_bundle: build_bundle(langid!("id-ID"), ID_RESOURCE, "id-ID"),
        en_bundle: build_bundle(langid!("en-US"), EN_RESOURCE, "en-US"),
    })
}

impl LocaleStore {
    fn bundle(&self, lang: Language) -> &FluentBundle<FluentResource> {
        match lang {
            Language::Id => &self.id_bundle,
            Language::En => &self.en_bundle,
        }
    }
}

fn format_in(
    bundle: &FluentBundle<FluentResource>,
    key: &str,
    args: Option<&FluentArgs>,
) -> Option<String> {
    let msg = bundle.get_message(key)?;
    let pattern = msg.value()?;
    let mut errors = vec![];
    let value = bundle.format_pattern(pattern, args, &mut errors);
    if errors.is_empty() {
        Some(value.to_string())
    } else {
        log::warn!("i18n: kunci '{key}' gagal diformat: {errors:?}");
        None
    }
}

/// Terjemahkan `key` dalam bahasa `lang`; fallback ke Indonesia lalu Inggris;
/// kunci dikembalikan apa adanya bila tidak ada di mana pun.
pub fn translate_lang(lang: Language, key: &str, args: Option<&FluentArgs>) -> String {
    let store = get_store();
    if let Some(s) = format_in(store.bundle(lang), key, args) {
        return s;
    }
    for fallback in Language::all() {
        if *fallback != lang {
            if let Some(s) = format_in(store.bundle(*fallback), key, args) {
                return s;
            }
        }
    }
    log::warn!("i18n: kunci '{key}' tidak ditemukan");
    key.to_string()
}

/// Terjemahkan `key` dalam bahasa aktif.
pub fn translate(key: &str, args: Option<&FluentArgs>) -> String {
    translate_lang(current_language(), key, args)
}

/// Benar bila `key` ada di bahasa `lang` (tanpa fallback).
pub fn has_key(lang: Language, key: &str) -> bool {
    get_store().bundle(lang).has_message(key)
}

pub use fluent_bundle;

/// Bangun `FluentArgs` dari pasangan `(nama, nilai)`.
pub fn make_args<'a, I>(iter: I) -> FluentArgs<'a>
where
    I: IntoIterator<Item = (&'a str, FluentValue<'a>)>,
{
    let mut args = FluentArgs::new();
    for (k, v) in iter {
        args.set(k, v);
    }
    args
}

/// Ambil teks terjemahan dalam bahasa aktif.
///
/// ```
/// use h3d_i18n::t;
/// let text = t!("mode-slides");
/// let formatted = t!("warn-margin-close", dir = "anterior");
/// ```
#[macro_export]
macro_rules! t {
    ($key:expr) => {
        $crate::translate($key, None)
    };
    ($key:expr, $($name:ident = $val:expr),* $(,)?) => {{
        let mut args = $crate::fluent_bundle::FluentArgs::new();
        $(
            args.set(stringify!($name), $val);
        )*
        $crate::translate($key, Some(&args))
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluent_syntax::ast::Entry;

    static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn keys_of(src: &str) -> Vec<String> {
        let res = FluentResource::try_new(src.to_string()).unwrap();
        res.entries()
            .filter_map(|e| match e {
                Entry::Message(m) => Some(m.id.name.to_string()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn default_language_is_indonesian() {
        let _guard = TEST_LOCK.lock().unwrap();
        set_language(Language::default());
        assert_eq!(current_language(), Language::Id);
        assert_eq!(t!("mode-slides"), "Irisan");
        assert_eq!(Language::all()[0], Language::Id);
    }

    #[test]
    fn switch_to_english_and_back() {
        let _guard = TEST_LOCK.lock().unwrap();
        set_language(Language::En);
        assert_eq!(current_language(), Language::En);
        assert_eq!(t!("mode-slides"), "Slides");
        set_language(Language::Id);
        assert_eq!(t!("mode-slides"), "Irisan");
    }

    #[test]
    fn interpolation() {
        let _guard = TEST_LOCK.lock().unwrap();
        set_language(Language::Id);
        assert_eq!(
            t!("warn-margin-close", dir = "anterior"),
            "Tumor dekat tepi reseksi arah anterior"
        );
        set_language(Language::En);
        assert_eq!(
            t!("warn-margin-close", dir = "anterior"),
            "Tumor close to resection margin, direction anterior"
        );
        let args = make_args([("name", FluentValue::from("HER2"))]);
        assert_eq!(
            translate_lang(Language::Id, "warn-membrane-area-score", Some(&args)),
            "Skor HER2 dihitung per luas, bukan per sel"
        );
        set_language(Language::Id);
    }

    #[test]
    fn research_notice_matches_core_constant() {
        // Teks harus identik dengan `h3d_core::RESEARCH_ONLY_NOTICE` (aturan 16);
        // h3d-i18n tidak bergantung pada h3d-core, jadi literalnya disalin di sini.
        assert_eq!(
            translate_lang(Language::Id, "about-research-only", None),
            "Hanya untuk riset, bukan untuk diagnosis"
        );
    }

    #[test]
    fn unknown_key_returns_key() {
        assert_eq!(
            translate_lang(Language::En, "kunci-tidak-ada", None),
            "kunci-tidak-ada"
        );
    }

    #[test]
    fn key_parity() {
        let id_keys = keys_of(ID_RESOURCE);
        let en_keys = keys_of(EN_RESOURCE);
        assert!(id_keys.len() > 100, "berkas id-ID terlalu kecil");
        for k in &id_keys {
            assert!(has_key(Language::En, k), "en-US tidak punya kunci: {k}");
        }
        for k in &en_keys {
            assert!(has_key(Language::Id, k), "id-ID tidak punya kunci: {k}");
        }
    }

    #[test]
    fn label_kind_keys_exist() {
        for k in [
            "label-kind-tumor",
            "label-kind-necrosis",
            "label-kind-stroma",
            "label-kind-normal",
            "label-kind-vessel",
            "label-kind-nerve",
            "label-kind-capsule",
            "label-kind-margin",
            "label-kind-surface",
            "label-kind-reference",
            "label-kind-other",
        ] {
            assert!(has_key(Language::Id, k), "kunci hilang: {k}");
        }
    }
}
