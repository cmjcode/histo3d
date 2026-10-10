//! Penjaga "tofu": setiap karakter non-ASCII di string UI harus punya glyph di
//! font proporsional aplikasi. Karakter yang tidak ada di font manapun
//! digambar egui sebagai kotak kosong. Pakai `egui_icons::icons::ICON_*`
//! untuk ikon (aturan keras 13), dan hanya glyph yang lolos tes ini untuk teks.
//!
//! `Fonts::has_glyph` tidak bisa dipakai: di epaint 0.36 ia memberi negatif
//! palsu untuk semua glyph milik face yang juga memegang karakter pengganti
//! (NotoEmoji). Jadi tes ini membandingkan UV glyph hasil layout dengan UV
//! karakter pengganti. Disalin dari `ducad-ui/tests/ui_glyph_coverage.rs`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

type Uv = ([u16; 2], [u16; 2]);

fn glyph_uv(ctx: &egui::Context, c: char, font: &egui::FontId) -> Option<Uv> {
    let galley =
        ctx.fonts_mut(|f| f.layout_no_wrap(c.to_string(), font.clone(), egui::Color32::WHITE));
    let glyph = galley.rows.first()?.glyphs.first()?;
    Some((glyph.uv_rect.min, glyph.uv_rect.max))
}

/// Karakter non-ASCII di dalam literal string Rust, di luar komentar `//` dan
/// di atas `#[cfg(test)]` (pesan assert tidak tampil di UI).
fn rust_literal_chars(src: &str) -> Vec<(usize, char)> {
    let mut out = Vec::new();
    for (lineno, line) in src.lines().enumerate() {
        if line.trim_start().starts_with("#[cfg(test)]") {
            break;
        }
        let mut in_str = false;
        let mut escaped = false;
        let mut prev = '\0';
        for c in line.chars() {
            if in_str {
                if escaped {
                    escaped = false;
                } else if c == '\\' {
                    escaped = true;
                } else if c == '"' {
                    in_str = false;
                } else if !c.is_ascii() {
                    out.push((lineno + 1, c));
                }
            } else if c == '"' {
                in_str = true;
            } else if c == '/' && prev == '/' {
                break;
            }
            prev = c;
        }
    }
    out
}

/// Karakter non-ASCII di nilai pesan Fluent (baris `#` = komentar).
fn ftl_chars(src: &str) -> Vec<(usize, char)> {
    src.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim_start().starts_with('#'))
        .flat_map(|(i, l)| l.chars().filter(|c| !c.is_ascii()).map(move |c| (i + 1, c)))
        .collect()
}

fn collect(dir: &Path, ext: &str, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, ext, out);
        } else if path.extension().is_some_and(|e| e == ext) {
            out.push(path);
        }
    }
}

#[test]
fn ui_strings_have_glyphs() {
    let ctx = egui::Context::default();
    h3d_ui::theme::apply(&ctx, h3d_ui::theme::ThemeMode::Dark);
    let mut warmup = ctx.run_ui(Default::default(), |_| {});
    warmup.textures_delta.clear();

    let font = egui::FontId::proportional(12.0);
    let replacement = glyph_uv(&ctx, '\u{10FFFD}', &font);

    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut files = Vec::new();
    collect(&crates.join("h3d-ui/src"), "rs", &mut files);
    collect(&crates.join("h3d-app/src"), "rs", &mut files);
    collect(&crates.join("h3d-i18n/locales"), "ftl", &mut files);
    assert!(!files.is_empty(), "sumber UI tidak ditemukan di {crates:?}");

    // Karakter hilang -> lokasi pertama yang memakainya.
    let mut missing: BTreeMap<char, String> = BTreeMap::new();
    for path in &files {
        let Ok(src) = std::fs::read_to_string(path) else {
            continue;
        };
        let chars = if path.extension().is_some_and(|e| e == "ftl") {
            ftl_chars(&src)
        } else {
            rust_literal_chars(&src)
        };
        for (line, c) in chars {
            // Pemilih variasi / ZWJ / private-use (codepoint ikon Material) tidak dicek.
            if matches!(
                c,
                '\u{FE0E}' | '\u{FE0F}' | '\u{200D}' | '\u{E000}'..='\u{F8FF}'
            ) || ('\u{F0000}'..='\u{10FFFF}').contains(&c)
                || missing.contains_key(&c)
            {
                continue;
            }
            if glyph_uv(&ctx, c, &font) == replacement {
                let rel = path.strip_prefix(&crates).unwrap_or(path);
                missing.insert(c, format!("{}:{line}", rel.display()));
            }
        }
    }

    let report: Vec<String> = missing
        .iter()
        .map(|(c, at)| format!("  '{c}' (U+{:04X}) di {at}", *c as u32))
        .collect();
    assert!(
        missing.is_empty(),
        "glyph berikut tidak ada di font UI dan akan tampil sebagai kotak; \
         ganti dengan egui_icons::icons::ICON_* atau glyph yang tersedia:\n{}",
        report.join("\n")
    );
}

/// Ikon bernama h3d-ui harus terender (bukan karakter pengganti) setelah
/// `theme::apply` memasang font `egui_icons`.
#[test]
fn named_icons_render() {
    let ctx = egui::Context::default();
    h3d_ui::theme::apply(&ctx, h3d_ui::theme::ThemeMode::Dark);
    let mut warmup = ctx.run_ui(Default::default(), |_| {});
    warmup.textures_delta.clear();

    let icons = [
        h3d_ui::icons::CLOSE,
        h3d_ui::icons::OPEN_FOLDER,
        h3d_ui::icons::SAVE,
        h3d_ui::icons::IMPORT,
        h3d_ui::icons::EXPORT,
        h3d_ui::icons::UNDO,
        h3d_ui::icons::REDO,
        h3d_ui::icons::PLAY,
        h3d_ui::icons::SETTINGS,
        h3d_ui::icons::WARNING,
        h3d_ui::icons::CHECK,
        h3d_ui::icons::BRUSH,
        h3d_ui::icons::ERASER,
        h3d_ui::icons::LASSO,
        h3d_ui::icons::POLYGON,
        h3d_ui::icons::ZOOM_IN,
        h3d_ui::icons::ZOOM_OUT,
        h3d_ui::icons::VISIBILITY_ON,
        h3d_ui::icons::VISIBILITY_OFF,
        h3d_ui::icons::LAYERS,
        h3d_ui::icons::CUBE,
        h3d_ui::icons::REPORT,
        h3d_ui::icons::GRID,
        h3d_ui::icons::COMPARE,
        h3d_ui::icons::DRAW,
        h3d_ui::icons::FLIP,
        h3d_ui::icons::ROTATE,
        h3d_ui::icons::INFO,
    ];
    let mut bad = Vec::new();
    for icon in icons {
        let font = egui::FontId::new(16.0, icon.font_family());
        let replacement = glyph_uv(&ctx, '\u{10FFFD}', &font);
        let Some(c) = icon.codepoint.chars().next() else {
            bad.push(format!("codepoint kosong: {icon:?}"));
            continue;
        };
        if glyph_uv(&ctx, c, &font) == replacement {
            bad.push(format!("U+{:04X}", c as u32));
        }
    }
    assert!(bad.is_empty(), "ikon tidak terender: {bad:?}");
}
