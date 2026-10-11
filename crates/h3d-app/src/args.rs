//! Argumen baris perintah `histo3d` (masukan luar: tidak pernah panik).

use std::path::PathBuf;

use h3d_ui::AppMode;

/// Teks bantuan.
pub const USAGE: &str = "Pemakaian: histo3d [--project <folder.h3d>] [--mode slides|register|segment|reconstruct|report] \
[--slide <nomor urut>] [--analyze] [--screenshot <berkas.png>] [--lang id|en] [--theme dark|light]";

/// Argumen yang sudah diurai.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CliArgs {
    /// Proyek yang dibuka saat mulai.
    pub project: Option<PathBuf>,
    /// Mode awal.
    pub mode: Option<AppMode>,
    /// Simpan tangkapan layar ke berkas ini lalu keluar.
    pub screenshot: Option<PathBuf>,
    /// Bahasa awal (`id`/`en`).
    pub lang: Option<h3d_i18n::Language>,
    /// Tema awal.
    pub theme: Option<h3d_ui::ThemeMode>,
    /// Irisan aktif awal (nomor urut 1-basis dalam tumpukan).
    pub slide: Option<usize>,
    /// Jalankan analisis spasial saat mulai (mengisi lapisan skalar mesh).
    pub analyze: bool,
    /// Hanya cetak bantuan.
    pub help: bool,
}

/// Nama mode di baris perintah → `AppMode`.
pub fn parse_mode(name: &str) -> Option<AppMode> {
    match name.to_ascii_lowercase().as_str() {
        "slides" | "irisan" => Some(AppMode::Slides),
        "register" | "registrasi" => Some(AppMode::Register),
        "segment" | "segmentasi" => Some(AppMode::Segment),
        "reconstruct" | "rekonstruksi" => Some(AppMode::Reconstruct),
        "report" | "laporan" => Some(AppMode::Report),
        _ => None,
    }
}

impl CliArgs {
    /// Urai argumen (tanpa nama program). Galat berupa pesan untuk pengguna.
    pub fn parse<I: IntoIterator<Item = String>>(args: I) -> Result<Self, String> {
        let mut out = CliArgs::default();
        let mut it = args.into_iter();
        while let Some(a) = it.next() {
            let mut value = |name: &str| {
                it.next()
                    .ok_or_else(|| format!("argumen {name} membutuhkan nilai"))
            };
            match a.as_str() {
                "-h" | "--help" => out.help = true,
                "--analyze" => out.analyze = true,
                "--slide" => {
                    let v = value("--slide")?;
                    let n: usize = v
                        .parse()
                        .ok()
                        .filter(|n| *n >= 1)
                        .ok_or_else(|| format!("nomor irisan tidak sah: {v}"))?;
                    out.slide = Some(n);
                }
                "--project" => out.project = Some(PathBuf::from(value("--project")?)),
                "--screenshot" => out.screenshot = Some(PathBuf::from(value("--screenshot")?)),
                "--mode" => {
                    let v = value("--mode")?;
                    out.mode =
                        Some(parse_mode(&v).ok_or_else(|| format!("mode tidak dikenal: {v}"))?);
                }
                "--lang" => {
                    let v = value("--lang")?;
                    out.lang = Some(match v.as_str() {
                        "id" => h3d_i18n::Language::Id,
                        "en" => h3d_i18n::Language::En,
                        _ => return Err(format!("bahasa tidak dikenal: {v}")),
                    });
                }
                "--theme" => {
                    let v = value("--theme")?;
                    out.theme = Some(match v.as_str() {
                        "dark" => h3d_ui::ThemeMode::Dark,
                        "light" => h3d_ui::ThemeMode::Light,
                        _ => return Err(format!("tema tidak dikenal: {v}")),
                    });
                }
                other => return Err(format!("argumen tidak dikenal: {other}")),
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(v: &[&str]) -> Result<CliArgs, String> {
        CliArgs::parse(v.iter().map(|s| s.to_string()))
    }

    #[test]
    fn parses_all_flags() {
        let a = parse(&[
            "--project",
            "/tmp/x.h3d",
            "--mode",
            "segment",
            "--screenshot",
            "a.png",
            "--lang",
            "en",
            "--theme",
            "light",
            "--slide",
            "3",
            "--analyze",
        ])
        .unwrap();
        assert_eq!(a.project, Some(PathBuf::from("/tmp/x.h3d")));
        assert_eq!(a.mode, Some(AppMode::Segment));
        assert_eq!(a.screenshot, Some(PathBuf::from("a.png")));
        assert_eq!(a.lang, Some(h3d_i18n::Language::En));
        assert_eq!(a.theme, Some(h3d_ui::ThemeMode::Light));
        assert_eq!(a.slide, Some(3));
        assert!(a.analyze);
    }

    #[test]
    fn bad_input_is_error_not_panic() {
        assert!(parse(&["--mode"]).is_err());
        assert!(parse(&["--mode", "xyz"]).is_err());
        assert!(parse(&["--wat"]).is_err());
        assert!(parse(&["--slide", "0"]).is_err());
        assert!(parse(&["--slide", "x"]).is_err());
        assert!(parse(&[]).unwrap().project.is_none());
    }
}
