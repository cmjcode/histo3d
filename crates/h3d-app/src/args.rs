//! Argumen baris perintah `histo3d` (masukan luar: tidak pernah panik).

use std::path::PathBuf;

use crate::state::{Drawer, View};

/// Teks bantuan.
pub const USAGE: &str = "Pemakaian: histo3d [--project <folder.h3d>] [--view 3d|2d] \
[--drawer layers|sample|labels|align|tumor3d] [--report] [--slide <nomor urut>] [--analyze] \
[--screenshot <berkas.png>] [--lang id|en] [--theme dark|light] [--look technical|tissue]";

/// Argumen yang sudah diurai.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CliArgs {
    /// Proyek yang dibuka saat mulai.
    pub project: Option<PathBuf>,
    /// Tampilan awal (tumpukan 3D atau irisan 2D).
    pub view: Option<View>,
    /// Laci yang dibuka saat mulai.
    pub drawer: Option<Drawer>,
    /// Buka lembar laporan saat mulai.
    pub report: bool,
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
    /// Gaya tampilan 3D awal.
    pub look: Option<crate::features::reconstruct::ViewLook>,
    /// Hanya cetak bantuan.
    pub help: bool,
}

/// Nama tampilan di baris perintah → `View`.
pub fn parse_view(name: &str) -> Option<View> {
    match name.to_ascii_lowercase().as_str() {
        "3d" | "stack" | "tumpukan" => Some(View::Stack3D),
        "2d" | "slice" | "irisan" => Some(View::Slice2D),
        _ => None,
    }
}

/// Nama laci di baris perintah → `Drawer`.
pub fn parse_drawer(name: &str) -> Option<Drawer> {
    match name.to_ascii_lowercase().as_str() {
        "layers" | "lapisan" => Some(Drawer::Layers),
        "sample" | "sampel" => Some(Drawer::Sample),
        "labels" | "label" => Some(Drawer::Labels),
        "align" | "sejajar" => Some(Drawer::Align),
        "tumor3d" | "tumor" => Some(Drawer::Tumor3D),
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
                "--report" => out.report = true,
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
                "--view" => {
                    let v = value("--view")?;
                    out.view =
                        Some(parse_view(&v).ok_or_else(|| format!("tampilan tidak dikenal: {v}"))?);
                }
                "--drawer" => {
                    let v = value("--drawer")?;
                    out.drawer =
                        Some(parse_drawer(&v).ok_or_else(|| format!("laci tidak dikenal: {v}"))?);
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
                "--look" => {
                    use crate::features::reconstruct::ViewLook;
                    let v = value("--look")?;
                    out.look = Some(match v.to_ascii_lowercase().as_str() {
                        "technical" | "teknis" => ViewLook::Technical,
                        "tissue" | "jaringan" => ViewLook::Tissue,
                        _ => return Err(format!("gaya tampilan tidak dikenal: {v}")),
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
            "--view",
            "2d",
            "--drawer",
            "labels",
            "--report",
            "--screenshot",
            "a.png",
            "--lang",
            "en",
            "--theme",
            "light",
            "--slide",
            "3",
            "--analyze",
            "--look",
            "jaringan",
        ])
        .unwrap();
        assert_eq!(a.look, Some(crate::features::reconstruct::ViewLook::Tissue));
        assert!(parse(&["--look", "mewah"]).is_err());

        assert_eq!(a.project, Some(PathBuf::from("/tmp/x.h3d")));
        assert_eq!(a.view, Some(View::Slice2D));
        assert_eq!(a.drawer, Some(Drawer::Labels));
        assert!(a.report);
        assert_eq!(a.screenshot, Some(PathBuf::from("a.png")));
        assert_eq!(a.lang, Some(h3d_i18n::Language::En));
        assert_eq!(a.theme, Some(h3d_ui::ThemeMode::Light));
        assert_eq!(a.slide, Some(3));
        assert!(a.analyze);
        assert_eq!(parse_view("tumpukan"), Some(View::Stack3D));
        assert_eq!(parse_drawer("tumor"), Some(Drawer::Tumor3D));
    }

    #[test]
    fn bad_input_is_error_not_panic() {
        assert!(parse(&["--view"]).is_err());
        assert!(parse(&["--view", "xyz"]).is_err());
        assert!(parse(&["--drawer", "xyz"]).is_err());
        assert!(
            parse(&["--mode", "slides"]).is_err(),
            "flag lama tidak dikenal"
        );
        assert!(parse(&["--wat"]).is_err());
        assert!(parse(&["--slide", "0"]).is_err());
        assert!(parse(&["--slide", "x"]).is_err());
        assert!(parse(&[]).unwrap().project.is_none());
    }
}
