//! Komponen UI bersama HISTO3D (histo3d.md 8.11), bebas dari `h3d-engine`
//! dan crate rendering; hanya bergantung pada `egui`, `egui_icons`,
//! `h3d-core`, dan `h3d-i18n`.
//!
//! - [`theme`]: mode terang/gelap, token warna, frame panel mengambang ala
//!   DUCAD, dan pemasangan font ikon (`theme::apply`).
//! - [`icons`]: ikon bernama dari `egui_icons::icons::ICON_*` (aturan keras 13:
//!   tidak ada glyph Unicode/emoji sebagai ikon).
//! - [`widgets`]: tombol ikon, tombol alat persegi, kepala laci, slider
//!   berlabel, pemberitahuan "hanya untuk riset", spanduk peringatan,
//!   legenda colormap, swatch warna.
//! - [`rail`]: rel alat vertikal mengambang (generik atas pengenal alat).
//! - [`viewcube`]: kubus orientasi kamera 3D.
//!
//! Teks UI selalu lewat `h3d_i18n::t!`; cakupan glyph string UI dijaga tes
//! `tests/ui_glyph_coverage.rs`.

pub mod icons;
pub mod rail;
pub mod theme;
pub mod viewcube;
pub mod widgets;

pub use rail::{show_rail, RailItem};
pub use theme::ThemeMode;
pub use viewcube::{ViewCube, ViewCubeAction};
