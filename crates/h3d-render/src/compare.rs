//! Pembanding registrasi: dua lapis tile (tetap dan bergerak), masing-masing
//! dengan affine `T_i` (ruang U irisan → ruang W, µm); matriks pandangan
//! memetakan W µm → NDC. Mode: tembus pandang, papan catur, selisih warna.

use crate::tiles::TileLayerDraw;

/// Mode pembanding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CompareMode {
    /// Lapisan bergerak digambar di atas lapisan tetap dengan opasitas `alpha`.
    Blend {
        /// Opasitas lapisan bergerak 0..1.
        alpha: f32,
    },
    /// Kotak bergantian tetap/bergerak berukuran `size_px` point layar.
    Checkerboard {
        /// Ukuran kotak (point egui).
        size_px: f32,
    },
    /// Selisih warna: tetap = magenta, bergerak = hijau, sejajar = abu-abu.
    /// Dipakai *kegelapan* (1 − luminansi) di atas latar hitam, sehingga
    /// jaringan terwarnai dan latar kaca kosong tetap hitam.
    Difference,
}

/// Parameter pembanding registrasi.
#[derive(Clone, Debug, PartialEq)]
pub struct CompareParams {
    /// Irisan tetap (acuan), `affine` = `T_fixed`.
    pub fixed: TileLayerDraw,
    /// Irisan bergerak, `affine` = `T_moving` (hasil registrasi yang sedang dinilai).
    pub moving: TileLayerDraw,
    /// Mode tampilan.
    pub mode: CompareMode,
}

/// Mode fragmen lapisan tile; dikirim ke shader di `Layer.viewport.w`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum LayerMode {
    /// Gambar biasa.
    Normal = 0,
    /// Papan catur: hanya kotak dengan paritas genap.
    CheckerEven = 1,
    /// Papan catur: hanya kotak dengan paritas ganjil.
    CheckerOdd = 2,
    /// Kegelapan → magenta, opak (lapisan tetap pada mode selisih).
    TintFixed = 3,
    /// Kegelapan → hijau, aditif (lapisan bergerak pada mode selisih).
    TintMoving = 4,
    /// Kotak hitam penuh viewport (latar mode selisih).
    SolidBlack = 5,
}

/// Lapisan mana yang digambar pada satu pass pembanding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompareSource {
    /// Latar hitam (tanpa tile).
    Black,
    /// Lapisan tetap.
    Fixed,
    /// Lapisan bergerak.
    Moving,
}

/// Satu pass gambar hasil penguraian `CompareMode`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ComparePass {
    /// Sumber tile.
    pub source: CompareSource,
    /// Mode fragmen.
    pub mode: LayerMode,
    /// Pengali opasitas.
    pub opacity: f32,
    /// Ukuran kotak papan catur (point), 0 bila tidak dipakai.
    pub checker_px: f32,
}

/// Urai mode pembanding menjadi urutan pass gambar (murni, dapat diuji).
pub fn compare_passes(
    mode: CompareMode,
    fixed_opacity: f32,
    moving_opacity: f32,
) -> Vec<ComparePass> {
    match mode {
        CompareMode::Blend { alpha } => vec![
            ComparePass {
                source: CompareSource::Fixed,
                mode: LayerMode::Normal,
                opacity: fixed_opacity,
                checker_px: 0.0,
            },
            ComparePass {
                source: CompareSource::Moving,
                mode: LayerMode::Normal,
                opacity: moving_opacity * alpha.clamp(0.0, 1.0),
                checker_px: 0.0,
            },
        ],
        CompareMode::Checkerboard { size_px } => {
            let size = if size_px.is_finite() {
                size_px.max(1.0)
            } else {
                64.0
            };
            vec![
                ComparePass {
                    source: CompareSource::Fixed,
                    mode: LayerMode::CheckerEven,
                    opacity: fixed_opacity,
                    checker_px: size,
                },
                ComparePass {
                    source: CompareSource::Moving,
                    mode: LayerMode::CheckerOdd,
                    opacity: moving_opacity,
                    checker_px: size,
                },
            ]
        }
        CompareMode::Difference => vec![
            ComparePass {
                source: CompareSource::Black,
                mode: LayerMode::SolidBlack,
                opacity: 1.0,
                checker_px: 0.0,
            },
            ComparePass {
                source: CompareSource::Fixed,
                mode: LayerMode::TintFixed,
                opacity: fixed_opacity,
                checker_px: 0.0,
            },
            ComparePass {
                source: CompareSource::Moving,
                mode: LayerMode::TintMoving,
                opacity: moving_opacity,
                checker_px: 0.0,
            },
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_scales_moving_opacity() {
        let p = compare_passes(CompareMode::Blend { alpha: 0.25 }, 1.0, 0.8);
        assert_eq!(p.len(), 2);
        assert_eq!(p[0].source, CompareSource::Fixed);
        assert_eq!(p[1].source, CompareSource::Moving);
        assert!((p[1].opacity - 0.2).abs() < 1e-6);
    }

    #[test]
    fn checkerboard_alternates_parity_and_clamps_size() {
        let p = compare_passes(CompareMode::Checkerboard { size_px: 0.0 }, 1.0, 1.0);
        assert_eq!(p[0].mode, LayerMode::CheckerEven);
        assert_eq!(p[1].mode, LayerMode::CheckerOdd);
        assert!(p.iter().all(|x| x.checker_px >= 1.0));
    }

    #[test]
    fn difference_starts_with_black_then_tints() {
        let p = compare_passes(CompareMode::Difference, 1.0, 1.0);
        assert_eq!(p[0].mode, LayerMode::SolidBlack);
        assert_eq!(p[1].mode, LayerMode::TintFixed);
        assert_eq!(p[2].mode, LayerMode::TintMoving);
    }
}
