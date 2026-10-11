//! Gambar laporan (8.12.8 butir 3): untuk hasil margin, irisan terdekat ke
//! titik tumor–tepi terdekat digambar pada ruangnya sendiri (U) dengan garis
//! luar label, kedua titik, dan panah dari tumor ke tepi.

use std::path::Path;

use h3d_core::{LabelKind, SlideId};
use h3d_io::{write_atomic, ReportFigure};
use h3d_volume::VoxelMask;

use super::Session;
use crate::error::EngineError;
use crate::figure::{Canvas, ARROW, HALO, MARGIN_MARKER, TUMOR_MARKER};

/// Sisi terpanjang gambar laporan, piksel.
pub const FIGURE_MAX_SIDE: u32 = 768;
/// Tebal garis luar label, piksel.
const FIGURE_OUTLINE_PX: i64 = 2;
/// Jari-jari cakram penanda titik, piksel.
pub const FIGURE_MARKER_RADIUS: f64 = 4.0;

/// Gambar margin yang sudah dirender beserta keterangannya.
#[derive(Clone, Debug, PartialEq)]
pub struct MarginFigure {
    /// Keterangan (nama berkas relatif, irisan, posisi titik, skala).
    pub meta: ReportFigure,
    /// Lebar, piksel.
    pub width: u32,
    /// Tinggi, piksel.
    pub height: u32,
    /// Piksel RGBA.
    pub rgba: Vec<u8>,
}

impl MarginFigure {
    /// Kode PNG.
    pub fn encode_png(&self) -> Result<Vec<u8>, EngineError> {
        Canvas::from_rgba(self.width, self.height, self.rgba.clone())
            .ok_or_else(|| EngineError::invalid("buffer gambar tidak konsisten"))?
            .encode_png()
    }
}

/// Benar bila masker memuat voxel di sekitar titik W `p` (µm): voxel yang
/// memuat `p` pada lapis z terdekat.
fn mask_contains(mask: &VoxelMask, p: [f64; 3]) -> bool {
    let g = &mask.grid;
    let Some([i, j]) = g.um_to_index([p[0], p[1]]) else {
        return false;
    };
    let mut best: Option<(f64, usize)> = None;
    for (k, z) in g.z_um.iter().enumerate() {
        let d = (z - p[2]).abs();
        if best.is_none_or(|(bd, _)| d < bd) {
            best = Some((d, k));
        }
    }
    best.is_some_and(|(_, k)| mask.get(i as usize, j as usize, k) != 0)
}

impl Session {
    /// Gambar margin untuk keadaan proyek saat ini (8.12.8 butir 3). Kosong
    /// bila analisis spasial belum dijalankan (atau sudah basi) atau tidak
    /// ada hasil margin. Tidak menulis berkas.
    ///
    /// Irisan yang digambar = irisan berlabel yang `z`-nya terdekat ke titik
    /// tumor terdekat (seri → urutan lebih awal). Kedua titik dipetakan dari
    /// ruang W ke ruang U irisan itu dengan `T_i⁻¹` (arah mundur), lalu ke
    /// piksel thumbnail. Bila berkas slide tidak bisa dibaca, latar gambar
    /// putih dan hanya garis luar label serta penanda yang tampil.
    pub fn margin_figures(&self) -> Result<Vec<MarginFigure>, EngineError> {
        let data = self.slice_data()?;
        if data.is_empty() {
            return Ok(Vec::new());
        }
        let key = self.recon_key(&data)?;
        let tre = self.tre_estimate_um();
        let Some(analysis) = self.current_analysis(key, tre)? else {
            return Ok(Vec::new());
        };
        let Some(margin) = &analysis.margin else {
            return Ok(Vec::new());
        };
        let (pt, pm) = (margin.nearest_tumor_um, margin.nearest_margin_um);

        // Label tepi reseksi yang memuat titik tepi terdekat.
        let margin_defs: Vec<_> = self
            .project
            .labels
            .iter()
            .filter(|l| l.kind == LabelKind::Margin)
            .collect();
        let recon = self.recon_for_key(key);
        let label_name = margin_defs
            .iter()
            .find(|l| {
                recon.as_ref().is_some_and(|r| {
                    r.masks
                        .iter()
                        .any(|(id, m)| *id == l.id && mask_contains(m, pm))
                })
            })
            .or(margin_defs.first())
            .map(|l| l.name.clone())
            .unwrap_or_else(|| "margin".to_owned());

        // Irisan berlabel terdekat pada z ke titik tumor.
        let mut best: Option<(f64, usize)> = None;
        for (i, d) in data.iter().enumerate() {
            let dz = (d.z_um - pt[2]).abs();
            if best.is_none_or(|(b, _)| dz < b) {
                best = Some((dz, i));
            }
        }
        let Some((_, bi)) = best else {
            return Ok(Vec::new());
        };
        let slice = &data[bi];
        let entry = self.slide(slice.id)?;
        let size_um = [
            f64::from(entry.size_l0[0]) * entry.mpp[0],
            f64::from(entry.size_l0[1]) * entry.mpp[1],
        ];
        if !(size_um[0] > 0.0 && size_um[1] > 0.0) {
            return Ok(Vec::new());
        }

        // Latar: thumbnail irisan pada ruangnya sendiri; putih bila berkas
        // slide tidak bisa dibaca (hilang atau berubah).
        let mut canvas = match self.thumbnail(slice.id, FIGURE_MAX_SIDE) {
            Ok(img) => {
                // Gambar kecil (slide beresolusi rendah) diperbesar dengan
                // faktor bulat, tetangga terdekat, agar penanda tidak
                // menutupi jaringan.
                let (w, h) = (img.width(), img.height());
                let k = (FIGURE_MAX_SIDE / w.max(h).max(1)).max(1);
                if k == 1 {
                    Canvas::from_rgba(w, h, img.into_raw())
                } else {
                    let raw = img.as_raw();
                    let (bw, bh) = (w * k, h * k);
                    let mut big = Vec::with_capacity(bw as usize * bh as usize * 4);
                    for y in 0..bh {
                        for x in 0..bw {
                            let i = ((y / k) as usize * w as usize + (x / k) as usize) * 4;
                            big.extend_from_slice(&raw[i..i + 4]);
                        }
                    }
                    Canvas::from_rgba(bw, bh, big)
                }
            }
            Err(e) => {
                log::warn!(
                    "gambar margin: thumbnail irisan {} tidak tersedia ({e}); latar putih dipakai",
                    slice.id
                );
                None
            }
        }
        .unwrap_or_else(|| {
            let long = size_um[0].max(size_um[1]);
            let side = |v: f64| ((v / long * f64::from(FIGURE_MAX_SIDE)).round() as u32).max(1);
            Canvas::new(side(size_um[0]), side(size_um[1]), [255, 255, 255, 255])
        });
        let (w, h) = (canvas.width(), canvas.height());
        let um_per_px = [size_um[0] / f64::from(w), size_um[1] / f64::from(h)];

        // Garis luar label: piksel berlabel yang, dalam `FIGURE_OUTLINE_PX`
        // piksel pada sumbu x atau y, bertemu label lain. Label dicuplik di
        // pusat piksel gambar (+0,5).
        let (wu, hu) = (w as usize, h as usize);
        let mut labels = vec![0u8; wu * hu];
        for j in 0..hu {
            for i in 0..wu {
                let u = [
                    (i as f64 + 0.5) * um_per_px[0],
                    (j as f64 + 0.5) * um_per_px[1],
                ];
                labels[j * wu + i] = slice.store.get_um(u);
            }
        }
        for j in 0..hu {
            for i in 0..wu {
                let l = labels[j * wu + i];
                if l == 0 {
                    continue;
                }
                let differs = |x: i64, y: i64| -> bool {
                    if x < 0 || y < 0 || x >= wu as i64 || y >= hu as i64 {
                        return true;
                    }
                    labels[y as usize * wu + x as usize] != l
                };
                let (x, y) = (i as i64, j as i64);
                // Tebal dua piksel (ke arah dalam) agar tetap terlihat saat
                // gambar ditampilkan lebih kecil.
                let edge = (1..=FIGURE_OUTLINE_PX).any(|d| {
                    differs(x - d, y) || differs(x + d, y) || differs(x, y - d) || differs(x, y + d)
                });
                if edge {
                    if let Some(def) = self.project.label(h3d_core::LabelId(l)) {
                        let c = def.color;
                        canvas.put(x, y, [c[0], c[1], c[2], 255]);
                    }
                }
            }
        }

        // Titik W → U irisan ini (T⁻¹) → piksel gambar.
        let to_px = |p: [f64; 3]| -> Option<[f64; 2]> {
            let u = slice.transform.apply_inverse([p[0], p[1]])?;
            let q = [u[0] / um_per_px[0], u[1] / um_per_px[1]];
            (q[0].is_finite() && q[1].is_finite()).then_some(q)
        };
        let (Some(tp), Some(mp)) = (to_px(pt), to_px(pm)) else {
            return Ok(Vec::new());
        };
        let r = FIGURE_MARKER_RADIUS;
        // Panah tumor → tepi (lingkar terang di bawahnya agar kontras).
        canvas.arrow(tp, mp, 5.0, 12.0, HALO);
        canvas.arrow(tp, mp, 2.0, 12.0, ARROW);
        // Penanda: tepi dulu (cakram + cincin), lalu tumor di atasnya.
        canvas.ring(mp, r + 3.0, r + 5.0, HALO);
        canvas.ring(mp, r + 1.5, r + 3.0, MARGIN_MARKER);
        canvas.disc(mp, r + 1.5, HALO);
        canvas.disc(mp, r, MARGIN_MARKER);
        canvas.disc(tp, r + 1.5, HALO);
        canvas.disc(tp, r, TUMOR_MARKER);

        let du = [
            (mp[0] - tp[0]) * um_per_px[0],
            (mp[1] - tp[1]) * um_per_px[1],
        ];
        let stem = h3d_io::meshconv::file_stem_of(&label_name, "margin");
        let meta = ReportFigure {
            file: format!("figures/margin_{stem}.png"),
            label: label_name,
            slide: slice.id,
            slide_file: entry.source.file_name(),
            slide_z_um: slice.z_um,
            size_px: [w, h],
            um_per_px,
            tumor_px: tp,
            margin_px: mp,
            in_plane_um: (du[0] * du[0] + du[1] * du[1]).sqrt(),
            dz_um: pm[2] - pt[2],
        };
        Ok(vec![MarginFigure {
            meta,
            width: w,
            height: h,
            rgba: canvas.into_rgba(),
        }])
    }

    /// Tulis gambar laporan ke `<dir>/figures/` (atomik) dan kembalikan
    /// keterangannya; `ReportFigure::file` relatif terhadap `dir`, tempat
    /// `report.md` berada.
    pub fn write_report_figures(&self, dir: &Path) -> Result<Vec<ReportFigure>, EngineError> {
        let mut out = Vec::new();
        for fig in self.margin_figures()? {
            write_atomic(&dir.join(&fig.meta.file), &fig.encode_png()?)?;
            out.push(fig.meta);
        }
        Ok(out)
    }

    /// Irisan yang digambar pada gambar margin, bila ada (untuk GUI).
    pub fn margin_figure_slide(&self) -> Result<Option<SlideId>, EngineError> {
        Ok(self.margin_figures()?.first().map(|f| f.meta.slide))
    }
}
