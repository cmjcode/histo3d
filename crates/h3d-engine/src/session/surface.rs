//! Hiasan tampilan mesh setelah rekonstruksi (histo3d.md 8.10): oklusi
//! ambien dan warna pulasan per verteks. Hanya tampilan/ekspor; tidak pernah
//! masukan pengukuran atau analisis spasial (aturan keras 18–19).
//!
//! Warna diambil dari thumbnail bermasker jaringan (alfa 0 di kaca/latar)
//! yang sudah di-cache di `cache/thumbs/`. Tiap thumbnail dipotong ke kotak
//! kisi rekonstruksi (dipetakan W → U dengan `T_i⁻¹`) agar memorinya kecil;
//! total potongan dibatasi `Budgets::recon_working_set_bytes`.

use h3d_core::{Progress, SlideTransform};
use h3d_volume::{sample_vertex_colors, ColorSampler, ColorSlice, GridPlan, LabelMesh};

use super::recon::SliceData;
use super::Session;

/// Sisi terpanjang thumbnail bermasker jaringan untuk warna pulasan, px.
const COLOR_THUMB_SIDE: u32 = 1024;
/// Jumlah sinar oklusi ambien per verteks.
const AO_DIRS: usize = 32;
/// Jangkauan sinar oklusi relatif terhadap diagonal kotak pembatas mesh.
const AO_REACH: f64 = 0.25;
/// Alfa masker jaringan minimum agar piksel dianggap jaringan.
const TISSUE_ALPHA_MIN: u8 = 128;

/// Potongan thumbnail RGBA dalam piksel thumbnail `[x0, x0 + w) × [y0, y0 + h)`.
/// Piksel `(i, j)` menutup `[i, i + 1) × um_per_px` di ruang U (pusat +0,5).
struct ThumbCrop {
    rgba: Vec<u8>,
    w: usize,
    h: usize,
    x0: usize,
    y0: usize,
    um_per_px: [f64; 2],
}

impl ColorSampler for ThumbCrop {
    fn rgb_at_um(&self, u: [f64; 2]) -> Option<[u8; 3]> {
        let fx = (u[0] / self.um_per_px[0]).floor();
        let fy = (u[1] / self.um_per_px[1]).floor();
        if !fx.is_finite() || !fy.is_finite() || fx < self.x0 as f64 || fy < self.y0 as f64 {
            return None;
        }
        let (xi, yi) = (fx as usize - self.x0, fy as usize - self.y0);
        if xi >= self.w || yi >= self.h {
            return None;
        }
        let o = 4 * (yi * self.w + xi);
        let px = self.rgba.get(o..o + 4)?;
        (px[3] >= TISSUE_ALPHA_MIN).then(|| [px[0], px[1], px[2]])
    }
}

impl Session {
    /// Bakar oklusi ambien tiap mesh dan, bila `ReconConfig::stain_colors`,
    /// warnai verteks dari irisan `data`. Kegagalan membaca citra hanya
    /// dicatat (warna adalah hiasan, bukan hasil).
    pub(crate) fn decorate_meshes(
        &self,
        meshes: &mut [LabelMesh],
        data: &[SliceData<'_>],
        plan: &GridPlan,
        p: &dyn Progress,
    ) {
        p.set(0.0, "oklusi ambien");
        for m in meshes.iter_mut() {
            if let Some((lo, hi)) = m.bbox() {
                let d = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
                let diag = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                m.bake_occlusion(AO_DIRS, AO_REACH * diag);
            }
        }
        if !self.project.recon.stain_colors || meshes.is_empty() {
            return;
        }
        p.set(0.5, "warna pulasan");
        let crops = self.color_crops(data, plan);
        if crops.is_empty() {
            return;
        }
        let slices: Vec<ColorSlice<'_>> = crops
            .iter()
            .map(|(z, t, c)| ColorSlice {
                z_um: *z,
                transform: t,
                sampler: c,
            })
            .collect();
        for m in meshes.iter_mut() {
            m.colors = sample_vertex_colors(m, &slices, plan.grid.s_xy);
        }
        p.set(1.0, "warna pulasan selesai");
    }

    /// Potongan thumbnail bermasker per irisan yang menutup kisi `plan`.
    fn color_crops(
        &self,
        data: &[SliceData<'_>],
        plan: &GridPlan,
    ) -> Vec<(f64, SlideTransform, ThumbCrop)> {
        let g = &plan.grid;
        let w0 = g.origin_um;
        let w1 = [
            w0[0] + f64::from(g.size[0]) * g.s_xy,
            w0[1] + f64::from(g.size[1]) * g.s_xy,
        ];
        let corners = [
            [w0[0], w0[1]],
            [w1[0], w0[1]],
            [w0[0], w1[1]],
            [w1[0], w1[1]],
        ];
        let mut out = Vec::new();
        let mut bytes: u64 = 0;
        for d in data {
            let Ok(entry) = self.slide(d.id) else {
                continue;
            };
            // mpp tidak pernah ditebak: irisan tanpa mpp sah tidak diwarnai.
            if !entry.has_valid_mpp() || entry.size_l0[0] == 0 || entry.size_l0[1] == 0 {
                continue;
            }
            let img = match self.thumbnail_tissue(d.id, COLOR_THUMB_SIDE) {
                Ok(i) => i,
                Err(e) => {
                    log::warn!("warna pulasan: thumbnail irisan {} gagal: {e}", d.id);
                    continue;
                }
            };
            let (tw, th) = (img.width() as usize, img.height() as usize);
            if tw == 0 || th == 0 {
                continue;
            }
            let um = [
                f64::from(entry.size_l0[0]) * entry.mpp[0] / tw as f64,
                f64::from(entry.size_l0[1]) * entry.mpp[1] / th as f64,
            ];
            let mut lo = [f64::INFINITY; 2];
            let mut hi = [f64::NEG_INFINITY; 2];
            let mut ok = true;
            for c in corners {
                match d.transform.apply_inverse(c) {
                    Some(u) => {
                        lo = [lo[0].min(u[0]), lo[1].min(u[1])];
                        hi = [hi[0].max(u[0]), hi[1].max(u[1])];
                    }
                    None => ok = false,
                }
            }
            if !ok {
                continue;
            }
            let x0 = (lo[0] / um[0]).floor().clamp(0.0, tw as f64) as usize;
            let y0 = (lo[1] / um[1]).floor().clamp(0.0, th as f64) as usize;
            let x1 = (hi[0] / um[0]).ceil().clamp(0.0, tw as f64) as usize;
            let y1 = (hi[1] / um[1]).ceil().clamp(0.0, th as f64) as usize;
            if x1 <= x0 || y1 <= y0 {
                continue;
            }
            let (w, h) = (x1 - x0, y1 - y0);
            bytes += (w * h * 4) as u64;
            if bytes > self.budgets.recon_working_set_bytes {
                log::warn!("warna pulasan dihentikan: potongan thumbnail melampaui anggaran");
                break;
            }
            let raw = img.as_raw();
            let mut rgba = Vec::with_capacity(w * h * 4);
            for y in y0..y1 {
                let s = (y * tw + x0) * 4;
                rgba.extend_from_slice(&raw[s..s + w * 4]);
            }
            out.push((
                d.z_um,
                d.transform.clone(),
                ThumbCrop {
                    rgba,
                    w,
                    h,
                    x0,
                    y0,
                    um_per_px: um,
                },
            ));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_sampler_respects_offset_pixel_cells_and_tissue_alpha() {
        // Potongan 2 × 1 mulai di piksel (3, 5); 10 µm/px.
        let c = ThumbCrop {
            rgba: vec![10, 20, 30, 255, 40, 50, 60, 0],
            w: 2,
            h: 1,
            x0: 3,
            y0: 5,
            um_per_px: [10.0, 10.0],
        };
        assert_eq!(c.rgb_at_um([30.0, 50.0]), Some([10, 20, 30]));
        assert_eq!(c.rgb_at_um([39.9, 59.9]), Some([10, 20, 30]));
        // Piksel kedua: latar (alfa 0).
        assert_eq!(c.rgb_at_um([45.0, 55.0]), None);
        assert_eq!(c.rgb_at_um([29.9, 55.0]), None);
        assert_eq!(c.rgb_at_um([35.0, 60.0]), None);
        assert_eq!(c.rgb_at_um([f64::NAN, 55.0]), None);
    }
}
