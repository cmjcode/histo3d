//! Impor slide (8.8, H1).

use std::cmp::Ordering;
use std::path::{Path, PathBuf};

use h3d_core::{OrderSource, Progress, SlideEntry, SlideId, SlideSourceRef, SlideStatus};

use super::Session;
use crate::error::EngineError;

/// Opsi impor.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ImportOptions {
    /// mpp isian pengguna `[x, y]`, dipakai HANYA bila metadata berkas tidak
    /// memuatnya. Tanpa keduanya impor gagal dengan `MppMissing`.
    pub mpp: Option<[f64; 2]>,
    /// Salin berkas ke dalam bundel (`slides/`); wajib di iPad.
    pub embed: bool,
    /// Jarak irisan untuk `z_um` bawaan; `None` = nilai proyek.
    pub spacing_um: Option<f64>,
}

/// Pembanding "urutan alami": deret angka dibandingkan sebagai bilangan
/// (`s2` < `s10`), sisanya per karakter tanpa peka huruf besar/kecil. Seri
/// dipecah dengan pembanding byte agar deterministik.
pub fn natural_cmp(a: &str, b: &str) -> Ordering {
    let (ab, bb) = (a.as_bytes(), b.as_bytes());
    let (mut i, mut j) = (0usize, 0usize);
    while i < ab.len() && j < bb.len() {
        if ab[i].is_ascii_digit() && bb[j].is_ascii_digit() {
            let si = i;
            while i < ab.len() && ab[i].is_ascii_digit() {
                i += 1;
            }
            let sj = j;
            while j < bb.len() && bb[j].is_ascii_digit() {
                j += 1;
            }
            let da = a[si..i].trim_start_matches('0');
            let db = b[sj..j].trim_start_matches('0');
            let ord = da.len().cmp(&db.len()).then_with(|| da.cmp(db));
            if ord != Ordering::Equal {
                return ord;
            }
        } else {
            let ca = ab[i].to_ascii_lowercase();
            let cb = bb[j].to_ascii_lowercase();
            if ca != cb {
                return ca.cmp(&cb);
            }
            i += 1;
            j += 1;
        }
    }
    (ab.len() - i).cmp(&(bb.len() - j)).then_with(|| a.cmp(b))
}

fn file_name_of(p: &Path) -> String {
    p.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.to_string_lossy().into_owned())
}

impl Session {
    /// Impor slide (8.8). mpp diambil dari metadata; bila tidak ada dipakai
    /// argumen `mpp`; bila keduanya tidak ada → [`EngineError::MppMissing`]
    /// (tidak pernah ditebak). Urutan = urutan alami nama berkas, ditaruh
    /// setelah irisan yang sudah ada. `z_um = urutan × jarak` dengan
    /// `z_is_default = true`. Berkas dirujuk sebagai `External` (path relatif
    /// + sidik jari).
    pub fn import_slides(
        &mut self,
        paths: &[PathBuf],
        mpp: Option<[f64; 2]>,
        p: &dyn Progress,
    ) -> Result<Vec<SlideId>, EngineError> {
        self.import_slides_with(
            paths,
            &ImportOptions {
                mpp,
                ..Default::default()
            },
            p,
        )
    }

    /// Impor dengan opsi lengkap (`--embed`, `--spacing-um`). Semua berkas
    /// diperiksa dulu; bila satu saja gagal, proyek tidak berubah.
    pub fn import_slides_with(
        &mut self,
        paths: &[PathBuf],
        opts: &ImportOptions,
        p: &dyn Progress,
    ) -> Result<Vec<SlideId>, EngineError> {
        if let Some(m) = opts.mpp {
            if !m.iter().all(|v| v.is_finite() && *v > 0.0) {
                return Err(EngineError::invalid(format!(
                    "mpp tidak sah: {m:?} (harus positif)"
                )));
            }
        }
        if let Some(s) = opts.spacing_um {
            if !(s.is_finite() && s > 0.0) {
                return Err(EngineError::invalid(format!("jarak irisan tidak sah: {s}")));
            }
        }
        let mut sorted: Vec<&PathBuf> = paths.iter().collect();
        sorted
            .sort_by(|a, b| natural_cmp(&file_name_of(a), &file_name_of(b)).then_with(|| a.cmp(b)));
        sorted.dedup();

        // Tahap 1: baca metadata semua berkas tanpa mengubah proyek.
        struct Pending {
            path: PathBuf,
            mpp: [f64; 2],
            size_l0: [u32; 2],
            vendor: Option<String>,
            size_bytes: u64,
            fingerprint: u64,
        }
        let mut pending: Vec<Pending> = Vec::with_capacity(sorted.len());
        let total = sorted.len().max(1) as f32;
        for (i, path) in sorted.iter().enumerate() {
            if p.is_cancelled() {
                return Err(EngineError::Cancelled);
            }
            p.set(i as f32 / total * 0.8, "membaca metadata slide");
            let src = h3d_slide::open_slide(path)?;
            let info = src.info();
            let mpp = match (info.mpp, opts.mpp) {
                (Some(m), _) if m.iter().all(|v| v.is_finite() && *v > 0.0) => m,
                (_, Some(m)) => m,
                _ => {
                    return Err(EngineError::MppMissing {
                        file: file_name_of(path),
                    })
                }
            };
            let size_l0 = info.level0_size();
            if size_l0[0] == 0 || size_l0[1] == 0 {
                return Err(EngineError::invalid(format!(
                    "slide {} berukuran nol",
                    file_name_of(path)
                )));
            }
            let (size_bytes, fingerprint) = h3d_slide::fingerprint(path)?;
            // Berkas `External` yang sama tidak boleh masuk dua kali.
            let rel = self.bundle.make_relative(path);
            let duplicate = self.project.slides.iter().any(|s| {
                matches!(&s.source, SlideSourceRef::External { path: existing, .. } if *existing == rel)
            });
            if duplicate && !opts.embed {
                return Err(EngineError::invalid(format!(
                    "slide {} sudah ada di proyek",
                    file_name_of(path)
                )));
            }
            pending.push(Pending {
                path: (*path).clone(),
                mpp,
                size_l0,
                vendor: info.vendor.clone(),
                size_bytes,
                fingerprint,
            });
        }

        // Tahap 2: masukkan ke proyek.
        let previous_spacing = self.project.recon.default_spacing_um;
        if let Some(s) = opts.spacing_um {
            self.project.recon.default_spacing_um = s;
        }
        let spacing = self.project.recon.default_spacing_um;
        let first_order = self
            .project
            .slides
            .iter()
            .map(|s| s.order + 1)
            .max()
            .unwrap_or(0);
        // Bila ada yang gagal (atau dibatalkan) di tengah jalan, semua
        // irisan dari panggilan ini ditarik kembali: proyek tidak berubah.
        let original_len = self.project.slides.len();
        let mut ids = Vec::with_capacity(pending.len());
        let mut failure: Option<EngineError> = None;
        for (i, item) in pending.into_iter().enumerate() {
            if p.is_cancelled() {
                failure = Some(EngineError::Cancelled);
                break;
            }
            p.set(0.8 + i as f32 / total * 0.2, "menambahkan irisan");
            let next_order = first_order + i as u32;
            let id = self.project.next_slide_id();
            let source = if opts.embed {
                match self.bundle.embed_slide(&item.path, id) {
                    Ok(s) => s,
                    Err(e) => {
                        failure = Some(e.into());
                        break;
                    }
                }
            } else {
                SlideSourceRef::External {
                    path: self.bundle.make_relative(&item.path),
                    size_bytes: item.size_bytes,
                    fingerprint: item.fingerprint,
                }
            };
            let mut entry = SlideEntry {
                id,
                source,
                order: next_order,
                order_source: OrderSource::FileName,
                z_um: f64::from(next_order) * spacing,
                z_is_default: true,
                mpp: item.mpp,
                size_l0: item.size_l0,
                status: SlideStatus::Ok,
                ..Default::default()
            };
            entry.preanalytic.scanner = item.vendor;
            self.project.slides.push(entry);
            ids.push(id);
        }
        if let Some(e) = failure {
            for s in self.project.slides.drain(original_len..) {
                if let SlideSourceRef::Embedded { rel_path } = &s.source {
                    let _ = std::fs::remove_file(self.bundle.root.join(rel_path));
                }
            }
            self.project.recon.default_spacing_um = previous_spacing;
            return Err(e);
        }
        // Irisan lama ber-`z` bawaan ikut jarak baru; isian tangan tidak disentuh.
        self.project.recompute_default_z(spacing);
        self.drop_stale_registrations();
        self.recompose_chain();
        self.save_project()?;
        p.set(1.0, "impor selesai");
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::natural_cmp;
    use std::cmp::Ordering;

    #[test]
    fn natural_order() {
        let mut v = vec!["s10.png", "s2.png", "S1.png", "s02.png", "a.png", "s2a.png"];
        v.sort_by(|a, b| natural_cmp(a, b));
        assert_eq!(
            v,
            ["a.png", "S1.png", "s02.png", "s2.png", "s2a.png", "s10.png"]
        );
        assert_eq!(natural_cmp("x", "x"), Ordering::Equal);
        assert_eq!(natural_cmp("slice_009", "slice_010"), Ordering::Less);
    }
}
