//! Ekspor (8.7.3). Tanpa `with_clinical`, tidak ada isi `clinical.json`
//! yang keluar (aturan keras 17).

use std::path::{Path, PathBuf};

use h3d_core::{Colormap, LabelId, NoProgress};
use h3d_io::{
    encode_analysis, encode_f32_nifti, encode_glb, encode_label_nifti, encode_obj, encode_stl,
    export_slide_geojson, render_csv, render_markdown, slide_stem, stl_file_name, write_atomic,
    ClinicalScope, GlbMesh, GlbScalar,
};
use h3d_volume::LabelMesh;

use super::{Session, APP_VERSION};
use crate::error::EngineError;

/// Apa yang diekspor.
#[derive(Clone, Debug, PartialEq)]
pub enum ExportKind {
    /// `mesh.glb`: mesh per label; `scalar` memilih medan untuk `COLOR_0`
    /// (`margin_distance`, `vessel_distance`, atau nama atribut).
    Glb {
        /// Nama skalar; `None` = warna label.
        scalar: Option<String>,
    },
    /// Satu `<label>_mm.stl` per label.
    Stl,
    /// `mesh_mm.obj`: semua label, satu grup per label.
    Obj,
    /// `labels.nii.gz` + satu `.nii.gz` `f32` per atribut dan medan jarak.
    Nifti,
    /// Satu `NNNN.geojson` per irisan berlabel.
    GeoJson,
    /// `report.md` + `report.csv` + `figures/margin_<label>.png` (bila ada
    /// hasil margin). `with_clinical` juga menyalin `clinical.json` dan
    /// mencetak konteks klinis lengkap.
    Report {
        /// Sertakan data klinis.
        with_clinical: bool,
    },
    /// `analysis.json`.
    Analysis,
}

fn file_stem(name: &str, fallback: &str) -> String {
    h3d_io::meshconv::file_stem_of(name, fallback)
}

impl Session {
    /// Ekspor ke folder `out` (8.8). Lihat [`Self::export_files`].
    pub fn export(&self, what: ExportKind, out: &Path) -> Result<(), EngineError> {
        self.export_files(what, out).map(|_| ())
    }

    fn meshes_or_err(&self) -> Result<Vec<LabelMesh>, EngineError> {
        let meshes = self.meshes()?;
        if meshes.iter().all(|m| m.indices.is_empty()) {
            return Err(EngineError::invalid(
                "belum ada mesh untuk keadaan proyek saat ini; jalankan rekonstruksi dulu",
            ));
        }
        Ok(meshes)
    }

    fn label_name(&self, id: LabelId) -> String {
        self.project
            .label(id)
            .map(|l| l.name.clone())
            .unwrap_or_else(|| format!("label{}", id.0))
    }

    /// Ekspor ke folder `out` dan kembalikan berkas yang ditulis (terurut).
    /// Semua berkas ditulis atomik.
    pub fn export_files(&self, what: ExportKind, out: &Path) -> Result<Vec<PathBuf>, EngineError> {
        let mut written: Vec<PathBuf> = Vec::new();
        let mut put = |name: &str, bytes: &[u8]| -> Result<(), EngineError> {
            let path = out.join(name);
            write_atomic(&path, bytes)?;
            written.push(path);
            Ok(())
        };
        match what {
            ExportKind::Glb { scalar } => {
                let meshes = self.meshes_or_err()?;
                let names: Vec<String> = meshes.iter().map(|m| self.label_name(m.label)).collect();
                let items: Vec<GlbMesh<'_>> = meshes
                    .iter()
                    .zip(names.iter())
                    .map(|(m, n)| GlbMesh {
                        name: n,
                        color: self
                            .project
                            .label(m.label)
                            .map_or([200, 200, 200, 255], |l| l.color),
                        mesh: m,
                    })
                    .collect();
                // Nama skalar: persis, atau nama atribut (`attr:<nama>`).
                let resolved = match &scalar {
                    None => None,
                    Some(want) => {
                        let candidates = [want.clone(), format!("attr:{want}")];
                        let found = candidates.into_iter().find(|c| {
                            meshes
                                .iter()
                                .any(|m| m.scalars.iter().any(|s| s.name.eq_ignore_ascii_case(c)))
                        });
                        let name = found.ok_or_else(|| {
                            let mut have: Vec<String> = meshes
                                .iter()
                                .flat_map(|m| m.scalars.iter().map(|s| s.name.clone()))
                                .collect();
                            have.sort();
                            have.dedup();
                            EngineError::invalid(format!(
                                "skalar \"{want}\" tidak ada pada mesh (tersedia: {}); \
                                 jalankan analisis spasial dulu",
                                if have.is_empty() {
                                    "tidak ada".to_owned()
                                } else {
                                    have.join(", ")
                                }
                            ))
                        })?;
                        // Ejaan persis seperti tersimpan di mesh.
                        meshes
                            .iter()
                            .flat_map(|m| m.scalars.iter())
                            .find(|s| s.name.eq_ignore_ascii_case(&name))
                            .map(|s| s.name.clone())
                    }
                };
                let attr = resolved.as_ref().and_then(|n| {
                    n.strip_prefix("attr:")
                        .and_then(|a| self.project.attrs.iter().find(|d| d.name == a))
                });
                let sc = resolved.as_ref().map(|n| GlbScalar {
                    name: n,
                    colormap: attr.map_or(Colormap::Viridis, |a| a.colormap),
                    range: attr.and_then(|a| a.range),
                });
                put("mesh.glb", &encode_glb(&items, sc.as_ref())?)?;
            }
            ExportKind::Stl => {
                for m in self.meshes_or_err()? {
                    if m.indices.is_empty() {
                        continue;
                    }
                    let name = stl_file_name(&self.label_name(m.label), m.label.0);
                    put(&name, &encode_stl(&m)?)?;
                }
            }
            ExportKind::Obj => {
                let meshes = self.meshes_or_err()?;
                let names: Vec<String> = meshes.iter().map(|m| self.label_name(m.label)).collect();
                let items: Vec<(&str, &LabelMesh)> = names
                    .iter()
                    .map(String::as_str)
                    .zip(meshes.iter())
                    .collect();
                put("mesh_mm.obj", encode_obj(&items)?.as_bytes())?;
            }
            ExportKind::Nifti => {
                let data = self.slice_data()?;
                let recon = if data.is_empty() {
                    None
                } else {
                    self.recon_for_key(self.recon_key(&data)?)
                };
                let recon = recon.ok_or_else(|| {
                    EngineError::invalid(
                        "belum ada volume untuk keadaan proyek saat ini; jalankan rekonstruksi dulu",
                    )
                })?;
                let masks: Vec<(u8, &h3d_volume::VoxelMask)> =
                    recon.masks.iter().map(|(l, m)| (l.0, m)).collect();
                let mut notes: Vec<String> = Vec::new();
                let o = encode_label_nifti(&masks)?;
                put("labels.nii.gz", &o.gz)?;
                notes.extend(o.notes);
                // Volume atribut dan medan jarak (f32).
                let analysis =
                    self.compute_analysis(&recon, self.tre_estimate_um(), &NoProgress)?;
                for (def, vol) in &analysis.attr_volumes {
                    let o = encode_f32_nifti(&vol.grid, &vol.data)?;
                    put(&format!("{}.nii.gz", def.file_stem()), &o.gz)?;
                    notes.extend(o.notes);
                }
                for (name, _, field) in &analysis.analysis.fields {
                    if name.starts_with("attr:") {
                        continue;
                    }
                    let o = encode_f32_nifti(&analysis.field_grid, field)?;
                    put(&format!("{}.nii.gz", file_stem(name, "field")), &o.gz)?;
                    notes.extend(o.notes);
                }
                notes.sort();
                notes.dedup();
                if !notes.is_empty() {
                    let mut text = notes.join("\n");
                    text.push('\n');
                    put("nifti_catatan.txt", text.as_bytes())?;
                }
            }
            ExportKind::GeoJson => {
                let attr_rows = self.attr_reports()?;
                for (id, store) in self.label_snapshot()? {
                    let Some(entry) = self.project.slide(id) else {
                        continue;
                    };
                    let present = store.labels_present();
                    let named: Vec<(LabelId, String)> = present
                        .iter()
                        .filter_map(|l| self.project.label(*l).map(|d| (*l, d.name.clone())))
                        .collect();
                    if named.is_empty() {
                        continue;
                    }
                    let labels: Vec<(LabelId, &str)> =
                        named.iter().map(|(l, n)| (*l, n.as_str())).collect();
                    // Rata-rata atribut irisan ini dilampirkan ke tiap kontur
                    // label jenis Tumor (rata-rata intratumoral bila ada).
                    let means: Vec<(LabelId, Vec<(String, f64)>)> = named
                        .iter()
                        .map(|(l, _)| {
                            let is_tumor = self
                                .project
                                .label(*l)
                                .is_some_and(|d| d.kind == h3d_core::LabelKind::Tumor);
                            let list = attr_rows
                                .iter()
                                .filter_map(|a| {
                                    let row = a.per_slide.iter().find(|r| r.slide == id)?;
                                    let v = if is_tumor {
                                        row.intratumoral.unwrap_or(row.mean)
                                    } else {
                                        return None;
                                    };
                                    Some((a.name.clone(), v))
                                })
                                .collect();
                            (*l, list)
                        })
                        .collect();
                    let text = export_slide_geojson(&store, &labels, entry.mpp, &means);
                    put(&format!("{}.geojson", slide_stem(id)), text.as_bytes())?;
                }
                if written.is_empty() {
                    return Err(EngineError::invalid(
                        "tidak ada irisan berlabel untuk diekspor",
                    ));
                }
            }
            ExportKind::Report { with_clinical } => {
                let mut report = self.measure()?;
                // Gambar margin (8.12.8 butir 3) ditulis di samping laporan
                // dan dirujuk dengan tautan relatif.
                for fig in self.margin_figures()? {
                    put(&fig.meta.file, &fig.encode_png()?)?;
                    report.figures.push(fig.meta);
                }
                let scope = match (&self.project.sample.clinical, with_clinical) {
                    (Some(c), true) => ClinicalScope::Full(c),
                    _ => ClinicalScope::Omit,
                };
                put("report.md", render_markdown(&report, scope).as_bytes())?;
                put("report.csv", render_csv(&report, scope).as_bytes())?;
                if with_clinical {
                    if let Some(c) = &self.project.sample.clinical {
                        let mut bytes = serde_json::to_vec_pretty(c).map_err(|e| {
                            h3d_io::IoError::json(out.join(h3d_io::bundle::CLINICAL_FILE), e)
                        })?;
                        bytes.push(b'\n');
                        put(h3d_io::bundle::CLINICAL_FILE, &bytes)?;
                    }
                }
            }
            ExportKind::Analysis => {
                let data = self.slice_data()?;
                let analysis = if data.is_empty() {
                    None
                } else {
                    let key = self.recon_key(&data)?;
                    let tre = self.tre_estimate_um();
                    match self.current_analysis(key, tre)? {
                        Some(a) => Some((a, self.analysis_key(key, tre)?)),
                        None => None,
                    }
                };
                let (a, akey) = analysis.ok_or_else(|| {
                    EngineError::invalid(
                        "analisis spasial belum dijalankan untuk keadaan proyek saat ini; \
                         jalankan analisis dulu",
                    )
                })?;
                put(
                    "analysis.json",
                    encode_analysis(&a, APP_VERSION, akey)?.as_bytes(),
                )?;
            }
        }
        written.sort();
        Ok(written)
    }
}
