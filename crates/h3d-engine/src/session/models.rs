//! Model segmentasi AI di bundel proyek: `models/<nama>/model.json` +
//! `model.onnx` (histo3d.md 9.2). Menyediakan daftar model terpasang,
//! pemasangan model dari folder luar (salinan atomik, aturan keras 10), dan
//! penyiapan label untuk kelas model agar hasil deteksi tidak hilang diam-diam
//! (kelas tanpa label padanan menjadi latar di `infer_slide`).
//!
//! Tidak ada jaringan: model selalu dipilih pengguna dari sistem berkas.

use std::fs;
use std::path::{Path, PathBuf};

use h3d_core::LabelKind;
use h3d_io::atomic::copy_atomic;
use h3d_segment::{ModelSpec, TractSegmenter};

use super::Session;
use crate::error::EngineError;

/// Nama berkas spesifikasi model.
pub const MODEL_SPEC_FILE: &str = "model.json";
/// Nama berkas bobot ONNX.
pub const MODEL_ONNX_FILE: &str = "model.onnx";

/// Satu folder model di `models/` bundel.
#[derive(Clone, Debug)]
pub struct ModelEntry {
    /// Folder model (`models/<folder>`).
    pub dir: PathBuf,
    /// Nama folder; dipakai sebagai nama tampil cadangan.
    pub folder: String,
    /// Spesifikasi bila `model.json` sah; pesan galat bila tidak.
    pub spec: Result<ModelSpec, String>,
}

impl ModelEntry {
    /// Nama tampil: `nama versi` dari spesifikasi, atau nama folder.
    pub fn display_name(&self) -> String {
        match &self.spec {
            Ok(s) => match &s.version {
                Some(v) if !v.is_empty() => format!("{} {}", s.name, v),
                _ => s.name.clone(),
            },
            Err(_) => self.folder.clone(),
        }
    }

    /// Nama kelas (tanpa latar), urut indeks.
    pub fn class_names(&self) -> Vec<String> {
        let Ok(spec) = &self.spec else {
            return Vec::new();
        };
        let mut classes: Vec<_> = spec
            .classes
            .iter()
            .filter_map(|c| c.label.as_ref().map(|l| (c.index, l.clone())))
            .collect();
        classes.sort_by_key(|(i, _)| *i);
        classes.into_iter().map(|(_, l)| l).collect()
    }
}

/// Jenis label yang ditebak dari nama kelas model (hanya untuk label baru;
/// pengguna bisa mengubahnya di panel label).
pub fn kind_for_class(name: &str) -> LabelKind {
    let n = name.to_ascii_lowercase();
    if n.contains("tumor") || n.contains("karsinoma") || n.contains("carcinoma") {
        LabelKind::Tumor
    } else if n.contains("nekrosis") || n.contains("necrosis") {
        LabelKind::Necrosis
    } else if n.contains("stroma") {
        LabelKind::Stroma
    } else if n.contains("pembuluh") || n.contains("vessel") {
        LabelKind::Vessel(h3d_core::VesselKind::Blood)
    } else if n.contains("saraf") || n.contains("nerve") {
        LabelKind::Nerve
    } else if n.contains("jaringan") || n.contains("tissue") || n.contains("normal") {
        LabelKind::Normal
    } else {
        LabelKind::Other
    }
}

/// Palet warna untuk label baru buatan model (dipilih yang belum dipakai).
const NEW_LABEL_COLORS: [[u8; 4]; 8] = [
    [220, 40, 40, 255],
    [60, 160, 60, 255],
    [80, 120, 220, 255],
    [240, 200, 40, 255],
    [230, 60, 160, 255],
    [140, 100, 40, 255],
    [120, 80, 200, 255],
    [40, 190, 200, 255],
];

/// Hasil penyiapan label untuk sebuah model.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModelLabelSetup {
    /// Nama label yang baru dibuat.
    pub created: Vec<String>,
    /// Nama kelas yang sudah punya label.
    pub existing: Vec<String>,
}

fn validate_model_dir(dir: &Path) -> Result<ModelSpec, EngineError> {
    let spec_path = dir.join(MODEL_SPEC_FILE);
    let text = fs::read_to_string(&spec_path).map_err(|e| h3d_io::IoError::io(&spec_path, e))?;
    let spec = ModelSpec::from_json(&text).map_err(EngineError::Seg)?;
    if !dir.join(MODEL_ONNX_FILE).is_file() {
        return Err(EngineError::invalid(format!(
            "folder model tidak memuat {MODEL_ONNX_FILE}: {}",
            dir.display()
        )));
    }
    Ok(spec)
}

impl Session {
    /// Daftar model di `models/` bundel, urut nama folder. Folder dengan
    /// `model.json` rusak tetap dicantumkan dengan pesan galatnya.
    pub fn list_models(&self) -> Vec<ModelEntry> {
        let dir = self.bundle.models_dir();
        let Ok(rd) = fs::read_dir(&dir) else {
            return Vec::new();
        };
        let mut out: Vec<ModelEntry> = rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.join(MODEL_SPEC_FILE).is_file())
            .map(|p| {
                let folder = p
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                let spec = validate_model_dir(&p).map_err(|e| e.to_string());
                ModelEntry {
                    dir: p,
                    folder,
                    spec,
                }
            })
            .collect();
        out.sort_by(|a, b| a.folder.cmp(&b.folder));
        out
    }

    /// Pasang model dari folder luar (`src` = folder berisi `model.json` +
    /// `model.onnx`, atau path `model.json`-nya) ke `models/<nama folder>`.
    /// Model divalidasi dan dimuat lewat tract sebelum disalin; salinan ke
    /// folder sementara lalu `rename` (aturan keras 10). Model dengan nama
    /// folder sama diganti. Mengembalikan folder tujuan.
    pub fn install_model(&self, src: &Path) -> Result<PathBuf, EngineError> {
        let src_dir = if src.is_file() {
            src.parent().ok_or_else(|| {
                EngineError::invalid(format!("path model tidak sah: {}", src.display()))
            })?
        } else {
            src
        };
        validate_model_dir(src_dir)?;
        TractSegmenter::load(src_dir).map_err(EngineError::Seg)?;

        let folder = src_dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .filter(|n| !n.is_empty() && n != "." && n != "..")
            .ok_or_else(|| {
                EngineError::invalid(format!(
                    "nama folder model tidak sah: {}",
                    src_dir.display()
                ))
            })?;
        let models = self.bundle.models_dir();
        let dest = models.join(&folder);
        if src_dir
            .canonicalize()
            .ok()
            .zip(dest.canonicalize().ok())
            .is_some_and(|(a, b)| a == b)
        {
            // Sudah berada di bundel: tidak ada yang perlu disalin.
            return Ok(dest);
        }
        let tmp = models.join(format!("{folder}.tmp"));
        if tmp.exists() {
            fs::remove_dir_all(&tmp).map_err(|e| h3d_io::IoError::io(&tmp, e))?;
        }
        fs::create_dir_all(&tmp).map_err(|e| h3d_io::IoError::io(&tmp, e))?;
        let copied = (|| -> Result<(), EngineError> {
            for name in [MODEL_SPEC_FILE, MODEL_ONNX_FILE] {
                copy_atomic(&src_dir.join(name), &tmp.join(name))?;
            }
            Ok(())
        })();
        if let Err(e) = copied {
            let _ = fs::remove_dir_all(&tmp);
            return Err(e);
        }
        if dest.exists() {
            fs::remove_dir_all(&dest).map_err(|e| h3d_io::IoError::io(&dest, e))?;
        }
        fs::rename(&tmp, &dest).map_err(|e| {
            let _ = fs::remove_dir_all(&tmp);
            h3d_io::IoError::io(&dest, e)
        })?;
        Ok(dest)
    }

    /// Buat label untuk setiap kelas model yang belum punya label bernama
    /// sama (tidak peka huruf besar/kecil). Label yang sudah ada tidak
    /// diubah. Ini hanya menambah **definisi** label; piksel label tidak
    /// tersentuh (aturan keras 8).
    pub fn ensure_model_labels(
        &mut self,
        spec: &ModelSpec,
    ) -> Result<ModelLabelSetup, EngineError> {
        let mut classes: Vec<_> = spec
            .classes
            .iter()
            .filter_map(|c| c.label.as_deref().map(|l| (c.index, l)))
            .collect();
        classes.sort_by_key(|(i, _)| *i);
        let mut setup = ModelLabelSetup::default();
        for (_, name) in classes {
            if self.project.label_by_name(name).is_some() {
                setup.existing.push(name.to_owned());
                continue;
            }
            let used: Vec<[u8; 4]> = self.project.labels.iter().map(|l| l.color).collect();
            let color = NEW_LABEL_COLORS
                .iter()
                .find(|c| !used.contains(c))
                .copied()
                .unwrap_or(NEW_LABEL_COLORS[self.project.labels.len() % NEW_LABEL_COLORS.len()]);
            self.add_label(name, color, kind_for_class(name))?;
            setup.created.push(name.to_owned());
        }
        Ok(setup)
    }

    /// Tolak (buang) semua usulan tertunda di semua irisan; mengembalikan
    /// jumlah irisan yang usulannya dibuang. Label tidak berubah.
    pub fn reject_all_pending(&mut self) -> usize {
        let n = self.pending_slides().len();
        self.pending.clear();
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_guess_from_class_name() {
        assert_eq!(kind_for_class("Tumor"), LabelKind::Tumor);
        assert_eq!(kind_for_class("Jaringan"), LabelKind::Normal);
        assert_eq!(kind_for_class("Stroma fibrosa"), LabelKind::Stroma);
        assert_eq!(kind_for_class("Nekrosis"), LabelKind::Necrosis);
        assert_eq!(kind_for_class("Entah"), LabelKind::Other);
    }

    #[test]
    fn display_name_falls_back_to_folder() {
        let e = ModelEntry {
            dir: PathBuf::from("models/x"),
            folder: "x".into(),
            spec: Err("rusak".into()),
        };
        assert_eq!(e.display_name(), "x");
        assert!(e.class_names().is_empty());
    }
}
