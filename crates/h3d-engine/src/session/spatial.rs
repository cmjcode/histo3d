//! Analisis spasial 3D (8.12.6) di atas masker voxel hasil rekonstruksi.

use h3d_core::{AttrDef, LabelDef, Progress, SubProgress};
use h3d_io::{decode_analysis, encode_analysis, write_atomic};
use h3d_volume::{
    project_fields_to_mesh, resample_uniform_z, AttrVolume, SpatialAnalysis, StackGrid, VoxelMask,
};

use super::{ReconState, Session, APP_VERSION};
use crate::error::EngineError;
use crate::hash::{json_bytes, Fnv1a64};

/// Hasil hitung analisis beserta bahan untuk proyeksi dan ekspor.
pub(crate) struct AnalysisOutput {
    pub analysis: SpatialAnalysis,
    /// Kisi tempat `analysis.fields` berada (kisi rekonstruksi, atau kisi
    /// ber-`s_z` median bila jarak lapis tidak seragam).
    pub field_grid: StackGrid,
    /// Volume atribut pada kisi rekonstruksi.
    pub attr_volumes: Vec<(AttrDef, AttrVolume)>,
}

impl Session {
    /// Kunci masukan analisis: kunci rekonstruksi + konfigurasi spasial +
    /// orientasi + definisi dan isi atribut + taksiran TRE (`tre`, dari
    /// [`Session::tre_estimate_um`]; diteruskan agar tidak dihitung berulang).
    pub(crate) fn analysis_key(
        &self,
        recon_key: u64,
        tre: Option<f64>,
    ) -> Result<u64, EngineError> {
        let mut h = Fnv1a64::new();
        h.str("h3d-analysis-v1").u64(recon_key);
        h.chunk(&json_bytes(&self.project.spatial)?);
        h.chunk(&json_bytes(&self.project.sample.orientation)?);
        for def in &self.project.attrs {
            h.chunk(&json_bytes(def)?);
            let stem = def.file_stem();
            for s in self.project.usable_slides_ordered() {
                let path = self.bundle.attr_path(s.id, &stem);
                if let Ok(bytes) = std::fs::read(&path) {
                    h.u64(u64::from(s.id.0));
                    h.chunk(&bytes);
                }
            }
        }
        match tre {
            Some(t) => h.u64(1).f64(t),
            None => h.u64(0),
        };
        Ok(h.finish())
    }

    /// Analisis yang masih sah untuk keadaan saat ini: dari memori, atau dari
    /// `out/analysis.json` bila kunci masukannya cocok. `None` bila belum
    /// dijalankan atau sudah basi.
    pub(crate) fn current_analysis(
        &self,
        recon_key: u64,
        tre: Option<f64>,
    ) -> Result<Option<SpatialAnalysis>, EngineError> {
        let key = self.analysis_key(recon_key, tre)?;
        if let Some((k, a)) = &self.analysis {
            if *k == key {
                return Ok(Some(a.clone()));
            }
        }
        let path = self.bundle.analysis_json();
        let Ok(text) = std::fs::read_to_string(&path) else {
            return Ok(None);
        };
        match decode_analysis(&text, &path) {
            Ok(file) if file.input_key == format!("{key:016x}") => Ok(Some(file.analysis)),
            Ok(_) => Ok(None),
            Err(e) => {
                log::warn!("analysis.json diabaikan: {e}");
                Ok(None)
            }
        }
    }

    /// Jalankan `h3d_volume::spatial::analyze` atas hasil rekonstruksi.
    /// Tidak mengubah sesi; dipakai `analyze_spatial` dan ekspor NIfTI.
    pub(crate) fn compute_analysis(
        &self,
        recon: &ReconState,
        tre: Option<f64>,
        p: &dyn Progress,
    ) -> Result<AnalysisOutput, EngineError> {
        let masks: Vec<(LabelDef, VoxelMask)> = recon
            .masks
            .iter()
            .filter_map(|(id, m)| self.project.label(*id).map(|d| (d.clone(), m.clone())))
            .collect();
        if masks.is_empty() {
            return Err(h3d_volume::VolError::NoLabelledSlices.into());
        }
        p.set(0.0, "menyiapkan volume atribut");
        let attr_volumes = self.attr_volumes(recon)?;
        // `ClinicalMeta` tidak pernah menjadi masukan (aturan keras 18);
        // hanya orientasi anatomis yang dipakai untuk menamai arah.
        let analysis = h3d_volume::analyze(
            &masks,
            &attr_volumes,
            self.project.sample.orientation.as_ref(),
            &self.project.spatial,
            tre,
            &SubProgress::new(p, 0.1, 1.0),
        )?;
        let field_grid = if analysis.grid.resampled_z {
            resample_uniform_z(&masks[0].1).grid
        } else {
            recon.plan.grid.clone()
        };
        Ok(AnalysisOutput {
            analysis,
            field_grid,
            attr_volumes,
        })
    }

    /// Analisis spasial 3D (8.12.6): margin, kontak, penetrasi kapsul,
    /// multifokalitas, kedalaman, ukuran staging, fraksi, hipoksia, dan
    /// statistik atribut; semuanya pada masker voxel (bukan mesh). Rekonstruksi
    /// dijalankan dulu bila cache tidak sah. Medan jarak dan atribut
    /// diproyeksikan ke verteks mesh (`LabelMesh::scalars`), dan hasilnya
    /// ditulis ke `out/analysis.json` (deterministik).
    pub fn analyze_spatial(&mut self, p: &dyn Progress) -> Result<SpatialAnalysis, EngineError> {
        self.reconstruct(&SubProgress::new(p, 0.0, 0.5))?;
        let mut recon = self
            .recon
            .take()
            .ok_or_else(|| EngineError::invalid("hasil rekonstruksi tidak tersedia"))?;
        // Taksiran TRE dihitung sekali (memuat masker registrasi tiap irisan).
        let tre = self.tre_estimate_um();
        let result = self.compute_analysis(&recon, tre, &SubProgress::new(p, 0.5, 0.95));
        let out = match result {
            Ok(o) => o,
            Err(e) => {
                self.recon = Some(recon);
                return Err(e);
            }
        };
        for mesh in &mut recon.meshes {
            project_fields_to_mesh(mesh, &out.field_grid, &out.analysis.fields);
        }
        let key = recon.key;
        // Mesh di cache diperbarui agar skalar tersedia bagi ekspor GLB.
        if let Err(e) = self.write_recon_cache(&recon) {
            log::warn!("gagal memperbarui cache mesh: {e}");
        }
        self.recon = Some(recon);
        let akey = self.analysis_key(key, tre)?;
        let text = encode_analysis(&out.analysis, APP_VERSION, akey)?;
        write_atomic(&self.bundle.analysis_json(), text.as_bytes())?;
        self.analysis = Some((akey, out.analysis.clone()));
        p.set(1.0, "analisis selesai");
        Ok(out.analysis)
    }
}
