//! Rekonstruksi, cache voxel, dan pengukuran (8.6, 8.8).

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::PathBuf;

use h3d_core::{LabelDef, LabelId, Progress, SlideId, SlideStatus, SlideTransform, SubProgress};
use h3d_io::{
    decode_mesh, decode_vox, encode_mesh, encode_vox, write_atomic, AttrReport, GridInfo,
    GroupMember, GroupReport, LabelReport, ObjectReport, ProvenanceReport, Report, SlideSummary,
    TracksFile,
};
use h3d_segment::LabelStore;
use h3d_volume::{
    build_voxel_mask, cavalieri, cavalieri_areas, measure_label, plan_grid, reconstruct_label,
    reconstruct_label_objects, track_label, GridPlan, LabelMesh, LabelSampler, SliceInput,
    TrackParams, TrackSet, VoxelMask, Warning,
};
use serde::{Deserialize, Serialize};

use super::{ReconState, Session, APP_VERSION};
use crate::error::EngineError;
use crate::hash::{json_bytes, Fnv1a64};

/// Simpangan `|det T_i|` dari 1 yang membuat laporan mencatat hanyutan skala
/// rantai registrasi.
const CHAIN_SCALE_DRIFT_NOTE: f64 = 0.05;

/// Adaptor `LabelStore` → [`LabelSampler`] milik `h3d-volume`.
pub(crate) struct StoreSampler<'a>(pub &'a LabelStore);

impl LabelSampler for StoreSampler<'_> {
    fn label_at_um(&self, u: [f64; 2]) -> u8 {
        self.0.get_um(u)
    }
    fn mpp_label(&self) -> f64 {
        self.0.mpp_label
    }
    fn bbox_um(&self, label: u8) -> Option<[f64; 4]> {
        self.0.bbox_um(LabelId(label))
    }
}

/// Data satu irisan berlabel untuk rekonstruksi/pengukuran.
pub(crate) struct SliceData<'a> {
    pub id: SlideId,
    pub z_um: f64,
    pub transform: SlideTransform,
    pub store: Cow<'a, LabelStore>,
    /// `(label, interpolasi?)`.
    pub annotated: Vec<(u8, bool)>,
    pub reviewed: Vec<u8>,
    /// Hash isi label (lihat [`Session::label_content_hash`]).
    pub hash: u64,
}

/// Bagian [`SliceData`] yang menentukan kunci cache rekonstruksi, tanpa
/// penyimpanan labelnya; bisa dibangun dari cache tanpa membaca `.lbl`.
pub(crate) struct SliceMeta {
    pub id: SlideId,
    pub z_um: f64,
    pub transform: SlideTransform,
    pub annotated: Vec<(u8, bool)>,
    pub hash: u64,
}

impl SliceData<'_> {
    fn meta(&self) -> SliceMeta {
        SliceMeta {
            id: self.id,
            z_um: self.z_um,
            transform: self.transform.clone(),
            annotated: self.annotated.clone(),
            hash: self.hash,
        }
    }
}

/// `SliceInput` untuk `h3d-volume` dari data irisan dan adaptornya.
pub(crate) fn slice_inputs<'b>(
    data: &'b [SliceData<'_>],
    samplers: &'b [StoreSampler<'b>],
) -> Vec<SliceInput<'b>> {
    data.iter()
        .zip(samplers.iter())
        .map(|(d, s)| SliceInput {
            id: d.id,
            z_um: d.z_um,
            transform: &d.transform,
            labels: s,
            annotated_labels: d.annotated.clone(),
            reviewed_labels: d.reviewed.clone(),
        })
        .collect()
}

/// Isi `plan.json` di folder cache; ditulis terakhir sebagai penanda lengkap.
#[derive(Serialize, Deserialize)]
struct CacheManifest {
    key: String,
    plan: GridPlan,
    mask_labels: Vec<u8>,
    mesh_labels: Vec<u8>,
    /// Id objek per mesh (sejajar `mesh_labels`); kosong = semua `None`.
    #[serde(default)]
    mesh_instances: Vec<Option<u32>>,
}

/// Mesh satu label untuk baris laporan per label: mesh label utuh bila ada,
/// atau gabungan semua mesh per objek label itu (rekonstruksi per objek).
/// `None` bila label tidak punya mesh.
fn label_mesh(meshes: &[LabelMesh], label: LabelId) -> Option<Cow<'_, LabelMesh>> {
    if let Some(m) = meshes
        .iter()
        .find(|m| m.label == label && m.instance_id.is_none())
    {
        return Some(Cow::Borrowed(m));
    }
    let parts: Vec<&LabelMesh> = meshes.iter().filter(|m| m.label == label).collect();
    if parts.is_empty() {
        return None;
    }
    let mut out = LabelMesh::empty(label);
    for m in parts {
        let base = out.positions_um.len() as u32;
        out.positions_um.extend_from_slice(&m.positions_um);
        out.normals.extend_from_slice(&m.normals);
        out.indices.extend(m.indices.iter().map(|i| i + base));
    }
    Some(Cow::Owned(out))
}

/// Nama berkas mesh di cache: `LLL.mesh` atau `LLL_IIII.mesh` per objek.
fn mesh_cache_name(label: u8, instance: Option<u32>) -> String {
    match instance {
        Some(i) => format!("{label:03}_{i:04}.mesh"),
        None => format!("{label:03}.mesh"),
    }
}

impl Session {
    /// Irisan `Ok` yang punya label, terurut, beserta provenance per label.
    pub(crate) fn slice_data(&self) -> Result<Vec<SliceData<'_>>, EngineError> {
        let mut out = Vec::new();
        for (id, store) in self.label_snapshot()? {
            let Some(entry) = self.project.slide(id) else {
                continue;
            };
            if entry.status != SlideStatus::Ok {
                continue;
            }
            let hash = self.label_content_hash(id, Some(&store))?;
            let present = self.labels_present_cached(id, hash, || store.labels_present());
            let (annotated, reviewed) = self.annotation_flags(id, &present);
            if annotated.is_empty() {
                continue;
            }
            out.push(SliceData {
                id,
                z_um: entry.z_um,
                transform: entry.transform.clone(),
                store,
                annotated,
                reviewed,
                hash,
            });
        }
        Ok(out)
    }

    /// `(annotated, reviewed)` untuk label `present` pada irisan `id`.
    fn annotation_flags(&self, id: SlideId, present: &[LabelId]) -> (Vec<(u8, bool)>, Vec<u8>) {
        let mut annotated = Vec::new();
        let mut reviewed = Vec::new();
        for &l in present {
            // Piksel berlabel tanpa definisi di tabel label diabaikan.
            if self.project.label(l).is_none() {
                continue;
            }
            let rec = self.project.annotation(id, l);
            annotated.push((l.0, rec.is_some_and(|a| a.source.is_interpolated())));
            if rec.is_some_and(|a| a.review.is_reviewed()) {
                reviewed.push(l.0);
            }
        }
        (annotated, reviewed)
    }

    /// Label yang ada di irisan `id`, dari cache bila hash isinya sama;
    /// `compute` dipanggil hanya saat cache tidak berlaku.
    fn labels_present_cached(
        &self,
        id: SlideId,
        hash: u64,
        compute: impl FnOnce() -> Vec<LabelId>,
    ) -> Vec<LabelId> {
        if let Ok(c) = self.present_cache.lock() {
            if let Some((h, v)) = c.get(&id) {
                if *h == hash {
                    return v.clone();
                }
            }
        }
        let v = compute();
        if let Ok(mut c) = self.present_cache.lock() {
            c.insert(id, (hash, v.clone()));
        }
        v
    }

    /// Seperti [`Self::slice_data`] tetapi hanya metadata kunci: label yang
    /// tidak berubah sejak panggilan terakhir tidak dibaca dari disk, sehingga
    /// `meshes()`/`voxel_masks()` murah saat dipanggil berulang oleh GUI.
    pub(crate) fn slice_metas(&self) -> Result<Vec<SliceMeta>, EngineError> {
        let mut out = Vec::new();
        for entry in self.project.slides_ordered() {
            if entry.status != SlideStatus::Ok {
                continue;
            }
            let id = entry.id;
            let (hash, present) = if let Some(slot) = self.labels.get(&id) {
                if slot.store.is_empty() {
                    continue;
                }
                let hash = self.label_content_hash(id, Some(&slot.store))?;
                (
                    hash,
                    self.labels_present_cached(id, hash, || slot.store.labels_present()),
                )
            } else {
                if !self.bundle.label_path(id).is_file() {
                    continue;
                }
                let hash = self.label_content_hash(id, None)?;
                let cached = self.present_cache.lock().ok().and_then(|c| {
                    c.get(&id)
                        .filter(|(h, _)| *h == hash)
                        .map(|(_, v)| v.clone())
                });
                let present = match cached {
                    Some(v) => v,
                    None => {
                        let Some(store) = self.bundle.load_labels(id)? else {
                            continue;
                        };
                        self.labels_present_cached(id, hash, || store.labels_present())
                    }
                };
                if present.is_empty() {
                    continue;
                }
                (hash, present)
            };
            let (annotated, _) = self.annotation_flags(id, &present);
            if annotated.is_empty() {
                continue;
            }
            out.push(SliceMeta {
                id,
                z_um: entry.z_um,
                transform: entry.transform.clone(),
                annotated,
                hash,
            });
        }
        Ok(out)
    }

    /// Label (terdefinisi) yang muncul di data irisan, terurut id.
    fn present_labels(&self, data: &[SliceData<'_>]) -> Vec<LabelDef> {
        let metas: Vec<SliceMeta> = data.iter().map(SliceData::meta).collect();
        self.present_labels_meta(&metas)
    }

    fn present_labels_meta(&self, metas: &[SliceMeta]) -> Vec<LabelDef> {
        self.project
            .labels
            .iter()
            .filter(|l| {
                metas
                    .iter()
                    .any(|d| d.annotated.iter().any(|(id, _)| *id == l.id.0))
            })
            .cloned()
            .collect()
    }

    /// Hash isi label satu irisan (FNV atas byte `.lbl`). Irisan yang
    /// disunting di memori: dihitung dari `encode_lbl` sekali per suntingan;
    /// selain itu dari byte berkas di disk, dikunci metadata berkas, sehingga
    /// `meshes()`/`measure()` tidak mengompres ulang semua label tiap panggilan.
    pub(crate) fn label_content_hash(
        &self,
        id: SlideId,
        store: Option<&LabelStore>,
    ) -> Result<u64, EngineError> {
        use std::sync::atomic::Ordering;
        if let Some(slot) = self.labels.get(&id) {
            if slot.dirty {
                let cached = slot.content_hash.load(Ordering::Relaxed);
                if cached != 0 {
                    return Ok(cached);
                }
                let h = Fnv1a64::new()
                    .bytes(&h3d_segment::encode_lbl(&slot.store))
                    .finish();
                slot.content_hash.store(h, Ordering::Relaxed);
                return Ok(h);
            }
        }
        let path = self.bundle.label_path(id);
        let meta = match (std::fs::metadata(&path), store) {
            (Ok(m), _) => m,
            // Tidak ada berkas: label hanya di memori (belum pernah disimpan).
            (Err(_), Some(store)) => {
                return Ok(Fnv1a64::new()
                    .bytes(&h3d_segment::encode_lbl(store))
                    .finish())
            }
            (Err(e), None) => return Err(h3d_io::IoError::io(&path, e).into()),
        };
        let (len, modified) = (meta.len(), meta.modified().ok());
        if let Ok(cache) = self.label_hashes.lock() {
            if let Some(fh) = cache.get(&id) {
                if fh.len == len && fh.modified == modified {
                    return Ok(fh.hash);
                }
            }
        }
        let bytes = std::fs::read(&path).map_err(|e| h3d_io::IoError::io(&path, e))?;
        let hash = Fnv1a64::new().bytes(&bytes).finish();
        if let Ok(mut cache) = self.label_hashes.lock() {
            cache.insert(
                id,
                super::FileHash {
                    len,
                    modified,
                    hash,
                },
            );
        }
        Ok(hash)
    }

    /// Kunci cache voxel: hash atas label (byte `.lbl`), transformasi, `z`,
    /// tanda provenance, definisi label terkait, `ReconConfig`, dan anggaran
    /// (yang menentukan kenaikan `s_xy`). Berubah salah satunya → cache basi.
    pub(crate) fn recon_key(&self, data: &[SliceData<'_>]) -> Result<u64, EngineError> {
        let metas: Vec<SliceMeta> = data.iter().map(SliceData::meta).collect();
        self.recon_key_meta(&metas)
    }

    /// [`Self::recon_key`] atas metadata irisan ([`Self::slice_metas`]);
    /// nilainya identik dengan kunci dari [`Self::slice_data`].
    pub(crate) fn recon_key_meta(&self, data: &[SliceMeta]) -> Result<u64, EngineError> {
        let mut h = Fnv1a64::new();
        h.str("h3d-recon-v2");
        h.chunk(&json_bytes(&self.project.recon)?);
        h.u64(self.budgets.recon_working_set_bytes);
        for l in self.present_labels_meta(data) {
            h.u64(u64::from(l.id.0));
            h.u64(u64::from(l.reconstruct));
            h.chunk(&json_bytes(&l.kind)?);
        }
        h.u64(data.len() as u64);
        for d in data {
            h.u64(u64::from(d.id.0));
            h.f64(d.z_um);
            h.chunk(&json_bytes(&d.transform)?);
            for (l, interp) in &d.annotated {
                h.u64(u64::from(*l));
                h.u64(u64::from(*interp));
            }
            h.u64(d.hash);
        }
        Ok(h.finish())
    }

    fn cache_dir_for(&self, key: u64) -> PathBuf {
        self.bundle.cache_vox_dir().join(format!("{key:016x}"))
    }

    /// Muat hasil rekonstruksi dari `cache/vox/<kunci>/`; `None` bila tidak
    /// ada atau rusak.
    fn load_recon_cache(&self, key: u64) -> Option<ReconState> {
        let dir = self.cache_dir_for(key);
        let bytes = std::fs::read(dir.join("plan.json")).ok()?;
        let manifest: CacheManifest = serde_json::from_slice(&bytes).ok()?;
        if manifest.key != format!("{key:016x}") {
            return None;
        }
        let mut masks = Vec::new();
        for l in &manifest.mask_labels {
            let path = dir.join(format!("{l:03}.vox"));
            let b = std::fs::read(&path).ok()?;
            masks.push((LabelId(*l), decode_vox(&b, &path).ok()?));
        }
        let mut meshes = Vec::new();
        for (k, l) in manifest.mesh_labels.iter().enumerate() {
            let inst = manifest.mesh_instances.get(k).copied().flatten();
            let path = dir.join(mesh_cache_name(*l, inst));
            let b = std::fs::read(&path).ok()?;
            meshes.push(decode_mesh(&b, &path).ok()?);
        }
        Some(ReconState {
            key,
            plan: manifest.plan,
            masks,
            meshes,
        })
    }

    /// Tulis cache dan buang cache berkunci lain (basi).
    pub(crate) fn write_recon_cache(&self, state: &ReconState) -> Result<(), EngineError> {
        let root = self.bundle.cache_vox_dir();
        let dir = self.cache_dir_for(state.key);
        if let Ok(entries) = std::fs::read_dir(&root) {
            for e in entries.flatten() {
                if e.path() != dir {
                    let _ = std::fs::remove_dir_all(e.path());
                }
            }
        }
        for (l, m) in &state.masks {
            write_atomic(&dir.join(format!("{:03}.vox", l.0)), &encode_vox(m)?)?;
        }
        for m in &state.meshes {
            write_atomic(
                &dir.join(mesh_cache_name(m.label.0, m.instance_id)),
                &encode_mesh(m)?,
            )?;
        }
        let manifest = CacheManifest {
            key: format!("{:016x}", state.key),
            plan: state.plan.clone(),
            mask_labels: state.masks.iter().map(|(l, _)| l.0).collect(),
            mesh_labels: state.meshes.iter().map(|m| m.label.0).collect(),
            mesh_instances: state.meshes.iter().map(|m| m.instance_id).collect(),
        };
        let json = serde_json::to_vec(&manifest)
            .map_err(|e| h3d_io::IoError::json(dir.join("plan.json"), e))?;
        write_atomic(&dir.join("plan.json"), &json)?;
        Ok(())
    }

    /// Hasil rekonstruksi untuk `key`: dari memori, atau dari cache disk.
    pub(crate) fn recon_for_key(&self, key: u64) -> Option<Cow<'_, ReconState>> {
        match &self.recon {
            Some(r) if r.key == key => Some(Cow::Borrowed(r)),
            _ => self.load_recon_cache(key).map(Cow::Owned),
        }
    }

    /// Rekonstruksi semua label (8.6.2). Kisi direncanakan atas semua label
    /// yang ada (termasuk sumber medan jarak seperti `Margin`, agar analisis
    /// spasial tidak terpotong); mesh hanya dibuat untuk label ber-
    /// `reconstruct`. Masker voxel dan mesh disimpan di `cache/vox/` dengan
    /// kunci hash label + transformasi + `ReconConfig`; pemanggilan ulang
    /// tanpa perubahan memakai cache. Volume label juga ditulis ke
    /// `out/volume/labels.nii.gz` ([`Self::write_label_volume`]).
    pub fn reconstruct(&mut self, p: &dyn Progress) -> Result<Vec<LabelMesh>, EngineError> {
        let (state, fresh) = {
            let data = self.slice_data()?;
            if data.is_empty() {
                return Err(h3d_volume::VolError::NoLabelledSlices.into());
            }
            let key = self.recon_key(&data)?;
            if let Some(r) = self.recon_for_key(key) {
                (r.into_owned(), false)
            } else {
                let labels = self.present_labels(&data);
                let samplers: Vec<StoreSampler<'_>> =
                    data.iter().map(|d| StoreSampler(&d.store)).collect();
                let inputs = slice_inputs(&data, &samplers);
                let ids: Vec<u8> = labels.iter().map(|l| l.id.0).collect();
                let cfg = &self.project.recon;
                p.set(0.0, "merencanakan kisi");
                let plan = plan_grid(&inputs, &ids, cfg, &self.budgets)?;
                let mut masks = Vec::new();
                let mut meshes = Vec::new();
                let mut sets: Vec<TrackSet> = Vec::new();
                let n = labels.len().max(1) as f32;
                for (i, l) in labels.iter().enumerate() {
                    if p.is_cancelled() {
                        return Err(EngineError::Cancelled);
                    }
                    let sub = SubProgress::new(p, i as f32 / n, (i + 1) as f32 / n);
                    if l.reconstruct && cfg.per_object {
                        // Per objek: masker label utuh untuk analisis spasial,
                        // mesh terpisah per objek hasil penautan antar irisan.
                        let sub_a = SubProgress::new(&sub, 0.0, 0.25);
                        masks.push((l.id, build_voxel_mask(&plan, &inputs, l.id.0, cfg, &sub_a)?));
                        let sub_b = SubProgress::new(&sub, 0.25, 1.0);
                        let params = self.track_params(None);
                        let (set, obj) = reconstruct_label_objects(
                            &plan, &inputs, l.id.0, cfg, &params, &sub_b,
                        )?;
                        meshes.extend(obj);
                        sets.push(set);
                    } else if l.reconstruct {
                        let (mask, mesh) = reconstruct_label(&plan, &inputs, l.id.0, cfg, &sub)?;
                        masks.push((l.id, mask));
                        meshes.push(mesh);
                    } else {
                        masks.push((l.id, build_voxel_mask(&plan, &inputs, l.id.0, cfg, &sub)?));
                    }
                }
                if p.is_cancelled() {
                    return Err(EngineError::Cancelled);
                }
                if !sets.is_empty() {
                    // Gagal menulis tracks.json bukan galat rekonstruksi.
                    if let Err(e) = self.save_track_sets(sets) {
                        log::warn!("gagal menulis out/tracks.json: {e}");
                    }
                }
                (
                    ReconState {
                        key,
                        plan,
                        masks,
                        meshes,
                    },
                    true,
                )
            }
        };
        if !fresh {
            p.set(1.0, "rekonstruksi (dari cache)");
            let meshes = state.meshes.clone();
            self.recon = Some(state);
            if !self.label_volume_path().is_file() {
                self.write_label_volume_logged();
            }
            return Ok(meshes);
        }
        // Cache gagal ditulis bukan galat rekonstruksi.
        if let Err(e) = self.write_recon_cache(&state) {
            log::warn!("gagal menulis cache voxel: {e}");
        }
        let meshes = state.meshes.clone();
        self.recon = Some(state);
        self.analysis = None;
        self.write_label_volume_logged();
        p.set(1.0, "rekonstruksi selesai");
        Ok(meshes)
    }

    /// Parameter penautan dari konfigurasi proyek; `min_iou` menimpa
    /// `recon.track_min_iou`.
    fn track_params(&self, min_iou: Option<f64>) -> TrackParams {
        TrackParams {
            min_iou: min_iou.unwrap_or(self.project.recon.track_min_iou),
            max_gap_um: self.project.recon.max_gap_um,
            split: self.project.recon.track_split,
            max_skip_slices: self.project.recon.track_max_skip,
        }
    }

    /// Gabungkan `sets` ke `out/tracks.json` (set lama label lain tetap).
    fn save_track_sets(&self, sets: Vec<TrackSet>) -> Result<(), EngineError> {
        let mut file = match self.bundle.load_tracks() {
            Ok(Some(f)) => f,
            Ok(None) => TracksFile::new(APP_VERSION),
            Err(e) => {
                log::warn!("tracks.json lama tidak terbaca, ditulis ulang: {e}");
                TracksFile::new(APP_VERSION)
            }
        };
        file.app_version = APP_VERSION.to_owned();
        for s in sets {
            file.put(s);
        }
        self.bundle.save_tracks(&file)?;
        Ok(())
    }

    /// `out/tracks.json` yang tersimpan; `None` bila belum ada.
    pub fn tracks(&self) -> Result<Option<TracksFile>, EngineError> {
        Ok(self.bundle.load_tracks()?)
    }

    /// Tautkan objek (instance) `label` antar irisan (`h3d_volume::track`)
    /// pada kisi rekonstruksi, simpan hasilnya ke `out/tracks.json`
    /// (atomik), dan kembalikan set-nya. `min_iou` menimpa
    /// `ReconConfig::track_min_iou`; celah maksimum mengikuti rekonstruksi.
    /// Tidak mengubah label, urutan, maupun orientasi irisan.
    pub fn track(
        &mut self,
        label: LabelId,
        min_iou: Option<f64>,
        p: &dyn Progress,
    ) -> Result<TrackSet, EngineError> {
        if self.project.label(label).is_none() {
            return Err(EngineError::invalid(format!("label {label} tidak ada")));
        }
        let data = self.slice_data()?;
        if data.is_empty() {
            return Err(h3d_volume::VolError::NoLabelledSlices.into());
        }
        let params = self.track_params(min_iou);
        let set = {
            let samplers: Vec<StoreSampler<'_>> =
                data.iter().map(|d| StoreSampler(&d.store)).collect();
            let inputs = slice_inputs(&data, &samplers);
            let cfg = &self.project.recon;
            p.set(0.0, "merencanakan kisi");
            let plan = plan_grid(&inputs, &[label.0], cfg, &self.budgets)?;
            let sub = SubProgress::new(p, 0.05, 0.95);
            track_label(&plan, &inputs, label.0, &params, &sub)?
        };
        self.save_track_sets(vec![set.clone()])?;
        p.set(1.0, "penautan objek selesai");
        Ok(set)
    }

    /// `out/volume/labels.nii.gz` (8.7.1).
    pub fn label_volume_path(&self) -> PathBuf {
        self.bundle.out_volume_dir().join("labels.nii.gz")
    }

    /// Tulis volume label `u8` hasil rekonstruksi yang masih sah ke
    /// `out/volume/labels.nii.gz` (8.7.1; `pixdim` = `s_xy`, `s_xy`, `s_z`
    /// dalam mm, 8.7.3) dan kembalikan path-nya. [`Self::reconstruct`]
    /// memanggilnya sendiri; berguna pula untuk menulis ulang berkas yang
    /// terhapus. Galat bila belum ada rekonstruksi untuk keadaan saat ini.
    pub fn write_label_volume(&self) -> Result<PathBuf, EngineError> {
        let data = self.slice_metas()?;
        let recon = if data.is_empty() {
            None
        } else {
            self.recon_for_key(self.recon_key_meta(&data)?)
        };
        let recon = recon.ok_or_else(|| {
            EngineError::invalid(
                "belum ada volume untuk keadaan proyek saat ini; jalankan rekonstruksi dulu",
            )
        })?;
        let masks: Vec<(u8, &VoxelMask)> = recon.masks.iter().map(|(l, m)| (l.0, m)).collect();
        let out = h3d_io::encode_label_nifti(&masks)?;
        for n in &out.notes {
            log::info!("labels.nii.gz: {n}");
        }
        let path = self.label_volume_path();
        write_atomic(&path, &out.gz)?;
        Ok(path)
    }

    /// Gagal menulis keluaran turunan bukan galat rekonstruksi (mis. bundel
    /// hanya-baca); hasilnya tetap ada di memori.
    fn write_label_volume_logged(&self) {
        if let Err(e) = self.write_label_volume() {
            log::warn!("gagal menulis out/volume/labels.nii.gz: {e}");
        }
    }

    /// Mesh hasil rekonstruksi terakhir yang masih sah (memori atau cache);
    /// kosong bila belum direkonstruksi atau sudah basi.
    pub fn meshes(&self) -> Result<Vec<LabelMesh>, EngineError> {
        let data = self.slice_metas()?;
        if data.is_empty() {
            return Ok(Vec::new());
        }
        let key = self.recon_key_meta(&data)?;
        Ok(self
            .recon_for_key(key)
            .map(|r| r.meshes.clone())
            .unwrap_or_default())
    }

    /// Masker voxel hasil rekonstruksi terakhir yang masih sah.
    pub fn voxel_masks(&self) -> Result<Vec<(LabelId, VoxelMask)>, EngineError> {
        let data = self.slice_metas()?;
        if data.is_empty() {
            return Ok(Vec::new());
        }
        let key = self.recon_key_meta(&data)?;
        Ok(self
            .recon_for_key(key)
            .map(|r| r.masks.clone())
            .unwrap_or_default())
    }

    /// Bangun laporan (8.6.4, 8.12.8). Volume Cavalieri (angka utama) hanya
    /// dari irisan beranotasi non-interpolasi; volume "ditinjau saja" dari
    /// irisan `Reviewed`; nilai terkoreksi susut dilaporkan berdampingan dan
    /// tidak pernah diterapkan ke geometri. Besaran mesh diisi bila hasil
    /// rekonstruksi untuk keadaan saat ini tersedia (memori atau cache).
    pub fn measure(&self) -> Result<Report, EngineError> {
        let project = &self.project;
        let mut report = Report::new(&project.sample.name, APP_VERSION);
        report.sample_notes = project.sample.notes.clone();
        report.orientation = project.sample.orientation;
        report.shrinkage_linear = project.sample.shrinkage_linear;
        report.recon = project.recon.clone();
        report.spatial = project.spatial.clone();

        let reference = project.effective_reference();
        let usable = project.usable_slides_ordered();
        report.slides = SlideSummary {
            total: project.slides.len(),
            usable: usable.len(),
            registered: usable
                .iter()
                .filter(|s| Some(s.id) == reference || s.transform.is_registered())
                .count(),
            flagged: usable
                .iter()
                .filter(|s| s.reg.as_ref().is_some_and(|r| r.flagged))
                .map(|s| s.id)
                .collect(),
            reference,
            median_spacing_um: project.median_spacing_um(),
            order_suggested: project
                .slides
                .iter()
                .any(|s| s.order_source == h3d_core::OrderSource::Suggested),
        };

        let data = self.slice_data()?;
        let labels = self.present_labels(&data);
        let samplers: Vec<StoreSampler<'_>> = data.iter().map(|d| StoreSampler(&d.store)).collect();
        let inputs = slice_inputs(&data, &samplers);
        let recon = if data.is_empty() {
            None
        } else {
            self.recon_for_key(self.recon_key(&data)?)
        };
        let shrink = project.sample.shrinkage_linear;

        for l in &labels {
            let id = l.id.0;
            let n_interp = data
                .iter()
                .filter(|d| d.annotated.iter().any(|(x, i)| *x == id && *i))
                .count();
            // Volume dari irisan `Reviewed` saja (non-interpolasi).
            let reviewed_areas: Vec<(f64, f64)> = inputs
                .iter()
                .filter(|s| s.label_flag(id) == Some(false) && s.reviewed_labels.contains(&id))
                .map(|s| (s.z_um, h3d_volume::label_area_um2(s.labels, id)))
                .collect();
            let n_reviewed = reviewed_areas.len();
            let volume_reviewed = (n_reviewed >= 2).then(|| cavalieri(&reviewed_areas));
            let mesh = recon.as_ref().and_then(|r| label_mesh(&r.meshes, l.id));
            let row = match mesh {
                Some(mesh) => {
                    let m = measure_label(&mesh, &inputs, id, shrink);
                    for w in &m.warnings {
                        if !report.warnings.contains(w) {
                            report.warnings.push(w.clone());
                        }
                    }
                    LabelReport {
                        label: l.id,
                        name: l.name.clone(),
                        kind: l.kind,
                        group: l.group_name().to_owned(),
                        volume_cavalieri_um3: m.volume_cavalieri_um3,
                        volume_reviewed_um3: volume_reviewed,
                        volume_corrected_um3: m.volume_corrected_um3,
                        volume_mesh_um3: Some(m.volume_mesh_um3),
                        surface_area_um2: Some(m.surface_area_um2),
                        bbox_um: m.bbox_um,
                        max_diameter_um: Some(m.max_diameter_um),
                        n_components: Some(m.n_components),
                        n_slices_annotated: m.n_slices_annotated,
                        n_slices_interpolated: n_interp,
                        n_slices_reviewed: n_reviewed,
                    }
                }
                None => {
                    let areas = cavalieri_areas(&inputs, id);
                    let v = cavalieri(&areas);
                    LabelReport {
                        label: l.id,
                        name: l.name.clone(),
                        kind: l.kind,
                        group: l.group_name().to_owned(),
                        volume_cavalieri_um3: v,
                        volume_reviewed_um3: volume_reviewed,
                        volume_corrected_um3: shrink.map(|f| v * f * f * f),
                        volume_mesh_um3: None,
                        surface_area_um2: None,
                        bbox_um: None,
                        max_diameter_um: None,
                        n_components: None,
                        n_slices_annotated: areas.len(),
                        n_slices_interpolated: n_interp,
                        n_slices_reviewed: n_reviewed,
                    }
                }
            };
            report.labels.push(row);
        }

        // Objek (rekonstruksi per objek): satu baris per mesh ber-`instance_id`.
        if let Some(r) = &recon {
            let tracks = match self.bundle.load_tracks() {
                Ok(t) => t,
                Err(e) => {
                    log::warn!("tracks.json tidak terbaca: {e}");
                    None
                }
            };
            for m in r.meshes.iter().filter(|m| m.instance_id.is_some()) {
                let Some(id) = m.instance_id else {
                    continue;
                };
                let name = labels
                    .iter()
                    .find(|l| l.id == m.label)
                    .map_or_else(|| m.label.to_string(), |l| l.name.clone());
                let set = tracks.as_ref().and_then(|t| t.set(m.label));
                let track = set.and_then(|s| s.track(id));
                report.objects.push(ObjectReport {
                    label: m.label,
                    name: m.display_name(&name),
                    instance_id: id,
                    volume_mesh_um3: m.signed_volume_um3(),
                    surface_area_um2: m.surface_area_um2(),
                    bbox_um: m.bbox(),
                    max_diameter_um: m.max_diameter_um(),
                    n_components: m.connected_components(),
                    n_slices: track.map(|t| t.members.len()),
                    z_range_um: track.and_then(|t| t.z_range_um()),
                    track_params_hash: set.map(|s| s.params_hash),
                    n_bridged_slices: set.map(|s| s.skipped_slides(id).len()),
                });
            }
        }

        // Grup: volume grup = jumlah volume anggota (8.12.4).
        let mut groups: Vec<GroupReport> = Vec::new();
        for l in &report.labels {
            // Label sumber medan jarak (tepi reseksi, permukaan, garis
            // rujukan) bukan komponen jaringan; tidak ikut agregasi grup.
            if l.kind.is_distance_source() {
                continue;
            }
            match groups.iter_mut().find(|g| g.group == l.group) {
                Some(g) => {
                    g.volume_um3 += l.volume_cavalieri_um3;
                    g.members.push(GroupMember {
                        name: l.name.clone(),
                        volume_um3: l.volume_cavalieri_um3,
                        fraction: 0.0,
                    });
                }
                None => groups.push(GroupReport {
                    group: l.group.clone(),
                    volume_um3: l.volume_cavalieri_um3,
                    members: vec![GroupMember {
                        name: l.name.clone(),
                        volume_um3: l.volume_cavalieri_um3,
                        fraction: 0.0,
                    }],
                }),
            }
        }
        for g in &mut groups {
            for m in &mut g.members {
                m.fraction = if g.volume_um3 > 0.0 {
                    m.volume_um3 / g.volume_um3
                } else {
                    0.0
                };
            }
        }
        report.groups = groups;

        // Provenance (8.12.7): hanya catatan irisan `Ok`.
        let mut by_source: BTreeMap<&'static str, usize> = BTreeMap::new();
        let mut prov = ProvenanceReport::default();
        for a in &project.annotations {
            if !project
                .slide(a.slide)
                .is_some_and(|s| s.status == SlideStatus::Ok)
            {
                continue;
            }
            prov.total += 1;
            *by_source.entry(a.source.short_name()).or_default() += 1;
            if a.review.is_reviewed() {
                prov.reviewed += 1;
            } else {
                let name = project
                    .label(a.label)
                    .map(|l| l.name.clone())
                    .unwrap_or_else(|| a.label.to_string());
                prov.drafts.push((a.slide, name));
            }
        }
        prov.by_source = by_source
            .into_iter()
            .map(|(k, v)| (k.to_owned(), v))
            .collect();
        if !prov.drafts.is_empty() && !report.warnings.contains(&Warning::HasDraft) {
            report.warnings.push(Warning::HasDraft);
        }
        let n_interp = project
            .annotations
            .iter()
            .filter(|a| a.source.is_interpolated())
            .count();
        if prov.total > 0
            && n_interp * 2 > prov.total
            && !report.warnings.contains(&Warning::MostlyInterpolated)
        {
            report.warnings.push(Warning::MostlyInterpolated);
        }
        report.provenance = prov;

        // Kisi, catatan, dan peringatan perencanaan.
        let tre = self.tre_estimate_um();
        let (s_xy, s_z) = match &recon {
            Some(r) => {
                let g = &r.plan.grid;
                report.grid = Some(GridInfo {
                    s_xy_um: g.s_xy,
                    s_z_um: g.median_dz(),
                    size: g.size,
                    nz: g.nz(),
                });
                for n in &r.plan.notes {
                    report.notes.push(n.clone());
                }
                for w in &r.plan.warnings {
                    if !report.warnings.contains(w) {
                        report.warnings.push(w.clone());
                    }
                }
                (g.s_xy, g.median_dz())
            }
            None => (
                project.recon.s_xy_um,
                project.median_spacing_um().unwrap_or(project.recon.s_xy_um),
            ),
        };
        report.dimension_uncertainty_um = 0.5 * s_xy.max(s_z) + tre.unwrap_or(s_xy);
        if tre.is_none() {
            report.notes.push(
                "Taksiran galat registrasi tidak tersedia (belum ada registrasi); \
                 ketidakpastian memakai s_xy sebagai penggantinya."
                    .to_owned(),
            );
        }
        // Hanyutan skala rantai (ADR 0002): model bebas yang diterima pada
        // banyak pasangan bisa menumpuk skala searah; volume mesh/voxel ikut
        // bergeser sebesar itu, sedangkan Cavalieri (ruang irisan) tidak.
        let drift = usable
            .iter()
            .filter(|s| s.transform.is_registered())
            .map(|s| (s.transform.affine().determinant().abs() - 1.0).abs())
            .fold(0.0f64, f64::max);
        if drift > CHAIN_SCALE_DRIFT_NOTE {
            report.notes.push(format!(
                "Skala luas kumulatif rantai registrasi menyimpang hingga {:.1} % dari 1; \
                 volume mesh dan jarak 3D pada irisan itu ikut terpengaruh (volume Cavalieri tidak).",
                drift * 100.0
            ));
        }
        if !project.z_is_monotonic() {
            report
                .notes
                .push("Nilai z_um tidak monoton naik sepanjang urutan irisan.".to_owned());
        }

        // Atribut (8.12.5) dan analisis spasial (bila masih sah).
        let attrs: Vec<AttrReport> = self.attr_reports()?;
        for a in &attrs {
            if a.method.contains("area") {
                let w = Warning::MembraneAreaScore {
                    name: a.name.clone(),
                };
                if !report.warnings.contains(&w) {
                    report.warnings.push(w);
                }
            }
            let is_ihc = a.method.starts_with("ihc");
            if is_ihc
                && a.range_between_slides
                    .is_some_and(|r| r > f64::from(project.spatial.ihc_hetero_warn))
                && !report.warnings.contains(&Warning::IhcHeterogeneity)
            {
                report.warnings.push(Warning::IhcHeterogeneity);
            }
        }
        report.attrs = attrs;
        for n in &self.attr_notes {
            if !report.notes.contains(n) {
                report.notes.push(n.clone());
            }
        }
        if let Some(r) = &recon {
            report.analysis = self.current_analysis(r.key, tre)?;
        }
        Ok(report)
    }
}
