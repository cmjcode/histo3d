//! Rekonstruksi, cache voxel, dan pengukuran (8.6, 8.8).

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::PathBuf;

use h3d_core::{LabelDef, LabelId, Progress, SlideId, SlideStatus, SlideTransform, SubProgress};
use h3d_io::{
    decode_mesh, decode_vox, encode_mesh, encode_vox, write_atomic, AttrReport, GridInfo,
    GroupMember, GroupReport, LabelReport, ProvenanceReport, Report, SlideSummary,
};
use h3d_segment::LabelStore;
use h3d_volume::{
    build_voxel_mask, cavalieri, cavalieri_areas, measure_label, plan_grid, reconstruct_label,
    GridPlan, LabelMesh, LabelSampler, SliceInput, VoxelMask, Warning,
};
use serde::{Deserialize, Serialize};

use super::{ReconState, Session, APP_VERSION};
use crate::error::EngineError;
use crate::hash::{json_bytes, Fnv1a64};

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
            let mut annotated = Vec::new();
            let mut reviewed = Vec::new();
            for l in store.labels_present() {
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
            });
        }
        Ok(out)
    }

    /// Label (terdefinisi) yang muncul di data irisan, terurut id.
    fn present_labels(&self, data: &[SliceData<'_>]) -> Vec<LabelDef> {
        self.project
            .labels
            .iter()
            .filter(|l| {
                data.iter()
                    .any(|d| d.annotated.iter().any(|(id, _)| *id == l.id.0))
            })
            .cloned()
            .collect()
    }

    /// Kunci cache voxel: hash atas label (byte `.lbl`), transformasi, `z`,
    /// tanda provenance, definisi label terkait, `ReconConfig`, dan anggaran
    /// (yang menentukan kenaikan `s_xy`). Berubah salah satunya → cache basi.
    pub(crate) fn recon_key(&self, data: &[SliceData<'_>]) -> Result<u64, EngineError> {
        let mut h = Fnv1a64::new();
        h.str("h3d-recon-v1");
        h.chunk(&json_bytes(&self.project.recon)?);
        h.u64(self.budgets.recon_working_set_bytes);
        for l in self.present_labels(data) {
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
            h.chunk(&h3d_segment::encode_lbl(&d.store));
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
        for l in &manifest.mesh_labels {
            let path = dir.join(format!("{l:03}.mesh"));
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
                &dir.join(format!("{:03}.mesh", m.label.0)),
                &encode_mesh(m)?,
            )?;
        }
        let manifest = CacheManifest {
            key: format!("{:016x}", state.key),
            plan: state.plan.clone(),
            mask_labels: state.masks.iter().map(|(l, _)| l.0).collect(),
            mesh_labels: state.meshes.iter().map(|m| m.label.0).collect(),
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
    /// tanpa perubahan memakai cache.
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
                let n = labels.len().max(1) as f32;
                for (i, l) in labels.iter().enumerate() {
                    if p.is_cancelled() {
                        return Err(EngineError::Cancelled);
                    }
                    let sub = SubProgress::new(p, i as f32 / n, (i + 1) as f32 / n);
                    if l.reconstruct {
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
            return Ok(meshes);
        }
        // Cache gagal ditulis bukan galat rekonstruksi.
        if let Err(e) = self.write_recon_cache(&state) {
            log::warn!("gagal menulis cache voxel: {e}");
        }
        let meshes = state.meshes.clone();
        self.recon = Some(state);
        self.analysis = None;
        p.set(1.0, "rekonstruksi selesai");
        Ok(meshes)
    }

    /// Mesh hasil rekonstruksi terakhir yang masih sah (memori atau cache);
    /// kosong bila belum direkonstruksi atau sudah basi.
    pub fn meshes(&self) -> Result<Vec<LabelMesh>, EngineError> {
        let data = self.slice_data()?;
        if data.is_empty() {
            return Ok(Vec::new());
        }
        let key = self.recon_key(&data)?;
        Ok(self
            .recon_for_key(key)
            .map(|r| r.meshes.clone())
            .unwrap_or_default())
    }

    /// Masker voxel hasil rekonstruksi terakhir yang masih sah.
    pub fn voxel_masks(&self) -> Result<Vec<(LabelId, VoxelMask)>, EngineError> {
        let data = self.slice_data()?;
        if data.is_empty() {
            return Ok(Vec::new());
        }
        let key = self.recon_key(&data)?;
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
            let mesh = recon
                .as_ref()
                .and_then(|r| r.meshes.iter().find(|m| m.label == l.id));
            let row = match mesh {
                Some(mesh) => {
                    let m = measure_label(mesh, &inputs, id, shrink);
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
        if !prov.drafts.is_empty() {
            report.warnings.push(Warning::HasDraft);
        }
        let n_interp = project
            .annotations
            .iter()
            .filter(|a| a.source.is_interpolated())
            .count();
        if prov.total > 0 && n_interp * 2 > prov.total {
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
