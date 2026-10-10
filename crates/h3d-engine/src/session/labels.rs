//! Label per irisan: muat malas, suntingan ber-undo, usulan, provenance
//! (8.5, 8.12.7).
//!
//! Aturan keras 8: usulan (klasik, AI, interpolasi) **tidak pernah** masuk ke
//! label tanpa diterima. Semua usulan lewat [`Session::accept_proposal`];
//! `segment_ai` dan `interpolate_labels` hanya mengisi antrean usulan
//! tertunda yang diterima lewat [`Session::accept_pending`].

use std::path::Path;

use h3d_core::{
    AnnotationRecord, AnnotationSource, LabelDef, LabelId, LabelKind, Progress, ReviewStatus,
    SlideId, SlideStatus, SubProgress, UndoStack,
};
use h3d_segment::{
    ClassicParams, ImportSummary, LabelEditCommand, LabelStore, PaintMode, Proposal, Segmenter,
    TractSegmenter,
};

use super::images::read_region_at_mpp;
use super::{LabelSlot, Session};
use crate::error::EngineError;

/// Usulan yang menunggu keputusan pengguna.
#[derive(Clone, Debug)]
pub struct PendingProposal {
    /// Usulan (masker + label + asal).
    pub proposal: Proposal,
    /// Rata-rata probabilitas argmax (AI); `None` untuk sumber lain.
    pub confidence: Option<f32>,
}

impl Session {
    /// Pastikan label irisan ada di memori (dimuat dari `labels/NNNN.lbl`,
    /// atau penyimpanan kosong seukuran slide bila belum ada berkasnya).
    pub(crate) fn ensure_labels(&mut self, slide: SlideId) -> Result<&mut LabelSlot, EngineError> {
        if !self.labels.contains_key(&slide) {
            let entry = self.slide(slide)?;
            if !entry.has_valid_mpp() {
                return Err(EngineError::MppMissing {
                    file: entry.source.file_name(),
                });
            }
            let store = match self.bundle.load_labels(slide)? {
                Some(s) => s,
                None => LabelStore::new(self.project.recon.mpp_label, entry.size_um()),
            };
            self.labels.insert(
                slide,
                LabelSlot {
                    store,
                    dirty: false,
                    undo: UndoStack::default(),
                },
            );
        }
        self.labels
            .get_mut(&slide)
            .ok_or(EngineError::SlideNotFound(slide))
    }

    /// Label irisan (hanya-baca; dimuat malas). Menyunting hanya lewat
    /// [`Self::edit_labels`] dan kawan-kawannya.
    pub fn labels(&mut self, slide: SlideId) -> Result<&LabelStore, EngineError> {
        Ok(&self.ensure_labels(slide)?.store)
    }

    /// Label irisan bila sudah ada di memori (tanpa memuat dari disk);
    /// untuk pembacaan dari kode render yang hanya memegang `&Session`.
    pub fn labels_loaded(&self, slide: SlideId) -> Option<&LabelStore> {
        self.labels.get(&slide).map(|s| &s.store)
    }

    /// Benar bila label irisan punya suntingan yang belum disimpan.
    pub fn labels_dirty(&self, slide: SlideId) -> bool {
        self.labels.get(&slide).is_some_and(|s| s.dirty)
    }

    /// Simpan label satu irisan (atomik) dan `project.json` (provenance).
    pub fn save_labels(&mut self, slide: SlideId) -> Result<(), EngineError> {
        if let Some(slot) = self.labels.get_mut(&slide) {
            if slot.dirty {
                self.bundle.save_labels(slide, &slot.store)?;
                slot.store.clear_dirty();
                slot.dirty = false;
            }
        }
        self.save_project()
    }

    /// Buang label irisan dari memori (mis. saat peringatan memori di iPad).
    /// Suntingan yang belum disimpan disimpan dulu.
    pub fn unload_labels(&mut self, slide: SlideId) -> Result<(), EngineError> {
        if self.labels_dirty(slide) {
            self.save_labels(slide)?;
        }
        self.labels.remove(&slide);
        Ok(())
    }

    // ---- provenance ----

    fn manual_source(&self, edited_from: Option<String>) -> AnnotationSource {
        AnnotationSource::Manual {
            by: self.project.user_initials.clone(),
            edited_from,
        }
    }

    /// Selaraskan `AnnotationRecord` irisan dengan isi label: catatan untuk
    /// label yang sudah tidak ada dibuang; label tanpa catatan mendapat
    /// `Manual`; `touched` (label yang baru disunting tangan) beralih ke
    /// `Manual` dengan `edited_from` bila sebelumnya hasil otomatis, dan
    /// status tinjauannya kembali `Draft`.
    fn sync_annotations(&mut self, slide: SlideId, touched: Option<LabelId>) {
        let Some(slot) = self.labels.get(&slide) else {
            return;
        };
        let present = slot.store.labels_present();
        let before = self.project.annotations.len();
        self.project
            .annotations
            .retain(|a| a.slide != slide || present.contains(&a.label));
        let mut changed = before != self.project.annotations.len();
        let now = self.now();
        for label in &present {
            let existing = self.project.annotation(slide, *label).cloned();
            let is_touched = touched == Some(*label);
            let new = match existing {
                None => Some(AnnotationRecord {
                    slide,
                    label: *label,
                    source: self.manual_source(None),
                    at: now.clone(),
                    review: ReviewStatus::Draft,
                    confidence: None,
                }),
                Some(rec) if is_touched => {
                    let source = match &rec.source {
                        AnnotationSource::Manual { edited_from, .. } => {
                            self.manual_source(edited_from.clone())
                        }
                        other => self.manual_source(Some(other.short_name().to_owned())),
                    };
                    Some(AnnotationRecord {
                        slide,
                        label: *label,
                        source,
                        at: now.clone(),
                        review: ReviewStatus::Draft,
                        confidence: None,
                    })
                }
                Some(_) => None,
            };
            if let Some(rec) = new {
                self.project.upsert_annotation(rec);
                changed = true;
            }
        }
        if changed {
            self.project_dirty = true;
        }
    }

    // ---- suntingan ber-undo ----

    /// Jalankan suntingan `mutate` atas label irisan sebagai satu langkah
    /// undo (`LabelEditCommand`). `touched` = label yang disunting tangan
    /// (untuk provenance); `None` bila suntingan tidak mewakili goresan pada
    /// label tertentu. Mengembalikan `true` bila ada piksel yang berubah.
    pub fn edit_labels<F>(
        &mut self,
        slide: SlideId,
        touched: Option<LabelId>,
        name: &str,
        mutate: F,
    ) -> Result<bool, EngineError>
    where
        F: FnOnce(&mut LabelStore),
    {
        let slot = self.ensure_labels(slide)?;
        let cmd = LabelEditCommand::capture(&mut slot.store, mutate, name);
        if cmd.is_noop() {
            return Ok(false);
        }
        // `capture` sudah menerapkan mutasi; `push` hanya menulis ulang isi
        // yang sama sambil mencatat langkah undo.
        slot.undo.push(Box::new(cmd), &mut slot.store);
        slot.dirty = true;
        self.sync_annotations(slide, touched);
        Ok(true)
    }

    /// Goresan kuas (titik dalam µm ruang U irisan).
    pub fn paint_stroke(
        &mut self,
        slide: SlideId,
        label: LabelId,
        points_um: &[[f64; 2]],
        radius_um: f64,
        mode: PaintMode,
    ) -> Result<bool, EngineError> {
        self.check_label(label)?;
        if !(radius_um.is_finite() && radius_um > 0.0) {
            return Err(EngineError::invalid("jari-jari kuas tidak sah"));
        }
        let touched = (mode != PaintMode::Erase).then_some(label);
        let name = if mode == PaintMode::Erase {
            "Hapus"
        } else {
            "Kuas"
        };
        self.edit_labels(slide, touched, name, |s| {
            s.paint_stroke(points_um, radius_um, label, mode);
        })
    }

    /// Isi poligon/laso tertutup (titik dalam µm ruang U irisan).
    pub fn fill_polygon(
        &mut self,
        slide: SlideId,
        label: LabelId,
        points_um: &[[f64; 2]],
        mode: PaintMode,
    ) -> Result<bool, EngineError> {
        self.check_label(label)?;
        let touched = (mode != PaintMode::Erase).then_some(label);
        self.edit_labels(slide, touched, "Poligon", |s| {
            s.fill_polygon(points_um, label, mode);
        })
    }

    /// Isi lubang sebuah label.
    pub fn fill_label_holes(
        &mut self,
        slide: SlideId,
        label: LabelId,
    ) -> Result<bool, EngineError> {
        self.check_label(label)?;
        self.edit_labels(slide, Some(label), "Isi lubang", |s| {
            s.fill_holes(label);
        })
    }

    /// Batalkan suntingan terakhir irisan; `Some(nama)` bila ada.
    pub fn undo(&mut self, slide: SlideId) -> Result<Option<String>, EngineError> {
        let slot = self.ensure_labels(slide)?;
        let name = slot.undo.undo(&mut slot.store);
        if name.is_some() {
            slot.dirty = true;
            self.sync_annotations(slide, None);
        }
        Ok(name)
    }

    /// Ulangi suntingan yang dibatalkan; `Some(nama)` bila ada.
    pub fn redo(&mut self, slide: SlideId) -> Result<Option<String>, EngineError> {
        let slot = self.ensure_labels(slide)?;
        let name = slot.undo.redo(&mut slot.store);
        if name.is_some() {
            slot.dirty = true;
            self.sync_annotations(slide, None);
        }
        Ok(name)
    }

    /// Ada yang bisa dibatalkan pada irisan ini.
    pub fn can_undo(&self, slide: SlideId) -> bool {
        self.labels.get(&slide).is_some_and(|s| s.undo.can_undo())
    }

    /// Ada yang bisa diulang pada irisan ini.
    pub fn can_redo(&self, slide: SlideId) -> bool {
        self.labels.get(&slide).is_some_and(|s| s.undo.can_redo())
    }

    // ---- usulan ----

    /// Terima usulan: satu-satunya jalur usulan → label. Membuat/memperbarui
    /// `AnnotationRecord` dengan asal usulan (status `Draft`). Catatan
    /// non-interpolasi yang sudah ada tidak diturunkan menjadi
    /// `Interpolated`.
    pub fn accept_proposal(
        &mut self,
        slide: SlideId,
        proposal: &Proposal,
        mode: PaintMode,
    ) -> Result<bool, EngineError> {
        self.accept_proposal_with_confidence(slide, proposal, mode, None)
    }

    /// Seperti [`Self::accept_proposal`], dengan keyakinan model AI.
    pub fn accept_proposal_with_confidence(
        &mut self,
        slide: SlideId,
        proposal: &Proposal,
        mode: PaintMode,
        confidence: Option<f32>,
    ) -> Result<bool, EngineError> {
        self.check_label(proposal.label)?;
        let label = proposal.label;
        // Catatan sebelum usulan diterima: anotasi nyata (bukan interpolasi)
        // tidak boleh diturunkan menjadi `Interpolated`.
        let keep_existing = proposal.source.is_interpolated()
            && self
                .project
                .annotation(slide, label)
                .is_some_and(|a| !a.source.is_interpolated());
        let changed = self.edit_labels(slide, None, "Terima usulan", |s| {
            h3d_segment::accept_proposal(s, proposal, mode);
        })?;
        if !changed {
            return Ok(false);
        }
        let present = self
            .labels
            .get(&slide)
            .is_some_and(|s| s.store.labels_present().contains(&label));
        if present && !keep_existing {
            self.project.upsert_annotation(AnnotationRecord {
                slide,
                label,
                source: proposal.source.clone(),
                at: self.now(),
                review: ReviewStatus::Draft,
                confidence,
            });
            self.project_dirty = true;
        }
        Ok(true)
    }

    /// Segmentasi klasik (8.5.3) → usulan. Tidak mengubah label.
    pub fn segment_classic(
        &self,
        slide: SlideId,
        label: LabelId,
        params: &ClassicParams,
    ) -> Result<Proposal, EngineError> {
        self.check_label(label)?;
        let raster =
            self.read_slide_at(slide, self.project.recon.mpp_label, &h3d_core::NoProgress)?;
        let tissue = h3d_imgproc::tissue_mask(&raster.rgba, raster.w, raster.h, raster.mpp)?;
        let mask = h3d_segment::segment_classic(
            &raster.rgba,
            raster.w,
            raster.h,
            raster.mpp,
            Some(&tissue),
            params,
        )?;
        Ok(h3d_segment::classic::classic_proposal(
            mask,
            raster.mpp,
            [0.0, 0.0],
            label,
            params,
        ))
    }

    /// Segmentasi AI (8.5.4) atas `slides` dengan model di folder `model`
    /// (`model.json` + `model.onnx`). Hasilnya masuk antrean usulan tertunda;
    /// label baru berubah setelah [`Self::accept_pending`].
    pub fn segment_ai(
        &mut self,
        model: &Path,
        slides: &[SlideId],
        p: &dyn Progress,
    ) -> Result<(), EngineError> {
        let seg = TractSegmenter::load(model)?;
        self.segment_with(&seg, slides, p)
    }

    /// Seperti [`Self::segment_ai`] dengan backend inferensi apa pun.
    pub fn segment_with(
        &mut self,
        seg: &dyn Segmenter,
        slides: &[SlideId],
        p: &dyn Progress,
    ) -> Result<(), EngineError> {
        let total = slides.len().max(1) as f32;
        for (i, &slide) in slides.iter().enumerate() {
            if p.is_cancelled() {
                return Err(EngineError::Cancelled);
            }
            let sub = SubProgress::new(p, i as f32 / total, (i + 1) as f32 / total);
            let entry = self.slide(slide)?.clone();
            let src = self.open_source(slide)?;
            let reg = self.registration_image(slide)?;
            let mpp_model = seg.spec().mpp;
            let mpp_l0 = entry.mpp;
            let read_tile = |x_um: f64, y_um: f64, w: usize, h: usize| {
                read_region_at_mpp(src.as_ref(), mpp_l0, x_um, y_um, w, h, mpp_model).map_err(|e| {
                    h3d_segment::SegError::InvalidParameter(format!("gagal membaca tile: {e}"))
                })
            };
            let project = &self.project;
            let label_map = |name: &str| project.label_by_name(name).map(|l| l.id);
            let result = h3d_segment::infer_slide(
                seg,
                &read_tile,
                entry.size_um(),
                Some((&reg.mask, reg.mpp)),
                &label_map,
                self.project.recon.mpp_label,
                &sub,
            )?;
            let confidence = Some(result.mean_confidence);
            let list: Vec<PendingProposal> = result
                .proposals()
                .into_iter()
                .filter(|pr| !pr.is_empty())
                .map(|proposal| PendingProposal {
                    proposal,
                    confidence,
                })
                .collect();
            self.pending.entry(slide).or_default().extend(list);
        }
        p.set(1.0, "segmentasi selesai");
        Ok(())
    }

    /// Interpolasi bentuk `label` (8.5.5) untuk irisan `Ok` yang terletak di
    /// antara dua irisan beranotasi (bukan interpolasi) dan belum memuat
    /// label itu. Hasilnya **usulan tertunda** ber-asal `Interpolated`;
    /// terima dengan [`Self::accept_pending`]. Irisan hasil interpolasi tidak
    /// ikut Cavalieri (8.6.4).
    pub fn interpolate_labels(
        &mut self,
        label: LabelId,
        p: &dyn Progress,
    ) -> Result<(), EngineError> {
        self.check_label(label)?;
        let cfg = self.project.recon.clone();
        let band_um = cfg.band_px * cfg.s_xy_um;
        let mut out: Vec<(SlideId, PendingProposal)> = Vec::new();
        {
            let stores = self.label_snapshot()?;
            let ordered = self.project.usable_slides_ordered();
            let has_real = |id: SlideId| -> bool {
                stores
                    .iter()
                    .find(|(sid, _)| *sid == id)
                    .is_some_and(|(_, st)| st.labels_present().contains(&label))
                    && !self
                        .project
                        .annotation(id, label)
                        .is_some_and(|a| a.source.is_interpolated())
            };
            let has_any = |id: SlideId| -> bool {
                stores
                    .iter()
                    .find(|(sid, _)| *sid == id)
                    .is_some_and(|(_, st)| st.labels_present().contains(&label))
            };
            let anchors: Vec<usize> = (0..ordered.len())
                .filter(|&i| has_real(ordered[i].id))
                .collect();
            let n_pairs = anchors.len().saturating_sub(1).max(1) as f32;
            for (pi, w) in anchors.windows(2).enumerate() {
                if p.is_cancelled() {
                    return Err(EngineError::Cancelled);
                }
                p.set(pi as f32 / n_pairs, "interpolasi label");
                let (a, b) = (ordered[w[0]], ordered[w[1]]);
                let (Some((_, sa)), Some((_, sb))) = (
                    stores.iter().find(|(id, _)| *id == a.id),
                    stores.iter().find(|(id, _)| *id == b.id),
                ) else {
                    continue;
                };
                for k in &ordered[w[0] + 1..w[1]] {
                    if has_any(k.id) {
                        continue;
                    }
                    let proposal = h3d_segment::interpolate_label(
                        sa,
                        &a.transform,
                        a.z_um,
                        sb,
                        &b.transform,
                        b.z_um,
                        &k.transform,
                        k.z_um,
                        [a.id, b.id],
                        label,
                        cfg.s_xy_um,
                        band_um,
                    )?;
                    if !proposal.is_empty() {
                        out.push((
                            k.id,
                            PendingProposal {
                                proposal,
                                confidence: None,
                            },
                        ));
                    }
                }
            }
        }
        for (id, pp) in out {
            self.pending.entry(id).or_default().push(pp);
        }
        p.set(1.0, "interpolasi selesai");
        Ok(())
    }

    /// Usulan tertunda untuk satu irisan.
    pub fn pending_proposals(&self, slide: SlideId) -> &[PendingProposal] {
        self.pending.get(&slide).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Irisan yang punya usulan tertunda.
    pub fn pending_slides(&self) -> Vec<SlideId> {
        self.pending
            .iter()
            .filter(|(_, v)| !v.is_empty())
            .map(|(id, _)| *id)
            .collect()
    }

    /// Terima semua usulan tertunda satu irisan; mengembalikan jumlah usulan
    /// yang mengubah label.
    pub fn accept_pending(
        &mut self,
        slide: SlideId,
        mode: PaintMode,
    ) -> Result<usize, EngineError> {
        let list = self.pending.remove(&slide).unwrap_or_default();
        let mut n = 0;
        for pp in &list {
            if self.accept_proposal_with_confidence(slide, &pp.proposal, mode, pp.confidence)? {
                n += 1;
            }
        }
        Ok(n)
    }

    /// Terima semua usulan tertunda di semua irisan.
    pub fn accept_all_pending(&mut self, mode: PaintMode) -> Result<usize, EngineError> {
        let mut n = 0;
        for slide in self.pending_slides() {
            n += self.accept_pending(slide, mode)?;
        }
        Ok(n)
    }

    /// Tolak (buang) usulan tertunda satu irisan. Label tidak berubah.
    pub fn reject_pending(&mut self, slide: SlideId) {
        self.pending.remove(&slide);
    }

    // ---- alat lain ----

    /// "Salin ke irisan berikut" (8.5.2): salin semua label irisan ini ke
    /// irisan `Ok` berikutnya lewat `T_berikut⁻¹ ∘ T_ini`, hanya ke piksel
    /// latar. Mengembalikan id irisan tujuan.
    pub fn copy_labels_to_next(&mut self, slide: SlideId) -> Result<SlideId, EngineError> {
        let order = self.ok_order();
        let pos = order
            .iter()
            .position(|id| *id == slide)
            .ok_or_else(|| EngineError::invalid(format!("irisan {slide} tidak berstatus Ok")))?;
        let next = *order
            .get(pos + 1)
            .ok_or_else(|| EngineError::invalid("tidak ada irisan berikutnya"))?;
        let t_src = self.slide(slide)?.transform.affine();
        let t_dst = self.slide(next)?.transform.affine();
        // other_to_self: U sumber → W → U tujuan.
        let map = t_dst.try_inverse()?.compose(&t_src);
        let src = self.ensure_labels(slide)?.store.clone();
        let labels = src.labels_present();
        if labels.is_empty() {
            return Err(EngineError::invalid(format!(
                "irisan {slide} belum punya label untuk disalin"
            )));
        }
        let before: Vec<LabelId> = self.ensure_labels(next)?.store.labels_present();
        let mut err: Option<h3d_segment::SegError> = None;
        self.edit_labels(next, None, "Salin dari irisan sebelumnya", |s| {
            if let Err(e) = s.copy_from(&src, &map, &labels, PaintMode::OnlyBackground) {
                err = Some(e);
            }
        })?;
        if let Some(e) = err {
            return Err(e.into());
        }
        // Label yang baru muncul berasal dari salinan: `Manual`, `edited_from`.
        let now = self.now();
        let present = self.ensure_labels(next)?.store.labels_present();
        for l in present {
            if !before.contains(&l) {
                let rec = AnnotationRecord {
                    slide: next,
                    label: l,
                    source: self.manual_source(Some("copied".to_owned())),
                    at: now.clone(),
                    review: ReviewStatus::Draft,
                    confidence: None,
                };
                self.project.upsert_annotation(rec);
                self.project_dirty = true;
            }
        }
        Ok(next)
    }

    /// Impor anotasi GeoJSON (QuPath) ke irisan sebagai satu langkah undo.
    /// Nama kelas dicocokkan dengan tabel label; provenance label yang
    /// berubah menjadi `Imported { file }`.
    pub fn import_geojson(
        &mut self,
        slide: SlideId,
        path: &Path,
    ) -> Result<ImportSummary, EngineError> {
        let text = std::fs::read_to_string(path).map_err(|e| h3d_io::IoError::io(path, e))?;
        let mpp = self.slide(slide)?.mpp;
        let defs: Vec<(String, LabelId)> = self
            .project
            .labels
            .iter()
            .map(|l| (l.name.trim().to_lowercase(), l.id))
            .collect();
        let lookup = |name: &str| -> Option<LabelId> {
            let key = name.trim().to_lowercase();
            defs.iter().find(|(n, _)| *n == key).map(|(_, id)| *id)
        };
        let ids: Vec<LabelId> = self.project.labels.iter().map(|l| l.id).collect();
        let before: Vec<u64> = {
            let store = &self.ensure_labels(slide)?.store;
            ids.iter().map(|id| store.area_px(*id)).collect()
        };
        let mut result: Option<Result<ImportSummary, h3d_segment::SegError>> = None;
        self.edit_labels(slide, None, "Impor GeoJSON", |s| {
            result = Some(h3d_segment::import_geojson(&text, s, mpp, &lookup));
        })?;
        let summary = match result {
            Some(r) => r?,
            None => return Err(EngineError::invalid("impor GeoJSON tidak dijalankan")),
        };
        let file = path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let now = self.now();
        let after: Vec<u64> = {
            let store = &self.ensure_labels(slide)?.store;
            ids.iter().map(|id| store.area_px(*id)).collect()
        };
        for ((id, b), a) in ids.iter().zip(before).zip(after) {
            if a != b && a > 0 {
                self.project.upsert_annotation(AnnotationRecord {
                    slide,
                    label: *id,
                    source: AnnotationSource::Imported { file: file.clone() },
                    at: now.clone(),
                    review: ReviewStatus::Draft,
                    confidence: None,
                });
                self.project_dirty = true;
            }
        }
        Ok(summary)
    }

    /// Tandai anotasi `(irisan, label)` sudah ditinjau (8.12.7).
    pub fn mark_reviewed(
        &mut self,
        slide: SlideId,
        label: LabelId,
        by: &str,
    ) -> Result<(), EngineError> {
        self.slide(slide)?;
        self.check_label(label)?;
        if by.trim().is_empty() {
            return Err(EngineError::invalid("nama peninjau tidak boleh kosong"));
        }
        // Pastikan catatan selaras dengan isi label sebelum menandai.
        self.ensure_labels(slide)?;
        self.sync_annotations(slide, None);
        let at = self.now();
        let rec = self
            .project
            .annotations
            .iter_mut()
            .find(|a| a.slide == slide && a.label == label)
            .ok_or_else(|| {
                EngineError::invalid(format!(
                    "irisan {slide} tidak punya anotasi untuk label {label}"
                ))
            })?;
        rec.review = ReviewStatus::Reviewed {
            by: by.trim().to_owned(),
            at,
        };
        self.project_dirty = true;
        self.save_project()
    }

    // ---- tabel label ----

    pub(crate) fn check_label(&self, label: LabelId) -> Result<(), EngineError> {
        if label.is_background() || self.project.label(label).is_none() {
            return Err(EngineError::LabelNotFound(label.to_string()));
        }
        Ok(())
    }

    /// Cari label berdasarkan nama (tidak peka huruf besar/kecil).
    pub fn label_by_name(&self, name: &str) -> Result<LabelId, EngineError> {
        self.project
            .label_by_name(name)
            .map(|l| l.id)
            .ok_or_else(|| EngineError::LabelNotFound(name.to_owned()))
    }

    /// Tambah label baru; `reconstruct` mengikuti `kind`.
    pub fn add_label(
        &mut self,
        name: &str,
        color: [u8; 4],
        kind: LabelKind,
    ) -> Result<LabelId, EngineError> {
        let name = name.trim();
        if name.is_empty() {
            return Err(EngineError::invalid("nama label tidak boleh kosong"));
        }
        if self.project.label_by_name(name).is_some() {
            return Err(EngineError::invalid(format!("label \"{name}\" sudah ada")));
        }
        let id = self
            .project
            .next_label_id()
            .ok_or_else(|| EngineError::invalid("tabel label penuh (maksimum 255)"))?;
        self.project
            .labels
            .push(LabelDef::new(id, name, color, kind));
        self.project.labels.sort_by_key(|l| l.id);
        self.project_dirty = true;
        Ok(id)
    }

    /// Ubah definisi label (nama, warna, jenis, grup, tampil, rekonstruksi).
    pub fn update_label(&mut self, def: LabelDef) -> Result<(), EngineError> {
        self.check_label(def.id)?;
        if def.name.trim().is_empty() {
            return Err(EngineError::invalid("nama label tidak boleh kosong"));
        }
        if self
            .project
            .label_by_name(&def.name)
            .is_some_and(|l| l.id != def.id)
        {
            return Err(EngineError::invalid(format!(
                "label \"{}\" sudah ada",
                def.name.trim()
            )));
        }
        if let Some(l) = self.project.labels.iter_mut().find(|l| l.id == def.id) {
            *l = def;
        }
        self.project_dirty = true;
        Ok(())
    }

    /// Hapus label dari tabel. Ditolak bila masih dipakai anotasi mana pun,
    /// agar piksel berlabel tidak kehilangan definisinya.
    pub fn remove_label(&mut self, label: LabelId) -> Result<(), EngineError> {
        self.check_label(label)?;
        if self.project.annotations.iter().any(|a| a.label == label) {
            return Err(EngineError::invalid(
                "label masih dipakai anotasi; hapus dulu anotasinya",
            ));
        }
        self.project.labels.retain(|l| l.id != label);
        self.project_dirty = true;
        Ok(())
    }

    /// Irisan `Ok` terurut yang memuat `label` (interpolasi atau bukan).
    pub fn slides_with_label(&self, label: LabelId) -> Vec<SlideId> {
        self.project
            .usable_slides_ordered()
            .iter()
            .filter(|s| s.status == SlideStatus::Ok)
            .filter(|s| self.project.annotation(s.id, label).is_some())
            .map(|s| s.id)
            .collect()
    }
}
