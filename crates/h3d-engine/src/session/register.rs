//! Registrasi tumpukan, usulan urutan, koreksi QC (8.4, 8.8, 9.4).

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use h3d_core::{Affine2, OrderSource, Progress, RegRecord, SlideId, SlideStatus, SlideTransform};
use h3d_imgproc::{dice, ncc_masked, warp_affine_f32, warp_affine_u8, Interp};
use h3d_register::{
    compose_chain, from_landmarks, neighbor_toward_reference, pair_residual, record_from_result,
    BuiltinRegistrar, OrderSuggestion, QcThresholds, RegImage, RegInput, RegOptions, Registrar,
};
use ndarray::Array2;
use rayon::prelude::*;
use serde::Serialize;

use super::images::REG_IMAGE_MPP;
use super::Session;
use crate::error::EngineError;

/// Opsi lengkap satu jalan registrasi.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RegisterRun {
    /// Opsi registrasi berpasangan (model, metrik, piramida).
    pub options: RegOptions,
    /// Ambang QC (8.4.5).
    pub qc: QcThresholds,
    /// Irisan acuan; `None` = pakai acuan proyek (bawaan: tengah tumpukan).
    pub reference: Option<SlideId>,
    /// Daftarkan ulang semua pasangan walau catatannya masih sah.
    pub force: bool,
    /// Jumlah pasangan per gelombang (hasil disimpan tiap gelombang);
    /// 0 = jumlah thread rayon.
    pub batch: usize,
}

/// Ringkasan jalan registrasi.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct RegSummary {
    /// Irisan acuan (`T` = identitas).
    pub reference: Option<SlideId>,
    /// Irisan yang didaftarkan pada jalan ini.
    pub registered: Vec<SlideId>,
    /// Irisan yang dilewati karena catatannya masih sah (job dilanjutkan).
    pub skipped: Vec<SlideId>,
    /// Irisan yang gagal ambang QC (semua, termasuk dari jalan sebelumnya).
    pub flagged: Vec<SlideId>,
    /// Irisan `Ok` yang rantainya tidak sampai ke acuan.
    pub unreachable: Vec<SlideId>,
}

/// Satu baris tabel QC.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct QcRow {
    /// Irisan.
    pub slide: SlideId,
    /// Urutan dalam tumpukan.
    pub order: u32,
    /// Nama berkas.
    pub file: String,
    /// Status irisan.
    pub status: SlideStatus,
    /// Benar bila irisan ini acuan.
    pub is_reference: bool,
    /// Benar bila `T_i` tersedia (acuan atau rantai sampai ke acuan).
    pub registered: bool,
    /// Irisan tetap pasangan registrasinya.
    pub fixed: Option<SlideId>,
    /// Metode.
    pub method: Option<String>,
    /// NCC bermasker.
    pub ncc: Option<f32>,
    /// Dice masker jaringan.
    pub mask_dice: Option<f32>,
    /// `|det|` bagian linear.
    pub area_ratio: Option<f32>,
    /// Gagal ambang QC.
    pub flagged: bool,
    /// Hasil koreksi manual.
    pub manual: bool,
}

/// Meneruskan pembatalan saja; fraksi global dihitung dari pasangan selesai.
struct CancelOnly<'a>(&'a dyn Progress);

impl Progress for CancelOnly<'_> {
    fn set(&self, _fraction: f32, _stage: &str) {}
    fn is_cancelled(&self) -> bool {
        self.0.is_cancelled()
    }
}

/// NCC bermasker dan Dice masker untuk pasangan dengan `p` = `P_{moving→fixed}`
/// (µm → µm): citra bergerak di-warp mundur ke kerangka piksel citra tetap.
fn pair_metrics(fixed: &RegImage, moving: &RegImage, p: &Affine2) -> Option<(f32, f32)> {
    let inv = p.inverse()?;
    // t_inv: piksel tetap → µm tetap → µm bergerak → piksel bergerak.
    let t_inv = moving
        .um_to_px_affine()
        .compose(&inv)
        .compose(&fixed.px_to_um_affine());
    let size = fixed.size_px();
    let wg = warp_affine_f32(&moving.gray, &t_inv, size, Interp::Bilinear, 0.0).ok()?;
    let wm = warp_affine_u8(&moving.mask, &t_inv, size, Interp::Nearest, 0).ok()?;
    let both = Array2::from_shape_fn(fixed.mask.dim(), |i| {
        u8::from(fixed.mask[i] != 0 && wm[i] != 0)
    });
    let ncc = ncc_masked(&fixed.gray, &wg, Some(&both)).ok()?;
    let d = dice(&fixed.mask, &wm).ok()?;
    (ncc.is_finite() && d.is_finite()).then_some((ncc, d))
}

impl Session {
    /// Irisan `Ok` terurut.
    pub(crate) fn ok_order(&self) -> Vec<SlideId> {
        self.project
            .usable_slides_ordered()
            .iter()
            .map(|s| s.id)
            .collect()
    }

    /// Tetangga `slide` ke arah acuan efektif.
    pub(crate) fn neighbor_of(&self, slide: SlideId) -> Option<SlideId> {
        let reference = self.project.effective_reference()?;
        neighbor_toward_reference(&self.ok_order(), slide, reference)
    }

    /// Buang `RegRecord` yang tetangganya (ke arah acuan) sudah berubah.
    /// Mengembalikan irisan yang catatannya dibuang (perlu registrasi ulang).
    pub(crate) fn drop_stale_registrations(&mut self) -> Vec<SlideId> {
        let order = self.ok_order();
        let reference = self.project.effective_reference();
        let mut dropped = Vec::new();
        for s in &mut self.project.slides {
            if s.status != SlideStatus::Ok {
                continue;
            }
            let Some(rec) = &s.reg else { continue };
            let nb = reference.and_then(|r| neighbor_toward_reference(&order, s.id, r));
            if nb != Some(rec.fixed) {
                s.reg = None;
                dropped.push(s.id);
            }
        }
        if !dropped.is_empty() {
            self.project_dirty = true;
        }
        dropped
    }

    /// Hitung ulang semua `T_i` (U → W) dari catatan berpasangan yang
    /// tersimpan, TANPA registrasi ulang (8.4.5): `T_acuan = identitas`,
    /// `T_i = T_tetangga ∘ P_{i→tetangga}`. Irisan yang rantainya putus
    /// kembali ke `Identity` (belum terdaftar).
    pub(crate) fn recompose_chain(&mut self) {
        let Some(reference) = self.project.effective_reference() else {
            return;
        };
        let order = self.ok_order();
        let records: Vec<(SlideId, RegRecord)> = self
            .project
            .slides
            .iter()
            .filter(|s| s.status == SlideStatus::Ok)
            .filter_map(|s| s.reg.clone().map(|r| (s.id, r)))
            .collect();
        let t = compose_chain(&records, reference, &order);
        for s in &mut self.project.slides {
            let new = match t.get(&s.id) {
                Some(a) if s.id != reference => SlideTransform::Affine(*a),
                _ => SlideTransform::Identity,
            };
            if s.transform != new {
                s.transform = new;
                self.project_dirty = true;
            }
        }
    }

    /// Irisan `Ok` (selain acuan) yang belum punya `T_i`.
    pub fn needs_registration(&self) -> Vec<SlideId> {
        let reference = self.project.effective_reference();
        self.project
            .usable_slides_ordered()
            .iter()
            .filter(|s| Some(s.id) != reference && s.reg.is_none())
            .map(|s| s.id)
            .collect()
    }

    /// Tebakan awal `P_{i→tetangga}` dari `orient_hint` (8.4.7):
    /// `orient_tetangga⁻¹ ∘ orient_i`. `None` bila keduanya tanpa petunjuk.
    fn init_guess(&self, moving: SlideId, fixed: SlideId) -> Option<Affine2> {
        let om = self.project.slide(moving)?.orient_hint;
        let of = self.project.slide(fixed)?.orient_hint;
        if om.is_none() && of.is_none() {
            return None;
        }
        let om = om.unwrap_or(Affine2::IDENTITY);
        let of = of.unwrap_or(Affine2::IDENTITY);
        Some(of.inverse()?.compose(&om))
    }

    /// Daftarkan semua irisan `Ok` ke tetangganya ke arah acuan (8.8) dengan
    /// ambang QC bawaan. Lihat [`Self::register_all_with`].
    pub fn register_all(
        &mut self,
        opts: &RegOptions,
        p: &dyn Progress,
    ) -> Result<RegSummary, EngineError> {
        self.register_all_with(
            &RegisterRun {
                options: opts.clone(),
                ..Default::default()
            },
            p,
        )
    }

    /// Registrasi dengan opsi lengkap. **Bisa dilanjutkan (9.4):** pasangan
    /// dijalankan bergelombang; setelah tiap gelombang catatan disimpan,
    /// rantai dikomposisi ulang, dan `project.json` ditulis. Irisan yang
    /// sudah punya `RegRecord` sah terhadap tetangga yang sama dilewati
    /// kecuali `force`. Pembatalan mengembalikan `Cancelled` dengan proyek
    /// konsisten di disk. Hasil tidak bergantung pada ukuran gelombang.
    pub fn register_all_with(
        &mut self,
        run: &RegisterRun,
        p: &dyn Progress,
    ) -> Result<RegSummary, EngineError> {
        if let Some(r) = run.reference {
            self.set_reference(Some(r))?;
        }
        let reference = self.project.effective_reference().ok_or_else(|| {
            EngineError::invalid("tidak ada irisan berstatus Ok untuk didaftarkan")
        })?;
        // Acuan dipatok begitu registrasi dijalankan: acuan bawaan ("tengah
        // tumpukan") tidak boleh bergeser diam-diam saat irisan lain
        // dikecualikan, karena itu menggeser seluruh ruang W.
        if self.project.reference != Some(reference) {
            self.project.reference = Some(reference);
            self.project_dirty = true;
        }
        for s in self.project.usable_slides_ordered() {
            if !s.has_valid_mpp() {
                return Err(EngineError::MppMissing {
                    file: s.source.file_name(),
                });
            }
        }
        self.drop_stale_registrations();
        let order = self.ok_order();
        let mut summary = RegSummary {
            reference: Some(reference),
            ..Default::default()
        };
        let mut todo: Vec<(SlideId, SlideId)> = Vec::new();
        for &id in &order {
            let Some(nb) = neighbor_toward_reference(&order, id, reference) else {
                continue;
            };
            let has = self.project.slide(id).is_some_and(|s| s.reg.is_some());
            if has && !run.force {
                summary.skipped.push(id);
            } else {
                todo.push((id, nb));
            }
        }

        let registrar = BuiltinRegistrar;
        let batch = if run.batch == 0 {
            rayon::current_num_threads().max(1)
        } else {
            run.batch
        };
        let total = todo.len().max(1);
        let done = AtomicUsize::new(0);
        p.set(0.0, "registrasi berpasangan");
        for chunk in todo.chunks(batch) {
            if p.is_cancelled() {
                self.recompose_chain();
                self.save_project()?;
                return Err(EngineError::Cancelled);
            }
            // Muat gambar registrasi yang dibutuhkan gelombang ini saja.
            let mut images: HashMap<SlideId, Arc<RegImage>> = HashMap::new();
            for &(m, f) in chunk {
                for id in [m, f] {
                    if let std::collections::hash_map::Entry::Vacant(e) = images.entry(id) {
                        e.insert(self.registration_image(id)?);
                    }
                }
            }
            let jobs: Vec<(SlideId, SlideId, Option<Affine2>)> = chunk
                .iter()
                .map(|&(m, f)| (m, f, self.init_guess(m, f)))
                .collect();
            let cancel = CancelOnly(p);
            let results: Vec<Result<(SlideId, RegRecord), EngineError>> = jobs
                .par_iter()
                .map(|&(m, f, init)| {
                    if p.is_cancelled() {
                        return Err(EngineError::Cancelled);
                    }
                    let (Some(fi), Some(mi)) = (images.get(&f), images.get(&m)) else {
                        return Err(EngineError::invalid("gambar registrasi hilang"));
                    };
                    let rec = match registrar.register(fi, mi, init, &run.options, &cancel) {
                        Ok(res) => {
                            let mut rec =
                                record_from_result(f, registrar.name(), &res, &run.qc, false);
                            let residual = pair_residual(fi, mi, &res.transform);
                            rec.residual_um = residual.map(|r| r.local_um);
                            rec.residual_fit_um = residual.map(|r| r.fit_um);
                            rec
                        }
                        Err(h3d_register::RegError::Cancelled) => {
                            return Err(EngineError::Cancelled)
                        }
                        Err(e) => {
                            // Satu pasangan gagal (mis. masker kosong) tidak
                            // menggagalkan tumpukan: dicatat sebagai bertanda
                            // agar pengguna membetulkannya lewat landmark.
                            log::warn!("registrasi irisan {m} → {f} gagal: {e}");
                            RegRecord {
                                fixed: f,
                                method: "builtin-failed".to_owned(),
                                pairwise: init.unwrap_or(Affine2::IDENTITY),
                                ncc: 0.0,
                                mask_dice: 0.0,
                                area_ratio: 1.0,
                                flagged: true,
                                manual: false,
                                residual_um: None,
                                residual_fit_um: None,
                            }
                        }
                    };
                    let n = done.fetch_add(1, Ordering::SeqCst) + 1;
                    p.set(n as f32 / total as f32, "registrasi berpasangan");
                    Ok((m, rec))
                })
                .collect();
            // Simpan hasil pasangan yang selesai walau gelombang ini dibatalkan.
            let mut cancelled = false;
            let mut first_err: Option<EngineError> = None;
            for r in results {
                match r {
                    Ok((id, rec)) => {
                        if let Some(s) = self.project.slide_mut(id) {
                            s.reg = Some(rec);
                        }
                        summary.registered.push(id);
                    }
                    Err(EngineError::Cancelled) => cancelled = true,
                    Err(e) => first_err = first_err.or(Some(e)),
                }
            }
            self.recompose_chain();
            self.save_project()?;
            if let Some(e) = first_err {
                return Err(e);
            }
            if cancelled {
                return Err(EngineError::Cancelled);
            }
        }
        self.recompose_chain();
        self.save_project()?;
        for s in self.project.usable_slides_ordered() {
            if s.reg.as_ref().is_some_and(|r| r.flagged) {
                summary.flagged.push(s.id);
            }
            if s.id != reference && !s.transform.is_registered() && s.reg.is_none() {
                summary.unreachable.push(s.id);
            }
        }
        p.set(1.0, "registrasi selesai");
        Ok(summary)
    }

    /// Usulan urutan dan orientasi (8.4.7). Tidak mengubah proyek.
    pub fn suggest_order(&self, p: &dyn Progress) -> Result<OrderSuggestion, EngineError> {
        let slides = self.project.slides_ordered();
        let mut inputs = Vec::with_capacity(slides.len());
        for s in &slides {
            if p.is_cancelled() {
                return Err(EngineError::Cancelled);
            }
            let image = if s.status == SlideStatus::Ok {
                self.registration_image(s.id)?
            } else {
                // Irisan non-Ok tidak dihitung; cukup citra pengganti mungil.
                Arc::new(RegImage::new(
                    Array2::zeros((1, 1)),
                    Array2::zeros((1, 1)),
                    REG_IMAGE_MPP,
                    [0.0, 0.0],
                )?)
            };
            inputs.push(RegInput {
                id: s.id,
                order: s.order,
                status: s.status,
                image,
                init: None,
            });
        }
        let current: Vec<SlideId> = slides.iter().map(|s| s.id).collect();
        Ok(h3d_register::suggest_order(&inputs, &current, p)?)
    }

    /// Terima usulan urutan untuk irisan `accept` saja (8.4.7). Irisan yang
    /// tidak diterima tetap di posisinya; irisan yang diterima mengisi slot
    /// sisanya menurut urutan usulan.
    ///
    /// Yang berubah: `order` (+ `order_source = Suggested` bagi yang pindah),
    /// `orient_hint` irisan yang diterima, `z_um` **hanya** bagi irisan yang
    /// `z`-nya masih bawaan, dan `RegRecord` irisan yang tetangganya berubah
    /// (dibuang → perlu registrasi ulang). Label dan `z_um` isian tangan
    /// tidak disentuh.
    pub fn apply_order_suggestion(
        &mut self,
        s: &OrderSuggestion,
        accept: &[SlideId],
    ) -> Result<(), EngineError> {
        self.check_permutation(&s.order)?;
        let current: Vec<SlideId> = self.project.slides_ordered().iter().map(|e| e.id).collect();
        for id in accept {
            if !current.contains(id) {
                return Err(EngineError::SlideNotFound(*id));
            }
        }
        let mut new_order = current.clone();
        let slots: Vec<usize> = current
            .iter()
            .enumerate()
            .filter(|(_, id)| accept.contains(id))
            .map(|(i, _)| i)
            .collect();
        let accepted: Vec<SlideId> = s
            .order
            .iter()
            .copied()
            .filter(|id| accept.contains(id))
            .collect();
        for (slot, id) in slots.iter().zip(accepted.iter()) {
            new_order[*slot] = *id;
        }
        for (i, id) in new_order.iter().enumerate() {
            if let Some(e) = self.project.slide_mut(*id) {
                if e.order != i as u32 {
                    e.order = i as u32;
                    e.order_source = OrderSource::Suggested;
                }
            }
        }
        for (id, orient) in &s.orient {
            if accept.contains(id) && orient.is_finite() {
                if let Some(e) = self.project.slide_mut(*id) {
                    e.orient_hint = Some(*orient);
                }
            }
        }
        let spacing = self.project.recon.default_spacing_um;
        self.project.recompute_default_z(spacing);
        self.project_dirty = true;
        self.drop_stale_registrations();
        self.recompose_chain();
        self.save_project()
    }

    /// Simpan `P_{slide→tetangga}` hasil koreksi manual lalu komposisi ulang
    /// rantai tanpa registrasi ulang (8.4.5).
    fn set_manual_pairwise(
        &mut self,
        slide: SlideId,
        fixed: SlideId,
        pairwise: Affine2,
        method: &str,
    ) -> Result<(), EngineError> {
        if !pairwise.is_finite() || pairwise.inverse().is_none() {
            return Err(EngineError::invalid(
                "transformasi koreksi tidak sah (singular atau tak hingga)",
            ));
        }
        // Metrik QC dihitung ulang bila gambar tersedia; bila tidak, nilai
        // lama dipertahankan agar tabel QC tidak berisi angka karangan.
        let (metrics, residual) = match (
            self.registration_image(fixed),
            self.registration_image(slide),
        ) {
            (Ok(f), Ok(m)) => (
                pair_metrics(&f, &m, &pairwise),
                pair_residual(&f, &m, &pairwise),
            ),
            _ => (None, None),
        };
        let old = self.slide(slide)?.reg.clone();
        let (ncc, mask_dice) = metrics
            .or_else(|| old.as_ref().map(|r| (r.ncc, r.mask_dice)))
            .unwrap_or((0.0, 0.0));
        self.slide_mut(slide)?.reg = Some(RegRecord {
            fixed,
            method: method.to_owned(),
            pairwise,
            ncc,
            mask_dice,
            area_ratio: pairwise.determinant().abs() as f32,
            flagged: false,
            manual: true,
            residual_um: residual.map(|r| r.local_um),
            residual_fit_um: residual.map(|r| r.fit_um),
        });
        self.recompose_chain();
        self.save_project()
    }

    /// Koreksi landmark (8.4.5). `pairs` = `(titik di U irisan ini, titik
    /// padanannya di U tetangga ke arah acuan)`, µm. ≥ 3 pasang → affine
    /// kuadrat terkecil; 2 pasang → similarity. `T` irisan ini dan semua
    /// irisan sesudahnya di rantai dihitung ulang; irisan lain tidak berubah.
    pub fn apply_landmarks(
        &mut self,
        slide: SlideId,
        pairs: &[([f64; 2], [f64; 2])],
    ) -> Result<(), EngineError> {
        let fixed = self.correction_target(slide)?;
        let moving: Vec<[f64; 2]> = pairs.iter().map(|p| p.0).collect();
        let fixed_pts: Vec<[f64; 2]> = pairs.iter().map(|p| p.1).collect();
        let p = from_landmarks(&moving, &fixed_pts)?;
        self.set_manual_pairwise(slide, fixed, p, "manual-landmarks")
    }

    /// Koreksi geser/putar manual (8.4.5). `t_new` = `T_i` baru irisan ini,
    /// memetakan **U irisan → W**. Disimpan sebagai
    /// `P_{i→tetangga} = T_tetangga⁻¹ ∘ t_new`, lalu rantai dikomposisi ulang.
    pub fn apply_manual_transform(
        &mut self,
        slide: SlideId,
        t_new: Affine2,
    ) -> Result<(), EngineError> {
        let fixed = self.correction_target(slide)?;
        let reference = self.project.effective_reference();
        let nb = self.slide(fixed)?;
        if Some(fixed) != reference && !nb.transform.is_registered() {
            return Err(EngineError::invalid(format!(
                "irisan tetangga {fixed} belum terdaftar; daftarkan atau koreksi irisan itu dulu"
            )));
        }
        let inv = nb.transform.affine().try_inverse()?;
        self.set_manual_pairwise(slide, fixed, inv.compose(&t_new), "manual-transform")
    }

    fn correction_target(&self, slide: SlideId) -> Result<SlideId, EngineError> {
        let entry = self.slide(slide)?;
        if entry.status != SlideStatus::Ok {
            return Err(EngineError::invalid(format!(
                "irisan {slide} tidak berstatus Ok"
            )));
        }
        self.neighbor_of(slide).ok_or_else(|| {
            EngineError::invalid(format!(
                "irisan {slide} adalah acuan (T = identitas) sehingga tidak bisa dikoreksi"
            ))
        })
    }

    /// Tabel QC per irisan (NCC, Dice, rasio luas, tanda).
    pub fn qc_table(&self) -> Vec<QcRow> {
        let reference = self.project.effective_reference();
        self.project
            .slides_ordered()
            .iter()
            .map(|s| {
                let is_reference = Some(s.id) == reference;
                QcRow {
                    slide: s.id,
                    order: s.order,
                    file: s.source.file_name(),
                    status: s.status,
                    is_reference,
                    registered: is_reference || s.transform.is_registered(),
                    fixed: s.reg.as_ref().map(|r| r.fixed),
                    method: s.reg.as_ref().map(|r| r.method.clone()),
                    ncc: s.reg.as_ref().map(|r| r.ncc),
                    mask_dice: s.reg.as_ref().map(|r| r.mask_dice),
                    area_ratio: s.reg.as_ref().map(|r| r.area_ratio),
                    flagged: s.reg.as_ref().is_some_and(|r| r.flagged),
                    manual: s.reg.as_ref().is_some_and(|r| r.manual),
                }
            })
            .collect()
    }

    /// Taksiran galat satu pasangan dari Dice masker (cadangan untuk catatan
    /// lama tanpa `residual_um`): pergeseran yang menjelaskan kekurangan Dice
    /// pada cakram berluas sama, `(1 − Dice) × π × R / 2` — batas atas yang
    /// hati-hati karena perubahan bentuk antar irisan ikut terhitung —
    /// dengan batas bawah setengah piksel registrasi.
    fn dice_proxy_um(&self, slide: SlideId, mask_dice: f32) -> Option<f64> {
        let img = self.registration_image(slide).ok()?;
        let area = img.mask.iter().filter(|v| **v != 0).count() as f64 * img.mpp * img.mpp;
        if area <= 0.0 {
            return None;
        }
        let r = (area / std::f64::consts::PI).sqrt();
        let d = f64::from(mask_dice).clamp(0.0, 1.0);
        Some(((1.0 - d) * std::f64::consts::PI * r / 2.0).max(img.mpp / 2.0))
    }

    /// Taksiran galat registrasi per irisan di ruang W (µm), terurut tumpukan.
    ///
    /// `T_i` adalah komposisi pasangan menuju acuan, sehingga galat
    /// **transformasi** tiap pasangan (`RegRecord::residual_fit_um`) menumpuk
    /// — secara kuadrat karena pasangan diregistrasi saling bebas — sedangkan
    /// galat **lokal** (`RegRecord::residual_um`: tekstur dan deformasi lokal
    /// irisan itu sendiri) tidak:
    /// `e_i = sqrt(Σ_k fit_k² + lokal_i²)`, `k` = pasangan di jalur `i` →
    /// acuan. Keduanya diukur dari citra setelah penjajaran
    /// (`h3d_register::pair_residual`).
    ///
    /// Catatan lama tanpa field itu memakai cadangan berbasis Dice untuk
    /// `e_i` (dan menyumbang nol ke irisan sesudahnya). Irisan acuan (galat
    /// nol menurut definisi W) dan irisan yang rantainya putus tidak ikut.
    pub fn tre_per_slide_um(&self) -> Vec<(SlideId, f64)> {
        let Some(reference) = self.project.effective_reference() else {
            return Vec::new();
        };
        let ok = |v: Option<f64>| v.filter(|r| r.is_finite() && *r >= 0.0);
        let usable = self.project.usable_slides_ordered();
        // Per irisan: (tetangga, fit², lokal).
        let mut link: HashMap<SlideId, (SlideId, f64, Option<f64>)> = HashMap::new();
        for s in &usable {
            let Some(rec) = &s.reg else { continue };
            let fit = ok(rec.residual_fit_um).unwrap_or(0.0);
            link.insert(s.id, (rec.fixed, fit * fit, ok(rec.residual_um)));
        }
        let mut out = Vec::new();
        for s in &usable {
            let Some((_, _, local)) = link.get(&s.id).copied() else {
                continue;
            };
            if s.id == reference {
                continue;
            }
            let mut sum = 0.0;
            let mut cur = s.id;
            let mut reached = false;
            // Jalur paling panjang = jumlah irisan (pengaman terhadap siklus).
            for _ in 0..=usable.len() {
                if cur == reference {
                    reached = true;
                    break;
                }
                let Some((next, fit2, _)) = link.get(&cur) else {
                    break;
                };
                sum += fit2;
                cur = *next;
            }
            if !reached {
                continue;
            }
            let e = match local {
                Some(l) => Some((sum + l * l).sqrt()),
                None => s
                    .reg
                    .as_ref()
                    .and_then(|r| self.dice_proxy_um(s.id, r.mask_dice)),
            };
            if let Some(e) = e {
                out.push((s.id, e));
            }
        }
        out
    }

    /// Taksiran galat registrasi (µm) untuk ketidakpastian jarak (8.12.6):
    /// median [`Self::tre_per_slide_um`] atas irisan yang terlibat. `None`
    /// bila belum ada registrasi (pemanggil lalu memakai `s_xy`).
    pub fn tre_estimate_um(&self) -> Option<f64> {
        let mut est: Vec<f64> = self
            .tre_per_slide_um()
            .into_iter()
            .map(|(_, e)| e)
            .collect();
        if est.is_empty() {
            return None;
        }
        est.sort_by(|a, b| a.total_cmp(b));
        Some(est[est.len() / 2])
    }
}
