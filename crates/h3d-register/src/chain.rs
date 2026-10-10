//! Rantai registrasi tumpukan (histo3d.md 5.3, 8.4.4, 8.4.5).
//!
//! # Arah transformasi
//! - `RegRecord::pairwise` = `P_{i→fixed}`: ruang U irisan `i` → ruang U
//!   irisan tetangganya, µm → µm.
//! - [`compose_chain`] menghasilkan `T_i`: ruang U irisan `i` → ruang W
//!   tumpukan, dengan `T_ref = identitas` dan
//!   `T_i = T_{tetangga(i)} ∘ P_{i→tetangga(i)}`.
//! - `RegInput::init` = tebakan awal `P_{i→tetangga}` (µm → µm), disusun
//!   pemanggil dari `orient_hint`: `orient_tetangga⁻¹ ∘ orient_i`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use h3d_core::{Affine2, Progress, RegRecord, SlideId, SlideStatus};
use rayon::prelude::*;

use crate::error::RegError;
use crate::qc::{record_from_result, QcThresholds};
use crate::types::{RegImage, RegOptions, Registrar};

/// Masukan satu irisan untuk registrasi tumpukan dan usulan urutan.
#[derive(Clone, Debug)]
pub struct RegInput {
    /// Pengenal irisan.
    pub id: SlideId,
    /// Urutan dalam tumpukan (0 = pertama).
    pub order: u32,
    /// Status; hanya `Ok` yang didaftarkan.
    pub status: SlideStatus,
    /// Citra registrasi (resolusi sumber; diturunkan di dalam).
    pub image: Arc<RegImage>,
    /// Tebakan awal `P_{i→tetangga}` (U irisan ini → U tetangganya ke arah
    /// acuan, µm → µm), bila ada. Pemanggil menyusunnya dari `orient_hint`
    /// (8.4.7): `init = orient_tetangga⁻¹ ∘ orient_i`. Diabaikan `suggest_order`.
    pub init: Option<Affine2>,
}

/// Progres pembungkus untuk pasangan paralel: meneruskan pembatalan,
/// mengabaikan fraksi per pasangan (fraksi global dihitung dari pasangan selesai).
struct PairProgress<'a> {
    parent: &'a dyn Progress,
}

impl Progress for PairProgress<'_> {
    fn set(&self, _fraction: f32, _stage: &str) {}
    fn is_cancelled(&self) -> bool {
        self.parent.is_cancelled()
    }
}

/// Irisan `Ok` terurut menurut `order` (seri → `SlideId`).
fn ok_order(slides: &[RegInput]) -> Vec<SlideId> {
    let mut v: Vec<&RegInput> = slides
        .iter()
        .filter(|s| s.status == SlideStatus::Ok)
        .collect();
    v.sort_by_key(|s| (s.order, s.id));
    v.iter().map(|s| s.id).collect()
}

/// Tetangga sah terdekat dari `i` ke arah `reference` dalam daftar irisan `Ok`
/// terurut. `None` bila `i` adalah acuan, atau `i`/`reference` tidak ada.
pub fn neighbor_toward_reference(
    order_ok: &[SlideId],
    i: SlideId,
    reference: SlideId,
) -> Option<SlideId> {
    let pi = order_ok.iter().position(|&s| s == i)?;
    let pr = order_ok.iter().position(|&s| s == reference)?;
    match pi.cmp(&pr) {
        std::cmp::Ordering::Equal => None,
        std::cmp::Ordering::Less => order_ok.get(pi + 1).copied(),
        std::cmp::Ordering::Greater => order_ok.get(pi - 1).copied(),
    }
}

/// Daftarkan semua irisan `Ok` (kecuali acuan) ke tetangga `Ok` terdekat ke
/// arah `reference`. Pasangan independen dijalankan paralel; hasil dikumpulkan
/// dalam urutan tumpukan sehingga keluaran deterministik. Irisan
/// `Excluded`/`Damaged` dilompati. Mengembalikan `(id, RegRecord)` per irisan
/// terdaftar, terurut menurut `order`; `RegRecord::pairwise` = `P_{i→fixed}`.
pub fn register_stack(
    slides: &[RegInput],
    reference: SlideId,
    registrar: &dyn Registrar,
    opts: &RegOptions,
    thr: &QcThresholds,
    progress: &dyn Progress,
) -> Result<Vec<(SlideId, RegRecord)>, RegError> {
    let order = ok_order(slides);
    if order.is_empty() {
        return Err(RegError::NoUsableSlides(
            "tidak ada irisan berstatus Ok".into(),
        ));
    }
    if !order.contains(&reference) {
        return Err(RegError::NoUsableSlides(format!(
            "irisan acuan {reference} tidak ada atau tidak berstatus Ok"
        )));
    }
    let by_id: HashMap<SlideId, &RegInput> = slides.iter().map(|s| (s.id, s)).collect();
    let pairs: Vec<(SlideId, SlideId)> = order
        .iter()
        .filter_map(|&i| neighbor_toward_reference(&order, i, reference).map(|f| (i, f)))
        .collect();
    let total = pairs.len().max(1);
    let done = AtomicUsize::new(0);
    progress.set(0.0, "registrasi berpasangan");
    let pair_progress = PairProgress { parent: progress };

    let results: Vec<Result<(SlideId, RegRecord), RegError>> = pairs
        .par_iter()
        .map(|&(mid, fid)| {
            if progress.is_cancelled() {
                return Err(RegError::Cancelled);
            }
            let (Some(m), Some(f)) = (by_id.get(&mid), by_id.get(&fid)) else {
                return Err(RegError::NoUsableSlides(format!(
                    "irisan {mid}/{fid} hilang"
                )));
            };
            let res = registrar.register(&f.image, &m.image, m.init, opts, &pair_progress)?;
            let n = done.fetch_add(1, Ordering::SeqCst) + 1;
            progress.set(n as f32 / total as f32, "registrasi berpasangan");
            Ok((
                mid,
                record_from_result(fid, registrar.name(), &res, thr, false),
            ))
        })
        .collect();

    let mut out = Vec::with_capacity(results.len());
    for r in results {
        out.push(r?);
    }
    Ok(out)
}

/// Komposisi rantai: `T_i` (U irisan `i` → W) untuk tiap irisan yang punya
/// catatan, dengan `T_ref = identitas` dan
/// `T_i = T_{fixed(i)} ∘ P_{i→fixed(i)}`. `order` = irisan `Ok` terurut; irisan
/// yang rantainya tidak sampai ke acuan tidak masuk hasil. Dapat dijalankan
/// ulang setelah koreksi manual tanpa registrasi ulang (8.4.5).
pub fn compose_chain(
    records: &[(SlideId, RegRecord)],
    reference: SlideId,
    order: &[SlideId],
) -> HashMap<SlideId, Affine2> {
    let rec: HashMap<SlideId, &RegRecord> = records.iter().map(|(id, r)| (*id, r)).collect();
    let mut t: HashMap<SlideId, Affine2> = HashMap::new();
    t.insert(reference, Affine2::IDENTITY);
    // Kandidat: irisan dalam `order` lalu sisanya dari `records` (urutan tetap).
    let mut pending: Vec<SlideId> = order.iter().copied().filter(|&s| s != reference).collect();
    for (id, _) in records {
        if *id != reference && !pending.contains(id) {
            pending.push(*id);
        }
    }
    // Resolusi berulang: tiap putaran menyelesaikan irisan yang tetangganya
    // sudah punya `T`; berhenti bila tidak ada kemajuan.
    loop {
        let mut progressed = false;
        let mut rest = Vec::with_capacity(pending.len());
        for id in pending {
            match rec.get(&id) {
                Some(r) if r.fixed != id => {
                    if let Some(tn) = t.get(&r.fixed).copied() {
                        t.insert(id, tn.compose(&r.pairwise));
                        progressed = true;
                    } else {
                        rest.push(id);
                    }
                }
                _ => {}
            }
        }
        pending = rest;
        if !progressed || pending.is_empty() {
            break;
        }
    }
    t
}
