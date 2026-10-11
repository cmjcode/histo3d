//! Atribut skalar per irisan (8.12.5): skor IHC, densitas inti, densitas
//! limfosit; resampling ke kisi tumpukan; tabel per irisan untuk laporan.

use std::borrow::Cow;
use std::sync::atomic::{AtomicUsize, Ordering};

use h3d_core::{
    Affine2, AttrDef, AttrId, AttrKind, AttrSource, IhcScoring, LabelId, LabelKind, Progress,
    SlideId, SlideStatus, Stain, StainPattern, SubProgress,
};
use h3d_imgproc::{
    allred_proportion_score, deconvolve, detect_nuclei, edt_2d_um, measure_chromogen, tps_cps,
    IhcThresholds, NucleiParams, StainMatrix,
};
use h3d_io::{AttrReport, AttrSlideRow};
use h3d_register::RegImage;
use h3d_segment::{AttrStore, LabelStore};
use h3d_slide::SlideSource;
use h3d_volume::{AttrVolume, StackGrid};
use ndarray::Array2;
use rayon::prelude::*;

use super::images::nearest_level;
use super::{ReconState, Session};
use crate::error::EngineError;
use crate::hash::Fnv1a64;

/// Parameter penghitungan atribut. Semuanya masuk `params_hash`; mengubah
/// satu saja menghasilkan atribut BARU, bukan menimpa diam-diam (8.12.5).
#[derive(Clone, Debug, PartialEq)]
pub struct AttrParams {
    /// Ambang OD kromogen (`t_D` dan batas intensitas 1/2/3).
    pub thresholds: IhcThresholds,
    /// Parameter deteksi inti (σ, NMS, ambang H).
    pub nuclei: NucleiParams,
    /// Resolusi kerja sasaran, µm/px (bawaan 1,0); dipakai level piramida
    /// terdekat.
    pub mpp_work: f64,
    /// Resolusi `AttrStore`, µm/px (bawaan 16).
    pub mpp_attr: f64,
    /// Sisi tile kerja, piksel (bawaan 1024).
    pub tile_px: u32,
    /// Tumpang tindih tile, piksel (bawaan 64).
    pub overlap_px: u32,
    /// Jari-jari jendela agregasi dalam piksel atribut (bawaan 2 → 5×5
    /// piksel = 80 µm): skor tiap piksel dihitung dari sel di jendela ini.
    pub window_px: u32,
    /// Batasi ke dalam label ini (masker label yang diminta), bila ada.
    pub restrict_label: Option<LabelId>,
}

impl Default for AttrParams {
    fn default() -> Self {
        AttrParams {
            thresholds: IhcThresholds::default(),
            nuclei: NucleiParams::default(),
            mpp_work: 1.0,
            mpp_attr: h3d_segment::attr_store::DEFAULT_MPP_ATTR,
            tile_px: 1024,
            overlap_px: 64,
            window_px: 2,
            restrict_label: None,
        }
    }
}

impl AttrParams {
    /// Hash FNV-1a stabil atas semua parameter.
    pub fn params_hash(&self) -> u64 {
        let mut h = Fnv1a64::new();
        h.str("h3d-attr-v1");
        h.f64(f64::from(self.thresholds.t_d));
        for t in self.thresholds.intensity {
            h.f64(f64::from(t));
        }
        h.f64(self.nuclei.smooth_sigma_um);
        h.u64(self.nuclei.log_sigmas_um.len() as u64);
        for s in &self.nuclei.log_sigmas_um {
            h.f64(*s);
        }
        h.f64(self.nuclei.nms_radius_um);
        match self.nuclei.h_threshold {
            Some(t) => h.u64(1).f64(f64::from(t)),
            None => h.u64(0),
        };
        h.f64(self.mpp_work);
        h.f64(self.mpp_attr);
        h.u64(u64::from(self.tile_px));
        h.u64(u64::from(self.overlap_px));
        h.u64(u64::from(self.window_px));
        h.u64(self.restrict_label.map_or(0, |l| u64::from(l.0) + 1));
        h.finish()
    }

    fn validate(&self) -> Result<(), EngineError> {
        let ok = self.mpp_work.is_finite()
            && self.mpp_work > 0.0
            && self.mpp_attr.is_finite()
            && self.mpp_attr > 0.0
            && self.tile_px >= 64
            && self.overlap_px * 2 < self.tile_px
            && self.thresholds.t_d.is_finite()
            && self.window_px <= 64;
        if ok {
            Ok(())
        } else {
            Err(EngineError::invalid("parameter atribut tidak sah"))
        }
    }
}

/// Cara satu atribut dihitung.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Plan {
    /// Skor per inti.
    Nuclear(IhcScoring),
    /// TPS/CPS dengan pemisahan tumor/imun lewat masker label.
    TumorImmune(IhcScoring),
    /// Skor berbasis luas (pola membran/sitoplasma).
    Area(IhcScoring),
    /// Densitas semua inti.
    NucleiDensity,
    /// Densitas inti positif (IHC CD3/CD8).
    TilDensity,
}

impl Plan {
    fn method(self, fallback_pct: bool) -> &'static str {
        match self {
            Plan::Nuclear(_) if fallback_pct => "ihc-tps-cps-unavailable-pctpos-v1",
            Plan::Nuclear(_) => "ihc-nuclear-v1",
            Plan::TumorImmune(IhcScoring::Cps) => "ihc-cps-v1",
            Plan::TumorImmune(_) => "ihc-tps-v1",
            Plan::Area(_) => "ihc-area-v1",
            Plan::NucleiDensity => "nuclei-density-v1",
            Plan::TilDensity => "til-ihc-density-v1",
        }
    }

    fn unit(self) -> &'static str {
        match self {
            Plan::NucleiDensity | Plan::TilDensity => "sel/mm²",
            Plan::Nuclear(s) | Plan::TumorImmune(s) | Plan::Area(s) => match s {
                IhcScoring::PercentPositive | IhcScoring::Tps => "%",
                IhcScoring::Cps => "CPS",
                IhcScoring::HScore => "H-score",
                IhcScoring::Allred => "Allred",
                IhcScoring::Intensity03 => "0–3",
            },
        }
    }

    /// Benar bila agregasi jendela dipisah per kompartemen (di dalam vs di
    /// luar masker Tumor) saat masker itu tersedia: skor per inti dan
    /// densitas. TPS/CPS sudah per kompartemen menurut definisinya; skor
    /// berbasis luas dihitung per piksel.
    fn compartment_aware(self) -> bool {
        matches!(
            self,
            Plan::Nuclear(_) | Plan::NucleiDensity | Plan::TilDensity
        )
    }

    /// Kuantisasi `AttrStore` (nilai = raw × scale + offset).
    fn scale(self) -> f32 {
        match self {
            Plan::NucleiDensity | Plan::TilDensity => 1.0,
            _ => 0.01,
        }
    }
}

/// Kelas region sebuah titik.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Region {
    Tumor,
    Stroma,
    Other,
}

/// Masker label untuk sebuah irisan: label sendiri, atau label irisan H&E
/// tetangga terdekat lewat `T_he⁻¹ ∘ T_ihc` (8.12.3).
pub(crate) struct RegionLookup<'a> {
    store: Cow<'a, LabelStore>,
    /// U irisan ini → U irisan sumber label; `None` = label sendiri.
    map: Option<Affine2>,
    tumor: Vec<u8>,
    stroma: Vec<u8>,
}

impl RegionLookup<'_> {
    fn label_at(&self, u: [f64; 2]) -> u8 {
        let q = match &self.map {
            Some(m) => m.apply(u),
            None => u,
        };
        self.store.get_um(q)
    }

    fn region_at(&self, u: [f64; 2]) -> Region {
        let l = self.label_at(u);
        if l == 0 {
            Region::Other
        } else if self.tumor.contains(&l) {
            Region::Tumor
        } else if self.stroma.contains(&l) {
            Region::Stroma
        } else {
            Region::Other
        }
    }

    fn has_stroma(&self) -> bool {
        let present = self.store.labels_present();
        present.iter().any(|l| self.stroma.contains(&l.0))
    }
}

/// Hitungan per piksel atribut.
#[derive(Clone, Default)]
struct Counts {
    /// Semua sel (atau piksel jaringan untuk skor berbasis luas).
    all: Vec<f64>,
    /// Positif.
    pos: Vec<f64>,
    /// Kelas intensitas 1..=3.
    cls: [Vec<f64>; 3],
    tumor_all: Vec<f64>,
    tumor_pos: Vec<f64>,
    /// Kelas intensitas 1..=3 sel di dalam tumor.
    tumor_cls: [Vec<f64>; 3],
    immune_pos: Vec<f64>,
}

impl Counts {
    fn new(w: usize, h: usize) -> Self {
        let z = || vec![0.0f64; w * h];
        Counts {
            all: z(),
            pos: z(),
            cls: [z(), z(), z()],
            tumor_all: z(),
            tumor_pos: z(),
            tumor_cls: [z(), z(), z()],
            immune_pos: z(),
        }
    }
}

/// Jumlah jendela kotak berjari-jari `r` lewat tabel luas terjumlah.
fn window_sum(src: &[f64], w: usize, h: usize, r: usize) -> Vec<f64> {
    if r == 0 {
        return src.to_vec();
    }
    let mut sat = vec![0.0f64; (w + 1) * (h + 1)];
    for y in 0..h {
        let mut row = 0.0;
        for x in 0..w {
            row += src[y * w + x];
            sat[(y + 1) * (w + 1) + x + 1] = sat[y * (w + 1) + x + 1] + row;
        }
    }
    let mut out = vec![0.0f64; w * h];
    for y in 0..h {
        let y0 = y.saturating_sub(r);
        let y1 = (y + r + 1).min(h);
        for x in 0..w {
            let x0 = x.saturating_sub(r);
            let x1 = (x + r + 1).min(w);
            out[y * w + x] =
                sat[y1 * (w + 1) + x1] - sat[y0 * (w + 1) + x1] - sat[y1 * (w + 1) + x0]
                    + sat[y0 * (w + 1) + x0];
        }
    }
    out
}

/// Hasil satu tile kerja.
enum TileOut {
    /// `(u_x, u_y, d_mean)` inti di dalam inti tile.
    Nuclei(Vec<(f64, f64, f32)>),
    /// `(indeks piksel atribut, piksel jaringan, piksel positif, kelas 1..3)`.
    Area(Vec<(usize, u32, u32, [u32; 3])>),
    Empty,
}

fn mask_at(reg: &RegImage, u: [f64; 2]) -> bool {
    let p = reg.um_to_px(u);
    if p[0] < 0.0 || p[1] < 0.0 {
        return false;
    }
    let (x, y) = (p[0] as usize, p[1] as usize);
    let (h, w) = reg.mask.dim();
    x < w && y < h && reg.mask[[y, x]] != 0
}

impl Session {
    /// Masker label untuk `slide`: label `Tumor` sendiri bila ada; selain
    /// itu label irisan H&E `Ok` terdekat (menurut urutan) yang punya label
    /// tumor, dipetakan lewat `T_he⁻¹ ∘ T_slide`. `None` bila tidak ada.
    pub(crate) fn region_lookup(
        &self,
        slide: SlideId,
    ) -> Result<Option<RegionLookup<'_>>, EngineError> {
        let ids = |pred: &dyn Fn(LabelKind) -> bool| -> Vec<u8> {
            self.project
                .labels
                .iter()
                .filter(|l| pred(l.kind))
                .map(|l| l.id.0)
                .collect()
        };
        let tumor = ids(&|k| k == LabelKind::Tumor);
        let stroma = ids(&|k| k == LabelKind::Stroma);
        let has_tumor = |st: &LabelStore| st.labels_present().iter().any(|l| tumor.contains(&l.0));
        let stores = self.label_snapshot()?;
        let me = self.slide(slide)?;
        let own = stores
            .iter()
            .position(|(id, st)| *id == slide && has_tumor(st));
        let pick = match own {
            Some(i) => Some((i, None)),
            None => {
                let mut best: Option<(u32, u32, usize)> = None;
                for (i, (id, st)) in stores.iter().enumerate() {
                    let Some(e) = self.project.slide(*id) else {
                        continue;
                    };
                    if *id == slide
                        || e.status != SlideStatus::Ok
                        || e.stain != Stain::He
                        || !has_tumor(st)
                    {
                        continue;
                    }
                    let d = e.order.abs_diff(me.order);
                    // Seri → urutan lebih kecil (deterministik).
                    if best.is_none_or(|(bd, bo, _)| (d, e.order) < (bd, bo)) {
                        best = Some((d, e.order, i));
                    }
                }
                match best {
                    Some((_, _, i)) => {
                        let src = self.slide(stores[i].0)?;
                        let map = src
                            .transform
                            .affine()
                            .try_inverse()?
                            .compose(&me.transform.affine());
                        Some((i, Some(map)))
                    }
                    None => None,
                }
            }
        };
        Ok(pick.and_then(|(i, map)| {
            stores.into_iter().nth(i).map(|(_, store)| RegionLookup {
                store,
                map,
                tumor,
                stroma,
            })
        }))
    }

    /// Hitung atribut (8.12.5) dengan parameter bawaan. Lihat
    /// [`Self::compute_attr_with`].
    pub fn compute_attr(
        &mut self,
        def: AttrDef,
        slides: &[SlideId],
        p: &dyn Progress,
    ) -> Result<AttrId, EngineError> {
        self.compute_attr_with(def, slides, &AttrParams::default(), p)
    }

    /// Hitung atribut `def.kind` untuk `slides`:
    ///
    /// - `IhcScore`: pola inti → deteksi inti LoG + `D_mean` per inti;
    ///   pola membran/sitoplasma → skor berbasis luas (dicatat sebagai
    ///   peringatan); TPS/CPS butuh masker `Tumor` (dan `Stroma` untuk CPS),
    ///   bila tidak ada yang dihitung persen positif dan metodenya ditandai
    ///   "tidak tersedia".
    /// - `NucleiDensity`: inti per mm² jaringan.
    /// - `TilDensity`: inti positif per mm² pada irisan IHC (CD3/CD8).
    ///
    /// Bekerja pada level piramida terdekat ke `mpp_work`, per tile dengan
    /// tumpang tindih, di dalam masker jaringan (dan masker label bila
    /// diminta). Hasil per irisan disimpan ke `attrs/NNNN.<stem>.atr`.
    /// `def.id` dan `def.source` diisi mesin. Atribut bernama sama dengan
    /// parameter sama dihitung ulang di tempat; parameter berbeda → atribut
    /// baru. Pembatalan tidak mengubah proyek.
    pub fn compute_attr_with(
        &mut self,
        mut def: AttrDef,
        slides: &[SlideId],
        params: &AttrParams,
        p: &dyn Progress,
    ) -> Result<AttrId, EngineError> {
        params.validate()?;
        if slides.is_empty() {
            return Err(EngineError::invalid("tidak ada irisan yang dipilih"));
        }
        if let Some(l) = params.restrict_label {
            self.check_label(l)?;
        }
        self.attr_notes.clear();

        // Tentukan cara hitung (harus sama untuk semua irisan terpilih).
        let mut plan: Option<Plan> = None;
        let mut fallback_pct = false;
        for &s in slides {
            let e = self.slide(s)?;
            if !e.has_valid_mpp() {
                return Err(EngineError::MppMissing {
                    file: e.source.file_name(),
                });
            }
            let pattern = e.ihc.as_ref().map_or(StainPattern::Nuclear, |i| i.pattern);
            let this = match &def.kind {
                AttrKind::IhcScore { scoring, .. } => match (pattern, *scoring) {
                    (StainPattern::Nuclear, IhcScoring::Tps | IhcScoring::Cps) => {
                        let lookup = self.region_lookup(s)?;
                        let ok = match (&lookup, *scoring) {
                            (Some(l), IhcScoring::Cps) => l.has_stroma(),
                            (Some(_), _) => true,
                            (None, _) => false,
                        };
                        if ok {
                            Plan::TumorImmune(*scoring)
                        } else {
                            fallback_pct = true;
                            Plan::Nuclear(IhcScoring::PercentPositive)
                        }
                    }
                    (StainPattern::Nuclear, sc) => Plan::Nuclear(sc),
                    (_, sc) => Plan::Area(sc),
                },
                AttrKind::NucleiDensity => Plan::NucleiDensity,
                AttrKind::TilDensity => {
                    if e.stain != Stain::Ihc {
                        return Err(EngineError::invalid(format!(
                            "densitas limfosit butuh irisan IHC CD3/CD8 (atau kelas Limfosit \
                             dari model AI); irisan {s} bukan IHC"
                        )));
                    }
                    Plan::TilDensity
                }
                AttrKind::Custom => {
                    return Err(EngineError::invalid(
                        "atribut jenis Custom tidak bisa dihitung; impor dari berkas",
                    ))
                }
            };
            match plan {
                None => plan = Some(this),
                Some(prev) if prev != this => {
                    return Err(EngineError::invalid(
                        "irisan terpilih memakai cara skor berbeda (pola pewarnaan atau \
                         ketersediaan masker); hitung terpisah",
                    ))
                }
                Some(_) => {}
            }
        }
        let plan = plan.ok_or_else(|| EngineError::invalid("tidak ada irisan yang dipilih"))?;
        let method = plan.method(fallback_pct);
        if fallback_pct {
            self.attr_notes.push(
                "TPS/CPS tidak tersedia: butuh masker label Tumor (dan Stroma untuk CPS); \
                 yang dihitung adalah persen positif."
                    .to_owned(),
            );
        }
        if matches!(plan, Plan::Area(_)) {
            self.attr_notes
                .push("Skor berbasis luas, bukan per sel.".to_owned());
        }
        // Agregasi per kompartemen (lihat `compute_slide_attr`) aktif bila
        // ada masker Tumor untuk salah satu irisan; itu mengubah hasil,
        // jadi ikut masuk hash (atribut lama tanpa masker tidak terpengaruh).
        let mut compartments = false;
        if plan.compartment_aware() {
            for &sl in slides {
                compartments |= self.region_lookup(sl)?.is_some();
            }
        }
        let mut h = Fnv1a64::new();
        h.u64(params.params_hash()).str(method).str(plan.unit());
        if compartments {
            h.str("kompartemen-tumor-v1");
        }
        let source = AttrSource::Computed {
            method: method.to_owned(),
            params_hash: h.finish(),
        };

        // Identitas atribut: sama nama + jenis + sumber → hitung ulang di
        // tempat; nama sama tetapi parameter berbeda → atribut baru.
        if def.name.trim().is_empty() {
            def.name = match plan {
                Plan::NucleiDensity => "Densitas inti".to_owned(),
                Plan::TilDensity => "Densitas limfosit".to_owned(),
                _ => "Skor IHC".to_owned(),
            };
        }
        if let AttrKind::IhcScore { slide, .. } = &mut def.kind {
            if !slides.contains(slide) {
                *slide = slides[0];
            }
        }
        if def.unit.trim().is_empty() {
            def.unit = plan.unit().to_owned();
        }
        let same_kind = |a: &AttrKind, b: &AttrKind| match (a, b) {
            (AttrKind::IhcScore { scoring: x, .. }, AttrKind::IhcScore { scoring: y, .. }) => {
                x == y
            }
            (x, y) => std::mem::discriminant(x) == std::mem::discriminant(y),
        };
        let existing = self
            .project
            .attrs
            .iter()
            .find(|a| a.name == def.name && a.source == source && same_kind(&a.kind, &def.kind))
            .map(|a| a.id);
        let id = match existing {
            Some(id) => id,
            None => {
                let base = def.name.clone();
                let mut n = 1;
                loop {
                    let stem = def.file_stem();
                    let clash = self
                        .project
                        .attrs
                        .iter()
                        .any(|a| a.name == def.name || a.file_stem() == stem);
                    if !clash {
                        break;
                    }
                    n += 1;
                    def.name = format!("{base} ({n})");
                }
                AttrId(
                    self.project
                        .attrs
                        .iter()
                        .map(|a| a.id.0 + 1)
                        .max()
                        .unwrap_or(0),
                )
            }
        };
        def.id = id;
        def.source = source;

        // Hitung semua irisan dulu; proyek baru diubah setelah semuanya selesai.
        let mut stores: Vec<(SlideId, AttrStore)> = Vec::with_capacity(slides.len());
        let total = slides.len() as f32;
        for (i, &s) in slides.iter().enumerate() {
            if p.is_cancelled() {
                return Err(EngineError::Cancelled);
            }
            let sub = SubProgress::new(p, i as f32 / total, (i + 1) as f32 / total);
            stores.push((s, self.compute_slide_attr(s, plan, params, &sub)?));
        }
        let stem = def.file_stem();
        for (s, store) in &stores {
            self.bundle.save_attr(*s, &stem, store, id.0)?;
        }
        match self.project.attrs.iter_mut().find(|a| a.id == id) {
            Some(a) => *a = def,
            None => {
                self.project.attrs.push(def);
                self.project.attrs.sort_by_key(|a| a.id);
            }
        }
        self.analysis = None;
        self.save_project()?;
        p.set(1.0, "atribut selesai");
        Ok(id)
    }

    /// Hitung `AttrStore` satu irisan.
    fn compute_slide_attr(
        &self,
        slide: SlideId,
        plan: Plan,
        params: &AttrParams,
        p: &dyn Progress,
    ) -> Result<AttrStore, EngineError> {
        let entry = self.slide(slide)?;
        let src = self.open_source(slide)?;
        let lc = nearest_level(src.as_ref(), entry.mpp, params.mpp_work)?;
        let [lw, lh] = lc.size;
        let [lmx, lmy] = lc.mpp;
        let m = (lmx + lmy) / 2.0;
        let reg = self.registration_image(slide)?;
        let lookup = if matches!(plan, Plan::TumorImmune(_))
            || plan.compartment_aware()
            || params.restrict_label.is_some()
        {
            self.region_lookup(slide)?
        } else {
            None
        };
        if params.restrict_label.is_some() && lookup.is_none() {
            return Err(EngineError::invalid(format!(
                "irisan {slide} tidak punya masker label untuk membatasi penghitungan"
            )));
        }
        let matrix = match (&entry.stain, entry.ihc.as_ref().and_then(|i| i.od_vectors)) {
            (_, Some(v)) => StainMatrix::new(v)?,
            (Stain::Ihc, None) => StainMatrix::h_dab(),
            (_, None) if matches!(plan, Plan::NucleiDensity) => StainMatrix::he(),
            (_, None) => StainMatrix::h_dab(),
        };

        let mut store = AttrStore::new(params.mpp_attr, entry.size_um(), plan.scale(), 0.0);
        let aw = store.size_px[0] as usize;
        let ah = store.size_px[1] as usize;
        let mpp_attr = store.mpp_attr;
        let attr_index = |u: [f64; 2]| -> Option<usize> {
            if u[0] < 0.0 || u[1] < 0.0 {
                return None;
            }
            let (x, y) = ((u[0] / mpp_attr) as usize, (u[1] / mpp_attr) as usize);
            (x < aw && y < ah).then_some(y * aw + x)
        };
        let allowed = |u: [f64; 2]| -> bool {
            mask_at(&reg, u)
                && match (params.restrict_label, &lookup) {
                    (Some(l), Some(lk)) => lk.label_at(u) == l.0,
                    _ => true,
                }
        };

        // Tile kerja: inti `core` piksel, dibaca dengan halo `overlap`.
        let overlap = params.overlap_px;
        let core = params.tile_px - 2 * overlap;
        let nx = lw.div_ceil(core);
        let ny = lh.div_ceil(core);
        let tiles: Vec<(u32, u32)> = (0..ny)
            .flat_map(|ty| (0..nx).map(move |tx| (tx, ty)))
            .collect();
        let done = AtomicUsize::new(0);
        let n_tiles = tiles.len().max(1);
        let source: &dyn SlideSource = src.as_ref();
        let area_mode = matches!(plan, Plan::Area(_));
        let thr = params.thresholds;

        let outs: Vec<Result<TileOut, EngineError>> = tiles
            .par_iter()
            .map(|&(tx, ty)| {
                if p.is_cancelled() {
                    return Err(EngineError::Cancelled);
                }
                let cx0 = tx * core;
                let cy0 = ty * core;
                let cw = core.min(lw - cx0);
                let ch = core.min(lh - cy0);
                let x0 = i64::from(cx0) - i64::from(overlap);
                let y0 = i64::from(cy0) - i64::from(overlap);
                let w = (cw + 2 * overlap) as usize;
                let h = (ch + 2 * overlap) as usize;
                // Masker tile: jaringan (dan label bila diminta), lewat
                // pencarian tetangga terdekat pada masker 16 µm.
                let mut any = false;
                let mask = Array2::from_shape_fn((h, w), |(j, i)| {
                    let u = [
                        (x0 as f64 + i as f64 + 0.5) * lmx,
                        (y0 as f64 + j as f64 + 0.5) * lmy,
                    ];
                    let v = allowed(u);
                    any |= v;
                    u8::from(v)
                });
                let out = if !any {
                    TileOut::Empty
                } else {
                    let rgba = source.read_region(lc.level, x0, y0, w as u32, h as u32)?;
                    let [hch, dch, _] = deconvolve(rgba.as_raw(), w, h, &matrix)?;
                    if area_mode {
                        let mut acc: std::collections::BTreeMap<usize, (u32, u32, [u32; 3])> =
                            std::collections::BTreeMap::new();
                        for j in overlap as usize..(overlap + ch) as usize {
                            for i in overlap as usize..(overlap + cw) as usize {
                                if mask[[j, i]] == 0 {
                                    continue;
                                }
                                let u = [
                                    (x0 as f64 + i as f64 + 0.5) * lmx,
                                    (y0 as f64 + j as f64 + 0.5) * lmy,
                                ];
                                let Some(idx) = attr_index(u) else { continue };
                                let e = acc.entry(idx).or_insert((0, 0, [0; 3]));
                                e.0 += 1;
                                let d = dch[[j, i]];
                                if d > thr.t_d {
                                    e.1 += 1;
                                    let c = thr.intensity_class(d).clamp(1, 3) as usize;
                                    e.2[c - 1] += 1;
                                }
                            }
                        }
                        TileOut::Area(acc.into_iter().map(|(k, v)| (k, v.0, v.1, v.2)).collect())
                    } else {
                        let mut nuclei = detect_nuclei(&hch, m, Some(&mask), &params.nuclei);
                        measure_chromogen(&mut nuclei, &dch, m);
                        let mut list = Vec::with_capacity(nuclei.len());
                        for n in &nuclei {
                            // Koordinat tile (piksel) → hanya inti tile.
                            let px = n.x_um / m;
                            let py = n.y_um / m;
                            if px < f64::from(overlap)
                                || py < f64::from(overlap)
                                || px >= f64::from(overlap + cw)
                                || py >= f64::from(overlap + ch)
                            {
                                continue;
                            }
                            list.push(((x0 as f64 + px) * lmx, (y0 as f64 + py) * lmy, n.d_mean));
                        }
                        TileOut::Nuclei(list)
                    }
                };
                let n = done.fetch_add(1, Ordering::SeqCst) + 1;
                p.set(n as f32 / n_tiles as f32, "menghitung atribut");
                Ok(out)
            })
            .collect();

        // Gabung berurutan (urutan tile tetap → deterministik).
        let mut c = Counts::new(aw, ah);
        for out in outs {
            match out? {
                TileOut::Empty => {}
                TileOut::Area(list) => {
                    for (idx, tissue, pos, cls) in list {
                        c.all[idx] += f64::from(tissue);
                        c.pos[idx] += f64::from(pos);
                        for (dst, n) in c.cls.iter_mut().zip(cls.iter()) {
                            dst[idx] += f64::from(*n);
                        }
                    }
                }
                TileOut::Nuclei(list) => {
                    for (ux, uy, d) in list {
                        let Some(idx) = attr_index([ux, uy]) else {
                            continue;
                        };
                        let positive = d > thr.t_d;
                        let k = thr.intensity_class(d).clamp(1, 3) as usize;
                        c.all[idx] += 1.0;
                        if positive {
                            c.pos[idx] += 1.0;
                            c.cls[k - 1][idx] += 1.0;
                        }
                        if let Some(lk) = &lookup {
                            match lk.region_at([ux, uy]) {
                                Region::Tumor => {
                                    c.tumor_all[idx] += 1.0;
                                    if positive {
                                        c.tumor_pos[idx] += 1.0;
                                        c.tumor_cls[k - 1][idx] += 1.0;
                                    }
                                }
                                Region::Stroma if positive => c.immune_pos[idx] += 1.0,
                                _ => {}
                            }
                        }
                    }
                }
            }
        }

        // Luas jaringan per piksel atribut (dari masker 16 µm).
        let tissue: Vec<f64> = (0..aw * ah)
            .map(|i| {
                let u = [
                    ((i % aw) as f64 + 0.5) * mpp_attr,
                    ((i / aw) as f64 + 0.5) * mpp_attr,
                ];
                f64::from(u8::from(allowed(u)))
            })
            .collect();

        // Agregasi jendela per kompartemen. Tanpa pemisahan, jendela 80 µm
        // mencampur sel di kedua sisi batas tumor: pada fantom CD8 (10 %
        // positif intratumoral, 60 % di pita peritumoral) densitas
        // intratumoral terukur +38 % dan peritumoral −11 %. Dengan masker
        // Tumor, tiap piksel hanya mengagregasi sel (dan luas jaringan)
        // kompartemennya sendiri: di dalam tumor bila pusat piksel di dalam
        // masker, di luar bila tidak.
        let r = params.window_px as usize;
        let in_tumor: Option<Vec<bool>> = match &lookup {
            Some(lk) if plan.compartment_aware() => Some(
                (0..aw * ah)
                    .map(|i| {
                        let u = [
                            ((i % aw) as f64 + 0.5) * mpp_attr,
                            ((i / aw) as f64 + 0.5) * mpp_attr,
                        ];
                        lk.region_at(u) == Region::Tumor
                    })
                    .collect(),
            ),
            _ => None,
        };
        // `pick(dalam, total)`: jumlah jendela kompartemen piksel itu.
        let split = |inside: &[f64], total: &[f64]| -> Vec<f64> {
            match &in_tumor {
                None => window_sum(total, aw, ah, r),
                Some(mask) => {
                    let outside: Vec<f64> = total
                        .iter()
                        .zip(inside.iter())
                        .map(|(t, i)| t - i)
                        .collect();
                    let wi = window_sum(inside, aw, ah, r);
                    let wo = window_sum(&outside, aw, ah, r);
                    (0..aw * ah)
                        .map(|i| if mask[i] { wi[i] } else { wo[i] })
                        .collect()
                }
            }
        };
        let tissue_inside: Vec<f64> = match &in_tumor {
            Some(mask) => tissue
                .iter()
                .zip(mask.iter())
                .map(|(t, m)| if *m { *t } else { 0.0 })
                .collect(),
            None => Vec::new(),
        };
        let all = split(&c.tumor_all, &c.all);
        let pos = split(&c.tumor_pos, &c.pos);
        let cls = [
            split(&c.tumor_cls[0], &c.cls[0]),
            split(&c.tumor_cls[1], &c.cls[1]),
            split(&c.tumor_cls[2], &c.cls[2]),
        ];
        let tumor_all = window_sum(&c.tumor_all, aw, ah, r);
        let tumor_pos = window_sum(&c.tumor_pos, aw, ah, r);
        let immune_pos = window_sum(&c.immune_pos, aw, ah, r);
        let tissue_win = split(&tissue_inside, &tissue);
        let px_area_mm2 = mpp_attr * mpp_attr / 1.0e6;

        let score = |sc: IhcScoring, i: usize| -> Option<f32> {
            let n = all[i];
            if n <= 0.0 {
                return None;
            }
            let pct = (100.0 * pos[i] / n) as f32;
            Some(match sc {
                IhcScoring::PercentPositive | IhcScoring::Tps | IhcScoring::Cps => pct,
                IhcScoring::HScore => {
                    (100.0 * (cls[0][i] + 2.0 * cls[1][i] + 3.0 * cls[2][i]) / n) as f32
                }
                IhcScoring::Allred | IhcScoring::Intensity03 => {
                    // Modus kelas intensitas sel positif; seri → kelas lebih tinggi.
                    let mut intensity = 0u8;
                    let mut best = 0.0;
                    for (k, cl) in cls.iter().enumerate() {
                        if cl[i] > 0.0 && cl[i] >= best {
                            best = cl[i];
                            intensity = k as u8 + 1;
                        }
                    }
                    if sc == IhcScoring::Allred {
                        f32::from(allred_proportion_score(pct) + intensity)
                    } else {
                        f32::from(intensity)
                    }
                }
            })
        };

        for y in 0..ah {
            for x in 0..aw {
                let i = y * aw + x;
                let (value, weight): (Option<f32>, f64) = match plan {
                    Plan::Nuclear(sc) | Plan::Area(sc) => (score(sc, i), all[i]),
                    Plan::TumorImmune(sc) => {
                        let (tps, cps) = tps_cps(
                            tumor_pos[i] as u32,
                            tumor_all[i] as u32,
                            immune_pos[i] as u32,
                        );
                        (if sc == IhcScoring::Cps { cps } else { tps }, tumor_all[i])
                    }
                    Plan::NucleiDensity | Plan::TilDensity => {
                        if tissue[i] > 0.0 && tissue_win[i] > 0.0 {
                            let n = if plan == Plan::TilDensity {
                                pos[i]
                            } else {
                                all[i]
                            };
                            (
                                Some((n / (tissue_win[i] * px_area_mm2)) as f32),
                                tissue_win[i],
                            )
                        } else {
                            (None, 0.0)
                        }
                    }
                };
                if let Some(v) = value {
                    if v.is_finite() && weight > 0.0 {
                        let wgt = weight.round().clamp(1.0, f64::from(u16::MAX)) as u16;
                        store.set_with_weight(x as u32, y as u32, v, wgt);
                    }
                }
            }
        }
        store.prune_empty_tiles();
        Ok(store)
    }

    /// `AttrStore` sebuah atribut pada sebuah irisan; `None` bila belum dihitung.
    pub fn attr_store(
        &self,
        slide: SlideId,
        attr: AttrId,
    ) -> Result<Option<AttrStore>, EngineError> {
        let def = self
            .project
            .attrs
            .iter()
            .find(|a| a.id == attr)
            .ok_or_else(|| EngineError::invalid(format!("atribut {attr} tidak ada")))?;
        Ok(self
            .bundle
            .load_attr(slide, &def.file_stem())?
            .map(|(s, _)| s))
    }

    /// Cari atribut berdasarkan nama (tidak peka huruf besar/kecil).
    pub fn attr_by_name(&self, name: &str) -> Option<&AttrDef> {
        let key = name.trim().to_lowercase();
        self.project
            .attrs
            .iter()
            .find(|a| a.name.trim().to_lowercase() == key)
    }

    /// Irisan `Ok` yang punya berkas `.atr` untuk `def`, terurut.
    fn attr_slices(&self, def: &AttrDef) -> Result<Vec<(SlideId, AttrStore)>, EngineError> {
        let stem = def.file_stem();
        let mut out = Vec::new();
        for s in self.project.usable_slides_ordered() {
            if let Some((store, _)) = self.bundle.load_attr(s.id, &stem)? {
                out.push((s.id, store));
            }
        }
        Ok(out)
    }

    /// Tabel atribut untuk laporan: rata-rata per irisan (berbobot bila bobot
    /// tersedia), rata-rata intratumoral dan peritumoral (pita
    /// `peri_band_um` di luar tumor) bila masker tumor ada.
    pub(crate) fn attr_reports(&self) -> Result<Vec<AttrReport>, EngineError> {
        let mut out = Vec::new();
        let band = self.project.spatial.peri_band_um;
        for def in &self.project.attrs {
            let mut rows = Vec::new();
            for (slide, store) in self.attr_slices(def)? {
                let values = store.to_array();
                let weights = store.weights_array();
                let (h, w) = values.dim();
                let lookup = self.region_lookup(slide)?;
                // Jarak ke tumor pada kisi atribut (untuk pita peritumoral).
                let tumor_mask = lookup.as_ref().map(|lk| {
                    Array2::from_shape_fn((h, w), |(y, x)| {
                        let u = [
                            (x as f64 + 0.5) * store.mpp_attr,
                            (y as f64 + 0.5) * store.mpp_attr,
                        ];
                        u8::from(lk.region_at(u) == Region::Tumor)
                    })
                });
                let dist = tumor_mask
                    .as_ref()
                    .filter(|m| m.iter().any(|v| *v != 0))
                    .map(|m| edt_2d_um(m, store.mpp_attr));
                let mut sum = [0.0f64; 3];
                let mut wsum = [0.0f64; 3];
                for y in 0..h {
                    for x in 0..w {
                        let v = values[[y, x]];
                        if !v.is_finite() {
                            continue;
                        }
                        let wt = f64::from(weights[[y, x]]).max(0.0);
                        sum[0] += f64::from(v) * wt;
                        wsum[0] += wt;
                        if let (Some(tm), Some(d)) = (&tumor_mask, &dist) {
                            if tm[[y, x]] != 0 {
                                sum[1] += f64::from(v) * wt;
                                wsum[1] += wt;
                            } else if f64::from(d[[y, x]]) <= band {
                                sum[2] += f64::from(v) * wt;
                                wsum[2] += wt;
                            }
                        }
                    }
                }
                if wsum[0] <= 0.0 {
                    continue;
                }
                let mean = |k: usize| (wsum[k] > 0.0).then(|| sum[k] / wsum[k]);
                rows.push(AttrSlideRow {
                    slide,
                    mean: sum[0] / wsum[0],
                    weight: wsum[0],
                    intratumoral: mean(1),
                    peritumoral: mean(2),
                });
            }
            let range = if rows.len() >= 2 {
                let lo = rows.iter().map(|r| r.mean).fold(f64::INFINITY, f64::min);
                let hi = rows
                    .iter()
                    .map(|r| r.mean)
                    .fold(f64::NEG_INFINITY, f64::max);
                Some(hi - lo)
            } else {
                None
            };
            let (method, params_hash) = match &def.source {
                AttrSource::Computed {
                    method,
                    params_hash,
                } => (method.clone(), Some(*params_hash)),
                AttrSource::Imported { file } => (format!("impor: {file}"), None),
            };
            out.push(AttrReport {
                name: def.name.clone(),
                unit: def.unit.clone(),
                method,
                params_hash,
                per_slide: rows,
                range_between_slides: range,
            });
        }
        Ok(out)
    }

    /// Volume atribut pada kisi rekonstruksi (8.12.5): tiap irisan
    /// beratribut diresampel mundur (`u = T_i⁻¹(w)`, bilinear dengan bobot
    /// NODATA = 0); lapis Z di antara dua irisan beratribut diisi interpolasi
    /// linear; di luar `max_gap_um` (atau lebih dari setengah jarak irisan
    /// dari irisan beratribut terluar) nilainya NODATA (`NaN`).
    pub(crate) fn attr_volumes(
        &self,
        recon: &ReconState,
    ) -> Result<Vec<(AttrDef, AttrVolume)>, EngineError> {
        let grid = &recon.plan.grid;
        let mut out = Vec::new();
        for def in &self.project.attrs {
            let slices = self.attr_slices(def)?;
            if slices.is_empty() {
                continue;
            }
            let mut layers: Vec<(f64, Vec<f32>)> = Vec::with_capacity(slices.len());
            for (slide, store) in &slices {
                let entry = self.slide(*slide)?;
                layers.push((
                    entry.z_um,
                    resample_attr_layer(grid, store, &entry.transform)?,
                ));
            }
            layers.sort_by(|a, b| a.0.total_cmp(&b.0));
            let data = interpolate_attr_layers(
                grid,
                &layers,
                recon.plan.max_gap_um,
                recon.plan.median_spacing_um / 2.0,
            );
            out.push((
                def.clone(),
                AttrVolume {
                    grid: grid.clone(),
                    data,
                },
            ));
        }
        Ok(out)
    }
}

/// Resampel satu `AttrStore` (ruang U) ke satu lapis kisi (ruang W) dengan
/// pemetaan mundur `u = T⁻¹(w)` dan bilinear berbobot NODATA = 0.
fn resample_attr_layer(
    grid: &StackGrid,
    store: &AttrStore,
    transform: &h3d_core::SlideTransform,
) -> Result<Vec<f32>, EngineError> {
    let values = store.to_array();
    let (h, w) = values.dim();
    let [nx, ny] = [grid.size[0] as usize, grid.size[1] as usize];
    if transform.affine_inverse().is_none() {
        return Err(EngineError::invalid("transformasi irisan singular"));
    }
    let mpp = store.mpp_attr;
    let rows: Vec<Vec<f32>> = (0..ny)
        .into_par_iter()
        .map(|j| {
            let mut row = vec![f32::NAN; nx];
            for (i, o) in row.iter_mut().enumerate() {
                let wpt = grid.index_to_um(i as u32, j as u32);
                let Some(u) = transform.apply_inverse(wpt) else {
                    continue;
                };
                // Koordinat piksel dengan pusat piksel pada bilangan bulat.
                let gx = u[0] / mpp - 0.5;
                let gy = u[1] / mpp - 0.5;
                let x0 = gx.floor();
                let y0 = gy.floor();
                let fx = gx - x0;
                let fy = gy - y0;
                let mut acc = 0.0f64;
                let mut wsum = 0.0f64;
                for (dy, wy) in [(0i64, 1.0 - fy), (1, fy)] {
                    for (dx, wx) in [(0i64, 1.0 - fx), (1, fx)] {
                        let xx = x0 as i64 + dx;
                        let yy = y0 as i64 + dy;
                        if xx < 0 || yy < 0 || xx >= w as i64 || yy >= h as i64 {
                            continue;
                        }
                        let v = values[[yy as usize, xx as usize]];
                        let wt = wx * wy;
                        if v.is_finite() && wt > 0.0 {
                            acc += f64::from(v) * wt;
                            wsum += wt;
                        }
                    }
                }
                if wsum > 1e-9 {
                    *o = (acc / wsum) as f32;
                }
            }
            row
        })
        .collect();
    Ok(rows.into_iter().flatten().collect())
}

/// Isi semua lapis kisi dari lapis beratribut `(z, nilai)` yang terurut z.
fn interpolate_attr_layers(
    grid: &StackGrid,
    layers: &[(f64, Vec<f32>)],
    max_gap_um: f64,
    edge_reach_um: f64,
) -> Vec<f32> {
    let n = grid.layer_len();
    let mut data = vec![f32::NAN; grid.voxel_count()];
    const EPS: f64 = 1e-6;
    for (k, &z) in grid.z_um.iter().enumerate() {
        let dst = &mut data[k * n..(k + 1) * n];
        let below = layers.iter().rposition(|(lz, _)| *lz <= z + EPS);
        let above = layers.iter().position(|(lz, _)| *lz >= z - EPS);
        match (below, above) {
            (Some(a), Some(b)) if a == b => dst.copy_from_slice(&layers[a].1),
            (Some(a), Some(b)) => {
                let (za, va) = &layers[a];
                let (zb, vb) = &layers[b];
                let gap = zb - za;
                if gap <= max_gap_um + EPS && gap > 0.0 {
                    let t = ((z - za) / gap) as f32;
                    for ((d, x), y) in dst.iter_mut().zip(va.iter()).zip(vb.iter()) {
                        *d = match (x.is_finite(), y.is_finite()) {
                            (true, true) => x + (y - x) * t,
                            _ => f32::NAN,
                        };
                    }
                } else if (z - za).abs() <= edge_reach_um.min(gap / 3.0) {
                    dst.copy_from_slice(va);
                } else if (zb - z).abs() <= edge_reach_um.min(gap / 3.0) {
                    dst.copy_from_slice(vb);
                }
            }
            (Some(a), None) if (z - layers[a].0).abs() <= edge_reach_um + EPS => {
                dst.copy_from_slice(&layers[a].1);
            }
            (None, Some(b)) if (layers[b].0 - z).abs() <= edge_reach_um + EPS => {
                dst.copy_from_slice(&layers[b].1);
            }
            _ => {}
        }
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_sum_matches_bruteforce() {
        let (w, h, r) = (7usize, 5usize, 2usize);
        let src: Vec<f64> = (0..w * h).map(|i| (i * 7 % 11) as f64).collect();
        let fast = window_sum(&src, w, h, r);
        for y in 0..h {
            for x in 0..w {
                let mut s = 0.0;
                for yy in y.saturating_sub(r)..(y + r + 1).min(h) {
                    for xx in x.saturating_sub(r)..(x + r + 1).min(w) {
                        s += src[yy * w + xx];
                    }
                }
                assert_eq!(fast[y * w + x], s);
            }
        }
        assert_eq!(window_sum(&src, w, h, 0), src);
    }

    #[test]
    fn attr_layers_interpolate_linearly_and_respect_gap() {
        let grid = StackGrid {
            origin_um: [0.0, 0.0],
            s_xy: 16.0,
            size: [1, 1],
            z_um: vec![-30.0, 0.0, 25.0, 50.0, 100.0, 300.0, 520.0, 560.0],
        };
        let layers = vec![
            (0.0, vec![10.0f32]),
            (50.0, vec![20.0]),
            (500.0, vec![80.0]),
        ];
        let d = interpolate_attr_layers(&grid, &layers, 250.0, 25.0);
        assert!(d[0].is_nan(), "lebih dari setengah jarak di luar ujung");
        assert_eq!(d[1], 10.0);
        assert_eq!(d[2], 15.0);
        assert_eq!(d[3], 20.0);
        assert!(d[4].is_nan(), "celah 450 µm > max_gap → NODATA di tengah");
        assert!(d[5].is_nan());
        assert_eq!(
            d[6], 80.0,
            "dalam jangkauan setengah jarak dari irisan terluar"
        );
        assert!(d[7].is_nan());
    }

    #[test]
    fn params_hash_changes_with_any_parameter() {
        let a = AttrParams::default();
        let mut b = a.clone();
        b.thresholds.t_d = 0.2;
        let mut c = a.clone();
        c.window_px = 3;
        assert_eq!(a.params_hash(), AttrParams::default().params_hash());
        assert_ne!(a.params_hash(), b.params_hash());
        assert_ne!(a.params_hash(), c.params_hash());
    }
}
