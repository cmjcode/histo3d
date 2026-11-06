//! Penautan objek (instance) antar irisan: komponen terhubung satu label di
//! tiap irisan ditautkan ke komponen irisan tetangga lewat IoU pada kisi W,
//! sehingga "objek yang sama" mendapat `instance_id` yang sama dan bisa
//! direkonstruksi sebagai mesh terpisah (`<Label>#<id>`).
//!
//! # Konvensi
//! - Masker tiap irisan diresampling ke kisi W dengan jalur yang sama
//!   dengan langkah 1 rekonstruksi (8.6.2): `u = T_i⁻¹(w)` pada pusat voxel.
//! - Komponen terhubung konektivitas 8, diurutkan deterministik menurut
//!   `(y_min, x_min, luas)`; nomor komponen 1..=n (0 = latar).
//! - Pencocokan rakus terurut tetap: IoU tertinggi dulu, seri dipecah oleh
//!   indeks komponen. Komponen tanpa pasangan → objek lahir/berakhir.
//!   Satu-ke-banyak dengan IoU ≥ ambang → peristiwa [`TrackEvent::Split`] /
//!   [`TrackEvent::Merge`]; anak selalu mendapat id objek baru.
//! - Objek yang bersentuhan pada satu irisan dapat dibelah dulu dengan
//!   watershed EDT ([`h3d_imgproc::split_touching`]) bila
//!   `TrackParams::split.enabled`; wilayah lebih kecil dari `min_area_um2`
//!   digabung ke tetangga terluas.
//! - Komponen tanpa pasangan pada irisan berikutnya boleh dicocokkan ke
//!   irisan berikutnya lagi (sampai `max_skip_slices` irisan berlabel
//!   dilompati, jarak z tetap ≤ `max_gap_um`): objek yang luput di satu
//!   irisan tetap satu objek, dicatat sebagai [`TrackEvent::Bridged`].
//! - Celah z > `max_gap_um` memutus semua rantai (dicatat sebagai
//!   [`Warning::GapSplit`]).
//! - Deterministik: urutan irisan menurut z (seri → indeks masukan), tidak
//!   ada acak.

use std::collections::BTreeMap;

use h3d_core::{LabelId, Progress, SlideId, SplitParams};
use h3d_imgproc::{label_components, split_touching};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::error::VolError;
use crate::measure::Warning;
use crate::recon::{resample_masks, GridPlan, SliceInput};

/// Ambang IoU bawaan untuk menautkan dua komponen.
pub const DEFAULT_MIN_IOU: f64 = 0.2;

/// Parameter penautan.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TrackParams {
    /// Ambang IoU minimum agar dua komponen dianggap objek yang sama.
    pub min_iou: f64,
    /// Celah z maksimum (µm) yang masih dijembatani; `None` = memakai
    /// `GridPlan::max_gap_um` (5 × jarak irisan median, sama dengan recon).
    pub max_gap_um: Option<f64>,
    /// Pemisah objek bersentuhan per irisan (bawaan nonaktif).
    pub split: SplitParams,
    /// Jumlah irisan berlabel yang boleh dilompati saat mencari pasangan
    /// (bawaan 1; 0 = hanya irisan bertetangga).
    pub max_skip_slices: u32,
}

impl Default for TrackParams {
    fn default() -> Self {
        TrackParams {
            min_iou: DEFAULT_MIN_IOU,
            max_gap_um: None,
            split: SplitParams::default(),
            max_skip_slices: 1,
        }
    }
}

/// FNV-1a 64-bit atas byte (stabil lintas platform; untuk `params_hash`).
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

impl TrackParams {
    /// Periksa nilai: `min_iou` harus terhingga di `[0, 1]`, `max_gap_um`
    /// (bila diisi) positif.
    pub fn validate(&self) -> Result<(), VolError> {
        if !(self.min_iou.is_finite() && (0.0..=1.0).contains(&self.min_iou)) {
            return Err(VolError::InvalidParameter(
                "min_iou harus terhingga dan berada di [0, 1]".into(),
            ));
        }
        if let Some(g) = self.max_gap_um {
            if !(g.is_finite() && g > 0.0) {
                return Err(VolError::InvalidParameter(
                    "max_gap_um harus terhingga dan positif".into(),
                ));
            }
        }
        let sp = &self.split;
        for (name, v) in [
            ("split.min_seed_distance_um", sp.min_seed_distance_um),
            ("split.h_um", sp.h_um),
            ("split.min_area_um2", sp.min_area_um2),
        ] {
            if !(v.is_finite() && v >= 0.0) {
                return Err(VolError::InvalidParameter(format!(
                    "{name} harus terhingga dan tidak negatif"
                )));
            }
        }
        Ok(())
    }

    /// Hash parameter efektif (`min_iou`, `max_gap_um` terselesaikan,
    /// pemisah, lompatan), dicetak di `tracks.json` dan laporan.
    pub fn hash(&self, resolved_max_gap_um: f64) -> u64 {
        let mut b = Vec::with_capacity(64);
        b.extend_from_slice(b"h3d-track-v2");
        b.extend_from_slice(&self.min_iou.to_bits().to_le_bytes());
        b.extend_from_slice(&resolved_max_gap_um.to_bits().to_le_bytes());
        b.push(u8::from(self.split.enabled));
        b.extend_from_slice(&self.split.min_seed_distance_um.to_bits().to_le_bytes());
        b.extend_from_slice(&self.split.h_um.to_bits().to_le_bytes());
        b.extend_from_slice(&self.split.min_area_um2.to_bits().to_le_bytes());
        b.extend_from_slice(&self.max_skip_slices.to_le_bytes());
        fnv1a(&b)
    }
}

/// Komponen terhubung satu label pada satu irisan, di kisi W.
#[derive(Clone, Debug, PartialEq)]
pub struct ComponentMask {
    /// Irisan asal.
    pub slide: SlideId,
    /// Posisi z irisan, µm.
    pub z_um: f64,
    /// Nomor komponen per voxel (1..=n; 0 = latar), bentuk `(ny, nx)`.
    pub labels: Array2<u32>,
    /// Luas tiap komponen dalam voxel; indeks `c − 1`.
    pub areas_px: Vec<usize>,
    /// Pusat massa tiap komponen di ruang W (µm); indeks `c − 1`.
    pub centroids_um: Vec<[f64; 2]>,
}

impl ComponentMask {
    /// Jumlah komponen.
    pub fn len(&self) -> usize {
        self.areas_px.len()
    }

    /// Benar bila tidak ada komponen.
    pub fn is_empty(&self) -> bool {
        self.areas_px.is_empty()
    }
}

/// Satu anggota track: satu komponen pada satu irisan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrackMember {
    /// Irisan.
    pub slide: SlideId,
    /// Nomor komponen pada irisan itu (1..=n).
    pub component: u32,
    /// Luas komponen, µm².
    pub area_um2: f64,
    /// Pusat massa di ruang W, µm.
    pub centroid_um: [f64; 2],
    /// Posisi z irisan, µm.
    pub z_um: f64,
}

/// Peristiwa topologi antar irisan bertetangga.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TrackEvent {
    /// Satu objek pada irisan sebelumnya bersambung ke ≥ 2 komponen pada
    /// irisan `slide`; tiap anak mendapat id baru. Disimpan pada track induk.
    Split {
        /// Irisan tempat anak-anak muncul.
        slide: SlideId,
        /// Posisi z irisan itu, µm.
        z_um: f64,
        /// Id objek induk.
        parent: u32,
        /// Id objek anak, terurut.
        children: Vec<u32>,
    },
    /// ≥ 2 objek pada irisan sebelumnya bersambung ke satu komponen pada
    /// irisan `slide`; anak mendapat id baru. Disimpan pada track anak.
    Merge {
        /// Irisan tempat gabungan muncul.
        slide: SlideId,
        /// Posisi z irisan itu, µm.
        z_um: f64,
        /// Id objek induk, terurut.
        parents: Vec<u32>,
        /// Id objek anak.
        child: u32,
    },
    /// Objek bersambung dari irisan `from` ke `to` dengan melompati irisan
    /// berlabel `skipped` yang tidak memuat objek ini (objek luput di
    /// irisan itu). Disimpan pada track yang bersangkutan.
    Bridged {
        /// Irisan sebelum lompatan.
        from: SlideId,
        /// Irisan sesudah lompatan.
        to: SlideId,
        /// Irisan berlabel yang dilompati, terurut z.
        skipped: Vec<SlideId>,
    },
}

/// Satu objek yang ditautkan lintas irisan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObjectTrack {
    /// Id objek (1..), unik per label.
    pub instance_id: u32,
    /// Label.
    pub label: LabelId,
    /// Anggota terurut z.
    pub members: Vec<TrackMember>,
    /// Peristiwa yang melibatkan objek ini sebagai induk (`Split`) atau
    /// anak (`Merge`).
    pub events: Vec<TrackEvent>,
}

impl ObjectTrack {
    /// Rentang z `(min, maks)` anggota, µm.
    pub fn z_range_um(&self) -> Option<(f64, f64)> {
        let first = self.members.first()?;
        let last = self.members.last()?;
        Some((first.z_um, last.z_um))
    }
}

/// Hasil penautan satu label.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrackSet {
    /// Label.
    pub label: LabelId,
    /// Parameter yang diminta.
    pub params: TrackParams,
    /// Hash parameter efektif (lihat [`TrackParams::hash`]).
    pub params_hash: u64,
    /// Ambang IoU yang dipakai.
    pub min_iou: f64,
    /// Celah maksimum yang dipakai, µm.
    pub max_gap_um: f64,
    /// Ukuran voxel kisi tempat IoU dihitung, µm.
    pub s_xy_um: f64,
    /// Track terurut `instance_id`.
    pub tracks: Vec<ObjectTrack>,
    /// Peringatan (celah yang memutus rantai).
    pub warnings: Vec<Warning>,
    /// Jumlah lompatan irisan ([`TrackEvent::Bridged`]) pada semua track.
    #[serde(default)]
    pub bridged: usize,
}

impl TrackSet {
    /// Jumlah peristiwa `Split` + `Merge` pada semua track (lompatan tidak
    /// dihitung; lihat `bridged`).
    pub fn event_count(&self) -> usize {
        self.tracks
            .iter()
            .flat_map(|t| t.events.iter())
            .filter(|e| !matches!(e, TrackEvent::Bridged { .. }))
            .count()
    }

    /// Irisan yang dilompati track `instance_id` (berlabel, tetapi objek
    /// tidak hadir di sana), terurut z; kosong bila tidak ada.
    pub fn skipped_slides(&self, instance_id: u32) -> Vec<SlideId> {
        let Some(t) = self.track(instance_id) else {
            return Vec::new();
        };
        let mut out: Vec<SlideId> = t
            .events
            .iter()
            .filter_map(|e| match e {
                TrackEvent::Bridged { skipped, .. } => Some(skipped.iter().copied()),
                _ => None,
            })
            .flatten()
            .collect();
        out.sort_by_key(|s| s.0);
        out.dedup();
        out
    }

    /// Track dengan `instance_id` tertentu.
    pub fn track(&self, instance_id: u32) -> Option<&ObjectTrack> {
        self.tracks.iter().find(|t| t.instance_id == instance_id)
    }
}

/// Komponen terhubung (konektivitas 8) `label` per irisan di kisi W, dengan
/// resampling mundur yang sama dengan rekonstruksi. `None` bila irisan tidak
/// memuat label. Komponen dinomori ulang menurut `(y_min, x_min, luas)`.
/// Tanpa pemisah objek bersentuhan; lihat [`component_masks_split`].
pub fn component_masks(
    plan: &GridPlan,
    inputs: &[SliceInput<'_>],
    label: u8,
    progress: &dyn Progress,
) -> Result<Vec<Option<ComponentMask>>, VolError> {
    component_masks_split(plan, inputs, label, &SplitParams::default(), progress)
}

/// Gabungkan wilayah berluas `< min_area_px` ke wilayah tetangga (8) yang
/// terluas (seri → nomor terkecil); wilayah kecil tanpa tetangga dibiarkan.
/// Diproses dari yang terkecil; luas diperbarui setelah tiap penggabungan.
/// Mengembalikan jumlah wilayah tersisa setelah penomoran ulang rapat.
fn merge_small_regions(labels: &mut Array2<u32>, n: usize, min_area_px: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let (h, w) = labels.dim();
    let mut area = vec![0usize; n + 1];
    for &l in labels.iter() {
        area[l as usize] += 1;
    }
    // `alias[l]` = wilayah tujuan (dirantai) setelah penggabungan.
    let mut alias: Vec<u32> = (0..=n as u32).collect();
    let resolve = |alias: &[u32], mut l: u32| {
        while alias[l as usize] != l {
            l = alias[l as usize];
        }
        l
    };
    let mut order: Vec<usize> = (1..=n).collect();
    order.sort_by_key(|&l| (area[l], l));
    for l in order {
        let l = l as u32;
        if resolve(&alias, l) != l || area[l as usize] >= min_area_px {
            continue;
        }
        // Tetangga wilayah lain di sekeliling wilayah `l`.
        let mut neigh: BTreeMap<u32, usize> = BTreeMap::new();
        for ((y, x), &v) in labels.indexed_iter() {
            if resolve(&alias, v) != l {
                continue;
            }
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let (ny, nx) = (y as i64 + dy, x as i64 + dx);
                    if ny < 0 || nx < 0 || ny >= h as i64 || nx >= w as i64 {
                        continue;
                    }
                    let o = labels[[ny as usize, nx as usize]];
                    if o == 0 {
                        continue;
                    }
                    let o = resolve(&alias, o);
                    if o != l {
                        *neigh.entry(o).or_insert(0) += 1;
                    }
                }
            }
        }
        let Some((&target, _)) = neigh.iter().max_by(|a, b| {
            area[*a.0 as usize]
                .cmp(&area[*b.0 as usize])
                .then(b.0.cmp(a.0))
        }) else {
            continue;
        };
        alias[l as usize] = target;
        area[target as usize] += area[l as usize];
        area[l as usize] = 0;
    }
    // Penomoran ulang rapat menurut nomor asal.
    let mut remap = vec![0u32; n + 1];
    let mut m = 0u32;
    for l in 1..=n as u32 {
        let r = resolve(&alias, l);
        if remap[r as usize] == 0 {
            m += 1;
            remap[r as usize] = m;
        }
    }
    for v in labels.iter_mut() {
        if *v != 0 {
            *v = remap[resolve(&alias, *v) as usize];
        }
    }
    m as usize
}

/// Seperti [`component_masks`], tetapi bila `split.enabled` tiap masker
/// irisan dibelah dulu dengan watershed EDT ([`split_touching`]) sehingga
/// objek yang bersentuhan menjadi komponen terpisah; wilayah berluas
/// `< split.min_area_um2` digabung ke tetangga terluas.
pub fn component_masks_split(
    plan: &GridPlan,
    inputs: &[SliceInput<'_>],
    label: u8,
    split: &SplitParams,
    progress: &dyn Progress,
) -> Result<Vec<Option<ComponentMask>>, VolError> {
    let grid = &plan.grid;
    let sub = h3d_core::SubProgress::new(progress, 0.0, 0.7);
    let masks = resample_masks(grid, inputs, label, &sub)?;
    let mut out = Vec::with_capacity(inputs.len());
    for (k, (s, mask)) in inputs.iter().zip(masks).enumerate() {
        if progress.is_cancelled() {
            return Err(VolError::Cancelled);
        }
        progress.set(
            0.7 + 0.3 * k as f32 / inputs.len().max(1) as f32,
            "Mencari komponen terhubung",
        );
        let Some(mask) = mask else {
            out.push(None);
            continue;
        };
        let (raw, n) = if split.enabled {
            let (mut lab, n) =
                split_touching(&mask, grid.s_xy, split.min_seed_distance_um, split.h_um);
            let min_px = (split.min_area_um2 / (grid.s_xy * grid.s_xy)).ceil();
            let n = if min_px > 1.0 {
                merge_small_regions(&mut lab, n, min_px as usize)
            } else {
                n
            };
            (lab, n)
        } else {
            label_components(&mask)
        };
        // Statistik per komponen mentah: (y_min, x_min, luas, Σx, Σy).
        let mut y_min = vec![usize::MAX; n];
        let mut x_min = vec![usize::MAX; n];
        let mut area = vec![0usize; n];
        let mut sx = vec![0f64; n];
        let mut sy = vec![0f64; n];
        for ((y, x), &c) in raw.indexed_iter() {
            if c == 0 {
                continue;
            }
            let i = c as usize - 1;
            y_min[i] = y_min[i].min(y);
            x_min[i] = x_min[i].min(x);
            area[i] += 1;
            sx[i] += x as f64;
            sy[i] += y as f64;
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            (y_min[a], x_min[a], area[a], a).cmp(&(y_min[b], x_min[b], area[b], b))
        });
        let mut remap = vec![0u32; n + 1];
        for (new, &old) in order.iter().enumerate() {
            remap[old + 1] = new as u32 + 1;
        }
        let labels = raw.mapv(|c| remap[c as usize]);
        let areas_px: Vec<usize> = order.iter().map(|&o| area[o]).collect();
        let centroids_um: Vec<[f64; 2]> = order
            .iter()
            .map(|&o| {
                let a = area[o].max(1) as f64;
                [
                    grid.origin_um[0] + (sx[o] / a + 0.5) * grid.s_xy,
                    grid.origin_um[1] + (sy[o] / a + 0.5) * grid.s_xy,
                ]
            })
            .collect();
        out.push(Some(ComponentMask {
            slide: s.id,
            z_um: s.z_um,
            labels,
            areas_px,
            centroids_um,
        }));
    }
    Ok(out)
}

/// Matriks IoU `na × nb` (indeks `i × nb + j`) antara komponen `a` dan `b`.
fn iou_matrix(a: &ComponentMask, b: &ComponentMask) -> Vec<f64> {
    let na = a.len();
    let nb = b.len();
    let mut inter = vec![0usize; na * nb];
    for (&ca, &cb) in a.labels.iter().zip(b.labels.iter()) {
        if ca != 0 && cb != 0 {
            inter[(ca as usize - 1) * nb + (cb as usize - 1)] += 1;
        }
    }
    inter
        .iter()
        .enumerate()
        .map(|(k, &x)| {
            let (i, j) = (k / nb, k % nb);
            let union = a.areas_px[i] + b.areas_px[j] - x;
            if union == 0 {
                0.0
            } else {
                x as f64 / union as f64
            }
        })
        .collect()
}

/// Hubungan komponen `a` → `b` di atas ambang: `children[i]` dan
/// `parents[j]` (terurut naik). Pencocokan rakus satu-ke-satu dulu (IoU
/// tertinggi, seri → indeks terkecil), lalu komponen tak berpasangan
/// ditautkan ke pasangan terbaiknya (split/merge).
fn match_components(
    iou: &[f64],
    na: usize,
    nb: usize,
    thr: f64,
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); na];
    let mut parents: Vec<Vec<usize>> = vec![Vec::new(); nb];
    let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
    for i in 0..na {
        for j in 0..nb {
            let v = iou[i * nb + j];
            if v >= thr && v > 0.0 {
                pairs.push((v, i, j));
            }
        }
    }
    pairs.sort_by(|p, q| q.0.total_cmp(&p.0).then(p.1.cmp(&q.1)).then(p.2.cmp(&q.2)));
    let mut matched_a = vec![false; na];
    let mut matched_b = vec![false; nb];
    for &(_, i, j) in &pairs {
        if !matched_a[i] && !matched_b[j] {
            matched_a[i] = true;
            matched_b[j] = true;
            children[i].push(j);
            parents[j].push(i);
        }
    }
    // Anak tambahan (split): b tak berpasangan → induk terbaik.
    for j in 0..nb {
        if matched_b[j] {
            continue;
        }
        if let Some(&(_, i, _)) = pairs.iter().find(|p| p.2 == j) {
            children[i].push(j);
            parents[j].push(i);
            matched_b[j] = true;
        }
    }
    // Induk tambahan (merge): a tak berpasangan → anak terbaik.
    for i in 0..na {
        if matched_a[i] {
            continue;
        }
        if let Some(&(_, _, j)) = pairs.iter().find(|p| p.1 == i) {
            children[i].push(j);
            parents[j].push(i);
            matched_a[i] = true;
        }
    }
    for c in &mut children {
        c.sort_unstable();
    }
    for p in &mut parents {
        p.sort_unstable();
    }
    (children, parents)
}

/// Tautkan komponen antar irisan bertetangga (urut z) menjadi [`TrackSet`].
/// `comps` berasal dari [`component_masks`] atas `plan` yang sama.
/// `params.max_gap_um = None` memakai `plan.max_gap_um`.
pub fn link_components(
    plan: &GridPlan,
    label: u8,
    comps: &[Option<ComponentMask>],
    params: &TrackParams,
) -> TrackSet {
    let max_gap = params.max_gap_um.unwrap_or(plan.max_gap_um);
    let thr = params.min_iou;
    let s_xy = plan.grid.s_xy;
    let area_um2 = s_xy * s_xy;
    let mut set = TrackSet {
        label: LabelId(label),
        params: *params,
        params_hash: params.hash(max_gap),
        min_iou: thr,
        max_gap_um: max_gap,
        s_xy_um: s_xy,
        tracks: Vec::new(),
        warnings: Vec::new(),
        bridged: 0,
    };
    // Irisan berkomponen, terurut z (seri → indeks masukan), unik per z.
    let mut ann: Vec<(f64, usize)> = comps
        .iter()
        .enumerate()
        .filter_map(|(i, c)| c.as_ref().map(|c| (c.z_um, i)))
        .collect();
    ann.sort_by(|p, q| p.0.total_cmp(&q.0).then(p.1.cmp(&q.1)));
    ann.dedup_by(|a, b| (a.0 - b.0).abs() <= 1e-6);

    let new_track = |set: &mut TrackSet, c: &ComponentMask, comp: usize| -> u32 {
        let id = set.tracks.len() as u32 + 1;
        set.tracks.push(ObjectTrack {
            instance_id: id,
            label: LabelId(label),
            members: vec![member(c, comp, area_um2)],
            events: Vec::new(),
        });
        id
    };

    // Riwayat irisan yang sudah diproses: (indeks comps, id objek per komponen).
    let mut history: Vec<(usize, Vec<u32>)> = Vec::new();
    let max_skip = params.max_skip_slices as usize;
    for &(z, idx) in &ann {
        let Some(cur) = comps[idx].as_ref() else {
            continue;
        };
        let nb = cur.len();
        let mut ids = vec![0u32; nb];
        let prev = history.last();
        match prev {
            Some((pidx, pids)) if comps[*pidx].as_ref().is_some_and(|p| z - p.z_um <= max_gap) => {
                let Some(p) = comps[*pidx].as_ref() else {
                    continue;
                };
                let iou = iou_matrix(p, cur);
                let (children, parents) = match_components(&iou, p.len(), nb, thr);
                // Penerusan.
                for j in 0..nb {
                    let continues = parents[j].len() == 1 && children[parents[j][0]].len() == 1;
                    if continues {
                        let id = pids[parents[j][0]];
                        ids[j] = id;
                        set.tracks[id as usize - 1]
                            .members
                            .push(member(cur, j, area_um2));
                    }
                }
                // Lompatan: komponen tanpa induk sama sekali dicoba ke irisan
                // lebih awal (≤ max_skip irisan dilompati) terhadap objek yang
                // berakhir di sana; satu objek hanya disambung sekali
                // (IoU tertinggi, seri → komponen terkecil).
                let orphans: Vec<usize> = (0..nb).filter(|&j| parents[j].is_empty()).collect();
                if !orphans.is_empty() && max_skip > 0 && history.len() >= 2 {
                    let n_hist = history.len();
                    let lo = n_hist.saturating_sub(1 + max_skip);
                    // Dari yang terdekat ke yang terjauh.
                    for e in (lo..n_hist - 1).rev() {
                        let (eidx, eids) = &history[e];
                        let Some(ec) = comps[*eidx].as_ref() else {
                            continue;
                        };
                        if z - ec.z_um > max_gap {
                            break;
                        }
                        // Objek yang berakhir di irisan `e`.
                        let ended: Vec<usize> = (0..ec.len())
                            .filter(|&i| {
                                let t = &set.tracks[eids[i] as usize - 1];
                                t.members.last().is_some_and(|m| m.slide == ec.slide)
                            })
                            .collect();
                        if ended.is_empty() {
                            continue;
                        }
                        let eiou = iou_matrix(ec, cur);
                        let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
                        for &i in &ended {
                            for &j in &orphans {
                                let v = eiou[i * nb + j];
                                if ids[j] == 0 && v >= thr && v > 0.0 {
                                    pairs.push((v, i, j));
                                }
                            }
                        }
                        pairs.sort_by(|p, q| {
                            q.0.total_cmp(&p.0).then(p.1.cmp(&q.1)).then(p.2.cmp(&q.2))
                        });
                        let mut used_i: Vec<usize> = Vec::new();
                        for &(_, i, j) in &pairs {
                            if ids[j] != 0 || used_i.contains(&i) {
                                continue;
                            }
                            let id = eids[i];
                            let skipped: Vec<SlideId> = history[e + 1..]
                                .iter()
                                .filter_map(|(k, _)| comps[*k].as_ref().map(|c| c.slide))
                                .collect();
                            let t = &mut set.tracks[id as usize - 1];
                            t.members.push(member(cur, j, area_um2));
                            t.events.push(TrackEvent::Bridged {
                                from: ec.slide,
                                to: cur.slide,
                                skipped,
                            });
                            set.bridged += 1;
                            ids[j] = id;
                            used_i.push(i);
                        }
                    }
                }
                // Kelahiran.
                for (j, id) in ids.iter_mut().enumerate() {
                    if *id == 0 {
                        *id = new_track(&mut set, cur, j);
                    }
                }
                // Peristiwa.
                for (i, ch) in children.iter().enumerate() {
                    if ch.len() >= 2 {
                        let parent = pids[i];
                        set.tracks[parent as usize - 1]
                            .events
                            .push(TrackEvent::Split {
                                slide: cur.slide,
                                z_um: cur.z_um,
                                parent,
                                children: ch.iter().map(|&j| ids[j]).collect(),
                            });
                    }
                }
                for (j, pa) in parents.iter().enumerate() {
                    if pa.len() >= 2 {
                        let child = ids[j];
                        set.tracks[child as usize - 1]
                            .events
                            .push(TrackEvent::Merge {
                                slide: cur.slide,
                                z_um: cur.z_um,
                                parents: pa.iter().map(|&i| pids[i]).collect(),
                                child,
                            });
                    }
                }
            }
            other => {
                if let Some((pidx, _)) = other {
                    if let Some(p) = comps[*pidx].as_ref() {
                        set.warnings.push(Warning::GapSplit {
                            from_z: p.z_um,
                            to_z: z,
                            gap: z - p.z_um,
                        });
                    }
                }
                for (j, id) in ids.iter_mut().enumerate() {
                    *id = new_track(&mut set, cur, j);
                }
            }
        }
        history.push((idx, ids));
    }
    set
}

fn member(c: &ComponentMask, comp: usize, area_um2: f64) -> TrackMember {
    TrackMember {
        slide: c.slide,
        component: comp as u32 + 1,
        area_um2: c.areas_px[comp] as f64 * area_um2,
        centroid_um: c.centroids_um[comp],
        z_um: c.z_um,
    }
}

/// Masker id objek per irisan di kisi W: nilai voxel = `instance_id`
/// (0 = latar). `None` bila irisan tidak memuat label; irisan yang
/// komponennya tidak ada di `set` (mis. z ganda) menghasilkan masker nol.
pub fn instance_masks(set: &TrackSet, comps: &[Option<ComponentMask>]) -> Vec<Option<Array2<u32>>> {
    let mut map: BTreeMap<(u32, u32), u32> = BTreeMap::new();
    for t in &set.tracks {
        for m in &t.members {
            map.insert((m.slide.0, m.component), t.instance_id);
        }
    }
    comps
        .iter()
        .map(|c| {
            c.as_ref().map(|c| {
                c.labels.mapv(|v| {
                    if v == 0 {
                        0
                    } else {
                        map.get(&(c.slide.0, v)).copied().unwrap_or(0)
                    }
                })
            })
        })
        .collect()
}

/// Penautan lengkap satu label: resampling + komponen ([`component_masks`])
/// lalu penautan ([`link_components`]). Masukan sama jenisnya dengan
/// [`crate::reconstruct_label`].
pub fn track_label(
    plan: &GridPlan,
    inputs: &[SliceInput<'_>],
    label: u8,
    params: &TrackParams,
    progress: &dyn Progress,
) -> Result<TrackSet, VolError> {
    params.validate()?;
    let comps = component_masks_split(plan, inputs, label, &params.split, progress)?;
    if progress.is_cancelled() {
        return Err(VolError::Cancelled);
    }
    progress.set(1.0, "Menautkan objek antar irisan");
    Ok(link_components(plan, label, &comps, params))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_matching_is_ordered_and_handles_split_merge() {
        // a0 → b0 (0,9), a0 → b1 (0,5): split. a1 → b2 (0,3), a2 → b2 (0,3): merge.
        let na = 3;
        let nb = 3;
        let mut iou = vec![0.0; na * nb];
        iou[0] = 0.9;
        iou[1] = 0.5;
        iou[nb + 2] = 0.3;
        iou[2 * nb + 2] = 0.3;
        let (children, parents) = match_components(&iou, na, nb, 0.2);
        assert_eq!(children, vec![vec![0, 1], vec![2], vec![2]]);
        assert_eq!(parents, vec![vec![0], vec![0], vec![1, 2]]);
        // Di bawah ambang tidak ditautkan.
        let (c2, _) = match_components(&iou, na, nb, 0.6);
        assert_eq!(c2, vec![vec![0], vec![], vec![]]);
    }

    #[test]
    fn params_validation_and_hash() {
        assert!(TrackParams::default().validate().is_ok());
        assert!(TrackParams {
            min_iou: 1.5,
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(TrackParams {
            min_iou: f64::NAN,
            ..Default::default()
        }
        .validate()
        .is_err());
        let p = TrackParams::default();
        assert_eq!(p.hash(500.0), p.hash(500.0));
        assert_ne!(p.hash(500.0), p.hash(600.0));
    }
}
