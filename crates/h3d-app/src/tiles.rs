//! Aliran tile viewer 2D (histo3d.md 8.10).
//!
//! Tiap frame, untuk tiap lapisan irisan yang tampil:
//! 1. kotak pandang W dibawa ke U lewat `T_i⁻¹` ([`visible_tiles`]);
//! 2. level dipilih dengan `h3d_slide::level_for_scale`;
//! 3. tile yang belum ada diminta ke `TileLoader` dengan prioritas
//!    pusat-dulu; generasi dinaikkan saat himpunan tile yang terlihat
//!    berubah sehingga permintaan basi dibuang;
//! 4. hasil dekode masuk `TileCache` (CPU, berbatas byte), lalu diunggah ke
//!    GPU dalam kuota per frame milik `SceneRenderer`;
//! 5. tile yang belum siap digantikan tile induk (level lebih kasar) yang
//!    sudah ada, supaya layar tidak pernah kosong.
//!
//! Tile label diunggah dari `LabelStore`; tile yang disunting mendapat
//! nomor versi baru (disimpan di field `level` kunci GPU) sehingga versi
//! lama tetap tampil sampai versi baru terunggah.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::mpsc::Receiver;
use std::sync::Arc;

use h3d_core::{Affine2, Budgets, LabelDef, SlideId};
use h3d_engine::{LabelStore, Session};
use h3d_render::{Camera2D, Palette256, TileDraw, TileKey as GpuKey};
use h3d_slide::coords::tiles_in_rect;
use h3d_slide::{level_for_scale, SlideInfo, TileCache, TileLoader, TileRequest, TileResult};

/// Batas jumlah tile yang diminta per lapisan per frame.
pub const MAX_VISIBLE_TILES: usize = 1024;

/// Tujuan unggah tile (GPU sungguhan atau tiruan di tes).
pub trait TileSink {
    /// Tile gambar sudah ada di GPU.
    fn has_tile(&self, key: &GpuKey) -> bool;
    /// Unggah tile RGBA8; `false` bila kuota frame habis atau gagal.
    fn upload_tile(&mut self, key: GpuKey, w: u32, h: u32, rgba: &[u8]) -> bool;
    /// Tile label sudah ada di GPU.
    fn has_label_tile(&self, key: &GpuKey) -> bool;
    /// Unggah tile label `u8`; `false` bila kuota frame habis atau gagal.
    fn upload_label_tile(&mut self, key: GpuKey, w: u32, h: u32, labels: &[u8]) -> bool;
}

/// Kotak pandang di ruang W (µm) beserta skala layar.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewW {
    /// Pojok minimum (µm).
    pub min: [f64; 2],
    /// Pojok maksimum (µm).
    pub max: [f64; 2],
    /// µm per piksel **fisik** layar (point egui dibagi `pixels_per_point`).
    pub um_per_px: f64,
}

impl ViewW {
    /// Dari kamera 2D dan viewport (point egui).
    pub fn from_camera(camera: &Camera2D, viewport: egui::Rect, pixels_per_point: f32) -> Self {
        let a = camera.screen_to_world_um(viewport.min, viewport);
        let b = camera.screen_to_world_um(viewport.max, viewport);
        ViewW {
            min: [a[0].min(b[0]), a[1].min(b[1])],
            max: [a[0].max(b[0]), a[1].max(b[1])],
            um_per_px: camera.um_per_px / f64::from(pixels_per_point.max(0.1)),
        }
    }

    /// Pusat pandangan (µm, W).
    pub fn center(&self) -> [f64; 2] {
        [
            (self.min[0] + self.max[0]) / 2.0,
            (self.min[1] + self.max[1]) / 2.0,
        ]
    }
}

/// Satu tile yang terlihat.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VisibleTile {
    /// Kolom tile.
    pub tx: u32,
    /// Baris tile.
    pub ty: u32,
    /// Jarak pusat tile ke pusat pandangan (µm, ruang U); kecil = dulu.
    pub priority: f32,
}

/// Tile yang terlihat pada satu level, terurut pusat-dulu.
#[derive(Clone, Debug, PartialEq)]
pub struct VisibleTiles {
    /// Level piramida terpilih.
    pub level: u32,
    /// Rentang inklusif `[tx0, ty0, tx1, ty1]`.
    pub range: [u32; 4],
    /// Tile terurut prioritas naik.
    pub tiles: Vec<VisibleTile>,
}

/// Kotak pembatas sumbu-sejajar di U dari kotak pandang W. `t` = `T_i`
/// (U → W); balikannya dipakai di sini. `None` bila `t` singular.
pub fn view_bbox_in_slide(t: &Affine2, view: &ViewW) -> Option<([f64; 2], [f64; 2])> {
    let inv = t.inverse()?;
    let corners = [
        [view.min[0], view.min[1]],
        [view.max[0], view.min[1]],
        [view.max[0], view.max[1]],
        [view.min[0], view.max[1]],
    ];
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for c in corners {
        let u = inv.apply(c);
        for k in 0..2 {
            lo[k] = lo[k].min(u[k]);
            hi[k] = hi[k].max(u[k]);
        }
    }
    (lo[0].is_finite() && lo[1].is_finite() && hi[0].is_finite() && hi[1].is_finite())
        .then_some((lo, hi))
}

/// Piksel level 0 per piksel level `level`, per sumbu (lebih teliti daripada
/// satu angka `downsample` untuk level yang ukurannya dibulatkan).
pub fn level_scale(info: &SlideInfo, level: u32) -> Option<[f64; 2]> {
    let l0 = info.levels.first()?;
    let l = info.levels.get(level as usize)?;
    if l.size[0] == 0 || l.size[1] == 0 {
        return None;
    }
    Some([
        f64::from(l0.size[0]) / f64::from(l.size[0]),
        f64::from(l0.size[1]) / f64::from(l.size[1]),
    ])
}

/// Level untuk skala layar `view.um_per_px` (di W). Skala `T_i` ikut
/// diperhitungkan: satu piksel layar mewakili `um_per_px / sqrt(|det T|)`
/// µm di ruang U.
pub fn choose_level(info: &SlideInfo, mpp: [f64; 2], t: &Affine2, view: &ViewW) -> u32 {
    let det = t.determinant().abs();
    let scale = if det.is_finite() && det > 0.0 {
        det.sqrt()
    } else {
        1.0
    };
    level_for_scale(info, view.um_per_px / scale, mpp)
}

fn tiles_at_level(
    info: &SlideInfo,
    mpp: [f64; 2],
    level: u32,
    lo: [f64; 2],
    hi: [f64; 2],
) -> Option<[u32; 4]> {
    let s = level_scale(info, level)?;
    let size = info.levels.get(level as usize)?.size;
    tiles_in_rect(
        size,
        info.tile_size,
        lo[0] / mpp[0] / s[0],
        lo[1] / mpp[1] / s[1],
        hi[0] / mpp[0] / s[0],
        hi[1] / mpp[1] / s[1],
    )
}

fn range_count(r: [u32; 4]) -> usize {
    (r[2] - r[0] + 1) as usize * (r[3] - r[1] + 1) as usize
}

/// Tile yang terlihat untuk irisan dengan `T_i = t` (U → W) pada pandangan
/// `view` (W). Level dipilih [`choose_level`]; bila jumlah tile melampaui
/// [`MAX_VISIBLE_TILES`] level dinaikkan (lebih kasar). `None` bila irisan
/// di luar pandangan.
pub fn visible_tiles(
    info: &SlideInfo,
    mpp: [f64; 2],
    t: &Affine2,
    view: &ViewW,
) -> Option<VisibleTiles> {
    if info.levels.is_empty() || info.tile_size == 0 || !(mpp[0] > 0.0 && mpp[1] > 0.0) {
        return None;
    }
    let (lo, hi) = view_bbox_in_slide(t, view)?;
    let last = (info.levels.len() - 1) as u32;
    let mut level = choose_level(info, mpp, t, view).min(last);
    let mut range = tiles_at_level(info, mpp, level, lo, hi)?;
    while range_count(range) > MAX_VISIBLE_TILES && level < last {
        level += 1;
        range = tiles_at_level(info, mpp, level, lo, hi)?;
    }
    let s = level_scale(info, level)?;
    let ts = f64::from(info.tile_size);
    let center = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0];
    let mut tiles = Vec::with_capacity(range_count(range).min(MAX_VISIBLE_TILES));
    for ty in range[1]..=range[3] {
        for tx in range[0]..=range[2] {
            let cx = (f64::from(tx) + 0.5) * ts * s[0] * mpp[0];
            let cy = (f64::from(ty) + 0.5) * ts * s[1] * mpp[1];
            let d = ((cx - center[0]).powi(2) + (cy - center[1]).powi(2)).sqrt();
            tiles.push(VisibleTile {
                tx,
                ty,
                priority: d as f32,
            });
        }
    }
    // Pusat dulu; seri dipecah (ty, tx) agar deterministik.
    tiles.sort_by(|a, b| {
        a.priority
            .total_cmp(&b.priority)
            .then(a.ty.cmp(&b.ty))
            .then(a.tx.cmp(&b.tx))
    });
    tiles.truncate(MAX_VISIBLE_TILES);
    Some(VisibleTiles {
        level,
        range,
        tiles,
    })
}

/// Persegi tile dalam µm ruang U `[x0, y0, x1, y1]` dan UV bagian tile yang
/// benar-benar berisi gambar (tile tepi diisi putih oleh pemuat).
pub fn tile_rect_um(
    info: &SlideInfo,
    mpp: [f64; 2],
    level: u32,
    tx: u32,
    ty: u32,
) -> Option<([f32; 4], [f32; 4])> {
    let s = level_scale(info, level)?;
    let size = info.levels.get(level as usize)?.size;
    let ts = info.tile_size;
    let x0 = tx.checked_mul(ts)?;
    let y0 = ty.checked_mul(ts)?;
    if ts == 0 || x0 >= size[0] || y0 >= size[1] {
        return None;
    }
    let vw = ts.min(size[0] - x0);
    let vh = ts.min(size[1] - y0);
    let rect = [
        (f64::from(x0) * s[0] * mpp[0]) as f32,
        (f64::from(y0) * s[1] * mpp[1]) as f32,
        (f64::from(x0 + vw) * s[0] * mpp[0]) as f32,
        (f64::from(y0 + vh) * s[1] * mpp[1]) as f32,
    ];
    let uv = [0.0, 0.0, vw as f32 / ts as f32, vh as f32 / ts as f32];
    Some((rect, uv))
}

/// Tile induk di `parent_level` (lebih kasar) yang memuat pusat tile
/// `(tx, ty)` pada `level`.
pub fn parent_tile(
    info: &SlideInfo,
    level: u32,
    tx: u32,
    ty: u32,
    parent_level: u32,
) -> Option<(u32, u32)> {
    if info.tile_size == 0 {
        return None;
    }
    let s = level_scale(info, level)?;
    let p = level_scale(info, parent_level)?;
    let size = info.levels.get(parent_level as usize)?.size;
    let cols = size[0].div_ceil(info.tile_size).max(1);
    let rows = size[1].div_ceil(info.tile_size).max(1);
    let px = ((f64::from(tx) + 0.5) * s[0] / p[0]).floor();
    let py = ((f64::from(ty) + 0.5) * s[1] / p[1]).floor();
    Some(((px as u32).min(cols - 1), (py as u32).min(rows - 1)))
}

/// Palet overlay dari tabel label: indeks 0 transparan, label tersembunyi
/// transparan, lainnya warna label (alfa penuh; opasitas global di shader).
pub fn palette_from_labels(labels: &[LabelDef]) -> Palette256 {
    let mut p = [[0u8; 4]; 256];
    for l in labels {
        if l.id.is_background() || !l.visible {
            continue;
        }
        p[l.id.0 as usize] = [l.color[0], l.color[1], l.color[2], 255];
    }
    p
}

/// Rentang tile label `[tx0, ty0, tx1, ty1]` yang tersentuh goresan dengan
/// titik `points_um` (U) dan jari-jari terbesar `radius_um`.
pub fn label_tiles_touched(
    store: &LabelStore,
    points_um: &[[f64; 2]],
    radius_um: f64,
) -> Option<[u32; 4]> {
    if points_um.is_empty() || store.tile_size == 0 {
        return None;
    }
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for p in points_um {
        for k in 0..2 {
            lo[k] = lo[k].min(p[k] - radius_um);
            hi[k] = hi[k].max(p[k] + radius_um);
        }
    }
    let m = store.mpp_label;
    tiles_in_rect(
        store.size_px,
        store.tile_size,
        lo[0] / m - 1.0,
        lo[1] / m - 1.0,
        hi[0] / m + 1.0,
        hi[1] / m + 1.0,
    )
}

/// Metadata irisan yang dibutuhkan aliran tile.
#[derive(Clone, Debug)]
pub struct SlideMeta {
    /// Level piramida dan ukuran tile.
    pub info: SlideInfo,
    /// µm per piksel level 0 (dari proyek; tidak pernah ditebak).
    pub mpp: [f64; 2],
}

type CpuKey = h3d_slide::TileKey;
type LabelTile = (SlideId, u32, u32);

/// Aliran tile: pemuat latar, cache CPU, dan pembukuan versi tile label.
pub struct TileStreamer {
    loader: TileLoader,
    rx: Receiver<TileResult>,
    cache: TileCache,
    metas: HashMap<SlideId, SlideMeta>,
    failed: HashSet<SlideId>,
    pending: HashSet<CpuKey>,
    generation: u64,
    frame_hasher: std::collections::hash_map::DefaultHasher,
    last_frame_key: u64,
    deferred: bool,
    next_label_version: u32,
    label_floor: HashMap<SlideId, u32>,
    label_versions: HashMap<LabelTile, u32>,
    label_uploaded: HashMap<LabelTile, u32>,
}

impl TileStreamer {
    /// Buat aliran tile dengan anggaran `budgets` (cache CPU dan jumlah
    /// thread dekode; aturan keras 15).
    pub fn new(budgets: &Budgets) -> Self {
        let (loader, rx) = TileLoader::new(budgets.decode_threads);
        TileStreamer {
            loader,
            rx,
            cache: TileCache::new(budgets.tile_cache_cpu_bytes),
            metas: HashMap::new(),
            failed: HashSet::new(),
            pending: HashSet::new(),
            generation: 1,
            frame_hasher: Default::default(),
            last_frame_key: 0,
            deferred: false,
            next_label_version: 1,
            label_floor: HashMap::new(),
            label_versions: HashMap::new(),
            label_uploaded: HashMap::new(),
        }
    }

    /// Awal frame: ambil hasil dekode ke cache CPU dan naikkan generasi bila
    /// himpunan tile yang terlihat pada frame lalu berbeda dari sebelumnya.
    pub fn begin_frame(&mut self) {
        let key = std::mem::take(&mut self.frame_hasher).finish();
        if key != self.last_frame_key {
            self.last_frame_key = key;
            self.generation += 1;
            self.loader.set_generation(self.generation);
            self.pending.clear();
        }
        self.deferred = false;
        while let Ok(res) = self.rx.try_recv() {
            let k = res.request.key();
            self.pending.remove(&k);
            match res.result {
                Ok(img) => {
                    self.cache.insert(k, Arc::new(img));
                }
                Err(e) => log::warn!("tile {k:?} gagal dimuat: {e}"),
            }
        }
    }

    /// Masih ada tile yang ditunggu (aplikasi perlu meminta repaint).
    pub fn needs_repaint(&self) -> bool {
        !self.pending.is_empty() || self.deferred || self.loader.pending() > 0
    }

    /// Generasi viewport saat ini.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Byte cache CPU terpakai.
    pub fn cpu_cache_bytes(&self) -> u64 {
        self.cache.bytes()
    }

    /// Pastikan sumber piksel irisan terbuka dan terdaftar di pemuat.
    /// Kegagalan dicatat sekali (tidak dicoba ulang tiap frame).
    pub fn ensure_source(&mut self, session: &Session, slide: SlideId) -> Option<&SlideMeta> {
        if !self.metas.contains_key(&slide) {
            if self.failed.contains(&slide) {
                return None;
            }
            let entry = session.slide(slide).ok()?;
            match session.open_source(slide) {
                Ok(src) => {
                    let src: Arc<dyn h3d_slide::SlideSource> = Arc::from(src);
                    let meta = SlideMeta {
                        info: src.info().clone(),
                        mpp: entry.mpp,
                    };
                    self.loader.add_slide(slide, src);
                    self.metas.insert(slide, meta);
                }
                Err(e) => {
                    log::warn!("irisan {slide} tidak bisa dibuka: {e}");
                    self.failed.insert(slide);
                    return None;
                }
            }
        }
        self.metas.get(&slide)
    }

    /// Daftarkan metadata secara langsung (tes dan sumber buatan).
    pub fn insert_meta(&mut self, slide: SlideId, meta: SlideMeta) {
        self.metas.insert(slide, meta);
    }

    /// Metadata irisan bila sumbernya sudah terbuka.
    pub fn meta(&self, slide: SlideId) -> Option<&SlideMeta> {
        self.metas.get(&slide)
    }

    /// Lupakan irisan (mpp diganti atau irisan dikeluarkan dari proyek).
    pub fn forget_slide(&mut self, slide: SlideId) {
        self.loader.remove_slide(slide);
        self.metas.remove(&slide);
        self.failed.remove(&slide);
        self.cache.remove_slide(slide);
        self.pending.retain(|k| k.slide != slide);
        self.invalidate_labels(slide);
    }

    /// Lupakan semua irisan (proyek ditutup/diganti).
    pub fn reset(&mut self) {
        let slides: Vec<SlideId> = self.metas.keys().copied().collect();
        for s in slides {
            self.loader.remove_slide(s);
        }
        self.metas.clear();
        self.failed.clear();
        self.clear_cpu_cache();
        self.label_floor.clear();
        self.label_versions.clear();
    }

    /// Kosongkan cache CPU dan antrean (peringatan memori, 9.4). Cache GPU
    /// dikosongkan pemanggil lewat `Gpu::clear_tile_caches`.
    pub fn clear_cpu_cache(&mut self) {
        self.cache.clear();
        self.loader.clear_pending();
        self.pending.clear();
        self.label_uploaded.clear();
    }

    fn gpu_key(slide: SlideId, level: u32, tx: u32, ty: u32) -> GpuKey {
        GpuKey {
            slide: slide.0,
            level,
            tx,
            ty,
        }
    }

    /// Pastikan tile ada di GPU. Mengembalikan `true` bila siap digambar.
    /// `priority = None` berarti tile tidak diminta ke pemuat bila belum ada.
    fn ensure_tile(
        &mut self,
        sink: &mut Option<&mut dyn TileSink>,
        slide: SlideId,
        tile: (u32, u32, u32),
        priority: Option<f32>,
    ) -> bool {
        let (level, tx, ty) = tile;
        let gkey = Self::gpu_key(slide, level, tx, ty);
        if sink.as_ref().is_some_and(|s| s.has_tile(&gkey)) {
            return true;
        }
        let ckey = CpuKey {
            slide,
            level,
            tx,
            ty,
        };
        if let Some(img) = self.cache.get(&ckey) {
            if let Some(s) = sink.as_mut() {
                if s.upload_tile(gkey, img.width(), img.height(), img.as_raw()) {
                    return true;
                }
                self.deferred = true;
            }
            return false;
        }
        if let Some(priority) = priority {
            if !self.pending.contains(&ckey) {
                let accepted = self.loader.request(TileRequest {
                    slide,
                    level,
                    tx,
                    ty,
                    generation: self.generation,
                    priority,
                });
                if accepted {
                    self.pending.insert(ckey);
                }
            }
        }
        false
    }

    /// Susun tile gambar lapisan irisan `slide` (`t` = `T_i`, U → W) untuk
    /// pandangan `view`. Tile yang belum siap diminta ke pemuat dan diganti
    /// tile induk yang sudah ada. Urutan hasil: kasar dulu, halus di atas.
    pub fn layer_tiles(
        &mut self,
        mut sink: Option<&mut dyn TileSink>,
        slide: SlideId,
        t: &Affine2,
        view: &ViewW,
    ) -> Vec<TileDraw> {
        let Some(meta) = self.metas.get(&slide).cloned() else {
            return Vec::new();
        };
        let Some(vis) = visible_tiles(&meta.info, meta.mpp, t, view) else {
            return Vec::new();
        };
        (slide, vis.level, vis.range).hash(&mut self.frame_hasher);

        let last = (meta.info.levels.len() - 1) as u32;
        let mut fine = Vec::with_capacity(vis.tiles.len());
        let mut missing = Vec::new();
        for tile in &vis.tiles {
            let key = (vis.level, tile.tx, tile.ty);
            if self.ensure_tile(&mut sink, slide, key, Some(tile.priority)) {
                fine.push(key);
            } else {
                missing.push(*tile);
            }
        }

        // Induk untuk tile yang belum siap: naik level sampai ada yang siap.
        // Hanya level terkasar yang ikut diminta (prioritas negatif = paling
        // dulu) supaya selalu ada sesuatu untuk digambar.
        let mut parents: BTreeSet<(std::cmp::Reverse<u32>, u32, u32)> = BTreeSet::new();
        for tile in &missing {
            for pl in (vis.level + 1)..=last {
                let Some((px, py)) = parent_tile(&meta.info, vis.level, tile.tx, tile.ty, pl)
                else {
                    continue;
                };
                let entry = (std::cmp::Reverse(pl), px, py);
                if parents.contains(&entry) {
                    break;
                }
                let priority = (pl == last).then_some(-1.0);
                if self.ensure_tile(&mut sink, slide, (pl, px, py), priority) {
                    parents.insert(entry);
                    break;
                }
            }
        }

        let mut out = Vec::with_capacity(parents.len() + fine.len());
        let coarse = parents
            .into_iter()
            .map(|(std::cmp::Reverse(l), tx, ty)| (l, tx, ty));
        for (l, tx, ty) in coarse.chain(fine) {
            if let Some((rect, uv)) = tile_rect_um(&meta.info, meta.mpp, l, tx, ty) {
                out.push(TileDraw {
                    key: Self::gpu_key(slide, l, tx, ty),
                    rect_um: rect,
                    uv,
                });
            }
        }
        out
    }

    // ---- label ----

    /// Tandai semua tile label irisan berubah (undo, usulan diterima, dsb.).
    pub fn invalidate_labels(&mut self, slide: SlideId) {
        let v = self.bump_label_version();
        self.label_floor.insert(slide, v);
    }

    /// Tandai tile label dalam rentang inklusif `[tx0, ty0, tx1, ty1]` berubah.
    pub fn invalidate_label_tiles(&mut self, slide: SlideId, range: [u32; 4]) {
        for ty in range[1]..=range[3] {
            for tx in range[0]..=range[2] {
                let v = self.bump_label_version();
                self.label_versions.insert((slide, tx, ty), v);
            }
        }
    }

    fn bump_label_version(&mut self) -> u32 {
        let v = self.next_label_version;
        self.next_label_version = self.next_label_version.wrapping_add(1).max(1);
        v
    }

    /// Versi tile label yang berlaku.
    pub fn label_version(&self, slide: SlideId, tx: u32, ty: u32) -> u32 {
        let floor = self.label_floor.get(&slide).copied().unwrap_or(0);
        let own = self
            .label_versions
            .get(&(slide, tx, ty))
            .copied()
            .unwrap_or(0);
        floor.max(own)
    }

    /// Susun tile overlay label irisan `slide` dari `store` (`t` = `T_i`,
    /// U → W). Tile yang versinya berubah diunggah ulang dalam kuota frame;
    /// sampai itu terjadi versi lama yang masih ada di GPU tetap digambar.
    pub fn label_tiles(
        &mut self,
        mut sink: Option<&mut dyn TileSink>,
        slide: SlideId,
        store: &LabelStore,
        t: &Affine2,
        view: &ViewW,
    ) -> Vec<TileDraw> {
        let Some((lo, hi)) = view_bbox_in_slide(t, view) else {
            return Vec::new();
        };
        let m = store.mpp_label;
        let ts = store.tile_size;
        if ts == 0 || m.is_nan() || m <= 0.0 {
            return Vec::new();
        }
        let Some(range) = tiles_in_rect(
            store.size_px,
            ts,
            lo[0] / m,
            lo[1] / m,
            hi[0] / m,
            hi[1] / m,
        ) else {
            return Vec::new();
        };
        let span = f64::from(ts) * m;
        let mut out = Vec::new();
        for (tx, ty) in store.tile_keys() {
            if tx < range[0] || tx > range[2] || ty < range[1] || ty > range[3] {
                continue;
            }
            let Some(data) = store.tile(tx, ty) else {
                continue;
            };
            let want = self.label_version(slide, tx, ty);
            let want_key = Self::gpu_key(slide, want, tx, ty);
            let mut shown = None;
            if let Some(s) = sink.as_mut() {
                if s.has_label_tile(&want_key) || s.upload_label_tile(want_key, ts, ts, data) {
                    self.label_uploaded.insert((slide, tx, ty), want);
                    shown = Some(want_key);
                } else {
                    self.deferred = true;
                    if let Some(old) = self.label_uploaded.get(&(slide, tx, ty)) {
                        let old_key = Self::gpu_key(slide, *old, tx, ty);
                        if s.has_label_tile(&old_key) {
                            shown = Some(old_key);
                        }
                    }
                }
            }
            if let Some(key) = shown {
                let x0 = f64::from(tx) * span;
                let y0 = f64::from(ty) * span;
                out.push(TileDraw::new(
                    key,
                    [x0 as f32, y0 as f32, (x0 + span) as f32, (y0 + span) as f32],
                ));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use h3d_slide::LevelInfo;

    /// Slide 4096 × 2048 px, mpp 0,5, tile 256, level 1/4/16.
    fn info() -> SlideInfo {
        SlideInfo {
            levels: [1.0f64, 4.0, 16.0]
                .iter()
                .map(|d| LevelInfo {
                    size: [(4096.0 / d) as u32, (2048.0 / d) as u32],
                    downsample: *d,
                })
                .collect(),
            mpp: Some([0.5, 0.5]),
            tile_size: 256,
            vendor: None,
        }
    }

    const MPP: [f64; 2] = [0.5, 0.5];

    fn view(min: [f64; 2], max: [f64; 2], um_per_px: f64) -> ViewW {
        ViewW {
            min,
            max,
            um_per_px,
        }
    }

    #[test]
    fn level_follows_screen_scale() {
        let i = info();
        let id = Affine2::IDENTITY;
        let v = |s| view([0.0, 0.0], [100.0, 100.0], s);
        // 0,5 µm/px layar = resolusi penuh.
        assert_eq!(choose_level(&i, MPP, &id, &v(0.5)), 0);
        // 2 µm/px → downsample 4.
        assert_eq!(choose_level(&i, MPP, &id, &v(2.0)), 1);
        // 1 µm/px → target 2; level pertama dengan downsample ≥ 2 adalah 4.
        assert_eq!(choose_level(&i, MPP, &id, &v(1.0)), 1);
        // Sangat jauh → level terkasar.
        assert_eq!(choose_level(&i, MPP, &id, &v(500.0)), 2);
        // T_i yang memperbesar 4× membuat satu piksel layar mewakili ¼ µm U.
        let big = Affine2::scale(4.0, 4.0);
        assert_eq!(choose_level(&i, MPP, &big, &v(2.0)), 0);
    }

    #[test]
    fn visible_tiles_cover_view_and_are_centre_first() {
        let i = info();
        // Pandang 256..768 µm × 128..384 µm = piksel level 0 512..1536 × 256..768.
        let v = view([256.0, 128.0], [768.0, 384.0], 0.5);
        let vis = visible_tiles(&i, MPP, &Affine2::IDENTITY, &v).unwrap();
        assert_eq!(vis.level, 0);
        assert_eq!(vis.range, [2, 1, 5, 2]);
        assert_eq!(vis.tiles.len(), 8);
        // Prioritas tidak turun, dan tile pertama menyentuh pusat pandangan.
        for w in vis.tiles.windows(2) {
            assert!(w[0].priority <= w[1].priority);
        }
        let first = vis.tiles[0];
        assert!((3..=4).contains(&first.tx) && (1..=2).contains(&first.ty));
        // Semua tile unik.
        let set: HashSet<(u32, u32)> = vis.tiles.iter().map(|t| (t.tx, t.ty)).collect();
        assert_eq!(set.len(), 8);
    }

    #[test]
    fn visible_tiles_go_through_inverse_transform() {
        let i = info();
        // T_i menggeser irisan 1000 µm ke kanan di W: pandangan W 1000..1128
        // jatuh di U 0..128 µm = tile (0,0) level 0.
        let t = Affine2::translation(1000.0, 0.0);
        let v = view([1000.0, 0.0], [1128.0, 128.0], 0.5);
        let vis = visible_tiles(&i, MPP, &t, &v).unwrap();
        assert_eq!(vis.range, [0, 0, 0, 0]);
        // Tanpa T_i pandangan yang sama jatuh di tile lain.
        let vis0 = visible_tiles(&i, MPP, &Affine2::IDENTITY, &v).unwrap();
        assert_eq!(vis0.range[0], 7);

        // Rotasi 90° + cermin: kotak pembatas U tetap memuat titik asalnya.
        let t = Affine2::translation(500.0, 300.0)
            .compose(&Affine2::rotation(std::f64::consts::FRAC_PI_2))
            .compose(&Affine2::mirror_x_about(200.0));
        let u = [300.0, 220.0];
        let w = t.apply(u);
        let v = view([w[0] - 10.0, w[1] - 10.0], [w[0] + 10.0, w[1] + 10.0], 0.5);
        let vis = visible_tiles(&i, MPP, &t, &v).unwrap();
        let (tx, ty) = ((u[0] / 0.5 / 256.0) as u32, (u[1] / 0.5 / 256.0) as u32);
        assert!(vis.tiles.iter().any(|t| t.tx == tx && t.ty == ty));
    }

    #[test]
    fn nothing_visible_outside_slide_or_with_singular_transform() {
        let i = info();
        let v = view([-900.0, -900.0], [-100.0, -100.0], 0.5);
        assert!(visible_tiles(&i, MPP, &Affine2::IDENTITY, &v).is_none());
        let singular = Affine2::scale(0.0, 1.0);
        let v = view([0.0, 0.0], [100.0, 100.0], 0.5);
        assert!(visible_tiles(&i, MPP, &singular, &v).is_none());
    }

    #[test]
    fn huge_view_falls_back_to_coarser_level_within_cap() {
        let i = SlideInfo {
            levels: vec![
                LevelInfo {
                    size: [200_000, 200_000],
                    downsample: 1.0,
                },
                LevelInfo {
                    size: [12_500, 12_500],
                    downsample: 16.0,
                },
            ],
            mpp: Some([0.25, 0.25]),
            tile_size: 256,
            vendor: None,
        };
        // Skala layar meminta level 0 tetapi pandangan mencakup seluruh slide.
        let v = view([0.0, 0.0], [50_000.0, 50_000.0], 0.25);
        let vis = visible_tiles(&i, [0.25, 0.25], &Affine2::IDENTITY, &v).unwrap();
        assert_eq!(vis.level, 1);
        assert!(vis.tiles.len() <= MAX_VISIBLE_TILES);
    }

    #[test]
    fn tile_rect_and_parent_are_consistent() {
        let i = info();
        // Tile penuh level 0: 256 px × 0,5 µm = 128 µm.
        let (r, uv) = tile_rect_um(&i, MPP, 0, 2, 1).unwrap();
        assert_eq!(r, [256.0, 128.0, 384.0, 256.0]);
        assert_eq!(uv, [0.0, 0.0, 1.0, 1.0]);
        // Level 2 berukuran 256 × 128: satu tile, separuh bawahnya kosong.
        let (r, uv) = tile_rect_um(&i, MPP, 2, 0, 0).unwrap();
        assert_eq!(r, [0.0, 0.0, 2048.0, 1024.0]);
        assert_eq!(uv, [0.0, 0.0, 1.0, 0.5]);
        assert!(tile_rect_um(&i, MPP, 2, 1, 0).is_none());
        // Induk tile (5, 3) level 0 di level 1 (4×): (1, 0); di level 2: (0, 0).
        assert_eq!(parent_tile(&i, 0, 5, 3, 1), Some((1, 0)));
        assert_eq!(parent_tile(&i, 0, 5, 3, 2), Some((0, 0)));
        assert_eq!(parent_tile(&i, 0, 15, 7, 1), Some((3, 1)));
        // Persegi induk memuat persegi anak.
        let (c, _) = tile_rect_um(&i, MPP, 0, 5, 3).unwrap();
        let (p, _) = tile_rect_um(&i, MPP, 1, 1, 0).unwrap();
        assert!(p[0] <= c[0] && p[1] <= c[1] && p[2] >= c[2] && p[3] >= c[3]);
    }

    /// GPU tiruan: mencatat kunci yang "diunggah" dengan kuota per frame.
    #[derive(Default)]
    struct FakeSink {
        tiles: HashSet<GpuKey>,
        labels: HashSet<GpuKey>,
        budget: usize,
    }

    impl TileSink for FakeSink {
        fn has_tile(&self, key: &GpuKey) -> bool {
            self.tiles.contains(key)
        }
        fn upload_tile(&mut self, key: GpuKey, _w: u32, _h: u32, _rgba: &[u8]) -> bool {
            if self.budget == 0 {
                return false;
            }
            self.budget -= 1;
            self.tiles.insert(key)
        }
        fn has_label_tile(&self, key: &GpuKey) -> bool {
            self.labels.contains(key)
        }
        fn upload_label_tile(&mut self, key: GpuKey, _w: u32, _h: u32, _l: &[u8]) -> bool {
            if self.budget == 0 {
                return false;
            }
            self.budget -= 1;
            self.labels.insert(key)
        }
    }

    fn streamer() -> TileStreamer {
        let mut s = TileStreamer::new(&Budgets::IPAD);
        s.insert_meta(
            SlideId(7),
            SlideMeta {
                info: info(),
                mpp: MPP,
            },
        );
        s
    }

    fn key(level: u32, tx: u32, ty: u32) -> GpuKey {
        GpuKey {
            slide: 7,
            level,
            tx,
            ty,
        }
    }

    #[test]
    fn missing_tiles_fall_back_to_parent_so_screen_is_never_blank() {
        let mut s = streamer();
        let mut sink = FakeSink {
            budget: 8,
            ..Default::default()
        };
        // Hanya tile terkasar yang sudah ada di GPU.
        sink.tiles.insert(key(2, 0, 0));
        let v = view([256.0, 128.0], [768.0, 384.0], 0.5);
        let draws = s.layer_tiles(Some(&mut sink), SlideId(7), &Affine2::IDENTITY, &v);
        assert_eq!(draws.len(), 1, "satu induk menutupi semua tile yang hilang");
        assert_eq!(draws[0].key, key(2, 0, 0));
        assert_eq!(draws[0].rect_um, [0.0, 0.0, 2048.0, 1024.0]);

        // Dua tile halus tersedia: induk tetap digambar dulu, halus di atasnya.
        sink.tiles.insert(key(0, 3, 1));
        sink.tiles.insert(key(0, 4, 2));
        let draws = s.layer_tiles(Some(&mut sink), SlideId(7), &Affine2::IDENTITY, &v);
        assert_eq!(draws.len(), 3);
        assert_eq!(draws[0].key.level, 2, "kasar dulu");
        assert!(draws[1..].iter().all(|d| d.key.level == 0));

        // Semua tile halus tersedia: induk tidak lagi digambar.
        for ty in 1..=2 {
            for tx in 2..=5 {
                sink.tiles.insert(key(0, tx, ty));
            }
        }
        let draws = s.layer_tiles(Some(&mut sink), SlideId(7), &Affine2::IDENTITY, &v);
        assert_eq!(draws.len(), 8);
        assert!(draws.iter().all(|d| d.key.level == 0));
    }

    #[test]
    fn generation_bumps_only_when_visible_set_changes() {
        let mut s = streamer();
        let v = view([256.0, 128.0], [768.0, 384.0], 0.5);
        s.begin_frame();
        s.layer_tiles(None, SlideId(7), &Affine2::IDENTITY, &v);
        s.begin_frame();
        let g = s.generation();
        // Pandangan sama → generasi tetap.
        s.layer_tiles(None, SlideId(7), &Affine2::IDENTITY, &v);
        s.begin_frame();
        assert_eq!(s.generation(), g);
        // Geser sedikit tanpa mengubah rentang tile → tetap.
        let v2 = view([260.0, 130.0], [760.0, 380.0], 0.5);
        s.layer_tiles(None, SlideId(7), &Affine2::IDENTITY, &v2);
        s.begin_frame();
        assert_eq!(s.generation(), g);
        // Geser jauh → generasi naik.
        let v3 = view([1256.0, 128.0], [1768.0, 384.0], 0.5);
        s.layer_tiles(None, SlideId(7), &Affine2::IDENTITY, &v3);
        s.begin_frame();
        assert_eq!(s.generation(), g + 1);
    }

    #[test]
    fn label_tiles_reupload_with_new_version_and_keep_old_until_then() {
        let mut s = streamer();
        let slide = SlideId(7);
        let mut store = LabelStore::new(4.0, [2048.0, 1024.0]);
        store.paint_disc(
            [100.0, 100.0],
            40.0,
            h3d_core::LabelId(1),
            h3d_engine::PaintMode::Replace,
        );
        assert_eq!(store.tile_keys(), vec![(0, 0)]);
        let v = view([0.0, 0.0], [2048.0, 1024.0], 4.0);

        let mut sink = FakeSink {
            budget: 8,
            ..Default::default()
        };
        let d = s.label_tiles(Some(&mut sink), slide, &store, &Affine2::IDENTITY, &v);
        assert_eq!(d.len(), 1);
        let v0 = d[0].key.level;
        assert_eq!(d[0].rect_um, [0.0, 0.0, 1024.0, 1024.0]);

        // Goresan menyentuh tile (0,0): versi naik, kuota habis → versi lama
        // tetap tampil.
        let range = label_tiles_touched(&store, &[[120.0, 110.0]], 30.0).unwrap();
        assert_eq!(range, [0, 0, 0, 0]);
        s.invalidate_label_tiles(slide, range);
        sink.budget = 0;
        let d = s.label_tiles(Some(&mut sink), slide, &store, &Affine2::IDENTITY, &v);
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].key.level, v0);
        assert!(s.needs_repaint());

        // Kuota tersedia lagi → versi baru terunggah.
        sink.budget = 8;
        let d = s.label_tiles(Some(&mut sink), slide, &store, &Affine2::IDENTITY, &v);
        assert_ne!(d[0].key.level, v0);

        // Undo dsb. menandai seluruh irisan.
        let v1 = d[0].key.level;
        s.invalidate_labels(slide);
        let d = s.label_tiles(Some(&mut sink), slide, &store, &Affine2::IDENTITY, &v);
        assert!(d[0].key.level > v1);
    }

    #[test]
    fn palette_hides_background_and_invisible_labels() {
        let mut labels = LabelDef::default_table();
        assert!(labels.len() >= 2);
        labels[1].visible = false;
        let p = palette_from_labels(&labels);
        assert_eq!(p[0], [0, 0, 0, 0]);
        let a = labels[0].id.0 as usize;
        let b = labels[1].id.0 as usize;
        assert_eq!(p[a][3], 255);
        assert_eq!(&p[a][..3], &labels[0].color[..3]);
        assert_eq!(p[b], [0, 0, 0, 0]);
    }
}
