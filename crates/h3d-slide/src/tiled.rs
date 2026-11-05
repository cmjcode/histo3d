//! Piramida bertile di disk (`cache/pyramid/<slide>/`) untuk gambar yang
//! tidak membawa piramida sendiri (PNG/JPEG, TIFF datar).
//!
//! Dibangun **sekali** (saat impor) secara streaming per pita baris sehingga
//! memori tidak bergantung pada ukuran gambar, lalu dibaca per tile sesuai
//! permintaan: tidak ada level yang pernah dimuat utuh ke RAM. Tiap level
//! disimpan dalam satu berkas `level_NN.tiles`:
//!
//! ```text
//! offset  ukuran  isi
//! 0       8       magic "H3DTILE1"
//! 8       4       lebar level (u32 LE)
//! 12      4       tinggi level
//! 16      4       ukuran tile (sisi, piksel)
//! 20      4       kolom tile
//! 24      4       baris tile
//! 28      4       kodek tile (1 = PNG RGBA8)
//! 32      12·n    indeks: (offset u64, panjang u32) per tile, urutan baris
//! …       …       muatan tile (tile tepi disimpan pada ukuran sebenarnya)
//! ```
//!
//! Manifes `pyramid.json` (versi 2) menyebut ukuran tiap level. Penulisan
//! atomik: semua berkas ditulis sebagai `.tmp` lalu di-`rename`, manifes
//! terakhir (aturan keras 10); manifes yang ada berarti piramida lengkap.

use std::fs::{self, File};
use std::io::{BufReader, Cursor, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use image::{ImageFormat, ImageReader, RgbaImage};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::error::SlideError;
use crate::source::{
    compose_on_white, downsample_2x_raw, downsample_of, half_size, Intersection, LevelInfo,
    SlideInfo, SlideSource,
};

/// Sisi tile piramida bertile.
pub const TILE_SIZE: u32 = 256;
/// Sisi terpanjang level teratas tidak melebihi nilai ini.
pub const MIN_TOP_SIDE: u32 = 512;

const MAGIC: &[u8; 8] = b"H3DTILE1";
const HEADER_LEN: u64 = 32;
const INDEX_ENTRY_LEN: u64 = 12;
const CODEC_PNG: u32 = 1;
const MANIFEST_VERSION: u32 = 2;
const MANIFEST_NAME: &str = "pyramid.json";

/// Manifes `pyramid.json`.
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
struct Manifest {
    version: u32,
    level_sizes: Vec<[u32; 2]>,
    tile_size: u32,
}

/// Sumber baris RGBA8 berurutan dari atas ke bawah (masukan pembangun).
pub trait RowSource {
    /// Ukuran gambar `[lebar, tinggi]`.
    fn size(&self) -> [u32; 2];
    /// Baris berikutnya, paling banyak `max_rows`; `None` bila habis.
    /// Panjang buffer = `baris × lebar × 4`.
    fn next_rows(&mut self, max_rows: u32) -> Result<Option<Vec<u8>>, SlideError>;
}

/// Baris dari gambar yang sudah ada di memori.
pub struct ImageRows {
    img: RgbaImage,
    y: u32,
}

impl ImageRows {
    /// Bungkus gambar.
    pub fn new(img: RgbaImage) -> Self {
        Self { img, y: 0 }
    }
}

impl RowSource for ImageRows {
    fn size(&self) -> [u32; 2] {
        [self.img.width(), self.img.height()]
    }

    fn next_rows(&mut self, max_rows: u32) -> Result<Option<Vec<u8>>, SlideError> {
        if self.y >= self.img.height() {
            return Ok(None);
        }
        let n = max_rows.min(self.img.height() - self.y);
        let stride = self.img.width() as usize * 4;
        let start = self.y as usize * stride;
        let out = self.img.as_raw()[start..start + n as usize * stride].to_vec();
        self.y += n;
        Ok(Some(out))
    }
}

/// Baris level 0 dari sumber slide apa pun (pita dibaca lewat `read_region`).
pub struct SourceRows<'a> {
    src: &'a dyn SlideSource,
    size: [u32; 2],
    y: u32,
}

impl<'a> SourceRows<'a> {
    /// Bungkus `src`; level 0 dipakai.
    pub fn new(src: &'a dyn SlideSource) -> Self {
        Self {
            src,
            size: src.info().level0_size(),
            y: 0,
        }
    }
}

impl RowSource for SourceRows<'_> {
    fn size(&self) -> [u32; 2] {
        self.size
    }

    fn next_rows(&mut self, max_rows: u32) -> Result<Option<Vec<u8>>, SlideError> {
        if self.y >= self.size[1] {
            return Ok(None);
        }
        let n = max_rows.min(self.size[1] - self.y);
        let img = self
            .src
            .read_region(0, 0, i64::from(self.y), self.size[0], n)?;
        self.y += n;
        Ok(Some(img.into_raw()))
    }
}

/// Baris PNG yang didekode mengalir (memori ≈ beberapa baris), apa pun
/// kedalaman bit atau tipe warnanya.
pub struct PngRows {
    reader: png::Reader<BufReader<File>>,
    size: [u32; 2],
    channels: usize,
    y: u32,
}

impl PngRows {
    /// Buka berkas PNG.
    pub fn open(path: &Path) -> Result<Self, SlideError> {
        let file = File::open(path).map_err(|e| SlideError::io(path, e))?;
        let mut decoder = png::Decoder::new(BufReader::new(file));
        decoder.set_transformations(png::Transformations::normalize_to_color8());
        let reader = decoder
            .read_info()
            .map_err(|e| SlideError::Decode(format!("png {}: {e}", path.display())))?;
        let (w, h) = (reader.info().width, reader.info().height);
        if w == 0 || h == 0 {
            return Err(SlideError::Decode("gambar berukuran nol".into()));
        }
        let (color, depth) = reader.output_color_type();
        if depth != png::BitDepth::Eight {
            return Err(SlideError::Decode(format!(
                "kedalaman bit PNG {depth:?} tidak didukung"
            )));
        }
        let channels = match color {
            png::ColorType::Grayscale => 1,
            png::ColorType::GrayscaleAlpha => 2,
            png::ColorType::Rgb => 3,
            png::ColorType::Rgba => 4,
            other => {
                return Err(SlideError::Decode(format!(
                    "tipe warna PNG {other:?} tidak didukung"
                )))
            }
        };
        Ok(Self {
            reader,
            size: [w, h],
            channels,
            y: 0,
        })
    }
}

impl RowSource for PngRows {
    fn size(&self) -> [u32; 2] {
        self.size
    }

    fn next_rows(&mut self, max_rows: u32) -> Result<Option<Vec<u8>>, SlideError> {
        if self.y >= self.size[1] {
            return Ok(None);
        }
        let n = max_rows.min(self.size[1] - self.y);
        let w = self.size[0] as usize;
        let mut out = Vec::with_capacity(n as usize * w * 4);
        for _ in 0..n {
            let row = self
                .reader
                .next_row()
                .map_err(|e| SlideError::Decode(format!("png: {e}")))?
                .ok_or_else(|| SlideError::Decode("png berakhir sebelum waktunya".into()))?;
            let data = row.data();
            if data.len() < w * self.channels {
                return Err(SlideError::Decode("baris png terlalu pendek".into()));
            }
            match self.channels {
                1 => {
                    for &g in &data[..w] {
                        out.extend_from_slice(&[g, g, g, 255]);
                    }
                }
                2 => {
                    for p in data[..w * 2].as_chunks::<2>().0 {
                        out.extend_from_slice(&[p[0], p[0], p[0], p[1]]);
                    }
                }
                3 => {
                    for p in data[..w * 3].as_chunks::<3>().0 {
                        out.extend_from_slice(&[p[0], p[1], p[2], 255]);
                    }
                }
                _ => out.extend_from_slice(&data[..w * 4]),
            }
        }
        self.y += n;
        Ok(Some(out))
    }
}

/// Sumber baris dari berkas gambar: PNG dialirkan, format lain didekode
/// utuh lewat crate `image`.
pub fn rows_from_path(path: &Path) -> Result<Box<dyn RowSource>, SlideError> {
    let mut head = [0u8; 8];
    {
        let mut f = File::open(path).map_err(|e| SlideError::io(path, e))?;
        let n = f.read(&mut head).map_err(|e| SlideError::io(path, e))?;
        if n == 8 && head == *b"\x89PNG\r\n\x1a\n" {
            return Ok(Box::new(PngRows::open(path)?));
        }
    }
    let reader = ImageReader::open(path)
        .map_err(|e| SlideError::io(path, e))?
        .with_guessed_format()
        .map_err(|e| SlideError::io(path, e))?;
    if reader.format().is_none() {
        return Err(SlideError::backend("flat", "format gambar tidak dikenali"));
    }
    let img = reader.decode()?.into_rgba8();
    if img.width() == 0 || img.height() == 0 {
        return Err(SlideError::Decode("gambar berukuran nol".into()));
    }
    Ok(Box::new(ImageRows::new(img)))
}

/// Ukuran level piramida 2× dari `level0` sampai sisi terpanjang ≤
/// [`MIN_TOP_SIDE`] (minimal satu level).
pub fn level_sizes(level0: [u32; 2]) -> Vec<[u32; 2]> {
    let mut out = vec![level0];
    loop {
        let last = out[out.len() - 1];
        if last[0].max(last[1]) <= MIN_TOP_SIDE {
            break;
        }
        out.push(half_size(last));
    }
    out
}

/// Penulis satu level: menampung baris sampai satu pita tile penuh, menulis
/// tile pita itu, lalu meneruskan pita yang diturunkan 2× ke level berikut.
struct LevelWriter {
    size: [u32; 2],
    tile: u32,
    cols: u32,
    rows: u32,
    file: File,
    tmp: PathBuf,
    path: PathBuf,
    index: Vec<(u64, u32)>,
    offset: u64,
    buf: Vec<u8>,
    buf_rows: u32,
    rows_done: u32,
    next: Option<Box<LevelWriter>>,
}

impl LevelWriter {
    fn create(dir: &Path, sizes: &[[u32; 2]], level: usize, tile: u32) -> Result<Self, SlideError> {
        let size = sizes[level];
        let cols = size[0].div_ceil(tile);
        let rows = size[1].div_ceil(tile);
        let path = dir.join(level_file_name(level));
        let tmp = path.with_extension("tiles.tmp");
        let mut file = File::create(&tmp).map_err(|e| SlideError::io(&tmp, e))?;
        let offset = HEADER_LEN + INDEX_ENTRY_LEN * u64::from(cols) * u64::from(rows);
        // Ruang header + indeks dicadangkan; diisi saat selesai.
        file.set_len(offset).map_err(|e| SlideError::io(&tmp, e))?;
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| SlideError::io(&tmp, e))?;
        let next = if level + 1 < sizes.len() {
            Some(Box::new(Self::create(dir, sizes, level + 1, tile)?))
        } else {
            None
        };
        Ok(Self {
            size,
            tile,
            cols,
            rows,
            file,
            tmp,
            path,
            index: Vec::with_capacity((cols * rows) as usize),
            offset,
            buf: Vec::new(),
            buf_rows: 0,
            rows_done: 0,
            next,
        })
    }

    fn push(&mut self, rows: &[u8], n: u32) -> Result<(), SlideError> {
        self.buf.extend_from_slice(rows);
        self.buf_rows += n;
        while self.buf_rows >= self.tile {
            self.flush_band(self.tile)?;
        }
        Ok(())
    }

    /// Tulis pita `n` baris teratas dari penyangga sebagai satu baris tile.
    fn flush_band(&mut self, n: u32) -> Result<(), SlideError> {
        let w = self.size[0];
        let stride = w as usize * 4;
        let band_len = n as usize * stride;
        let band: Vec<u8> = self.buf.drain(..band_len).collect();
        self.buf_rows -= n;
        let ty = self.rows_done / self.tile;
        if ty >= self.rows {
            return Err(SlideError::Decode(
                "sumber baris lebih panjang dari tinggi".into(),
            ));
        }
        let tile = self.tile;
        let encoded: Vec<Result<Vec<u8>, SlideError>> = (0..self.cols)
            .into_par_iter()
            .map(|tx| {
                let x0 = tx * tile;
                let dw = tile.min(w - x0);
                let mut px = Vec::with_capacity(dw as usize * n as usize * 4);
                for r in 0..n as usize {
                    let s = r * stride + x0 as usize * 4;
                    px.extend_from_slice(&band[s..s + dw as usize * 4]);
                }
                encode_tile(&px, dw, n)
            })
            .collect();
        for enc in encoded {
            let bytes = enc?;
            self.file
                .write_all(&bytes)
                .map_err(|e| SlideError::io(&self.tmp, e))?;
            self.index.push((self.offset, bytes.len() as u32));
            self.offset += bytes.len() as u64;
        }
        self.rows_done += n;
        if let Some(next) = self.next.as_mut() {
            let (down, _, orows) = downsample_2x_raw(&band, w, n);
            next.push(&down, orows)?;
        }
        Ok(())
    }

    fn finish(mut self) -> Result<(), SlideError> {
        if self.buf_rows > 0 {
            let n = self.buf_rows;
            self.flush_band(n)?;
        }
        if self.rows_done != self.size[1] || self.index.len() != (self.cols * self.rows) as usize {
            return Err(SlideError::Decode(format!(
                "sumber baris berakhir pada baris {} dari {}",
                self.rows_done, self.size[1]
            )));
        }
        let mut head = Vec::with_capacity(HEADER_LEN as usize + self.index.len() * 12);
        head.extend_from_slice(MAGIC);
        for v in [
            self.size[0],
            self.size[1],
            self.tile,
            self.cols,
            self.rows,
            CODEC_PNG,
        ] {
            head.extend_from_slice(&v.to_le_bytes());
        }
        for (off, len) in &self.index {
            head.extend_from_slice(&off.to_le_bytes());
            head.extend_from_slice(&len.to_le_bytes());
        }
        self.file
            .seek(SeekFrom::Start(0))
            .map_err(|e| SlideError::io(&self.tmp, e))?;
        self.file
            .write_all(&head)
            .map_err(|e| SlideError::io(&self.tmp, e))?;
        self.file
            .sync_all()
            .map_err(|e| SlideError::io(&self.tmp, e))?;
        fs::rename(&self.tmp, &self.path).map_err(|e| SlideError::io(&self.path, e))?;
        if let Some(next) = self.next.take() {
            next.finish()?;
        }
        Ok(())
    }
}

fn encode_tile(rgba: &[u8], w: u32, h: u32) -> Result<Vec<u8>, SlideError> {
    let mut buf = Vec::with_capacity(rgba.len() / 4);
    let enc = image::codecs::png::PngEncoder::new_with_quality(
        &mut buf,
        image::codecs::png::CompressionType::Default,
        image::codecs::png::FilterType::Adaptive,
    );
    image::ImageEncoder::write_image(enc, rgba, w, h, image::ExtendedColorType::Rgba8)?;
    Ok(buf)
}

fn decode_tile(bytes: &[u8], w: u32, h: u32) -> Result<RgbaImage, SlideError> {
    let img = ImageReader::with_format(Cursor::new(bytes), ImageFormat::Png)
        .decode()?
        .into_rgba8();
    if img.width() != w || img.height() != h {
        return Err(SlideError::Decode(format!(
            "tile cache berukuran {}x{}, diharapkan {w}x{h}",
            img.width(),
            img.height()
        )));
    }
    Ok(img)
}

/// Bangun piramida bertile dari `rows` ke `dir`. `progress(baris_selesai,
/// tinggi)` dipanggil per pita; mengembalikan `false` untuk membatalkan
/// ([`SlideError::Cancelled`]; berkas sementara dihapus, `dir` tanpa manifes).
pub fn build_tiled_pyramid(
    dir: &Path,
    rows: &mut dyn RowSource,
    mut progress: impl FnMut(u32, u32) -> bool,
) -> Result<(), SlideError> {
    let size = rows.size();
    if size[0] == 0 || size[1] == 0 {
        return Err(SlideError::Decode("gambar berukuran nol".into()));
    }
    fs::create_dir_all(dir).map_err(|e| SlideError::io(dir, e))?;
    // Manifes lama dibuang dulu: tanpa manifes = piramida tidak sah.
    let _ = fs::remove_file(dir.join(MANIFEST_NAME));
    let sizes = level_sizes(size);
    let result = (|| {
        let mut writer = LevelWriter::create(dir, &sizes, 0, TILE_SIZE)?;
        let mut done = 0u32;
        while let Some(band) = rows.next_rows(TILE_SIZE)? {
            let n = (band.len() / (size[0] as usize * 4)) as u32;
            writer.push(&band, n)?;
            done += n;
            if !progress(done, size[1]) {
                return Err(SlideError::Cancelled);
            }
        }
        writer.finish()?;
        let manifest = Manifest {
            version: MANIFEST_VERSION,
            level_sizes: sizes.clone(),
            tile_size: TILE_SIZE,
        };
        let json = serde_json::to_vec_pretty(&manifest)
            .map_err(|e| SlideError::Decode(format!("serialisasi manifes: {e}")))?;
        write_atomic(&dir.join(MANIFEST_NAME), &json)
    })();
    if result.is_err() {
        if let Ok(entries) = fs::read_dir(dir) {
            for e in entries.flatten() {
                if e.path().extension().is_some_and(|x| x == "tmp") {
                    let _ = fs::remove_file(e.path());
                }
            }
        }
    }
    result
}

/// Bangun piramida bertile dari berkas gambar (PNG dialirkan).
pub fn build_tiled_pyramid_from_path(
    dir: &Path,
    path: &Path,
    progress: impl FnMut(u32, u32) -> bool,
) -> Result<(), SlideError> {
    let mut rows = rows_from_path(path)?;
    build_tiled_pyramid(dir, rows.as_mut(), progress)
}

/// Bangun piramida bertile dari sumber slide (level 0 dibaca per pita).
pub fn build_tiled_pyramid_from_source(
    dir: &Path,
    src: &dyn SlideSource,
    progress: impl FnMut(u32, u32) -> bool,
) -> Result<(), SlideError> {
    let mut rows = SourceRows::new(src);
    build_tiled_pyramid(dir, &mut rows, progress)
}

/// Benar bila `dir` berisi piramida bertile yang lengkap.
pub fn tiled_pyramid_exists(dir: &Path) -> bool {
    dir.join(MANIFEST_NAME).is_file()
}

struct TiledLevel {
    file: File,
    size: [u32; 2],
    cols: u32,
    index: Vec<(u64, u32)>,
}

/// Piramida bertile di disk sebagai sumber slide. Hanya header dan indeks
/// tiap level yang tinggal di memori; tile didekode saat diminta.
pub struct TiledPyramid {
    info: SlideInfo,
    levels: Vec<TiledLevel>,
}

impl TiledPyramid {
    /// Buka piramida di `dir` (ditulis [`build_tiled_pyramid`]).
    pub fn open(dir: &Path) -> Result<Self, SlideError> {
        let manifest_path = dir.join(MANIFEST_NAME);
        let json = fs::read(&manifest_path).map_err(|e| SlideError::io(&manifest_path, e))?;
        let manifest: Manifest = serde_json::from_slice(&json)
            .map_err(|e| SlideError::Decode(format!("manifes cache rusak: {e}")))?;
        if manifest.version != MANIFEST_VERSION {
            return Err(SlideError::Decode(format!(
                "versi manifes cache {} tidak didukung",
                manifest.version
            )));
        }
        let Some(&l0) = manifest.level_sizes.first() else {
            return Err(SlideError::Decode("manifes cache tanpa level".into()));
        };
        let tile = manifest.tile_size;
        if tile == 0 {
            return Err(SlideError::Decode("ukuran tile nol".into()));
        }
        let mut levels = Vec::with_capacity(manifest.level_sizes.len());
        let mut infos = Vec::with_capacity(manifest.level_sizes.len());
        let mut expected = l0;
        for (i, &size) in manifest.level_sizes.iter().enumerate() {
            if size != expected {
                return Err(SlideError::Decode(format!(
                    "level {i} berukuran {}x{}, diharapkan {}x{}",
                    size[0], size[1], expected[0], expected[1]
                )));
            }
            levels.push(Self::open_level(&dir.join(level_file_name(i)), size, tile)?);
            infos.push(LevelInfo {
                size,
                downsample: downsample_of(l0, size),
            });
            expected = half_size(size);
        }
        Ok(Self {
            info: SlideInfo {
                levels: infos,
                mpp: None,
                tile_size: tile,
                vendor: None,
            },
            levels,
        })
    }

    fn open_level(path: &Path, size: [u32; 2], tile: u32) -> Result<TiledLevel, SlideError> {
        let mut file = File::open(path).map_err(|e| SlideError::io(path, e))?;
        let mut head = [0u8; HEADER_LEN as usize];
        file.read_exact(&mut head)
            .map_err(|e| SlideError::io(path, e))?;
        if &head[..8] != MAGIC {
            return Err(SlideError::Decode(format!(
                "{}: bukan berkas tile HISTO3D",
                path.display()
            )));
        }
        let u = |o: usize| u32::from_le_bytes([head[o], head[o + 1], head[o + 2], head[o + 3]]);
        let (w, h, ts, cols, rows, codec) = (u(8), u(12), u(16), u(20), u(24), u(28));
        if [w, h] != size || ts != tile || codec != CODEC_PNG {
            return Err(SlideError::Decode(format!(
                "{}: header tidak cocok dengan manifes",
                path.display()
            )));
        }
        if cols != w.div_ceil(tile) || rows != h.div_ceil(tile) {
            return Err(SlideError::Decode(format!(
                "{}: kisi tile tidak cocok",
                path.display()
            )));
        }
        let n = (cols as usize) * (rows as usize);
        let mut idx = vec![0u8; n * INDEX_ENTRY_LEN as usize];
        file.read_exact(&mut idx)
            .map_err(|e| SlideError::io(path, e))?;
        let file_len = file.metadata().map_err(|e| SlideError::io(path, e))?.len();
        let mut index = Vec::with_capacity(n);
        for e in idx.as_chunks::<12>().0 {
            let off = u64::from_le_bytes([e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7]]);
            let len = u32::from_le_bytes([e[8], e[9], e[10], e[11]]);
            if off
                .checked_add(u64::from(len))
                .is_none_or(|end| end > file_len)
            {
                return Err(SlideError::Decode(format!(
                    "{}: indeks tile di luar berkas",
                    path.display()
                )));
            }
            index.push((off, len));
        }
        Ok(TiledLevel {
            file,
            size,
            cols,
            index,
        })
    }

    /// Baca dan dekode satu tile `(tx, ty)` pada ukuran sebenarnya.
    fn tile(&self, level: &TiledLevel, tx: u32, ty: u32) -> Result<RgbaImage, SlideError> {
        let ts = self.info.tile_size;
        let i = (ty * level.cols + tx) as usize;
        let &(off, len) = level
            .index
            .get(i)
            .ok_or_else(|| SlideError::Decode("indeks tile di luar kisi".into()))?;
        let mut bytes = vec![0u8; len as usize];
        read_at(&level.file, off, &mut bytes)?;
        let dw = ts.min(level.size[0] - tx * ts);
        let dh = ts.min(level.size[1] - ty * ts);
        decode_tile(&bytes, dw, dh)
    }
}

impl SlideSource for TiledPyramid {
    fn info(&self) -> &SlideInfo {
        &self.info
    }

    fn read_region(
        &self,
        level: u32,
        x: i64,
        y: i64,
        w: u32,
        h: u32,
    ) -> Result<RgbaImage, SlideError> {
        let lvl = self
            .levels
            .get(level as usize)
            .ok_or(SlideError::LevelOutOfRange {
                level,
                count: self.levels.len() as u32,
            })?;
        let ts = self.info.tile_size;
        compose_on_white(lvl.size, x, y, w, h, |s| {
            let tx0 = s.x / ts;
            let tx1 = (s.x + s.w - 1) / ts;
            let ty0 = s.y / ts;
            let ty1 = (s.y + s.h - 1) / ts;
            let coords: Vec<(u32, u32)> = (ty0..=ty1)
                .flat_map(|ty| (tx0..=tx1).map(move |tx| (tx, ty)))
                .collect();
            if coords.len() == 1 {
                // Jalur viewer: satu tile, tanpa pool thread.
                let (tx, ty) = coords[0];
                let img = self.tile(lvl, tx, ty)?;
                if tx * ts == s.x && ty * ts == s.y && img.width() == s.w && img.height() == s.h {
                    return Ok(img);
                }
                let mut out = RgbaImage::new(s.w, s.h);
                blit_tile(&img, (tx * ts, ty * ts), &mut out, s);
                return Ok(out);
            }
            // Wilayah lebar (strip registrasi, thumbnail): dekode paralel,
            // digabung dalam urutan tetap (aturan keras 9).
            let tiles: Vec<Result<RgbaImage, SlideError>> = coords
                .par_iter()
                .map(|&(tx, ty)| self.tile(lvl, tx, ty))
                .collect();
            let mut out = RgbaImage::new(s.w, s.h);
            for (&(tx, ty), img) in coords.iter().zip(tiles) {
                blit_tile(&img?, (tx * ts, ty * ts), &mut out, s);
            }
            Ok(out)
        })
    }
}

/// Salin bagian `tile` (berasal di `origin` koordinat level) yang beririsan
/// dengan `s` ke `out` (berasal di `(s.x, s.y)`).
pub(crate) fn blit_tile(
    tile: &RgbaImage,
    origin: (u32, u32),
    out: &mut RgbaImage,
    s: Intersection,
) {
    let x0 = origin.0.max(s.x);
    let y0 = origin.1.max(s.y);
    let x1 = (origin.0 + tile.width()).min(s.x + s.w);
    let y1 = (origin.1 + tile.height()).min(s.y + s.h);
    if x1 <= x0 || y1 <= y0 {
        return;
    }
    let row_bytes = ((x1 - x0) * 4) as usize;
    for yy in y0..y1 {
        let src_start = (((yy - origin.1) * tile.width() + (x0 - origin.0)) * 4) as usize;
        let dst_start = (((yy - s.y) * out.width() + (x0 - s.x)) * 4) as usize;
        out.as_mut()[dst_start..dst_start + row_bytes]
            .copy_from_slice(&tile.as_raw()[src_start..src_start + row_bytes]);
    }
}

#[cfg(unix)]
fn read_at(file: &File, off: u64, buf: &mut [u8]) -> Result<(), SlideError> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, off)
        .map_err(|e| SlideError::Decode(format!("baca tile: {e}")))
}

#[cfg(windows)]
fn read_at(file: &File, off: u64, buf: &mut [u8]) -> Result<(), SlideError> {
    use std::os::windows::fs::FileExt;
    let mut done = 0usize;
    while done < buf.len() {
        let n = file
            .seek_read(&mut buf[done..], off + done as u64)
            .map_err(|e| SlideError::Decode(format!("baca tile: {e}")))?;
        if n == 0 {
            return Err(SlideError::Decode("berkas tile terpotong".into()));
        }
        done += n;
    }
    Ok(())
}

fn level_file_name(i: usize) -> String {
    format!("level_{i:02}.tiles")
}

/// Tulis `bytes` ke `path` secara atomik (`.tmp` lalu `rename`).
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), SlideError> {
    let tmp: PathBuf = {
        let mut name = path
            .file_name()
            .map(|n| n.to_os_string())
            .unwrap_or_default();
        name.push(".tmp");
        path.with_file_name(name)
    };
    let mut f = File::create(&tmp).map_err(|e| SlideError::io(&tmp, e))?;
    f.write_all(bytes).map_err(|e| SlideError::io(&tmp, e))?;
    f.sync_all().map_err(|e| SlideError::io(&tmp, e))?;
    drop(f);
    fs::rename(&tmp, path).map_err(|e| SlideError::io(path, e))
}
