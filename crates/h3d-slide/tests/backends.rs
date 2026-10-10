//! Uji integrasi backend `h3d-slide`: piramida datar, cache piramida, TIFF
//! multi-IFD, berkas rusak, cache tile, pemuat tile, thumbnail, sidik jari.

use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use h3d_core::SlideId;
use h3d_slide::{
    fingerprint, open_slide, parse_aperio_mpp, thumbnail, FlatBackend, SlideError, SlideInfo,
    SlideSource, TiffBackend, TileCache, TileKey, TileLoader, TileRequest, BACKGROUND,
};
use image::{Rgba, RgbaImage};
use tiff::encoder::colortype::RGB8;
use tiff::encoder::{Rational, TiffEncoder};
use tiff::tags::{ResolutionUnit, Tag};

/// Gambar uji deterministik: gradien + pola kotak.
fn test_image(w: u32, h: u32) -> RgbaImage {
    RgbaImage::from_fn(w, h, |x, y| {
        Rgba([
            (x % 256) as u8,
            (y % 256) as u8,
            ((x / 7 + y / 5) % 256) as u8,
            255,
        ])
    })
}

fn assert_same_pixels(a: &RgbaImage, b: &RgbaImage) {
    assert_eq!((a.width(), a.height()), (b.width(), b.height()));
    assert_eq!(a.as_raw(), b.as_raw());
}

fn rgb_bytes(img: &RgbaImage) -> Vec<u8> {
    img.pixels().flat_map(|p| [p[0], p[1], p[2]]).collect()
}

// ---------------------------------------------------------------- Flat

#[test]
fn flat_pyramid_levels_are_ceil_half_until_512() {
    let img = test_image(1537, 1001);
    let b = FlatBackend::from_image(img).unwrap();
    let levels = &b.info().levels;
    assert_eq!(levels[0].size, [1537, 1001]);
    for w in levels.windows(2) {
        assert_eq!(
            w[1].size,
            [w[0].size[0].div_ceil(2), w[0].size[1].div_ceil(2)]
        );
        assert!(w[0].size[0].max(w[0].size[1]) > FlatBackend::MIN_TOP_SIDE);
    }
    let top = levels.last().unwrap();
    assert!(top.size[0].max(top.size[1]) <= FlatBackend::MIN_TOP_SIDE);
    // 1537 → 769 → 385: tiga level.
    assert_eq!(levels.len(), 3);
    assert_eq!(levels[2].size, [385, 251]);
    assert!((levels[0].downsample - 1.0).abs() < 1e-12);
    assert!(levels[1].downsample > 1.9 && levels[1].downsample < 2.1);
    assert_eq!(b.info().mpp, None, "mpp gambar datar tidak boleh ditebak");
}

#[test]
fn flat_small_image_has_single_level() {
    let b = FlatBackend::from_image(test_image(512, 300)).unwrap();
    assert_eq!(b.info().levels.len(), 1);
}

#[test]
fn flat_read_region_matches_source_and_fills_white_outside() {
    let img = test_image(700, 600);
    let b = FlatBackend::from_image(img.clone()).unwrap();

    // Di dalam gambar: identik dengan sumber.
    let r = b.read_region(0, 100, 200, 50, 40).unwrap();
    for y in 0..40 {
        for x in 0..50 {
            assert_eq!(r.get_pixel(x, y), img.get_pixel(100 + x, 200 + y));
        }
    }

    // Sepenuhnya di luar: putih semua.
    let out = b.read_region(0, 1000, 1000, 16, 16).unwrap();
    assert!(out.pixels().all(|p| *p == BACKGROUND));
    let out = b.read_region(0, -100, -100, 16, 16).unwrap();
    assert!(out.pixels().all(|p| *p == BACKGROUND));

    // Sebagian di luar (tepi kiri-atas negatif).
    let part = b.read_region(0, -5, -3, 10, 10).unwrap();
    assert_eq!(*part.get_pixel(0, 0), BACKGROUND);
    assert_eq!(*part.get_pixel(4, 2), BACKGROUND);
    assert_eq!(part.get_pixel(5, 3), img.get_pixel(0, 0));
    assert_eq!(part.get_pixel(9, 9), img.get_pixel(4, 6));

    // Sebagian di luar (tepi kanan-bawah).
    let part = b.read_region(0, 695, 595, 10, 10).unwrap();
    assert_eq!(part.get_pixel(0, 0), img.get_pixel(695, 595));
    assert_eq!(part.get_pixel(4, 4), img.get_pixel(699, 599));
    assert_eq!(*part.get_pixel(5, 5), BACKGROUND);

    // Level di luar rentang → error, bukan panik.
    assert!(matches!(
        b.read_region(9, 0, 0, 1, 1),
        Err(SlideError::LevelOutOfRange { level: 9, .. })
    ));
    // Ukuran nol → error.
    assert!(b.read_region(0, 0, 0, 0, 5).is_err());
}

#[test]
fn flat_pyramid_cache_roundtrip_is_identical() {
    let dir = tempfile::tempdir().unwrap();
    let img = test_image(1100, 900);
    let b = FlatBackend::from_image(img).unwrap();
    b.write_pyramid_cache(dir.path()).unwrap();

    let loaded = FlatBackend::load_pyramid_cache(dir.path()).unwrap();
    assert_eq!(loaded.info(), b.info());
    for l in 0..b.info().levels.len() as u32 {
        assert_same_pixels(loaded.level_image(l).unwrap(), b.level_image(l).unwrap());
    }
    // Tidak ada berkas .tmp tersisa (penulisan atomik).
    assert!(fs::read_dir(dir.path()).unwrap().all(|e| !e
        .unwrap()
        .path()
        .to_string_lossy()
        .ends_with(".tmp")));
}

#[test]
fn flat_open_with_cache_uses_cache_on_second_open() {
    let dir = tempfile::tempdir().unwrap();
    let png = dir.path().join("foto.png");
    let cache = dir.path().join("cache");
    test_image(900, 700).save(&png).unwrap();

    let first = FlatBackend::open_with_cache(&png, &cache).unwrap();
    assert!(cache.join("pyramid.json").is_file());
    // Hapus berkas asal: pembukaan kedua harus tetap berhasil dari cache.
    fs::remove_file(&png).unwrap();
    let second = FlatBackend::open_with_cache(&png, &cache).unwrap();
    assert_eq!(first.info(), second.info());
    assert_same_pixels(
        first.level_image(1).unwrap(),
        second.level_image(1).unwrap(),
    );
}

#[test]
fn open_slide_reads_png_and_jpeg() {
    let dir = tempfile::tempdir().unwrap();
    let img = test_image(640, 480);
    let png = dir.path().join("a.png");
    img.save(&png).unwrap();
    let s = open_slide(&png).unwrap();
    assert_eq!(s.info().level0_size(), [640, 480]);
    assert_eq!(s.info().mpp, None);
    let r = s.read_region(0, 10, 10, 4, 4).unwrap();
    assert_eq!(r.get_pixel(0, 0), img.get_pixel(10, 10));

    let jpg = dir.path().join("b.jpg");
    image::DynamicImage::ImageRgba8(img)
        .to_rgb8()
        .save(&jpg)
        .unwrap();
    let s = open_slide(&jpg).unwrap();
    assert_eq!(s.info().level0_size(), [640, 480]);
    assert!(s.read_region(0, 0, 0, 8, 8).is_ok());
}

// ---------------------------------------------------------------- TIFF

/// Tulis TIFF multi-IFD berstrip (RGB8) dengan level `sizes`, deskripsi
/// `description` dan resolusi opsional (px/cm) pada IFD pertama.
fn write_striped_pyramid(
    path: &Path,
    level0: &RgbaImage,
    description: Option<&str>,
    res_px_per_cm: Option<u32>,
) -> Vec<[u32; 2]> {
    let mut sizes = Vec::new();
    let file = fs::File::create(path).unwrap();
    let mut enc = TiffEncoder::new(std::io::BufWriter::new(file)).unwrap();
    let mut cur = level0.clone();
    loop {
        sizes.push([cur.width(), cur.height()]);
        let mut img = enc.new_image::<RGB8>(cur.width(), cur.height()).unwrap();
        if sizes.len() == 1 {
            if let Some(d) = description {
                img.encoder().write_tag(Tag::ImageDescription, d).unwrap();
            }
            if let Some(r) = res_px_per_cm {
                img.resolution(ResolutionUnit::Centimeter, Rational { n: r, d: 1 });
            }
        }
        img.rows_per_strip(16).unwrap();
        img.write_data(&rgb_bytes(&cur)).unwrap();
        if cur.width().max(cur.height()) <= 64 {
            break;
        }
        cur = h3d_slide::downsample_2x(&cur);
    }
    sizes
}

#[test]
fn tiff_multi_ifd_levels_regions_and_mpp() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pyr.tif");
    let img = test_image(300, 200);
    let sizes = write_striped_pyramid(
        &path,
        &img,
        Some("Aperio Image Library v12.0.15|AppMag = 20|MPP = 0.4990|Filename = x"),
        None,
    );

    let b = TiffBackend::open(&path).unwrap();
    let info: &SlideInfo = b.info();
    assert_eq!(info.levels.len(), sizes.len());
    for (l, s) in info.levels.iter().zip(&sizes) {
        assert_eq!(l.size, *s);
    }
    assert!((info.levels[1].downsample - 2.0).abs() < 0.05);
    let mpp = info.mpp.unwrap();
    assert!((mpp[0] - 0.499).abs() < 1e-9 && (mpp[1] - 0.499).abs() < 1e-9);
    assert_eq!(info.vendor.as_deref(), Some("Aperio"));

    // Region level 0 melintasi batas strip (16 baris) dan tepi gambar.
    let r = b.read_region(0, 290, 10, 20, 20).unwrap();
    for y in 0..20u32 {
        for x in 0..10u32 {
            assert_eq!(
                r.get_pixel(x, y),
                img.get_pixel(290 + x, 10 + y),
                "({x},{y})"
            );
        }
        for x in 10..20u32 {
            assert_eq!(*r.get_pixel(x, y), BACKGROUND);
        }
    }
    // Region level 1 identik dengan penurunan 2× sumber.
    let l1 = h3d_slide::downsample_2x(&img);
    let r = b.read_region(1, 0, 0, l1.width(), l1.height()).unwrap();
    assert_same_pixels(&r, &l1);
    // Sepenuhnya di luar → putih.
    let r = b.read_region(1, -50, -50, 8, 8).unwrap();
    assert!(r.pixels().all(|p| *p == BACKGROUND));

    // Lewat open_slide juga berhasil dan punya ≥ 1 level.
    let s = open_slide(&path).unwrap();
    assert_eq!(s.info().level0_size(), [300, 200]);
    let r = s.read_region(0, 0, 0, 4, 4).unwrap();
    assert_eq!(r.get_pixel(3, 3), img.get_pixel(3, 3));
}

#[test]
fn tiff_mpp_from_resolution_tags() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("res.tif");
    // 20000 px/cm → 0,5 µm/px.
    write_striped_pyramid(&path, &test_image(128, 96), None, Some(20_000));
    let b = TiffBackend::open(&path).unwrap();
    let mpp = b.info().mpp.unwrap();
    assert!((mpp[0] - 0.5).abs() < 1e-9 && (mpp[1] - 0.5).abs() < 1e-9);
    assert_eq!(b.info().vendor, None);

    // Tanpa metadata apa pun → None.
    let path = dir.path().join("nores.tif");
    write_striped_pyramid(&path, &test_image(128, 96), None, None);
    assert_eq!(TiffBackend::open(&path).unwrap().info().mpp, None);
}

#[test]
fn aperio_mpp_parsing() {
    assert_eq!(
        parse_aperio_mpp("Aperio Image Library|AppMag = 40|MPP = 0.5|"),
        Some(0.5)
    );
    assert_eq!(parse_aperio_mpp("Aperio|MPP = 0.2527"), Some(0.2527));
    assert_eq!(parse_aperio_mpp("Hamamatsu"), None);
}

// ---------------------------------------------------------------- Rusak

#[test]
fn corrupted_files_return_err_not_panic() {
    let dir = tempfile::tempdir().unwrap();
    // Byte acak deterministik (LCG).
    let mut seed: u32 = 0x1234_5678;
    let junk: Vec<u8> = (0..4096)
        .map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 24) as u8
        })
        .collect();
    for name in ["junk.tif", "junk.png", "junk.jpg", "junk.svs"] {
        let p = dir.path().join(name);
        fs::write(&p, &junk).unwrap();
        assert!(open_slide(&p).is_err(), "{name} harus gagal");
        assert!(TiffBackend::open(&p).is_err());
        assert!(FlatBackend::open(&p).is_err());
    }

    // TIFF sah yang dipotong di tengah.
    let good = dir.path().join("good.tif");
    write_striped_pyramid(&good, &test_image(256, 256), None, None);
    let bytes = fs::read(&good).unwrap();
    let cut = dir.path().join("cut.tif");
    fs::write(&cut, &bytes[..bytes.len() / 3]).unwrap();
    match TiffBackend::open(&cut) {
        Err(_) => {}
        Ok(b) => {
            // Header mungkin masih terbaca; membaca piksel harus gagal
            // atau mengembalikan gambar, tetapi tidak panik.
            let _ = b.read_region(0, 0, 0, 256, 256);
        }
    }

    // PNG sah yang dipotong.
    let png = dir.path().join("good.png");
    test_image(256, 256).save(&png).unwrap();
    let bytes = fs::read(&png).unwrap();
    let cutpng = dir.path().join("cut.png");
    fs::write(&cutpng, &bytes[..bytes.len() / 2]).unwrap();
    assert!(FlatBackend::open(&cutpng).is_err());

    // Berkas tidak ada → Io.
    assert!(matches!(
        open_slide(&dir.path().join("tidak-ada.svs")),
        Err(SlideError::Io { .. })
    ));
}

// ---------------------------------------------------------------- Cache & loader

#[test]
fn tile_cache_evicts_lru_within_byte_budget() {
    let tile = || Arc::new(RgbaImage::new(16, 16)); // 1024 byte
    let budget = 1024 * 4;
    let mut c = TileCache::new(budget);
    for i in 0..4 {
        assert!(c.insert(TileKey::from((SlideId(1), 0, i, 0)), tile()));
    }
    assert_eq!(c.bytes(), budget);
    // Sentuh tile 0 dan 1; tile 2 jadi yang paling lama.
    c.get(&TileKey::from((SlideId(1), 0, 0, 0)));
    c.get(&TileKey::from((SlideId(1), 0, 1, 0)));
    c.insert(TileKey::from((SlideId(1), 0, 9, 0)), tile());
    assert!(c.bytes() <= budget);
    assert!(!c.contains(&TileKey::from((SlideId(1), 0, 2, 0))));
    assert!(c.contains(&TileKey::from((SlideId(1), 0, 0, 0))));
    assert!(c.contains(&TileKey::from((SlideId(1), 0, 9, 0))));
    c.evict_to(1024);
    assert_eq!(c.len(), 1);
    c.clear();
    assert_eq!(c.bytes(), 0);
}

/// Sumber lambat untuk menguji pembuangan generasi.
struct SlowSource {
    inner: FlatBackend,
    delay: Duration,
}

impl SlideSource for SlowSource {
    fn info(&self) -> &SlideInfo {
        self.inner.info()
    }
    fn read_region(
        &self,
        level: u32,
        x: i64,
        y: i64,
        w: u32,
        h: u32,
    ) -> Result<RgbaImage, SlideError> {
        std::thread::sleep(self.delay);
        self.inner.read_region(level, x, y, w, h)
    }
}

fn req(tx: u32, generation: u64, priority: f32) -> TileRequest {
    TileRequest {
        slide: SlideId(7),
        level: 0,
        tx,
        ty: 0,
        generation,
        priority,
    }
}

#[test]
fn tile_loader_loads_tiles_and_discards_stale_generations() {
    let src = Arc::new(SlowSource {
        inner: FlatBackend::from_image(test_image(1024, 256)).unwrap(),
        delay: Duration::from_millis(60),
    });
    let (loader, rx) = TileLoader::new(1);
    loader.add_slide(SlideId(7), src.clone());

    // Permintaan dengan generasi lampau langsung ditolak.
    loader.set_generation(5);
    assert!(!loader.request(req(0, 4, 0.0)));
    assert_eq!(loader.discarded(), 1);

    // Tiga permintaan generasi 5; pekerja tunggal lambat, lalu generasi naik.
    assert!(loader.request(req(0, 5, 0.0)));
    assert!(loader.request(req(1, 5, 1.0)));
    assert!(loader.request(req(2, 5, 2.0)));
    std::thread::sleep(Duration::from_millis(10));
    loader.set_generation(6);

    let mut received = Vec::new();
    while let Ok(r) = rx.recv_timeout(Duration::from_millis(500)) {
        received.push(r);
        if received.len() + (loader.discarded() as usize - 1) >= 3 && loader.pending() == 0 {
            break;
        }
    }
    // Tunggu pekerja menyelesaikan/menolak sisa antrean.
    for _ in 0..50 {
        if received.len() + (loader.discarded() as usize - 1) >= 3 {
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
        while let Ok(r) = rx.try_recv() {
            received.push(r);
        }
    }
    let stale_discarded = loader.discarded() - 1;
    assert_eq!(received.len() as u64 + stale_discarded, 3);
    assert!(
        stale_discarded >= 2,
        "minimal dua permintaan antre harus dibuang"
    );
    for r in &received {
        let tile = r.result.as_ref().unwrap();
        assert_eq!((tile.width(), tile.height()), (256, 256));
    }

    // Permintaan generasi baru tetap dilayani dan isinya benar.
    assert!(loader.request(req(3, 6, 0.0)));
    let r = rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(r.request.tx, 3);
    let tile = r.result.unwrap();
    let expect = src.inner.read_region(0, 768, 0, 256, 256).unwrap();
    assert_same_pixels(&tile, &expect);

    // Slide tak terdaftar → hasil error, bukan panik.
    let mut bad = req(0, 6, 0.0);
    bad.slide = SlideId(99);
    assert!(loader.request(bad));
    let r = rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert!(r.result.is_err());
    drop(loader);
}

#[test]
fn tile_loader_orders_by_priority() {
    let src = Arc::new(SlowSource {
        inner: FlatBackend::from_image(test_image(2048, 256)).unwrap(),
        delay: Duration::from_millis(40),
    });
    let (loader, rx) = TileLoader::new(1);
    loader.add_slide(SlideId(7), src);
    // Permintaan pertama menyibukkan pekerja; sisanya diurutkan prioritas.
    loader.request(req(0, 0, 0.0));
    std::thread::sleep(Duration::from_millis(10));
    loader.request(req(5, 0, 5.0));
    loader.request(req(1, 0, 1.0));
    loader.request(req(3, 0, 3.0));
    let mut order = Vec::new();
    for _ in 0..4 {
        order.push(rx.recv_timeout(Duration::from_secs(5)).unwrap().request.tx);
    }
    assert_eq!(order, vec![0, 1, 3, 5]);
}

// ---------------------------------------------------------------- Thumbnail & sidik jari

#[test]
fn thumbnail_fits_max_side_and_averages() {
    let img = RgbaImage::from_fn(1200, 800, |x, _| {
        if x < 600 {
            Rgba([0, 0, 0, 255])
        } else {
            Rgba([255, 255, 255, 255])
        }
    });
    let b = FlatBackend::from_image(img).unwrap();
    let t = thumbnail(&b, 200).unwrap();
    assert_eq!((t.width(), t.height()), (200, 133));
    assert_eq!(*t.get_pixel(10, 50), Rgba([0, 0, 0, 255]));
    assert_eq!(*t.get_pixel(190, 50), Rgba([255, 255, 255, 255]));

    // max_side lebih besar dari level terkecil: level terkecil apa adanya.
    let t = thumbnail(&b, 4096).unwrap();
    let top = b.info().levels.last().unwrap().size;
    assert_eq!([t.width(), t.height()], top);

    assert!(thumbnail(&b, 0).is_err());
}

#[test]
fn fingerprint_is_deterministic_and_sensitive() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("a.bin");
    let big: Vec<u8> = (0..(3 * 1024 * 1024u32)).map(|i| (i % 251) as u8).collect();
    fs::write(&p, &big).unwrap();
    let (size, h1) = fingerprint(&p).unwrap();
    assert_eq!(size, big.len() as u64);
    assert_eq!(fingerprint(&p).unwrap(), (size, h1));

    // Ubah satu byte di akhir → hash berubah.
    let mut tail = big.clone();
    *tail.last_mut().unwrap() ^= 0xff;
    fs::write(&p, &tail).unwrap();
    assert_ne!(fingerprint(&p).unwrap().1, h1);

    // Ubah byte di tengah (di luar jendela) → hash sama (sesuai spesifikasi).
    let mut mid = big.clone();
    mid[big.len() / 2] ^= 0xff;
    fs::write(&p, &mid).unwrap();
    assert_eq!(fingerprint(&p).unwrap().1, h1);

    // Berkas kecil dan kosong.
    let s = dir.path().join("s.bin");
    fs::write(&s, b"abc").unwrap();
    assert_eq!(fingerprint(&s).unwrap().0, 3);
    fs::write(&s, b"").unwrap();
    assert_eq!(fingerprint(&s).unwrap().0, 0);
}
