//! Uji integrasi backend `h3d-slide`: piramida datar, piramida bertile, TIFF
//! multi-IFD, berkas rusak, cache tile, pemuat tile, thumbnail, sidik jari.

use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use h3d_core::SlideId;
use h3d_slide::{
    build_tiled_pyramid, build_tiled_pyramid_from_path, build_tiled_pyramid_from_source,
    fingerprint, open_slide, parse_aperio_mpp, thumbnail, tiled_pyramid_exists, FlatBackend,
    SlideError, SlideInfo, SlideSource, TiffBackend, TileCache, TileKey, TileLoader, TileRequest,
    TiledPyramid, BACKGROUND,
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

// ---------------------------------------------------------------- Tiled

#[test]
fn tiled_pyramid_matches_in_memory_pyramid_pixel_exact() {
    let dir = tempfile::tempdir().unwrap();
    // 1100×900 → tiga level; lebar bukan kelipatan 256 (tile tepi), tinggi
    // bukan kelipatan pita.
    let img = test_image(1100, 900);
    let flat = FlatBackend::from_image(img.clone()).unwrap();
    let mut rows = h3d_slide::tiled::ImageRows::new(img);
    let mut calls = 0;
    build_tiled_pyramid(dir.path(), &mut rows, |done, total| {
        calls += 1;
        assert!(done <= total);
        true
    })
    .unwrap();
    assert!(calls >= 4);
    assert!(tiled_pyramid_exists(dir.path()));
    // Tidak ada berkas .tmp tersisa (penulisan atomik).
    assert!(fs::read_dir(dir.path()).unwrap().all(|e| !e
        .unwrap()
        .path()
        .to_string_lossy()
        .ends_with(".tmp")));

    let tiled = TiledPyramid::open(dir.path()).unwrap();
    assert_eq!(tiled.info().levels, flat.info().levels);
    assert_eq!(tiled.info().tile_size, 256);
    for l in 0..flat.info().levels.len() as u32 {
        let size = flat.info().levels[l as usize].size;
        let whole = tiled.read_region(l, 0, 0, size[0], size[1]).unwrap();
        assert_same_pixels(&whole, flat.level_image(l).unwrap());
    }
    // Wilayah lintas tile, sebagian di luar gambar: sama dengan sumber memori.
    let a = tiled.read_region(0, 200, 700, 400, 300).unwrap();
    let b = flat.read_region(0, 200, 700, 400, 300).unwrap();
    assert_same_pixels(&a, &b);
    let a = tiled.read_region(1, -10, -10, 50, 50).unwrap();
    let b = flat.read_region(1, -10, -10, 50, 50).unwrap();
    assert_same_pixels(&a, &b);
    // Satu tile persis (jalur viewer).
    let a = tiled.read_region(0, 256, 256, 256, 256).unwrap();
    let b = flat.read_region(0, 256, 256, 256, 256).unwrap();
    assert_same_pixels(&a, &b);
    assert!(matches!(
        tiled.read_region(9, 0, 0, 1, 1),
        Err(SlideError::LevelOutOfRange { level: 9, .. })
    ));
}

#[test]
fn tiled_pyramid_from_png_streams_rows_and_survives_source_removal() {
    let dir = tempfile::tempdir().unwrap();
    let png = dir.path().join("foto.png");
    let cache = dir.path().join("cache");
    let img = test_image(900, 700);
    // PNG RGB (tanpa alfa) agar jalur konversi 3 kanal teruji.
    image::DynamicImage::ImageRgba8(img.clone())
        .to_rgb8()
        .save(&png)
        .unwrap();

    build_tiled_pyramid_from_path(&cache, &png, |_, _| true).unwrap();
    fs::remove_file(&png).unwrap();
    let tiled = TiledPyramid::open(&cache).unwrap();
    let flat = FlatBackend::from_image(img).unwrap();
    assert_eq!(tiled.info(), flat.info());
    let a = tiled.read_region(0, 0, 0, 900, 700).unwrap();
    assert_same_pixels(&a, flat.level_image(0).unwrap());
}

#[test]
fn tiled_pyramid_cancel_leaves_no_manifest_or_tmp() {
    let dir = tempfile::tempdir().unwrap();
    let mut rows = h3d_slide::tiled::ImageRows::new(test_image(700, 1000));
    let err = build_tiled_pyramid(dir.path(), &mut rows, |done, _| done < 512).unwrap_err();
    assert!(matches!(err, SlideError::Cancelled));
    assert!(!tiled_pyramid_exists(dir.path()));
    assert!(fs::read_dir(dir.path()).unwrap().all(|e| !e
        .unwrap()
        .path()
        .to_string_lossy()
        .ends_with(".tmp")));
    assert!(TiledPyramid::open(dir.path()).is_err());
}

#[test]
fn tiled_pyramid_from_source_uses_level0_bands() {
    let dir = tempfile::tempdir().unwrap();
    let flat = FlatBackend::from_image(test_image(600, 520)).unwrap();
    build_tiled_pyramid_from_source(dir.path(), &flat, |_, _| true).unwrap();
    let tiled = TiledPyramid::open(dir.path()).unwrap();
    assert_eq!(tiled.info().levels, flat.info().levels);
    let a = tiled.read_region(1, 0, 0, 300, 260).unwrap();
    assert_same_pixels(&a, flat.level_image(1).unwrap());
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
fn tiff_parallel_reads_with_chunk_cache_are_consistent() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("par.tif");
    let img = test_image(400, 300);
    write_striped_pyramid(&path, &img, None, None);
    let b = Arc::new(TiffBackend::open_with_cache_bytes(&path, 8 * 1024 * 1024).unwrap());
    // Banyak thread membaca region tumpang tindih bersamaan: hasil harus
    // identik dengan sumber (kolam dekoder + cache chunk tidak merusak data).
    let handles: Vec<_> = (0..8)
        .map(|i| {
            let b = Arc::clone(&b);
            let img = img.clone();
            std::thread::spawn(move || {
                for k in 0..20u32 {
                    let x = (i * 37 + k * 11) % 380;
                    let y = (i * 13 + k * 7) % 280;
                    let r = b
                        .read_region(0, i64::from(x), i64::from(y), 20, 20)
                        .unwrap();
                    for yy in 0..20u32 {
                        for xx in 0..20u32 {
                            assert_eq!(r.get_pixel(xx, yy), img.get_pixel(x + xx, y + yy));
                        }
                    }
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
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

    // TIFF multi-IFD yang dipotong sebelum rantai IFD selesai harus ditolak
    // saat dibuka (mis. unduhan BigTIFF yang belum selesai).
    let good = dir.path().join("good.tif");
    write_striped_pyramid(&good, &test_image(256, 256), None, None);
    let bytes = fs::read(&good).unwrap();
    let cut = dir.path().join("cut.tif");
    fs::write(&cut, &bytes[..bytes.len() / 3]).unwrap();
    assert!(
        TiffBackend::open(&cut).is_err(),
        "TIFF dengan rantai IFD terputus harus ditolak"
    );
    assert!(open_slide(&cut).is_err());

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

#[test]
fn tile_loader_evicts_least_recently_used_slide() {
    let make = || -> Arc<dyn SlideSource> {
        Arc::new(FlatBackend::from_image(test_image(512, 256)).unwrap())
    };
    let (loader, rx) = TileLoader::with_max_slides(1, 2);
    assert!(loader.add_slide(SlideId(1), make()).is_empty());
    assert!(loader.add_slide(SlideId(2), make()).is_empty());
    // Slide 1 dipakai lagi → slide 2 menjadi yang paling lama.
    let mut r1 = req(0, 0, 0.0);
    r1.slide = SlideId(1);
    assert!(loader.request(r1));
    assert!(rx
        .recv_timeout(Duration::from_secs(5))
        .unwrap()
        .result
        .is_ok());

    let evicted = loader.add_slide(SlideId(3), make());
    assert_eq!(evicted, vec![SlideId(2)]);
    assert_eq!(loader.slide_count(), 2);
    assert!(loader.has_slide(SlideId(1)));
    assert!(!loader.has_slide(SlideId(2)));
    assert!(loader.has_slide(SlideId(3)));

    // Sumber yang dicabut benar-benar dilepas (tidak ada salinan Arc tersisa).
    let tracked = make();
    let weak = Arc::downgrade(&tracked);
    loader.add_slide(SlideId(4), tracked);
    loader.add_slide(SlideId(5), make());
    loader.add_slide(SlideId(6), make());
    assert!(
        weak.upgrade().is_none(),
        "sumber yang dicabut harus dilepas"
    );

    // Daftarkan ulang setelah dicabut tetap bisa dilayani.
    loader.add_slide(SlideId(2), make());
    let mut r2 = req(1, 0, 0.0);
    r2.slide = SlideId(2);
    assert!(loader.request(r2));
    assert!(rx
        .recv_timeout(Duration::from_secs(5))
        .unwrap()
        .result
        .is_ok());
}

#[test]
fn tile_loader_retain_slides_drops_others_and_their_queue() {
    let slow = |d: u64| -> Arc<dyn SlideSource> {
        Arc::new(SlowSource {
            inner: FlatBackend::from_image(test_image(2048, 256)).unwrap(),
            delay: Duration::from_millis(d),
        })
    };
    let (loader, rx) = TileLoader::new(1);
    loader.add_slide(SlideId(7), slow(40));
    loader.add_slide(SlideId(8), slow(0));
    loader.add_slide(SlideId(9), slow(0));
    // Sibukkan pekerja, lalu antrekan tile slide 8.
    loader.request(req(0, 0, 0.0));
    std::thread::sleep(Duration::from_millis(10));
    let mut r8 = req(1, 0, 1.0);
    r8.slide = SlideId(8);
    loader.request(r8);
    assert_eq!(loader.pending(), 1);

    let removed = loader.retain_slides(&[SlideId(7)]);
    assert_eq!(removed, vec![SlideId(8), SlideId(9)]);
    assert_eq!(loader.pending(), 0, "antrean slide yang dicabut dibuang");
    assert_eq!(loader.slide_count(), 1);
    assert_eq!(
        rx.recv_timeout(Duration::from_secs(5))
            .unwrap()
            .request
            .slide,
        SlideId(7)
    );
    assert!(rx.recv_timeout(Duration::from_millis(100)).is_err());
}

#[test]
fn old_version_1_cache_is_rejected_and_cleaned_on_rebuild() {
    // Cache format lama: manifes versi 1 + `level_NN.png` utuh (ratusan MB
    // di proyek nyata). Harus dianggap "belum ada" agar dibangun ulang oleh
    // job latar, bukan saat pertama dibaca di thread UI.
    let dir = tempfile::tempdir().unwrap();
    let img = test_image(600, 500);
    img.save(dir.path().join("level_00.png")).unwrap();
    fs::write(
        dir.path().join("pyramid.json"),
        r#"{"version":1,"level_sizes":[[600,500],[300,250]],"tile_size":256}"#,
    )
    .unwrap();
    assert!(!tiled_pyramid_exists(dir.path()));
    assert!(TiledPyramid::open(dir.path()).is_err());

    // Manifes versi benar tetapi berkas level hilang: juga belum ada.
    let half = tempfile::tempdir().unwrap();
    let mut rows = h3d_slide::tiled::ImageRows::new(img.clone());
    build_tiled_pyramid(half.path(), &mut rows, |_, _| true).unwrap();
    assert!(tiled_pyramid_exists(half.path()));
    fs::remove_file(half.path().join("level_01.tiles")).unwrap();
    assert!(!tiled_pyramid_exists(half.path()));

    // Pembangunan ulang di folder lama membuang `level_00.png`.
    let mut rows = h3d_slide::tiled::ImageRows::new(img.clone());
    build_tiled_pyramid(dir.path(), &mut rows, |_, _| true).unwrap();
    assert!(tiled_pyramid_exists(dir.path()));
    assert!(!dir.path().join("level_00.png").exists());
    let tiled = TiledPyramid::open(dir.path()).unwrap();
    let whole = tiled.read_region(0, 0, 0, 600, 500).unwrap();
    assert_same_pixels(&whole, &img);
}
