//! Pembantu uji ujung-ke-ujung: fantom berbenih + proyek dasar terdaftar
//! yang dibuat sekali lalu disalin per tes.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use h3d_core::{Affine2, NoProgress, SlideId};
use h3d_engine::phantom::{write_phantom, PhantomManifest, PhantomOptions, PhantomShapeKind};
use h3d_engine::{FixedClock, ImportOptions, RegOptions, Session};

pub const FIXED_TIME: &str = "2026-01-02T03:04:05+07:00";

/// Opsi fantom ellipsoid kecil untuk tes (192 px, 24 irisan).
pub fn ellipsoid_options() -> PhantomOptions {
    PhantomOptions {
        shape: PhantomShapeKind::Ellipsoid,
        slices: 24,
        seed: 7,
        size_px: 192,
        mpp: 16.0,
        ..Default::default()
    }
}

/// Folder kerja bersama satu proses tes: `slides/` berisi fantom, `base/p.h3d`
/// proyek yang sudah diimpor, didaftarkan, dan diberi label.
pub struct Shared {
    pub root: tempfile::TempDir,
    pub manifest: PhantomManifest,
    /// Waktu (detik) membangun proyek dasar, untuk laporan.
    pub build_seconds: f64,
}

impl Shared {
    pub fn slides_dir(&self) -> PathBuf {
        self.root.path().join("slides")
    }

    pub fn slide_paths(&self) -> Vec<PathBuf> {
        self.manifest
            .slices
            .iter()
            .map(|s| self.slides_dir().join(&s.file))
            .collect()
    }

    pub fn base_project(&self) -> PathBuf {
        self.root.path().join("base").join("p.h3d")
    }
}

pub fn fixed_clock() -> Arc<FixedClock> {
    Arc::new(FixedClock(FIXED_TIME.to_owned()))
}

/// Impor → registrasi → impor label dari masker fantom. Proyek dibuat di
/// `<root>/<run>/p.h3d` sehingga path relatif ke `slides/` sama di tiap jalan.
pub fn build_project(shared_root: &Path, manifest: &PhantomManifest, run: &str) -> Session {
    let dir = shared_root.join(run).join("p.h3d");
    std::fs::create_dir_all(dir.parent().unwrap()).unwrap();
    let mut s = Session::create(&dir).expect("buat proyek");
    s.set_clock(fixed_clock());
    s.set_user_initials("UJI");
    let slides = shared_root.join("slides");
    let paths: Vec<PathBuf> = manifest
        .slices
        .iter()
        .map(|x| slides.join(&x.file))
        .collect();
    let ids = s
        .import_slides_with(
            &paths,
            &ImportOptions {
                mpp: Some([manifest.mpp, manifest.mpp]),
                embed: false,
                spacing_um: Some(manifest.spacing_um),
            },
            &NoProgress,
        )
        .expect("impor");
    assert_eq!(ids.len(), manifest.slices.len());
    s.register_all(&RegOptions::default(), &NoProgress)
        .expect("registrasi");
    import_phantom_labels(&mut s, &slides, manifest);
    s.save().expect("simpan");
    s
}

/// Impor GeoJSON masker fantom ke irisan yang bersesuaian (urutan berkas =
/// urutan impor = id irisan).
pub fn import_phantom_labels(s: &mut Session, slides: &Path, manifest: &PhantomManifest) {
    for (i, sl) in manifest.slices.iter().enumerate() {
        if let Some(g) = &sl.geojson {
            let sum = s
                .import_geojson(SlideId(i as u32), &slides.join(g))
                .expect("impor geojson");
            assert!(sum.imported > 0, "{g}: tidak ada fitur terimpor");
        }
    }
}

pub fn shared() -> &'static Shared {
    static CELL: OnceLock<Shared> = OnceLock::new();
    CELL.get_or_init(|| {
        let root = tempfile::tempdir().expect("tempdir");
        let manifest =
            write_phantom(&root.path().join("slides"), &ellipsoid_options()).expect("fantom");
        let t0 = std::time::Instant::now();
        let s = build_project(root.path(), &manifest, "base");
        drop(s);
        Shared {
            root,
            manifest,
            build_seconds: t0.elapsed().as_secs_f64(),
        }
    })
}

fn copy_dir(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).unwrap();
    for e in std::fs::read_dir(src).unwrap() {
        let e = e.unwrap();
        let to = dst.join(e.file_name());
        if e.file_type().unwrap().is_dir() {
            copy_dir(&e.path(), &to);
        } else {
            std::fs::copy(e.path(), &to).unwrap();
        }
    }
}

/// Salinan proyek dasar di `<root>/<name>/p.h3d`, dibuka dengan jam beku.
pub fn open_copy(name: &str) -> Session {
    let sh = shared();
    let dst = sh.root.path().join(name).join("p.h3d");
    copy_dir(&sh.base_project(), &dst);
    let mut s = Session::open(&dst).expect("buka salinan");
    s.set_clock(fixed_clock());
    s
}

/// `T_i` sebenarnya (U → W kanonik) per id irisan.
pub fn true_transforms(manifest: &PhantomManifest) -> Vec<Affine2> {
    manifest
        .slices
        .iter()
        .map(|s| Affine2::new(s.transform_u_to_w))
        .collect()
}

/// TRE RMS (µm) `T_est` terhadap kebenaran, atas kisi titik di area jaringan
/// (lingkaran 0,3 × sisi di sekitar pusat). Kebenaran dinyatakan relatif
/// terhadap irisan acuan: `T_i_benar_rel = T_acuan_benar⁻¹ ∘ T_i_benar`.
pub fn tre_rms_um(s: &Session, manifest: &PhantomManifest) -> (f64, f64) {
    let truth = true_transforms(manifest);
    let reference = s.project().effective_reference().unwrap();
    let t_ref_inv = truth[reference.0 as usize].inverse().unwrap();
    let field = manifest.mpp * f64::from(manifest.size_px[0]);
    let c = field / 2.0;
    let mut worst = 0.0f64;
    let mut sum = 0.0;
    let mut n = 0.0;
    for e in s.project().slides_ordered() {
        let rel = t_ref_inv.compose(&truth[e.id.0 as usize]);
        let est = e.transform.affine();
        let mut ss = 0.0;
        let mut m = 0.0f64;
        for i in 0..9 {
            for j in 0..9 {
                let u = [
                    c + (i as f64 - 4.0) / 4.0 * 0.3 * field,
                    c + (j as f64 - 4.0) / 4.0 * 0.3 * field,
                ];
                let a = est.apply(u);
                let b = rel.apply(u);
                ss += (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2);
                m += 1.0;
            }
        }
        let rms: f64 = (ss / m).sqrt();
        worst = worst.max(rms);
        sum += rms;
        n += 1.0;
    }
    (sum / n, worst)
}
