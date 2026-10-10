//! Uji rantai komposisi, QC, landmark, dan `register_stack`.

mod common;

use std::sync::Arc;

use common::{textured_image, warp_image, Field3};
use h3d_core::{Affine2, NoProgress, Progress, RegRecord, SlideId, SlideStatus};
use h3d_imgproc::phantom::slice_ellipsoid;
use h3d_register::{
    compose_chain, flag, from_landmarks, neighbor_toward_reference, record_from_result,
    register_stack, tre, QcThresholds, RegError, RegImage, RegInput, RegOptions, RegResult,
    Registrar,
};

fn rec(fixed: u32, p: Affine2) -> RegRecord {
    RegRecord {
        fixed: SlideId(fixed),
        method: "uji".into(),
        pairwise: p,
        ncc: 0.9,
        mask_dice: 0.95,
        area_ratio: 1.0,
        flagged: false,
        manual: false,
    }
}

#[test]
fn compose_chain_matches_ground_truth() {
    // Lima irisan 0..5, acuan 2. T_i kebenaran dasar acak; P_{i→tetangga} = T_nb⁻¹ ∘ T_i.
    let order: Vec<SlideId> = (0..5).map(SlideId).collect();
    let t_true: Vec<Affine2> = (0..5)
        .map(|i| {
            if i == 2 {
                Affine2::IDENTITY
            } else {
                Affine2::translation(10.0 * i as f64, -7.0 * i as f64)
                    .compose(&Affine2::rotation(0.1 * i as f64))
                    .compose(&Affine2::scale(
                        1.0 + 0.01 * i as f64,
                        1.0 - 0.01 * i as f64,
                    ))
            }
        })
        .collect();
    let mut records = Vec::new();
    for i in 0..5usize {
        let Some(nb) = neighbor_toward_reference(&order, SlideId(i as u32), SlideId(2)) else {
            continue;
        };
        let p = t_true[nb.0 as usize].inverse().unwrap().compose(&t_true[i]);
        records.push((SlideId(i as u32), rec(nb.0, p)));
    }
    assert_eq!(
        neighbor_toward_reference(&order, SlideId(0), SlideId(2)),
        Some(SlideId(1))
    );
    assert_eq!(
        neighbor_toward_reference(&order, SlideId(4), SlideId(2)),
        Some(SlideId(3))
    );
    assert_eq!(
        neighbor_toward_reference(&order, SlideId(2), SlideId(2)),
        None
    );
    assert_eq!(
        neighbor_toward_reference(&order, SlideId(9), SlideId(2)),
        None
    );

    let t = compose_chain(&records, SlideId(2), &order);
    assert_eq!(t.len(), 5);
    for i in 0..5 {
        assert!(
            t[&SlideId(i as u32)].approx_eq(&t_true[i], 1e-9),
            "T_{i}: {:?} vs {:?}",
            t[&SlideId(i as u32)],
            t_true[i]
        );
    }

    // Koreksi manual pada irisan 1 (fixed 2): hanya irisan 0 dan 1 (hulu dari 1) berubah.
    let fix = Affine2::translation(100.0, 0.0).compose(&records[1].1.pairwise);
    assert_eq!(records[1].0, SlideId(1));
    records[1].1.pairwise = fix;
    records[1].1.manual = true;
    let t2 = compose_chain(&records, SlideId(2), &order);
    for i in [2u32, 3, 4] {
        assert!(
            t2[&SlideId(i)].approx_eq(&t[&SlideId(i)], 1e-12),
            "T_{i} tidak boleh berubah"
        );
    }
    assert!(t2[&SlideId(1)].approx_eq(&fix, 1e-12));
    assert!(!t2[&SlideId(0)].approx_eq(&t[&SlideId(0)], 1e-6));
    assert!(t2[&SlideId(0)].approx_eq(&fix.compose(&records[0].1.pairwise), 1e-9));
}

#[test]
fn compose_chain_skips_unreachable() {
    let order = vec![SlideId(0), SlideId(1), SlideId(2)];
    // Irisan 0 menunjuk ke irisan 7 yang tidak ada → tidak ikut.
    let records = vec![
        (SlideId(0), rec(7, Affine2::IDENTITY)),
        (SlideId(2), rec(1, Affine2::translation(1.0, 2.0))),
    ];
    let t = compose_chain(&records, SlideId(1), &order);
    assert_eq!(t.len(), 2);
    assert!(t.contains_key(&SlideId(1)) && t.contains_key(&SlideId(2)));
}

#[test]
fn qc_flags_each_threshold() {
    let thr = QcThresholds::default();
    let ok = RegResult {
        transform: Affine2::IDENTITY,
        ncc: 0.7,
        mask_dice: 0.9,
        area_ratio: 1.0,
        iterations: 3,
    };
    assert!(!flag(&ok, &thr));
    assert!(flag(&RegResult { ncc: 0.49, ..ok }, &thr));
    assert!(!flag(&RegResult { ncc: 0.5, ..ok }, &thr));
    assert!(flag(
        &RegResult {
            mask_dice: 0.79,
            ..ok
        },
        &thr
    ));
    assert!(flag(
        &RegResult {
            area_ratio: 0.84,
            ..ok
        },
        &thr
    ));
    assert!(flag(
        &RegResult {
            area_ratio: 1.16,
            ..ok
        },
        &thr
    ));
    assert!(!flag(
        &RegResult {
            area_ratio: 1.15,
            ..ok
        },
        &thr
    ));
    assert!(flag(
        &RegResult {
            ncc: f32::NAN,
            ..ok
        },
        &thr
    ));
    let r = record_from_result(SlideId(4), "m", &RegResult { ncc: 0.2, ..ok }, &thr, true);
    assert_eq!(r.fixed, SlideId(4));
    assert_eq!(r.method, "m");
    assert!(r.flagged && r.manual);
    assert_eq!(r.pairwise, Affine2::IDENTITY);
}

#[test]
fn landmarks_recover_affine_and_similarity() {
    let a = Affine2::new([1.1, 0.2, 30.0, -0.1, 0.9, -12.0]);
    let moving = [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [70.0, 40.0]];
    let fixed: Vec<[f64; 2]> = moving.iter().map(|p| a.apply(*p)).collect();
    let est = from_landmarks(&moving[..3], &fixed[..3]).unwrap();
    assert!(est.approx_eq(&a, 1e-9), "{est:?}");
    assert!(tre(&est, &moving, &fixed) < 1e-9);

    let s = Affine2::translation(5.0, -3.0)
        .compose(&Affine2::rotation(0.4))
        .compose(&Affine2::scale(1.3, 1.3));
    let m2 = [[10.0, 10.0], [60.0, -20.0]];
    let f2: Vec<[f64; 2]> = m2.iter().map(|p| s.apply(*p)).collect();
    let est2 = from_landmarks(&m2, &f2).unwrap();
    assert!(est2.approx_eq(&s, 1e-9), "{est2:?}");

    assert!(matches!(
        from_landmarks(&m2[..1], &f2[..1]),
        Err(RegError::InvalidInput(_))
    ));
    assert!(matches!(
        from_landmarks(&m2, &f2[..1]),
        Err(RegError::InvalidInput(_))
    ));
    let col = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    assert!(matches!(from_landmarks(&col, &col), Err(RegError::Core(_))));
}

/// Registrar palsu: mengembalikan `init` bila ada, selain itu geseran tetap.
struct FakeRegistrar;

impl Registrar for FakeRegistrar {
    fn name(&self) -> &'static str {
        "palsu"
    }
    fn register(
        &self,
        _fixed: &RegImage,
        _moving: &RegImage,
        init: Option<Affine2>,
        _opts: &RegOptions,
        progress: &dyn Progress,
    ) -> Result<RegResult, RegError> {
        if progress.is_cancelled() {
            return Err(RegError::Cancelled);
        }
        Ok(RegResult {
            transform: init.unwrap_or(Affine2::translation(1.0, 0.0)),
            ncc: 0.9,
            mask_dice: 0.9,
            area_ratio: 1.0,
            iterations: 1,
        })
    }
}

fn tiny_image() -> Arc<RegImage> {
    let mask = slice_ellipsoid([800.0, 600.0, 500.0], 0.0, 32.0, [64, 64]);
    let field = Field3::new(1, 500.0, 100.0, 8);
    Arc::new(textured_image(&mask, 32.0, 0.0, &field, 1, [0.0, 0.0], 0.5))
}

fn inputs(statuses: &[SlideStatus]) -> Vec<RegInput> {
    let img = tiny_image();
    statuses
        .iter()
        .enumerate()
        .map(|(i, &st)| RegInput {
            id: SlideId(10 + i as u32),
            order: i as u32,
            status: st,
            image: img.clone(),
            init: None,
        })
        .collect()
}

#[test]
fn register_stack_pairs_toward_reference_and_skips_bad_slides() {
    use SlideStatus::*;
    let slides = inputs(&[Ok, Damaged, Ok, Ok, Excluded, Ok]);
    let recs = register_stack(
        &slides,
        SlideId(13),
        &FakeRegistrar,
        &RegOptions::default(),
        &QcThresholds::default(),
        &NoProgress,
    )
    .unwrap();
    let ids: Vec<u32> = recs.iter().map(|(i, _)| i.0).collect();
    assert_eq!(ids, vec![10, 12, 15]);
    assert_eq!(recs[0].1.fixed, SlideId(12)); // 10 → 12 (11 rusak dilompati)
    assert_eq!(recs[1].1.fixed, SlideId(13));
    assert_eq!(recs[2].1.fixed, SlideId(13)); // 15 → 13 (14 dikecualikan)
    assert!(recs.iter().all(|(_, r)| r.method == "palsu" && !r.flagged));

    // Acuan bukan Ok → galat.
    assert!(matches!(
        register_stack(
            &slides,
            SlideId(11),
            &FakeRegistrar,
            &RegOptions::default(),
            &QcThresholds::default(),
            &NoProgress
        ),
        Err(RegError::NoUsableSlides(_))
    ));
}

#[test]
fn register_stack_passes_init_to_registrar() {
    let mut slides = inputs(&[SlideStatus::Ok, SlideStatus::Ok]);
    let of = Affine2::rotation(0.3);
    let om = Affine2::rotation(-0.2).compose(&Affine2::translation(5.0, 6.0));
    slides[0].init = Some(of.inverse().unwrap().compose(&om));
    let recs = register_stack(
        &slides,
        SlideId(11),
        &FakeRegistrar,
        &RegOptions::default(),
        &QcThresholds::default(),
        &NoProgress,
    )
    .unwrap();
    let expected = of.inverse().unwrap().compose(&om);
    assert!(recs[0].1.pairwise.approx_eq(&expected, 1e-12));
}

struct AlwaysCancelled;
impl Progress for AlwaysCancelled {
    fn set(&self, _: f32, _: &str) {}
    fn is_cancelled(&self) -> bool {
        true
    }
}

#[test]
fn register_stack_propagates_cancellation() {
    let slides = inputs(&[SlideStatus::Ok, SlideStatus::Ok, SlideStatus::Ok]);
    let r = register_stack(
        &slides,
        SlideId(11),
        &FakeRegistrar,
        &RegOptions::default(),
        &QcThresholds::default(),
        &AlwaysCancelled,
    );
    assert_eq!(r.err(), Some(RegError::Cancelled));
}

#[test]
fn register_stack_with_builtin_registrar_recovers_known_shift() {
    // Tiga irisan: tengah acuan; tetangga digeser/putar sedikit.
    let base = tiny_image();
    let mpp = 32.0;
    let n = 64;
    let center = [n as f64 * mpp / 2.0, n as f64 * mpp / 2.0];
    let g0 = Affine2::translation(96.0, -64.0).compose(&Affine2::rotation_about(0.2, center));
    let g2 = Affine2::translation(-64.0, 32.0).compose(&Affine2::rotation_about(-0.15, center));
    let img0 = Arc::new(warp_image(&base, &g0, [n, n], mpp, [0.0, 0.0]));
    let img2 = Arc::new(warp_image(&base, &g2, [n, n], mpp, [0.0, 0.0]));
    let mk = |id: u32, order: u32, image: Arc<RegImage>| RegInput {
        id: SlideId(id),
        order,
        status: SlideStatus::Ok,
        image,
        init: None,
    };
    let slides = vec![mk(0, 0, img0), mk(1, 1, base.clone()), mk(2, 2, img2)];
    let opts = RegOptions {
        model: h3d_register::RegModel::Rigid,
        levels_um: vec![64.0, 32.0],
        ..Default::default()
    };
    let recs = register_stack(
        &slides,
        SlideId(1),
        &h3d_register::BuiltinRegistrar,
        &opts,
        &QcThresholds::default(),
        &NoProgress,
    )
    .unwrap();
    let order = vec![SlideId(0), SlideId(1), SlideId(2)];
    let t = compose_chain(&recs, SlideId(1), &order);
    // T_0 = P_{0→1} ≈ g0⁻¹ (U_0 → U_1 = U_base).
    for (id, g) in [(0u32, g0), (2, g2)] {
        let img = &slides[id as usize].image;
        let e = common::tre_um(img, &t[&SlideId(id)], &g.inverse().unwrap(), 2) / mpp;
        eprintln!("irisan {id}: TRE {e:.3} px");
        assert!(e <= 2.0, "irisan {id}: TRE {e:.3} px");
    }
}
