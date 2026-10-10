//! Uji usulan urutan dan orientasi (histo3d.md 8.4.7, H6).

mod common;

use std::sync::Arc;

use common::{random_transform, textured_image, warp_image, Field3};
use h3d_core::{Affine2, NoProgress, SlideId, SlideStatus};
use h3d_imgproc::phantom::{slice_branch, Rng};
use h3d_register::{suggest_order, RegInput};

const N_SLICES: usize = 60;
const MPP: f64 = 32.0;
const SIZE: usize = 192;

/// Irisan ideal ke-`k` (tanpa putar/cermin) dari fantom cabang.
fn ideal_slices(seed: u64) -> Vec<h3d_register::RegImage> {
    let field = Field3::waves(seed, 32, (250.0, 900.0), (1500.0, 5000.0));
    (0..N_SLICES)
        .map(|k| {
            let z = -2000.0 + 4200.0 * k as f64 / (N_SLICES - 1) as f64;
            let mask = slice_branch(2600.0, 2600.0, 800.0, 0.5, z, MPP, [SIZE, SIZE]);
            textured_image(
                &mask,
                MPP,
                z,
                &field,
                seed * 1000 + k as u64,
                [0.0, 0.0],
                0.7,
            )
        })
        .collect()
}

struct Shuffled {
    inputs: Vec<RegInput>,
    /// Indeks irisan asli per `SlideId`.
    true_index: Vec<usize>,
    /// Benar bila irisan dicerminkan.
    mirrored: Vec<bool>,
    current: Vec<SlideId>,
}

/// Acak urutan, putar/cermin tiap irisan dengan benih, lalu bungkus `RegInput`.
fn shuffled(seed: u64, transform: bool) -> Shuffled {
    let ideal = ideal_slices(seed);
    let mut rng = Rng::new(seed + 17);
    let mut perm: Vec<usize> = (0..N_SLICES).collect();
    for i in (1..N_SLICES).rev() {
        let j = rng.below(i + 1);
        perm.swap(i, j);
    }
    let width_um = SIZE as f64 * MPP;
    let center = [width_um / 2.0, width_um / 2.0];
    let mut inputs = Vec::new();
    let mut true_index = vec![0; N_SLICES];
    let mut mirrored = vec![false; N_SLICES];
    for (pos, &k) in perm.iter().enumerate() {
        let (img, m) = if transform {
            let (g, m) = random_transform(&mut rng, center, width_um, 0.04, true, (1.0, 1.0));
            (warp_image(&ideal[k], &g, [SIZE, SIZE], MPP, [0.0, 0.0]), m)
        } else {
            (ideal[k].clone(), false)
        };
        true_index[pos] = k;
        mirrored[pos] = m;
        inputs.push(RegInput {
            id: SlideId(pos as u32),
            order: pos as u32,
            status: SlideStatus::Ok,
            image: Arc::new(img.downsample_to(64.0)),
            init: None,
        });
    }
    let current = (0..N_SLICES as u32).map(SlideId).collect();
    Shuffled {
        inputs,
        true_index,
        mirrored,
        current,
    }
}

#[test]
fn suggest_order_recovers_shuffled_branch_phantom() {
    let s = shuffled(3, true);
    let sug = suggest_order(&s.inputs, &s.current, &NoProgress).unwrap();
    assert_eq!(sug.order.len(), N_SLICES);
    let seq: Vec<usize> = sug
        .order
        .iter()
        .map(|id| s.true_index[id.0 as usize])
        .collect();
    let adjacent_ok = seq
        .windows(2)
        .filter(|w| (w[0] as i64 - w[1] as i64).abs() == 1)
        .count();
    let frac = adjacent_ok as f64 / (N_SLICES - 1) as f64;
    eprintln!("urutan usulan (indeks asli): {seq:?}");
    eprintln!(
        "pasangan bertetangga benar: {adjacent_ok}/{} = {frac:.3}",
        N_SLICES - 1
    );

    // Cermin: tanda det(orient_i) harus = cermin_i XOR cermin_acuan.
    let ref_id = sug.order[N_SLICES / 2];
    let ref_m = s.mirrored[ref_id.0 as usize];
    let mirror_ok = sug
        .orient
        .iter()
        .filter(|(id, o)| (o.determinant() < 0.0) == (s.mirrored[id.0 as usize] != ref_m))
        .count();
    let mfrac = mirror_ok as f64 / N_SLICES as f64;
    eprintln!("cermin benar: {mirror_ok}/{N_SLICES} = {mfrac:.3}");
    eprintln!("tidak yakin: {} irisan", sug.uncertain.len());
    assert!(frac >= 0.95, "pasangan bertetangga benar {frac:.3} < 0,95");
    assert!(mfrac >= 0.95, "cermin benar {mfrac:.3} < 0,95");
    assert_eq!(sug.confidence.len(), N_SLICES);
    assert!(sug.confidence.iter().all(|(_, c)| (0.0..=1.0).contains(c)));
    assert!(sug.uncertain.iter().all(|id| sug
        .confidence
        .iter()
        .any(|(i, c)| i == id && *c < h3d_register::UNCERTAIN_THRESHOLD)));
}

#[test]
fn already_ordered_input_is_unchanged_and_deterministic() {
    // Masukan berurutan tanpa putar/cermin (perm diabaikan: susun ulang menurut indeks asli).
    let mut s = shuffled(5, false);
    s.inputs.sort_by_key(|i| s.true_index[i.id.0 as usize]);
    for (pos, inp) in s.inputs.iter_mut().enumerate() {
        inp.id = SlideId(pos as u32);
        inp.order = pos as u32;
    }
    let current: Vec<SlideId> = (0..N_SLICES as u32).map(SlideId).collect();
    let a = suggest_order(&s.inputs, &current, &NoProgress).unwrap();
    assert_eq!(a.order, current, "urutan berubah: {:?}", a.order);
    for (_, o) in &a.orient {
        let parts = o.decompose();
        assert!(!parts.mirrored);
        assert!(
            parts.rotation.abs() < 6f64.to_radians(),
            "rotasi {:.2}°",
            parts.rotation.to_degrees()
        );
    }
    let b = suggest_order(&s.inputs, &current, &NoProgress).unwrap();
    assert_eq!(a, b);
}

#[test]
fn non_ok_slides_are_appended_and_single_slide_is_trivial() {
    let s = shuffled(9, false);
    let mut inputs: Vec<RegInput> = s.inputs.into_iter().take(6).collect();
    inputs[2].status = SlideStatus::Damaged;
    let current: Vec<SlideId> = inputs.iter().map(|i| i.id).collect();
    let sug = suggest_order(&inputs, &current, &NoProgress).unwrap();
    assert_eq!(sug.order.len(), 6);
    assert_eq!(*sug.order.last().unwrap(), SlideId(2));
    assert!(sug.orient.iter().all(|(id, _)| *id != SlideId(2)));

    let one = suggest_order(&inputs[..1], &current[..1], &NoProgress).unwrap();
    assert_eq!(one.order, vec![inputs[0].id]);
    assert_eq!(one.orient, vec![(inputs[0].id, Affine2::IDENTITY)]);
}
