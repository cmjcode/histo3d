//! Logika murni (tanpa egui/GPU) yang dipakai mode-mode: konversi layar → W
//! → U, tekanan → jari-jari kuas, mpp dari skala batang, pengumpul landmark,
//! selisih usulan urutan, tetangga ke arah acuan, dan pendeteksi geser dua
//! jari. Semua panjang dalam µm (`f64`) kecuali disebut lain.
//!
//! Arah transformasi mengikuti histo3d.md 5.3: `T_i` memetakan **U (slide)
//! → W (tumpukan)**; dari layar selalu lewat W dulu lalu `T_i⁻¹` ke U.

use h3d_core::{Affine2, SlideId, SlideTransform};
use h3d_engine::OrderSuggestion;
use h3d_render::Camera2D;

/// Layar (point egui) → W → U irisan. `t` = `T_i` (U → W). `None` bila
/// transformasinya tidak punya balikan.
pub fn screen_to_slide_um(
    camera: &Camera2D,
    viewport: egui::Rect,
    pos: egui::Pos2,
    t: &SlideTransform,
) -> Option<[f64; 2]> {
    let w = camera.screen_to_world_um(pos, viewport);
    t.apply_inverse(w)
}

/// U irisan → W → layar (point egui). `t` = `T_i` (U → W).
pub fn slide_um_to_screen(
    camera: &Camera2D,
    viewport: egui::Rect,
    u: [f64; 2],
    t: &SlideTransform,
) -> egui::Pos2 {
    camera.world_um_to_screen(t.apply(u), viewport)
}

/// Faktor jari-jari kuas dari tekanan Pencil (8.5.2: `r × f(tekanan)`).
/// Tanpa tekanan (tetikus/jari) → 1. Tekanan 0..1 dipetakan linear ke
/// 0,3..1 supaya sentuhan ringan tetap meninggalkan jejak.
pub fn pressure_radius_factor(force: Option<f32>) -> f64 {
    match force {
        Some(f) if f.is_finite() => 0.3 + 0.7 * f64::from(f.clamp(0.0, 1.0)),
        _ => 1.0,
    }
}

/// Jari-jari kuas (µm) untuk jari-jari dasar dan tekanan tertentu.
pub fn brush_radius_um(base_radius_um: f64, force: Option<f32>) -> f64 {
    base_radius_um.max(0.0) * pressure_radius_factor(force)
}

/// mpp dari skala batang: pengguna menarik garis `p0`–`p1` (piksel **level
/// 0**) di atas skala batang yang panjang aslinya `length_um`. Hasilnya
/// µm per piksel level 0. `None` bila garis terlalu pendek atau panjangnya
/// tidak sah (mpp tidak pernah ditebak, aturan keras 7).
pub fn scale_bar_mpp(p0_px: [f64; 2], p1_px: [f64; 2], length_um: f64) -> Option<f64> {
    let d = ((p1_px[0] - p0_px[0]).powi(2) + (p1_px[1] - p0_px[1]).powi(2)).sqrt();
    if !(d.is_finite() && d >= 2.0 && length_um.is_finite() && length_um > 0.0) {
        return None;
    }
    Some(length_um / d)
}

/// Seperti [`scale_bar_mpp`] untuk garis yang ditarik di atas thumbnail:
/// `thumb_scale` = piksel level 0 per piksel thumbnail.
pub fn scale_bar_mpp_from_thumb(
    p0_thumb: [f64; 2],
    p1_thumb: [f64; 2],
    thumb_scale: f64,
    length_um: f64,
) -> Option<f64> {
    if !(thumb_scale.is_finite() && thumb_scale > 0.0) {
        return None;
    }
    scale_bar_mpp(
        [p0_thumb[0] * thumb_scale, p0_thumb[1] * thumb_scale],
        [p1_thumb[0] * thumb_scale, p1_thumb[1] * thumb_scale],
        length_um,
    )
}

/// Sisi mana yang menerima klik landmark berikutnya.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LandmarkSide {
    /// Irisan tetap (tetangga ke arah acuan).
    Fixed,
    /// Irisan bergerak (irisan yang dikoreksi).
    Moving,
}

/// Pengumpul pasangan landmark (8.4.5): klik bergantian, pertama di irisan
/// tetap lalu di irisan bergerak. Titik disimpan di ruang U masing-masing.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LandmarkCollector {
    /// Titik di U irisan tetap yang menunggu pasangannya.
    pub pending_fixed: Option<[f64; 2]>,
    /// Pasangan `(titik di U irisan ini, titik di U tetangga)` — bentuk yang
    /// diminta `Session::apply_landmarks`.
    pub pairs: Vec<([f64; 2], [f64; 2])>,
}

impl LandmarkCollector {
    /// Sisi yang ditunggu klik berikutnya.
    pub fn next_side(&self) -> LandmarkSide {
        if self.pending_fixed.is_some() {
            LandmarkSide::Moving
        } else {
            LandmarkSide::Fixed
        }
    }

    /// Catat klik di titik `w` (ruang W). `t_fixed`/`t_moving` = `T`
    /// (U → W) kedua irisan; titik dibawa ke U lewat balikannya. Klik pada
    /// transformasi tanpa balikan diabaikan. `true` bila satu pasangan
    /// baru saja lengkap.
    pub fn push_click(
        &mut self,
        w: [f64; 2],
        t_fixed: &SlideTransform,
        t_moving: &SlideTransform,
    ) -> bool {
        match self.pending_fixed {
            None => {
                if let Some(u) = t_fixed.apply_inverse(w) {
                    self.pending_fixed = Some(u);
                }
                false
            }
            Some(uf) => match t_moving.apply_inverse(w) {
                Some(um) => {
                    self.pairs.push((um, uf));
                    self.pending_fixed = None;
                    true
                }
                None => false,
            },
        }
    }

    /// Buang klik terakhir (titik menunggu, atau pasangan terakhir).
    pub fn pop(&mut self) {
        if self.pending_fixed.take().is_none() {
            self.pairs.pop();
        }
    }

    /// Kosongkan.
    pub fn clear(&mut self) {
        self.pending_fixed = None;
        self.pairs.clear();
    }

    /// Cukup pasangan untuk diterapkan (2 → similarity, ≥ 3 → affine).
    pub fn can_apply(&self) -> bool {
        self.pairs.len() >= 2
    }

    /// Benar bila hasilnya affine penuh (≥ 3 pasang).
    pub fn is_affine(&self) -> bool {
        self.pairs.len() >= 3
    }
}

/// Satu baris pratinjau usulan urutan: irisan yang posisinya berubah.
#[derive(Clone, Debug, PartialEq)]
pub struct OrderChange {
    /// Irisan.
    pub slide: SlideId,
    /// Posisi sekarang (0 = pertama).
    pub from: usize,
    /// Posisi usulan.
    pub to: usize,
    /// Putaran usulan, derajat.
    pub rotation_deg: f64,
    /// Usulan menyebut irisan tercermin.
    pub mirrored: bool,
    /// Keyakinan 0..1.
    pub confidence: Option<f32>,
    /// Ditandai ragu oleh pengusul.
    pub uncertain: bool,
}

/// Hanya irisan yang posisinya berubah (8.4.7), terurut menurut posisi
/// usulan. Irisan yang tidak ada di salah satu daftar dilewati.
pub fn order_changes(current: &[SlideId], suggestion: &OrderSuggestion) -> Vec<OrderChange> {
    let mut out = Vec::new();
    for (to, id) in suggestion.order.iter().enumerate() {
        let Some(from) = current.iter().position(|c| c == id) else {
            continue;
        };
        if from == to {
            continue;
        }
        let parts = suggestion
            .orient
            .iter()
            .find(|(s, _)| s == id)
            .map(|(_, a)| a.decompose());
        out.push(OrderChange {
            slide: *id,
            from,
            to,
            rotation_deg: parts.map(|p| p.rotation.to_degrees()).unwrap_or(0.0),
            mirrored: parts.is_some_and(|p| p.mirrored),
            confidence: suggestion
                .confidence
                .iter()
                .find(|(s, _)| s == id)
                .map(|(_, c)| *c),
            uncertain: suggestion.uncertain.contains(id),
        });
    }
    out
}

/// "Balik urutan" (8.4.7 U3): gambar tidak bisa membedakan naik dari turun.
/// Hanya `order` yang dibalik; orientasi per irisan tetap.
pub fn flip_suggestion(s: &mut OrderSuggestion) {
    s.order.reverse();
}

/// Tetangga irisan `slide` ke arah acuan di dalam daftar irisan `Ok`
/// terurut `usable` (5.3). `None` untuk acuan sendiri atau bila salah satu
/// tidak ada di daftar.
pub fn neighbor_toward_reference(
    usable: &[SlideId],
    reference: SlideId,
    slide: SlideId,
) -> Option<SlideId> {
    let i = usable.iter().position(|s| *s == slide)?;
    let r = usable.iter().position(|s| *s == reference)?;
    match i.cmp(&r) {
        std::cmp::Ordering::Less => usable.get(i + 1).copied(),
        std::cmp::Ordering::Greater => usable.get(i - 1).copied(),
        std::cmp::Ordering::Equal => None,
    }
}

/// Arah navigasi irisan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NavDir {
    /// Irisan sebelumnya.
    Prev,
    /// Irisan berikutnya.
    Next,
}

/// Irisan tujuan navigasi dalam daftar terurut; berhenti di ujung.
pub fn navigate(order: &[SlideId], current: Option<SlideId>, dir: NavDir) -> Option<SlideId> {
    if order.is_empty() {
        return None;
    }
    let i = current
        .and_then(|c| order.iter().position(|s| *s == c))
        .unwrap_or(0);
    let j = match dir {
        NavDir::Prev => i.saturating_sub(1),
        NavDir::Next => (i + 1).min(order.len() - 1),
    };
    order.get(j).copied()
}

/// Pendeteksi geser dua jari untuk navigasi irisan (8.11). Selama gestur
/// dua jari, geseran dan zoom diakumulasi; saat jari diangkat, gestur yang
/// dominan horizontal, cukup panjang, dan nyaris tanpa cubit dianggap
/// "geser" (ke kiri = irisan berikutnya).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SwipeDetector {
    active: bool,
    dx: f32,
    dy: f32,
    log_zoom: f32,
}

/// Panjang geser minimum (point).
pub const SWIPE_MIN_PT: f32 = 90.0;
/// Batas |ln zoom| agar gestur masih dianggap geser, bukan cubit.
pub const SWIPE_MAX_LOG_ZOOM: f32 = 0.12;

impl SwipeDetector {
    /// Umpankan keadaan frame ini: `Some((geseran, zoom))` selama ≥ 2 jari
    /// menyentuh, `None` bila tidak. Mengembalikan arah saat gestur selesai
    /// dan memenuhi syarat.
    pub fn update(&mut self, two_finger: Option<(egui::Vec2, f32)>) -> Option<NavDir> {
        match two_finger {
            Some((delta, zoom)) => {
                self.active = true;
                self.dx += delta.x;
                self.dy += delta.y;
                if zoom.is_finite() && zoom > 0.0 {
                    self.log_zoom += zoom.ln();
                }
                None
            }
            None => {
                if !self.active {
                    return None;
                }
                let done = *self;
                *self = SwipeDetector::default();
                let horizontal =
                    done.dx.abs() >= SWIPE_MIN_PT && done.dx.abs() > 2.0 * done.dy.abs();
                if horizontal && done.log_zoom.abs() < SWIPE_MAX_LOG_ZOOM {
                    Some(if done.dx < 0.0 {
                        NavDir::Next
                    } else {
                        NavDir::Prev
                    })
                } else {
                    None
                }
            }
        }
    }

    /// Gestur dua jari sedang berlangsung.
    pub fn is_active(&self) -> bool {
        self.active
    }
}

/// `T` baru hasil pegangan geser/putar di layar (8.4.5): geser `shift_um`
/// (W) dan putar `angle_rad` di sekitar `pivot_w` (W), diterapkan **setelah**
/// `t` lama. Hasilnya tetap memetakan U → W.
pub fn manual_adjusted_transform(
    t: &Affine2,
    shift_um: [f64; 2],
    angle_rad: f64,
    pivot_w: [f64; 2],
) -> Affine2 {
    let rot = Affine2::rotation_about(angle_rad, pivot_w);
    let shift = Affine2::translation(shift_um[0], shift_um[1]);
    shift.compose(&rot).compose(t)
}

/// Format nilai ± ketidakpastian dalam mm (tiga desimal) lewat kunci
/// terjemahan `common-value-with-uncertainty`.
pub fn fmt_um_pm(value_um: f64, plus_minus_um: f64) -> String {
    h3d_i18n::t!(
        "common-value-with-uncertainty",
        value = format!("{:.3}", value_um / 1000.0),
        uncertainty = format!("{:.3}", plus_minus_um / 1000.0),
        unit = h3d_i18n::t!("unit-mm")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect() -> egui::Rect {
        egui::Rect::from_min_size(egui::pos2(40.0, 60.0), egui::vec2(800.0, 600.0))
    }

    /// `T_i` berputar 37° + cermin + geser.
    fn rotated_mirrored() -> SlideTransform {
        let a = Affine2::translation(1200.0, -340.0)
            .compose(&Affine2::rotation(37f64.to_radians()))
            .compose(&Affine2::mirror_x_about(900.0));
        assert!(a.determinant() < 0.0, "harus tercermin");
        SlideTransform::Affine(a)
    }

    #[test]
    fn screen_w_u_round_trip_through_rotated_mirrored_transform() {
        let cam = Camera2D {
            center_um: [2500.0, 1800.0],
            um_per_px: 7.5,
        };
        let t = rotated_mirrored();
        for pos in [
            egui::pos2(40.0, 60.0),
            egui::pos2(440.0, 360.0),
            egui::pos2(812.5, 77.25),
        ] {
            let u = screen_to_slide_um(&cam, rect(), pos, &t).unwrap();
            let back = slide_um_to_screen(&cam, rect(), u, &t);
            assert!((back.x - pos.x).abs() < 1e-2, "{back:?} vs {pos:?}");
            assert!((back.y - pos.y).abs() < 1e-2, "{back:?} vs {pos:?}");
            // Lewat W secara eksplisit harus sama.
            let w = cam.screen_to_world_um(pos, rect());
            let u2 = t.apply_inverse(w).unwrap();
            assert!((u[0] - u2[0]).abs() < 1e-9 && (u[1] - u2[1]).abs() < 1e-9);
            let w2 = t.apply(u);
            assert!((w[0] - w2[0]).abs() < 1e-6 && (w[1] - w2[1]).abs() < 1e-6);
        }
    }

    #[test]
    fn screen_to_slide_differs_from_world_when_transform_is_not_identity() {
        let cam = Camera2D {
            center_um: [0.0, 0.0],
            um_per_px: 2.0,
        };
        let pos = egui::pos2(500.0, 400.0);
        let w = cam.screen_to_world_um(pos, rect());
        let u = screen_to_slide_um(&cam, rect(), pos, &rotated_mirrored()).unwrap();
        assert!((w[0] - u[0]).abs() > 1.0 || (w[1] - u[1]).abs() > 1.0);
        let id = screen_to_slide_um(&cam, rect(), pos, &SlideTransform::Identity).unwrap();
        assert_eq!(id, w);
    }

    #[test]
    fn pressure_maps_to_radius_monotonically() {
        assert_eq!(pressure_radius_factor(None), 1.0);
        assert!((pressure_radius_factor(Some(0.0)) - 0.3).abs() < 1e-9);
        assert!((pressure_radius_factor(Some(1.0)) - 1.0).abs() < 1e-9);
        assert!((pressure_radius_factor(Some(7.0)) - 1.0).abs() < 1e-9);
        assert_eq!(pressure_radius_factor(Some(f32::NAN)), 1.0);
        let mut prev = 0.0;
        for i in 0..=10 {
            let f = pressure_radius_factor(Some(i as f32 / 10.0));
            assert!(f > prev);
            prev = f;
        }
        assert!((brush_radius_um(40.0, Some(0.5)) - 26.0).abs() < 1e-6);
        assert_eq!(brush_radius_um(-5.0, None), 0.0);
    }

    #[test]
    fn scale_bar_gives_mpp() {
        // Garis 400 px level 0 di atas skala batang 200 µm → 0,5 µm/px.
        let mpp = scale_bar_mpp([100.0, 50.0], [500.0, 50.0], 200.0).unwrap();
        assert!((mpp - 0.5).abs() < 1e-12);
        // Garis miring 3-4-5.
        let mpp = scale_bar_mpp([0.0, 0.0], [300.0, 400.0], 1000.0).unwrap();
        assert!((mpp - 2.0).abs() < 1e-12);
        // Di atas thumbnail berskala 8 px level 0 per piksel thumbnail.
        let mpp = scale_bar_mpp_from_thumb([10.0, 10.0], [60.0, 10.0], 8.0, 100.0).unwrap();
        assert!((mpp - 0.25).abs() < 1e-12);
        // Tidak pernah menebak.
        assert!(scale_bar_mpp([0.0, 0.0], [0.5, 0.0], 100.0).is_none());
        assert!(scale_bar_mpp([0.0, 0.0], [100.0, 0.0], 0.0).is_none());
        assert!(scale_bar_mpp([0.0, 0.0], [100.0, 0.0], f64::NAN).is_none());
        assert!(scale_bar_mpp_from_thumb([0.0, 0.0], [100.0, 0.0], 0.0, 10.0).is_none());
    }

    #[test]
    fn landmark_pairs_alternate_fixed_then_moving() {
        let t_fixed = SlideTransform::Affine(Affine2::translation(100.0, 0.0));
        let t_moving = rotated_mirrored();
        let mut c = LandmarkCollector::default();
        assert_eq!(c.next_side(), LandmarkSide::Fixed);
        assert!(!c.can_apply());

        let clicks = [
            [500.0, 600.0],
            [520.0, 590.0],
            [900.0, 100.0],
            [880.0, 130.0],
        ];
        assert!(!c.push_click(clicks[0], &t_fixed, &t_moving));
        assert_eq!(c.next_side(), LandmarkSide::Moving);
        assert!(c.push_click(clicks[1], &t_fixed, &t_moving));
        assert!(!c.push_click(clicks[2], &t_fixed, &t_moving));
        assert!(c.push_click(clicks[3], &t_fixed, &t_moving));
        assert_eq!(c.pairs.len(), 2);
        assert!(c.can_apply() && !c.is_affine());

        // Pasangan = (U irisan bergerak, U irisan tetap).
        let (um, uf) = c.pairs[0];
        assert_eq!(uf, [400.0, 600.0]);
        let w = t_moving.apply(um);
        assert!((w[0] - 520.0).abs() < 1e-6 && (w[1] - 590.0).abs() < 1e-6);

        // Titik menunggu dibuang dulu, baru pasangan.
        c.push_click([1.0, 2.0], &t_fixed, &t_moving);
        c.pop();
        assert_eq!(c.pairs.len(), 2);
        assert_eq!(c.next_side(), LandmarkSide::Fixed);
        c.pop();
        assert_eq!(c.pairs.len(), 1);
        c.push_click([0.0, 0.0], &t_fixed, &t_moving);
        c.push_click([0.0, 0.0], &t_fixed, &t_moving);
        c.push_click([5.0, 0.0], &t_fixed, &t_moving);
        c.push_click([5.0, 0.0], &t_fixed, &t_moving);
        assert!(c.is_affine());
        c.clear();
        assert!(c.pairs.is_empty() && c.pending_fixed.is_none());
    }

    #[test]
    fn landmarks_recover_known_offset() {
        // Irisan bergerak tergeser (30, -20) µm terhadap tetangga; tiga
        // pasang klik pada titik jaringan yang sama harus menghasilkan P
        // yang memetakan U bergerak → U tetap dengan geseran itu.
        let t_fixed = SlideTransform::Identity;
        let truth = Affine2::translation(30.0, -20.0);
        // Tampilan memakai T bergerak yang masih salah (identitas).
        let t_shown = SlideTransform::Identity;
        let mut c = LandmarkCollector::default();
        for uf in [[100.0, 100.0], [900.0, 150.0], [400.0, 800.0]] {
            let um = truth.inverse().unwrap().apply(uf);
            c.push_click(t_fixed.apply(uf), &t_fixed, &t_shown);
            c.push_click(t_shown.apply(um), &t_fixed, &t_shown);
        }
        let p = Affine2::fit_affine(&c.pairs).unwrap();
        assert!(p.approx_eq(&truth, 1e-6), "{p:?}");
    }

    fn ids(v: &[u32]) -> Vec<SlideId> {
        v.iter().map(|i| SlideId(*i)).collect()
    }

    #[test]
    fn order_preview_lists_only_moved_slides() {
        let current = ids(&[0, 1, 2, 3, 4]);
        let mut s = OrderSuggestion {
            order: ids(&[0, 2, 1, 3, 4]),
            orient: vec![
                (SlideId(1), Affine2::rotation(90f64.to_radians())),
                (SlideId(2), Affine2::mirror_x()),
            ],
            confidence: vec![(SlideId(1), 0.05), (SlideId(2), 0.8)],
            uncertain: ids(&[1]),
        };
        let ch = order_changes(&current, &s);
        assert_eq!(ch.len(), 2, "hanya irisan 1 dan 2 yang pindah");
        assert_eq!((ch[0].slide, ch[0].from, ch[0].to), (SlideId(2), 2, 1));
        assert_eq!((ch[1].slide, ch[1].from, ch[1].to), (SlideId(1), 1, 2));
        assert!(ch[0].mirrored && !ch[0].uncertain);
        assert!(ch[1].uncertain && (ch[1].rotation_deg - 90.0).abs() < 1e-6);
        assert_eq!(ch[1].confidence, Some(0.05));

        // Usulan sama dengan urutan sekarang → tidak ada perubahan.
        s.order = current.clone();
        assert!(order_changes(&current, &s).is_empty());

        // "Balik urutan": semua kecuali irisan tengah pindah.
        flip_suggestion(&mut s);
        let ch = order_changes(&current, &s);
        assert_eq!(ch.len(), 4);
        assert!(ch.iter().all(|c| c.slide != SlideId(2)));
        assert_eq!(s.orient.len(), 2, "orientasi tidak disentuh");
    }

    #[test]
    fn neighbor_points_toward_reference() {
        let usable = ids(&[10, 11, 12, 13, 14]);
        let r = SlideId(12);
        assert_eq!(
            neighbor_toward_reference(&usable, r, SlideId(10)),
            Some(SlideId(11))
        );
        assert_eq!(
            neighbor_toward_reference(&usable, r, SlideId(11)),
            Some(SlideId(12))
        );
        assert_eq!(
            neighbor_toward_reference(&usable, r, SlideId(14)),
            Some(SlideId(13))
        );
        assert_eq!(neighbor_toward_reference(&usable, r, r), None);
        assert_eq!(neighbor_toward_reference(&usable, r, SlideId(99)), None);
    }

    #[test]
    fn navigation_clamps_at_ends() {
        let order = ids(&[3, 1, 2]);
        assert_eq!(
            navigate(&order, Some(SlideId(3)), NavDir::Prev),
            Some(SlideId(3))
        );
        assert_eq!(
            navigate(&order, Some(SlideId(3)), NavDir::Next),
            Some(SlideId(1))
        );
        assert_eq!(
            navigate(&order, Some(SlideId(2)), NavDir::Next),
            Some(SlideId(2))
        );
        assert_eq!(navigate(&order, None, NavDir::Next), Some(SlideId(1)));
        assert_eq!(navigate(&[], None, NavDir::Next), None);
    }

    #[test]
    fn two_finger_swipe_detected_only_without_pinch() {
        let mut d = SwipeDetector::default();
        assert_eq!(d.update(None), None);
        for _ in 0..10 {
            assert_eq!(d.update(Some((egui::vec2(-15.0, 1.0), 1.0))), None);
        }
        assert!(d.is_active());
        assert_eq!(d.update(None), Some(NavDir::Next));
        assert_eq!(d.update(None), None);

        for _ in 0..10 {
            d.update(Some((egui::vec2(15.0, 0.0), 1.0)));
        }
        assert_eq!(d.update(None), Some(NavDir::Prev));

        // Cubit: bukan geser.
        for _ in 0..10 {
            d.update(Some((egui::vec2(-15.0, 0.0), 1.05)));
        }
        assert_eq!(d.update(None), None);

        // Vertikal: bukan geser.
        for _ in 0..10 {
            d.update(Some((egui::vec2(-10.0, 20.0), 1.0)));
        }
        assert_eq!(d.update(None), None);
    }

    #[test]
    fn manual_adjustment_keeps_pivot_under_pure_rotation() {
        let t = Affine2::translation(50.0, 60.0);
        let pivot = [300.0, 400.0];
        let rotated = manual_adjusted_transform(&t, [0.0, 0.0], 0.4, pivot);
        // Titik U yang jatuh di pivot tetap di pivot.
        let u = t.inverse().unwrap().apply(pivot);
        let w = rotated.apply(u);
        assert!((w[0] - pivot[0]).abs() < 1e-9 && (w[1] - pivot[1]).abs() < 1e-9);
        let shifted = manual_adjusted_transform(&t, [10.0, -5.0], 0.0, pivot);
        assert!(shifted.approx_eq(&Affine2::translation(60.0, 55.0), 1e-9));
    }
}
