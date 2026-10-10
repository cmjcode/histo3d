//! Kamera: `OrbitCamera` 3D (disalin dari DUCAD, tanpa mode Ortho2D) dan
//! `Camera2D` ortografis untuk viewer slide.
//!
//! Satuan `OrbitCamera`: milimeter ruang render R (Z ke atas). Satuan
//! `Camera2D`: mikrometer ruang W/U (`f64`), karena hanya matematika
//! koordinat; konversi ke GPU lewat [`View2D`] (`f32`, relatif pusat).

use glam::{Mat4, Vec2, Vec3};

/// Preset sudut pandang kamera standar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewPreset {
    /// Dari atas (+Z) — melihat tumpukan dari irisan terakhir.
    Top,
    /// Dari bawah (−Z).
    Bottom,
    /// Dari depan (−Y).
    Front,
    /// Dari belakang (+Y).
    Back,
    /// Dari kanan (+X).
    Right,
    /// Dari kiri (−X).
    Left,
    /// Isometrik.
    Isometric,
}

/// Kamera orbit gaya turntable: berputar mengelilingi `target`, sumbu Z
/// dunia selalu "atas" — tidak pernah roll. Satuan milimeter (ruang R).
#[derive(Debug, Clone, PartialEq)]
pub struct OrbitCamera {
    /// Titik yang dikelilingi (mm).
    pub target: Vec3,
    /// Rotasi sekitar sumbu Z (radian).
    pub yaw: f32,
    /// Elevasi dari bidang XY (radian), dibatasi < ±90° agar tidak gimbal-flip.
    pub pitch: f32,
    /// Jarak mata ke target (mm).
    pub distance: f32,
    /// Sudut pandang vertikal (radian).
    pub fov_y: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            yaw: -45f32.to_radians(),
            pitch: 30f32.to_radians(),
            distance: 50.0,
            fov_y: 45f32.to_radians(),
        }
    }
}

/// Rentang jarak kamera (mm) yang diizinkan.
const DISTANCE_MIN: f32 = 0.05;
const DISTANCE_MAX: f32 = 150_000.0;

/// Margin tambahan saat "pas ke layar" supaya objek tidak menempel tepi.
const FIT_MARGIN: f32 = 1.15;

impl OrbitCamera {
    /// Rasio piksel per milimeter pada bidang target untuk tinggi viewport
    /// `viewport_h_px`.
    pub fn px_per_mm(&self, viewport_h_px: f32) -> f32 {
        let world_h_mm = 2.0 * self.distance * (self.fov_y * 0.5).tan();
        viewport_h_px / world_h_mm.max(1e-4)
    }

    /// Posisi mata (mm).
    pub fn eye(&self) -> Vec3 {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        self.target + self.distance * Vec3::new(cos_pitch * cos_yaw, cos_pitch * sin_yaw, sin_pitch)
    }

    /// Matriks pandang (dunia → mata), tangan-kanan, Z ke atas.
    pub fn view(&self) -> Mat4 {
        glam::camera::rh::view::look_at_mat4(self.eye(), self.target, Vec3::Z)
    }

    /// Matriks proyeksi perspektif untuk wgpu (kedalaman 0..1).
    pub fn proj(&self, aspect: f32) -> Mat4 {
        let near = (self.distance * 0.001).max(0.001);
        let far = (self.distance * 100.0).max(1_000.0);
        glam::camera::rh::proj::directx::perspective(self.fov_y, aspect.max(0.01), near, far)
    }

    /// `proj × view`: dunia (mm) → clip.
    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        self.proj(aspect) * self.view()
    }

    /// Basis layar `(right, up)` dalam koordinat dunia.
    fn screen_basis(&self) -> (Vec3, Vec3) {
        let forward = (self.target - self.eye()).normalize_or_zero();
        let right = forward.cross(Vec3::Z).normalize_or_zero();
        let up = right.cross(forward).normalize_or_zero();
        (right, up)
    }

    /// Putar kamera; delta dalam piksel layar.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        const SENSITIVITY: f32 = 0.008;
        self.yaw -= dx * SENSITIVITY;
        self.pitch = (self.pitch + dy * SENSITIVITY).clamp(-89f32.to_radians(), 89f32.to_radians());
    }

    /// Geser target sejajar bidang layar; delta dalam piksel, diskalakan agar
    /// titik pada kedalaman target mengikuti kursor 1:1.
    pub fn pan(&mut self, dx: f32, dy: f32, viewport_height_px: f32) {
        let world_per_pixel =
            2.0 * self.distance * (self.fov_y * 0.5).tan() / viewport_height_px.max(1.0);
        let (right, up) = self.screen_basis();
        self.target -= right * dx * world_per_pixel;
        self.target += up * dy * world_per_pixel;
    }

    /// `factor` > 1 mendekat (zoom in), dipusatkan di tengah layar.
    pub fn zoom(&mut self, factor: f32) {
        self.zoom_at(factor, Vec2::ZERO, 1.0);
    }

    /// Zoom mengelilingi titik kursor dalam NDC layar (−1..1, y ke atas).
    /// Titik pada bidang target di bawah kursor tetap di posisi layar yang
    /// sama sebelum dan sesudah zoom.
    pub fn zoom_at(&mut self, factor: f32, cursor_ndc: Vec2, aspect: f32) {
        if !factor.is_finite() || factor <= 0.0 {
            return;
        }
        let old_distance = self.distance;
        let new_distance = (self.distance / factor).clamp(DISTANCE_MIN, DISTANCE_MAX);
        if (new_distance - old_distance).abs() < 1e-7 {
            return;
        }
        let tan_half = (self.fov_y * 0.5).tan();
        let half_h_old = old_distance * tan_half;
        let half_w_old = half_h_old * aspect.max(0.01);
        let half_h_new = new_distance * tan_half;
        let half_w_new = half_h_new * aspect.max(0.01);
        let delta_w = half_w_old - half_w_new;
        let delta_h = half_h_old - half_h_new;
        let (right, up) = self.screen_basis();
        self.target += right * cursor_ndc.x * delta_w + up * cursor_ndc.y * delta_h;
        self.distance = new_distance;
    }

    /// Terapkan preset orientasi.
    pub fn set_preset(&mut self, preset: ViewPreset) {
        let (yaw_deg, pitch_deg): (f32, f32) = match preset {
            ViewPreset::Top => (-90.0, 89.0),
            ViewPreset::Bottom => (-90.0, -89.0),
            ViewPreset::Front => (-90.0, 0.0),
            ViewPreset::Back => (90.0, 0.0),
            ViewPreset::Right => (0.0, 0.0),
            ViewPreset::Left => (180.0, 0.0),
            ViewPreset::Isometric => (-45.0, 35.264),
        };
        self.yaw = yaw_deg.to_radians();
        self.pitch = pitch_deg.to_radians();
    }

    /// Geser `target` ke pusat kotak `(min, max)` (mm) dan atur jarak supaya
    /// seluruh kotak muat di viewport beraspek `aspect`. Orientasi tetap.
    pub fn fit_bounds(&mut self, min: Vec3, max: Vec3, aspect: f32) {
        if !(min.is_finite() && max.is_finite()) {
            return;
        }
        let lo = min.min(max);
        let hi = min.max(max);
        self.target = (lo + hi) * 0.5;
        let radius = ((hi - lo).length() * 0.5).max(0.1);
        let half_h = radius / aspect.clamp(0.01, 1.0) * FIT_MARGIN;
        let distance = half_h / (self.fov_y * 0.5).tan();
        self.distance = distance.clamp(DISTANCE_MIN, DISTANCE_MAX);
    }
}

/// Kamera ortografis 2D untuk viewer slide. Bekerja dalam µm (`f64`) ruang
/// W (atau U bila affine lapisan identitas). Sumbu y µm dan y layar sama-sama
/// ke bawah, jadi tidak ada pembalikan di sini.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera2D {
    /// Titik dunia (µm) yang berada di pusat viewport.
    pub center_um: [f64; 2],
    /// Skala: mikrometer per piksel (point egui).
    pub um_per_px: f64,
}

impl Default for Camera2D {
    fn default() -> Self {
        Self {
            center_um: [0.0, 0.0],
            um_per_px: 1.0,
        }
    }
}

/// Rentang skala 2D yang diizinkan (µm/px).
const UM_PER_PX_MIN: f64 = 1e-3;
const UM_PER_PX_MAX: f64 = 1e5;

impl Camera2D {
    /// Titik layar (point egui) → dunia µm.
    pub fn screen_to_world_um(&self, pt_px: egui::Pos2, viewport_rect: egui::Rect) -> [f64; 2] {
        let c = viewport_rect.center();
        [
            self.center_um[0] + (pt_px.x - c.x) as f64 * self.um_per_px,
            self.center_um[1] + (pt_px.y - c.y) as f64 * self.um_per_px,
        ]
    }

    /// Dunia µm → titik layar (point egui).
    pub fn world_um_to_screen(&self, w_um: [f64; 2], viewport_rect: egui::Rect) -> egui::Pos2 {
        let c = viewport_rect.center();
        egui::pos2(
            (c.x as f64 + (w_um[0] - self.center_um[0]) / self.um_per_px) as f32,
            (c.y as f64 + (w_um[1] - self.center_um[1]) / self.um_per_px) as f32,
        )
    }

    /// Zoom di sekitar `cursor_px`; `factor` > 1 memperbesar. Titik dunia di
    /// bawah kursor tetap di posisi layar yang sama.
    pub fn zoom_at(&mut self, factor: f64, cursor_px: egui::Pos2, viewport_rect: egui::Rect) {
        if !factor.is_finite() || factor <= 0.0 {
            return;
        }
        let new_scale = (self.um_per_px / factor).clamp(UM_PER_PX_MIN, UM_PER_PX_MAX);
        let anchor = self.screen_to_world_um(cursor_px, viewport_rect);
        let c = viewport_rect.center();
        self.um_per_px = new_scale;
        self.center_um = [
            anchor[0] - (cursor_px.x - c.x) as f64 * new_scale,
            anchor[1] - (cursor_px.y - c.y) as f64 * new_scale,
        ];
    }

    /// Geser pandangan mengikuti seretan `delta_px` (konten bergerak searah kursor).
    pub fn pan(&mut self, delta_px: egui::Vec2) {
        self.center_um[0] -= delta_px.x as f64 * self.um_per_px;
        self.center_um[1] -= delta_px.y as f64 * self.um_per_px;
    }

    /// Pusatkan dan skalakan supaya kotak `(min_um, max_um)` muat di viewport.
    pub fn fit_rect_um(&mut self, min_um: [f64; 2], max_um: [f64; 2], viewport_rect: egui::Rect) {
        let w = (max_um[0] - min_um[0]).abs().max(1e-9);
        let h = (max_um[1] - min_um[1]).abs().max(1e-9);
        let vw = (viewport_rect.width() as f64).max(1.0);
        let vh = (viewport_rect.height() as f64).max(1.0);
        self.um_per_px =
            ((w / vw).max(h / vh) * FIT_MARGIN as f64).clamp(UM_PER_PX_MIN, UM_PER_PX_MAX);
        self.center_um = [(min_um[0] + max_um[0]) * 0.5, (min_um[1] + max_um[1]) * 0.5];
    }

    /// Turunkan parameter pandangan `f32` untuk GPU.
    pub fn view(&self, viewport_rect: egui::Rect) -> View2D {
        View2D {
            center_um: [self.center_um[0] as f32, self.center_um[1] as f32],
            um_per_px: self.um_per_px as f32,
            viewport_min_pt: [viewport_rect.min.x, viewport_rect.min.y],
            viewport_size_pt: [
                viewport_rect.width().max(1.0),
                viewport_rect.height().max(1.0),
            ],
        }
    }
}

/// Pandangan 2D siap-GPU (`f32`). Shader menghitung
/// `ndc = ((w − center) / um_per_px) / (viewport/2)` dengan y dibalik
/// (layar y ke bawah → NDC y ke atas).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct View2D {
    /// Pusat pandangan (µm ruang W).
    pub center_um: [f32; 2],
    /// Mikrometer per point.
    pub um_per_px: f32,
    /// Pojok kiri-atas viewport (point egui); dipakai untuk menjangkar papan catur.
    pub viewport_min_pt: [f32; 2],
    /// Ukuran viewport (point egui).
    pub viewport_size_pt: [f32; 2],
}

impl View2D {
    /// Titik W µm → NDC, cermin dari perhitungan di shader (untuk uji).
    pub fn ndc_of(&self, w_um: [f32; 2]) -> [f32; 2] {
        let px = (w_um[0] - self.center_um[0]) / self.um_per_px;
        let py = (w_um[1] - self.center_um[1]) / self.um_per_px;
        [
            px / (self.viewport_size_pt[0] * 0.5),
            -py / (self.viewport_size_pt[1] * 0.5),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect() -> egui::Rect {
        egui::Rect::from_min_size(egui::pos2(100.0, 50.0), egui::vec2(800.0, 600.0))
    }

    #[test]
    fn eye_respects_distance() {
        let cam = OrbitCamera::default();
        assert!((cam.eye().distance(cam.target) - cam.distance).abs() < 1e-3);
    }

    #[test]
    fn pitch_is_clamped() {
        let mut cam = OrbitCamera::default();
        cam.orbit(0.0, 1e6);
        assert!(cam.pitch <= 89f32.to_radians() + 1e-6);
        cam.orbit(0.0, -1e6);
        assert!(cam.pitch >= -89f32.to_radians() - 1e-6);
    }

    #[test]
    fn zoom_never_reaches_zero() {
        let mut cam = OrbitCamera::default();
        for _ in 0..100 {
            cam.zoom(10.0);
        }
        assert!(cam.distance >= DISTANCE_MIN);
        cam.zoom_at(f32::NAN, Vec2::ZERO, 1.0);
        cam.zoom_at(-1.0, Vec2::ZERO, 1.0);
        assert!(cam.distance >= DISTANCE_MIN);
    }

    #[test]
    fn orbit_zoom_at_keeps_target_plane_point_fixed() {
        let mut cam = OrbitCamera::default();
        let aspect = 1.6;
        let cursor_ndc = Vec2::new(0.4, -0.6);
        // Titik pada bidang target (tegak lurus arah pandang) di bawah kursor.
        let (right, up) = cam.screen_basis();
        let half_h = cam.distance * (cam.fov_y * 0.5).tan();
        let half_w = half_h * aspect;
        let p_world = cam.target + right * (cursor_ndc.x * half_w) + up * (cursor_ndc.y * half_h);

        let clip0 = cam.view_proj(aspect) * p_world.extend(1.0);
        let ndc0 = clip0.truncate() / clip0.w;
        assert!((ndc0.x - cursor_ndc.x).abs() < 1e-3 && (ndc0.y - cursor_ndc.y).abs() < 1e-3);

        cam.zoom_at(2.5, cursor_ndc, aspect);
        let clip = cam.view_proj(aspect) * p_world.extend(1.0);
        let ndc = clip.truncate() / clip.w;
        assert!(
            (ndc.x - cursor_ndc.x).abs() < 1e-3,
            "x: {} vs {}",
            ndc.x,
            cursor_ndc.x
        );
        assert!(
            (ndc.y - cursor_ndc.y).abs() < 1e-3,
            "y: {} vs {}",
            ndc.y,
            cursor_ndc.y
        );
    }

    #[test]
    fn fit_bounds_contains_box() {
        let mut cam = OrbitCamera::default();
        let min = Vec3::new(-10.0, -20.0, 0.0);
        let max = Vec3::new(30.0, 20.0, 50.0);
        cam.fit_bounds(min, max, 16.0 / 9.0);
        assert!((cam.target - Vec3::new(10.0, 0.0, 25.0)).length() < 1e-4);
        let vp = cam.view_proj(16.0 / 9.0);
        for &x in &[min.x, max.x] {
            for &y in &[min.y, max.y] {
                for &z in &[min.z, max.z] {
                    let clip = vp * Vec3::new(x, y, z).extend(1.0);
                    let ndc = clip.truncate() / clip.w;
                    assert!(ndc.x.abs() <= 1.0 && ndc.y.abs() <= 1.0, "di luar: {ndc:?}");
                }
            }
        }
        let d = cam.distance;
        cam.fit_bounds(Vec3::NAN, Vec3::ONE, 1.0);
        assert_eq!(cam.distance, d);
    }

    #[test]
    fn presets_apply() {
        let mut cam = OrbitCamera::default();
        cam.set_preset(ViewPreset::Top);
        assert!((cam.pitch - 89f32.to_radians()).abs() < 1e-4);
        cam.set_preset(ViewPreset::Front);
        assert!(cam.pitch.abs() < 1e-4);
        // Dari depan (−Y) melihat ke +Y: mata di y negatif.
        assert!(cam.eye().y < cam.target.y);
    }

    #[test]
    fn camera2d_screen_world_roundtrip() {
        let cam = Camera2D {
            center_um: [12_345.5, -678.25],
            um_per_px: 0.73,
        };
        let r = rect();
        let pts = [
            egui::pos2(100.0, 50.0),
            egui::pos2(500.0, 350.0),
            egui::pos2(899.5, 649.5),
        ];
        for p in pts {
            let w = cam.screen_to_world_um(p, r);
            let back = cam.world_um_to_screen(w, r);
            assert!(
                (back.x - p.x).abs() < 1e-3 && (back.y - p.y).abs() < 1e-3,
                "{p:?} → {back:?}"
            );
        }
        // Pusat viewport = center_um.
        let w = cam.screen_to_world_um(r.center(), r);
        assert!((w[0] - cam.center_um[0]).abs() < 1e-9 && (w[1] - cam.center_um[1]).abs() < 1e-9);
        // y layar ke bawah = y µm bertambah.
        let below = cam.screen_to_world_um(r.center() + egui::vec2(0.0, 10.0), r);
        assert!(below[1] > w[1]);
    }

    #[test]
    fn camera2d_zoom_at_keeps_cursor_point_fixed() {
        let mut cam = Camera2D {
            center_um: [5000.0, 3000.0],
            um_per_px: 2.0,
        };
        let r = rect();
        let cursor = egui::pos2(300.0, 500.0);
        let anchor = cam.screen_to_world_um(cursor, r);
        cam.zoom_at(1.7, cursor, r);
        assert!((cam.um_per_px - 2.0 / 1.7).abs() < 1e-12);
        let after = cam.screen_to_world_um(cursor, r);
        assert!((after[0] - anchor[0]).abs() < 1e-6 && (after[1] - anchor[1]).abs() < 1e-6);
        // Faktor tak sah diabaikan.
        let before = cam;
        cam.zoom_at(0.0, cursor, r);
        cam.zoom_at(f64::NAN, cursor, r);
        assert_eq!(cam, before);
    }

    #[test]
    fn camera2d_pan_moves_content_with_cursor() {
        let mut cam = Camera2D {
            center_um: [0.0, 0.0],
            um_per_px: 4.0,
        };
        let r = rect();
        let p_screen = egui::pos2(400.0, 300.0);
        let w = cam.screen_to_world_um(p_screen, r);
        cam.pan(egui::vec2(10.0, -5.0));
        let p_after = cam.world_um_to_screen(w, r);
        assert!((p_after.x - (p_screen.x + 10.0)).abs() < 1e-3);
        assert!((p_after.y - (p_screen.y - 5.0)).abs() < 1e-3);
    }

    #[test]
    fn camera2d_fit_rect_and_view_ndc() {
        let mut cam = Camera2D::default();
        let r = rect();
        cam.fit_rect_um([0.0, 0.0], [8000.0, 3000.0], r);
        assert!(
            (cam.center_um[0] - 4000.0).abs() < 1e-9 && (cam.center_um[1] - 1500.0).abs() < 1e-9
        );
        let view = cam.view(r);
        // Pusat → NDC (0,0); pojok kiri-atas kotak → x negatif, y positif (NDC y ke atas).
        let c = view.ndc_of([4000.0, 1500.0]);
        assert!(c[0].abs() < 1e-6 && c[1].abs() < 1e-6);
        let tl = view.ndc_of([0.0, 0.0]);
        assert!(tl[0] < 0.0 && tl[1] > 0.0 && tl[0] >= -1.0 && tl[1] <= 1.0);
        // Konsisten dengan world_um_to_screen.
        let p = cam.world_um_to_screen([0.0, 0.0], r);
        let ndc_x = ((p.x - r.min.x) / r.width()) * 2.0 - 1.0;
        let ndc_y = 1.0 - ((p.y - r.min.y) / r.height()) * 2.0;
        assert!((ndc_x - tl[0]).abs() < 1e-4 && (ndc_y - tl[1]).abs() < 1e-4);
    }
}
