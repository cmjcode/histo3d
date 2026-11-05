//! Kamera: `OrbitCamera` 3D (arcball bebas; turunan dari kamera turntable
//! DUCAD, tanpa mode Ortho2D) dan `Camera2D` ortografis untuk viewer slide.
//!
//! Satuan `OrbitCamera`: milimeter ruang render R (Z = arah tumpukan). Satuan
//! `Camera2D`: mikrometer ruang W/U (`f64`), karena hanya matematika
//! koordinat; konversi ke GPU lewat [`View2D`] (`f32`, relatif pusat).

use glam::{Mat3, Mat4, Quat, Vec2, Vec3};

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
    /// Tumpukan menyamping: dilihat dari depan (−Y) dengan sumbu Z (arah
    /// tumpukan) mendatar ke kanan layar, sehingga irisan pertama di kiri
    /// dan irisan terakhir di kanan.
    Side,
}

/// Kamera orbit bebas (arcball): berputar mengelilingi `target` ke segala
/// arah tanpa kunci kutub dan tanpa gimbal lock. Orientasi disimpan sebagai
/// kuaternion `rotation` yang memetakan basis kamera ke dunia: `+X` lokal =
/// kanan layar, `+Y` lokal = atas layar, `+Z` lokal = arah dari target ke
/// mata. Satuan milimeter (ruang R, Z dunia = arah tumpukan).
#[derive(Debug, Clone, PartialEq)]
pub struct OrbitCamera {
    /// Titik yang dikelilingi (mm).
    pub target: Vec3,
    /// Orientasi kamera (basis kamera → dunia), selalu ternormalisasi.
    pub rotation: Quat,
    /// Jarak mata ke target (mm).
    pub distance: f32,
    /// Sudut pandang vertikal (radian).
    pub fov_y: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            rotation: rotation_from_eye_up(eye_dir_from_yaw_pitch(-45.0, 30.0), Vec3::Z),
            distance: 50.0,
            fov_y: 45f32.to_radians(),
        }
    }
}

/// Arah target → mata untuk sudut turntable `yaw`/`pitch` (derajat).
/// Hanya dipakai untuk menyatakan preset dalam istilah yang akrab.
fn eye_dir_from_yaw_pitch(yaw_deg: f32, pitch_deg: f32) -> Vec3 {
    let (sin_yaw, cos_yaw) = yaw_deg.to_radians().sin_cos();
    let (sin_pitch, cos_pitch) = pitch_deg.to_radians().sin_cos();
    Vec3::new(cos_pitch * cos_yaw, cos_pitch * sin_yaw, sin_pitch)
}

/// Orientasi kamera dari arah target → mata `eye_dir` dan petunjuk "atas"
/// `up_hint` (tidak perlu tegak lurus). Bila keduanya sejajar, dipilih
/// sumbu dunia lain supaya basis tetap sah.
fn rotation_from_eye_up(eye_dir: Vec3, up_hint: Vec3) -> Quat {
    let z = eye_dir.normalize_or_zero();
    let z = if z == Vec3::ZERO { Vec3::Z } else { z };
    let mut up = up_hint.normalize_or_zero();
    if up == Vec3::ZERO || up.cross(z).length_squared() < 1e-6 {
        up = if z.z.abs() < 0.9 { Vec3::Z } else { Vec3::Y };
    }
    // Kanan layar = up × eye_dir (sama dengan `forward × up`, forward = −z).
    let x = up.cross(z).normalize();
    let y = z.cross(x).normalize();
    Quat::from_mat3(&Mat3::from_cols(x, y, z)).normalize()
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

    /// Arah satuan dari target ke mata (dunia).
    pub fn eye_dir(&self) -> Vec3 {
        (self.rotation * Vec3::Z).normalize_or_zero()
    }

    /// Posisi mata (mm).
    pub fn eye(&self) -> Vec3 {
        self.target + self.distance * self.eye_dir()
    }

    /// Matriks pandang (dunia → mata), tangan-kanan; "atas" mengikuti
    /// orientasi kamera (boleh miring terhadap Z dunia).
    pub fn view(&self) -> Mat4 {
        let (_, up) = self.screen_basis();
        glam::camera::rh::view::look_at_mat4(self.eye(), self.target, up)
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
    pub fn screen_basis(&self) -> (Vec3, Vec3) {
        (
            (self.rotation * Vec3::X).normalize_or_zero(),
            (self.rotation * Vec3::Y).normalize_or_zero(),
        )
    }

    /// Putar kamera bebas (arcball); delta dalam piksel layar. Seret
    /// mendatar memutar sekitar sumbu "atas" layar, seret tegak sekitar
    /// sumbu "kanan" layar; tidak ada batas kutub, kamera boleh melewati
    /// atas/bawah spesimen dengan mulus.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        const SENSITIVITY: f32 = 0.008;
        if !(dx.is_finite() && dy.is_finite()) {
            return;
        }
        let (right, up) = self.screen_basis();
        let around_up = Quat::from_axis_angle(up, -dx * SENSITIVITY);
        let around_right = Quat::from_axis_angle(right, -dy * SENSITIVITY);
        self.rotation = (around_up * around_right * self.rotation).normalize();
    }

    /// Tegakkan kamera: pertahankan arah pandang, tetapi putar sekitar sumbu
    /// pandang agar "atas" layar sejajar Z dunia (arah tumpukan) lagi. Bila
    /// kamera memandang hampir tepat sepanjang Z, "atas" diarahkan ke +Y.
    pub fn upright(&mut self) {
        let eye_dir = self.eye_dir();
        let hint = if eye_dir.cross(Vec3::Z).length_squared() < 1e-4 {
            Vec3::Y
        } else {
            Vec3::Z
        };
        self.rotation = rotation_from_eye_up(eye_dir, hint);
    }

    /// Sudut elevasi mata di atas bidang XY dunia (radian, −π/2..π/2).
    pub fn elevation(&self) -> f32 {
        self.eye_dir().z.clamp(-1.0, 1.0).asin()
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

    /// Terapkan preset orientasi; `target`, jarak, dan `fov_y` tidak berubah.
    pub fn set_preset(&mut self, preset: ViewPreset) {
        let (eye_dir, up) = match preset {
            // Z dunia menjadi arah pandang, jadi "atas" layar = +Y (atas) atau
            // −Y (bawah), sama dengan orientasi turntable di tepi kutub.
            ViewPreset::Top => (Vec3::Z, Vec3::Y),
            ViewPreset::Bottom => (-Vec3::Z, -Vec3::Y),
            ViewPreset::Front => (-Vec3::Y, Vec3::Z),
            ViewPreset::Back => (Vec3::Y, Vec3::Z),
            ViewPreset::Right => (Vec3::X, Vec3::Z),
            ViewPreset::Left => (-Vec3::X, Vec3::Z),
            ViewPreset::Isometric => (eye_dir_from_yaw_pitch(-45.0, 35.264), Vec3::Z),
            // Dari depan, dengan +Z ke kanan layar: kanan = up × eye_dir
            // = (−X) × (−Y) = +Z.
            ViewPreset::Side => (-Vec3::Y, -Vec3::X),
        };
        self.rotation = rotation_from_eye_up(eye_dir, up);
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
    fn default_matches_turntable_yaw_pitch() {
        // Bawaan setara yaw −45°, pitch 30° dengan Z dunia ke atas layar.
        let cam = OrbitCamera::default();
        let expect = eye_dir_from_yaw_pitch(-45.0, 30.0);
        assert!((cam.eye_dir() - expect).length() < 1e-5);
        assert!((cam.elevation() - 30f32.to_radians()).abs() < 1e-5);
        let (right, up) = cam.screen_basis();
        assert!(right.z.abs() < 1e-5, "kanan layar mendatar: {right:?}");
        assert!(up.z > 0.0);
    }

    #[test]
    fn orbit_passes_over_the_pole_without_lock() {
        let mut cam = OrbitCamera::default();
        // Seret ke bawah terus: elevasi naik melewati +90° lalu turun lagi
        // di sisi seberang (tidak ada kunci di ±89°).
        let before = cam.eye_dir();
        let mut max_elev = f32::MIN;
        let mut crossed = false;
        for _ in 0..400 {
            cam.orbit(0.0, 2.0);
            max_elev = max_elev.max(cam.elevation());
            if cam.eye_dir().dot(before) < 0.0 {
                crossed = true;
            }
            assert!((cam.rotation.length() - 1.0).abs() < 1e-4);
        }
        assert!(max_elev > 85f32.to_radians(), "{max_elev}");
        assert!(crossed, "kamera harus bisa ke sisi seberang");
    }

    #[test]
    fn orbit_full_turn_returns_home() {
        let mut cam = OrbitCamera::default();
        let home = cam.clone();
        // 2π / 0.008 piksel ≈ 785,398 piksel; dibagi 1000 langkah.
        let step = std::f32::consts::TAU / 0.008 / 1000.0;
        for _ in 0..1000 {
            cam.orbit(step, 0.0);
        }
        assert!((cam.eye_dir() - home.eye_dir()).length() < 1e-2);
        let (r0, _) = home.screen_basis();
        let (r1, _) = cam.screen_basis();
        assert!((r0 - r1).length() < 1e-2);
    }

    #[test]
    fn orbit_direction_matches_turntable_feel() {
        let mut cam = OrbitCamera::default();
        cam.set_preset(ViewPreset::Front);
        // Seret ke bawah: mata naik (melihat lebih dari atas).
        cam.orbit(0.0, 20.0);
        assert!(cam.eye_dir().z > 0.0);
        // Seret ke kanan dari depan: mata bergeser ke −X (searah jarum jam
        // dari atas), seperti yaw berkurang pada turntable.
        let mut cam = OrbitCamera::default();
        cam.set_preset(ViewPreset::Front);
        cam.orbit(20.0, 0.0);
        assert!(cam.eye_dir().x < 0.0, "{:?}", cam.eye_dir());
        // Masukan tak finit diabaikan.
        let before = cam.clone();
        cam.orbit(f32::NAN, 0.0);
        assert_eq!(cam, before);
    }

    #[test]
    fn upright_realigns_screen_up_to_world_z() {
        let mut cam = OrbitCamera::default();
        for _ in 0..37 {
            cam.orbit(13.0, 7.0);
        }
        let dir = cam.eye_dir();
        cam.upright();
        assert!((cam.eye_dir() - dir).length() < 1e-5, "arah pandang tetap");
        let (right, up) = cam.screen_basis();
        assert!(right.z.abs() < 1e-5);
        assert!(up.z > 0.0);
        // Memandang lurus sepanjang Z: tidak NaN, atas = +Y.
        cam.set_preset(ViewPreset::Top);
        cam.upright();
        assert!(cam.rotation.is_finite());
        assert!((cam.screen_basis().1 - Vec3::Y).length() < 1e-5);
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
        assert!((cam.elevation() - std::f32::consts::FRAC_PI_2).abs() < 1e-4);
        assert!((cam.screen_basis().1 - Vec3::Y).length() < 1e-5);
        cam.set_preset(ViewPreset::Bottom);
        assert!((cam.screen_basis().1 + Vec3::Y).length() < 1e-5);
        cam.set_preset(ViewPreset::Front);
        assert!(cam.elevation().abs() < 1e-4);
        // Dari depan (−Y) melihat ke +Y: mata di y negatif.
        assert!(cam.eye().y < cam.target.y);
        // Tampilan ortogonal selalu mengarah tepat ke sumbu.
        for (p, dir) in [
            (ViewPreset::Right, Vec3::X),
            (ViewPreset::Left, -Vec3::X),
            (ViewPreset::Back, Vec3::Y),
        ] {
            cam.set_preset(p);
            assert!((cam.eye_dir() - dir).length() < 1e-5, "{p:?}");
            assert!((cam.screen_basis().1 - Vec3::Z).length() < 1e-5, "{p:?}");
        }
    }

    #[test]
    fn side_preset_lays_stack_left_to_right() {
        let mut cam = OrbitCamera::default();
        cam.set_preset(ViewPreset::Side);
        let (right, up) = cam.screen_basis();
        // +Z dunia (irisan berikutnya) ke kanan layar; dilihat dari −Y.
        assert!((right - Vec3::Z).length() < 1e-5, "{right:?}");
        assert!((cam.eye_dir() + Vec3::Y).length() < 1e-5);
        assert!(up.z.abs() < 1e-5);
        // Titik dengan z lebih besar jatuh lebih ke kanan di layar.
        let vp = cam.view_proj(1.0);
        let lo = vp * Vec3::new(0.0, 0.0, -5.0).extend(1.0);
        let hi = vp * Vec3::new(0.0, 0.0, 5.0).extend(1.0);
        assert!(hi.x / hi.w > lo.x / lo.w);
    }

    #[test]
    fn view_matrix_is_consistent_with_basis() {
        let mut cam = OrbitCamera::default();
        cam.orbit(-31.0, 44.0);
        let v = cam.view();
        let (right, up) = cam.screen_basis();
        // Titik target + right harus muncul di +x mata, target + up di +y mata.
        let r = v.transform_point3(cam.target + right);
        let u = v.transform_point3(cam.target + up);
        assert!(r.x > 0.99 && r.y.abs() < 1e-4, "{r:?}");
        assert!(u.y > 0.99 && u.x.abs() < 1e-4, "{u:?}");
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
