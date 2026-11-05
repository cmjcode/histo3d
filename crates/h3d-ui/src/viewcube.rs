//! ViewCube: kubus orientasi di pojok kanvas 3D (disalin dari
//! `ducad-ui/src/viewcube.rs`). Berputar mengikuti yaw/pitch kamera orbit;
//! mengeklik salah satu muka mengembalikan preset pandangan.
//!
//! Konvensi kamera sama dengan `h3d_render::OrbitCamera`: Z ke atas,
//! `arah_mata = (cos p · cos y, cos p · sin y, sin p)` dari target ke mata.
//! Crate ini tidak bergantung pada `h3d-render`, jadi preset dinyatakan
//! sebagai [`ViewCubeAction`] dan dipetakan oleh aplikasi.

use egui::{Color32, FontId, Pos2, Rect, Stroke, Vec2};
use glam::Vec3;

use crate::theme;

/// Preset pandangan yang dihasilkan saat sebuah muka diklik.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewCubeAction {
    /// Dari atas (+Z): irisan terakhir di depan.
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
}

impl ViewCubeAction {
    /// Semua muka.
    pub const ALL: [ViewCubeAction; 6] = [
        ViewCubeAction::Top,
        ViewCubeAction::Bottom,
        ViewCubeAction::Front,
        ViewCubeAction::Back,
        ViewCubeAction::Right,
        ViewCubeAction::Left,
    ];

    /// Kunci terjemahan label muka (`viewcube-*`).
    pub fn i18n_key(self) -> &'static str {
        match self {
            ViewCubeAction::Top => "viewcube-top",
            ViewCubeAction::Bottom => "viewcube-bottom",
            ViewCubeAction::Front => "viewcube-front",
            ViewCubeAction::Back => "viewcube-back",
            ViewCubeAction::Right => "viewcube-right",
            ViewCubeAction::Left => "viewcube-left",
        }
    }
}

/// Kubus orientasi.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewCube {
    /// Panjang sisi kubus (point).
    pub size: f32,
}

impl Default for ViewCube {
    fn default() -> Self {
        Self { size: 54.0 }
    }
}

struct FaceDef {
    action: ViewCubeAction,
    normal: Vec3,
    corners: [Vec3; 4],
}

struct ProjectedFace<'a> {
    def: &'a FaceDef,
    pts: [Pos2; 4],
    center_2d: Pos2,
    z_depth: f32,
}

impl ViewCube {
    /// Luas area yang dipakai (termasuk sumbu di bawah kubus).
    pub fn footprint(&self) -> Vec2 {
        Vec2::splat(self.size * 1.6)
    }

    /// Gambar kubus berpusat di `center`; `Some(aksi)` bila sebuah muka diklik.
    pub fn show(
        &self,
        ui: &mut egui::Ui,
        center: Pos2,
        camera_yaw: f32,
        camera_pitch: f32,
    ) -> Option<ViewCubeAction> {
        let rect = Rect::from_center_size(center, self.footprint());
        let response = ui.allocate_rect(rect, egui::Sense::click());
        let painter = ui.painter_at(rect);

        let (sin_y, cos_y) = camera_yaw.sin_cos();
        let (sin_p, cos_p) = camera_pitch.sin_cos();
        let eye_dir = Vec3::new(cos_p * cos_y, cos_p * sin_y, sin_p).normalize_or_zero();
        let cam_right = Vec3::new(-sin_y, cos_y, 0.0);
        let cam_up = Vec3::new(-sin_p * cos_y, -sin_p * sin_y, cos_p);

        // Proyeksi ortografis ke kanvas (y egui ke bawah).
        let project = |p: Vec3| -> (Pos2, f32) {
            let sx = p.dot(cam_right);
            let sy = -p.dot(cam_up);
            (center + Vec2::new(sx, sy), p.dot(eye_dir))
        };

        let h = self.size * 0.5;
        let faces = [
            FaceDef {
                action: ViewCubeAction::Top,
                normal: Vec3::Z,
                corners: [
                    Vec3::new(-h, -h, h),
                    Vec3::new(h, -h, h),
                    Vec3::new(h, h, h),
                    Vec3::new(-h, h, h),
                ],
            },
            FaceDef {
                action: ViewCubeAction::Bottom,
                normal: -Vec3::Z,
                corners: [
                    Vec3::new(-h, h, -h),
                    Vec3::new(h, h, -h),
                    Vec3::new(h, -h, -h),
                    Vec3::new(-h, -h, -h),
                ],
            },
            FaceDef {
                action: ViewCubeAction::Front,
                normal: -Vec3::Y,
                corners: [
                    Vec3::new(-h, -h, -h),
                    Vec3::new(h, -h, -h),
                    Vec3::new(h, -h, h),
                    Vec3::new(-h, -h, h),
                ],
            },
            FaceDef {
                action: ViewCubeAction::Back,
                normal: Vec3::Y,
                corners: [
                    Vec3::new(h, h, -h),
                    Vec3::new(-h, h, -h),
                    Vec3::new(-h, h, h),
                    Vec3::new(h, h, h),
                ],
            },
            FaceDef {
                action: ViewCubeAction::Right,
                normal: Vec3::X,
                corners: [
                    Vec3::new(h, -h, -h),
                    Vec3::new(h, h, -h),
                    Vec3::new(h, h, h),
                    Vec3::new(h, -h, h),
                ],
            },
            FaceDef {
                action: ViewCubeAction::Left,
                normal: -Vec3::X,
                corners: [
                    Vec3::new(-h, h, -h),
                    Vec3::new(-h, -h, -h),
                    Vec3::new(-h, -h, h),
                    Vec3::new(-h, h, h),
                ],
            },
        ];

        // Hanya muka yang menghadap kamera yang digambar (back-face culling).
        let mut projected: Vec<ProjectedFace<'_>> = Vec::with_capacity(3);
        for face in &faces {
            if face.normal.dot(eye_dir) <= 0.001 {
                continue;
            }
            let mut pts = [center; 4];
            let mut sum_z = 0.0;
            let mut sum_2d = Vec2::ZERO;
            for (i, corner) in face.corners.iter().enumerate() {
                let (p2d, z) = project(*corner);
                pts[i] = p2d;
                sum_z += z;
                sum_2d += p2d.to_vec2();
            }
            projected.push(ProjectedFace {
                def: face,
                pts,
                center_2d: Pos2::new(sum_2d.x / 4.0, sum_2d.y / 4.0),
                z_depth: sum_z / 4.0,
            });
        }
        projected.sort_by(|a, b| a.z_depth.total_cmp(&b.z_depth));

        let hover_pos = ui.input(|i| i.pointer.hover_pos());
        let hovered_face = hover_pos
            .filter(|p| rect.contains(*p))
            .and_then(|p| projected.iter().rposition(|pf| point_in_quad(p, pf.pts)));

        let mut clicked = None;
        if response.clicked() {
            // Di layar sentuh `hover_pos()` sudah `None` pada frame jari
            // dilepas, jadi muka yang diketuk dicari dari posisi interaksi.
            let click_pos = response.interact_pointer_pos().or(hover_pos);
            let idx = click_pos
                .filter(|p| rect.contains(*p))
                .and_then(|p| projected.iter().rposition(|pf| point_in_quad(p, pf.pts)))
                .or(hovered_face);
            if let Some(idx) = idx {
                clicked = Some(projected[idx].def.action);
            }
        }

        let light_dir = Vec3::new(0.4, -0.5, 0.75).normalize();
        let dark = ui.visuals().dark_mode;
        for (idx, pf) in projected.iter().enumerate() {
            let hovered = hovered_face == Some(idx);
            let (fill, stroke, text) = if hovered {
                (
                    theme::ACCENT,
                    Stroke::new(1.5, theme::ACCENT_LIGHT),
                    Color32::WHITE,
                )
            } else {
                let diffuse = pf.def.normal.dot(light_dir).max(0.0);
                if dark {
                    let v = (46.0 + diffuse * 36.0).clamp(42.0, 96.0) as u8;
                    (
                        Color32::from_rgb(v, v + 2, v + 7),
                        Stroke::new(1.0, Color32::from_rgb(82, 88, 102)),
                        Color32::from_rgb(225, 230, 240),
                    )
                } else {
                    let v = (205.0 + diffuse * 40.0).clamp(200.0, 250.0) as u8;
                    (
                        Color32::from_rgb(v, v, v),
                        Stroke::new(1.0, Color32::from_rgb(150, 155, 165)),
                        Color32::from_rgb(40, 44, 52),
                    )
                }
            };
            painter.add(egui::Shape::convex_polygon(pf.pts.to_vec(), fill, stroke));
            painter.text(
                pf.center_2d,
                egui::Align2::CENTER_CENTER,
                h3d_i18n::t!(pf.def.action.i18n_key()),
                FontId::proportional(9.5),
                text,
            );
        }

        // Sumbu RGB kecil di bawah kubus.
        let axes_center = center + Vec2::new(0.0, self.size * 0.72);
        let axis_len = 16.0;
        let to_2d =
            |axis: Vec3| axes_center + Vec2::new(axis.dot(cam_right), -axis.dot(cam_up)) * axis_len;
        for (axis, color, name) in [
            (Vec3::X, theme::ERROR, "x"),
            (Vec3::Y, theme::SUCCESS, "y"),
            (Vec3::Z, theme::ACCENT, "z"),
        ] {
            let p = to_2d(axis);
            painter.line_segment([axes_center, p], Stroke::new(1.5, color));
            painter.text(
                p,
                egui::Align2::CENTER_CENTER,
                name,
                FontId::proportional(9.0),
                color,
            );
        }
        clicked
    }
}

/// Uji titik di dalam segi empat (even-odd).
pub fn point_in_quad(p: Pos2, pts: [Pos2; 4]) -> bool {
    let mut inside = false;
    let mut j = 3;
    for i in 0..4 {
        let (pi, pj) = (pts[i], pts[j]);
        if ((pi.y > p.y) != (pj.y > p.y))
            && (p.x < (pj.x - pi.x) * (p.y - pi.y) / (pj.y - pi.y) + pi.x)
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_in_quad_basic() {
        let quad = [
            Pos2::new(0.0, 0.0),
            Pos2::new(10.0, 0.0),
            Pos2::new(10.0, 10.0),
            Pos2::new(0.0, 10.0),
        ];
        assert!(point_in_quad(Pos2::new(5.0, 5.0), quad));
        assert!(!point_in_quad(Pos2::new(15.0, 5.0), quad));
        assert!(!point_in_quad(Pos2::new(-1.0, 5.0), quad));
    }

    #[test]
    fn preset_faces_face_the_camera() {
        let eye = |yaw_deg: f32, pitch_deg: f32| {
            let (sy, cy) = yaw_deg.to_radians().sin_cos();
            let (sp, cp) = pitch_deg.to_radians().sin_cos();
            Vec3::new(cp * cy, cp * sy, sp)
        };
        // Atas: yaw −90°, pitch 89° (sama dengan `ViewPreset::Top` h3d-render).
        assert!(Vec3::Z.dot(eye(-90.0, 89.0)) > 0.0);
        assert!((-Vec3::Z).dot(eye(-90.0, 89.0)) < 0.0);
        // Depan: −Y.
        assert!((-Vec3::Y).dot(eye(-90.0, 0.0)) > 0.99);
        // Kanan: +X.
        assert!(Vec3::X.dot(eye(0.0, 0.0)) > 0.99);
        // Isometrik: tiga muka tampak.
        let iso = eye(-45.0, 35.264);
        assert!(Vec3::Z.dot(iso) > 0.0 && (-Vec3::Y).dot(iso) > 0.0 && Vec3::X.dot(iso) > 0.0);
    }

    #[test]
    fn renders_without_click_and_keys_exist() {
        let ctx = egui::Context::default();
        theme::apply(&ctx, theme::ThemeMode::Dark);
        let cube = ViewCube::default();
        let mut out = ctx.run_ui(Default::default(), |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                let a = cube.show(ui, Pos2::new(100.0, 100.0), -0.8, 0.5);
                assert!(a.is_none());
            });
        });
        out.textures_delta.clear();
        for a in ViewCubeAction::ALL {
            for lang in h3d_i18n::Language::all() {
                assert!(h3d_i18n::has_key(*lang, a.i18n_key()), "{a:?}");
            }
        }
    }
}
