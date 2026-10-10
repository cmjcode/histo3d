//! Pembantu uji bersama: mesh bola tertutup, masker voxel, laporan contoh.
#![allow(dead_code)]

use h3d_core::{
    AnnotationRecord, AnnotationSource, LabelId, LabelKind, NoProgress, Project, ReconConfig,
    ReviewStatus, SlideEntry, SlideId, SlideSourceRef,
};
use h3d_io::{LabelReport, Report};
use h3d_volume::{extract_mesh, LabelMesh, SphereSdf, StackGrid, VoxelMask};

pub const RADIUS_UM: f64 = 200.0;
pub const CENTER_UM: [f64; 3] = [320.0, 320.0, 312.0];

pub fn sphere_grid() -> StackGrid {
    StackGrid {
        origin_um: [0.0, 0.0],
        s_xy: 16.0,
        size: [40, 40],
        z_um: (0..40).map(|k| k as f64 * 16.0).collect(),
    }
}

/// Mesh bola tertutup dari SDF analitik (surface nets + Taubin + decimate).
pub fn sphere_mesh() -> LabelMesh {
    let sdf = SphereSdf {
        grid: sphere_grid(),
        center_um: CENTER_UM,
        radius_um: RADIUS_UM,
    };
    extract_mesh(&sdf, LabelId(1), &ReconConfig::default(), &NoProgress).expect("mesh bola")
}

/// Masker voxel bola pada `grid`.
pub fn sphere_mask(grid: &StackGrid) -> VoxelMask {
    let mut m = VoxelMask::empty(grid.clone());
    let [nx, ny, nz] = grid.dims();
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let c = grid.center_um(x, y, z);
                let d2 = (c[0] - CENTER_UM[0]).powi(2)
                    + (c[1] - CENTER_UM[1]).powi(2)
                    + (c[2] - CENTER_UM[2]).powi(2);
                if d2 <= RADIUS_UM * RADIUS_UM {
                    let i = grid.linear(x, y, z);
                    m.data[i] = 1;
                }
            }
        }
    }
    m
}

/// Proyek contoh dengan beragam nilai pecahan.
pub fn sample_project() -> Project {
    let mut p = Project::new("sampel-uji");
    p.user_initials = "YJ".into();
    p.sample.shrinkage_linear = Some(1.0 / 0.9);
    for i in 0..4u32 {
        p.slides.push(SlideEntry {
            id: SlideId(i),
            source: SlideSourceRef::External {
                path: format!("../irisan/s{i:02}.png").into(),
                size_bytes: 1000 + u64::from(i),
                fingerprint: 0xdead_beef_0000_0000 + u64::from(i),
            },
            order: i,
            z_um: f64::from(i) * 50.0 + 0.1,
            mpp: [0.2531, 0.2529],
            size_l0: [40_000, 30_000],
            transform: h3d_core::SlideTransform::Affine(h3d_core::Affine2::rotation_about(
                0.1 * f64::from(i) + 1.0 / 3.0,
                [1234.5678, 987.654_321],
            )),
            ..Default::default()
        });
    }
    p.annotations.push(AnnotationRecord {
        slide: SlideId(1),
        label: LabelId(1),
        source: AnnotationSource::Manual {
            by: "YJ".into(),
            edited_from: None,
        },
        at: "2026-01-02T03:04:05+07:00".into(),
        review: ReviewStatus::Draft,
        confidence: None,
    });
    p
}

/// Laporan contoh dengan satu label.
pub fn sample_report() -> Report {
    let mut r = Report::new("sampel-uji", "0.1.0");
    r.dimension_uncertainty_um = 24.0;
    r.labels.push(LabelReport {
        label: LabelId(1),
        name: "Tumor".into(),
        kind: LabelKind::Tumor,
        group: "Tumor".into(),
        volume_cavalieri_um3: 3.3e7,
        volume_reviewed_um3: None,
        volume_corrected_um3: Some(4.0e7),
        volume_mesh_um3: Some(3.31e7),
        surface_area_um2: Some(5.0e5),
        bbox_um: Some(([0.0; 3], [400.0; 3])),
        max_diameter_um: Some(400.0),
        n_components: Some(1),
        n_slices_annotated: 20,
        n_slices_interpolated: 0,
        n_slices_reviewed: 0,
    });
    r
}
