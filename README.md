# HISTO3D

> **High-Precision 3D Tumor Reconstruction & Quantitative Spatial Pathology from Serial Histology Slices**

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.96%2B%20(Stable)-orange.svg)](rust-toolchain.toml)
[![Platform: macOS & iPadOS](https://img.shields.io/badge/Platform-macOS%20%7C%20iPadOS-lightgrey.svg)](apple/)
[![Offline](https://img.shields.io/badge/Network-100%25%20Offline-success.svg)](deny.toml)
[![CI](https://github.com/cmjcode/histo3d/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

---

> [!IMPORTANT]
> **Research Notice**: **For research use only. Not for diagnostic use or clinical decision-making.**
> HISTO3D provides quantitative 3D measurements and spatial estimations with explicit uncertainty bounds; it does not output clinical recommendations or diagnostic determinations.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Workspace Architecture](#workspace-architecture)
- [Coordinate Conventions & Physical Units](#coordinate-conventions--physical-units)
- [Prerequisites & Installation](#prerequisites--installation)
- [Running the GUI Application](#running-the-gui-application)
  - [macOS](#macos)
  - [iPadOS & Apple Pencil](#ipados--apple-pencil)
- [Headless CLI Guide (`h3d`)](#headless-cli-guide-h3d)
  - [Command Reference](#command-reference)
  - [End-to-End Pipeline Example](#end-to-end-pipeline-example)
  - [Synthetic Phantoms & Benchmarks](#synthetic-phantoms--benchmarks)
- [Spatial Analysis & Quantitative Pathology](#spatial-analysis--quantitative-pathology)
- [Project Directory Layout (`.h3d`)](#project-directory-layout-h3d)
- [Quality Gates & Testing](#quality-gates--testing)
- [License](#license)

---

## Overview

**HISTO3D** reconstructs true 3D tumor morphology and tissue architecture from serial physical histology sections (microtome cuts). 

Physical histology slicing introduces non-linear deformations, rotation shifts, stretching, and tissue tears. Unlike optical z-stacks, physical serial cuts require robust alignment (registration) across successive sections to restore anatomical continuity.

HISTO3D delivers an end-to-end, deterministic, and completely offline pipeline written entirely in **pure Rust**. It pairs a high-performance compute engine with an interactive cross-platform GUI (macOS & iPadOS powered by `wgpu` and `egui`) alongside a headless CLI (`h3d`) for automated batch processing.

```
+-----------------------------------------------------------------------------------+
|                                 HISTO3D PIPELINE                                  |
+-----------------------------------------------------------------------------------+
|  [1. Ingestion]     -> Multi-format WSIs (SVS, NDPI, TIFF) & standard microscopy  |
|  [2. Ordering]      -> Automated similarity-based sequence & flip/rotation search |
|  [3. Alignment]     -> Coarse-to-fine rigid & affine registration + QC validation |
|  [4. Segmentation]  -> Tiled raster store, Apple Pencil tools, local ONNX models  |
|  [5. Interpolation] -> Signed Distance Field (SDF) morphing across sparse slices  |
|  [6. Meshing]       -> Fast Surface Nets, Taubin smoothing & quadric decimation   |
|  [7. Spatial Path]  -> Cavalieri volume, 3D margins, invasion depth, IHC density  |
|  [8. Visualization] -> GPU-accelerated 2D multiscale viewer & 3D mesh orbit       |
|  [9. Export]        -> STL, OBJ, GLB (vertex scalars), NIfTI, GeoJSON & reports   |
+-----------------------------------------------------------------------------------+
```

---

## Key Features

- **Multi-Format Ingestion**:
  - Whole Slide Images (WSI): Aperio SVS, Hamamatsu NDPI, and pyramidal BigTIFF via `wsi-rs`.
  - Standard microscopy camera captures: PNG, JPEG, and TIFF.
  - Physical scale is **strictly preserved**: Microns-per-pixel (`mpp`) is read from metadata or explicitly declared—never guessed.
- **Intelligent Slice Ordering & Orientation**:
  - Proposes optimal slice sequence and flip/rotation orientations using classical pairwise normalized cross-correlation (R1–R2 heuristics).
  - Preserves scientific provenance: algorithms propose; users review and confirm.
- **Multi-Resolution Registration**:
  - Hierarchical coarse-to-fine registration pyramid.
  - Rigid and affine transformation models using Normalized Cross-Correlation (NCC).
  - Automated Quality Control (QC): flags problematic slices using pairwise NCC, tissue mask Dice scores, and linear area deformation ratios.
  - Interactive manual landmark editor for difficult or partially torn sections.
- **Annotation & Segmentation**:
  - Multi-resolution, tiled, sparse raster label store (`.lbl`) indexed in slide space (4 µm/pixel resolution, 256×256 tiles).
  - Interactive brush and lasso tools with Apple Pencil pressure sensitivity on iPadOS.
  - Embedded local AI inference via `tract-onnx` (pure Rust, zero external C++ dependencies).
  - Morphological shape interpolation across unannotated intermediate slices using signed distance field blending.
  - Full interoperability with QuPath via GeoJSON import and export.
- **Watertight 3D Reconstruction**:
  - Dual volume computation: Stereological **Cavalieri estimator** (gold standard) alongside watertight mesh volume.
  - Fast Surface Nets extraction from 3D Signed Distance Fields (SDF).
  - Volume-preserving Taubin smoothing and topology-preserving quadric error decimation.
- **Quantitative Spatial Pathology**:
  - True 3D resection margin analysis with directional anatomical breakdown (Superior, Inferior, Anterior, Posterior, Medial, Lateral, Deep, Superficial).
  - Maximum 3D tumor invasion depth and multifocal satellite lesion separation.
  - Structural contact and penetration analysis (blood vessels, lymphatic channels, perineural structures, capsule).
  - Regional biomarker quantification (IHC positivity, cell density, TIL proxies).
  - All 3D spatial measurements report explicit confidence intervals and uncertainty bounds ($\pm u$).
- **Privacy & Security by Design**:
  - **100% Offline**: Zero networking crates or HTTP clients in dependency trees (`ureq`, `reqwest`, and `hyper` are banned via `deny.toml`).
  - **Strict Clinical Data Isolation**: Patient demographics and clinical metadata are pseudonymized and sequestered exclusively in `clinical.json`. They are never stored in `project.json` or exported without explicit `--with-clinical` flags.
  - **Fully Deterministic**: Parallel reductions are sorted deterministically; identical inputs yield byte-identical outputs.
  - **Atomic Persistence**: All project modifications use temporary files (`.tmp`) followed by atomic filesystem renames to prevent partial write corruption.

---

## Workspace Architecture

HISTO3D is partitioned into 13 modular crates enforcing strict architectural boundaries:

```
crates/
├── h3d-core       # Primitive domain types, affine math (µm), budgets, traits
├── h3d-slide      # Slide decoders (WSI, TIFF, standard images), tile cache
├── h3d-imgproc    # Color deconvolution, tissue masking, morphology, distance transform
├── h3d-register   # Pairwise/chain registration, QC metrics, landmark solver
├── h3d-segment    # Tiled label store, drawing tools, ONNX runtime, shape interpolation
├── h3d-volume     # 3D stack grid, SDF generation, surface nets, mesh decimation & measure
├── h3d-io         # Project serialisation, atomic file I/O, NIfTI, mesh & report export
├── h3d-engine     # Headless pipeline orchestration, session management, background jobs
├── h3d-cli        # Standalone CLI binary (`h3d`)
├── h3d-render     # Pure wgpu rendering pipelines (2D tile pyramids, label overlays, 3D mesh)
├── h3d-i18n       # Internationalisation via Fluent (Indonesian default, English secondary)
├── h3d-ui         # Reusable egui widgets, panels, and glyph coverage verification
└── h3d-app        # Cross-platform GUI binary (`histo3d`) targeting macOS & iPadOS
```

### Dependency Hierarchy & Boundary Rules

1. **GUI Independence**: `h3d-engine` and all crates underneath it (`core`, `slide`, `imgproc`, `register`, `segment`, `volume`, `io`) have **zero** GUI dependencies (`egui`, `eframe`, `wgpu`, `rfd`). Enforced by automated architecture tests (`engine_has_no_gui_dependency`).
2. **Pure Rust Pipeline**: No OCCT, ITK, VTK, or OpenCV.
3. **Rendering Isolation**: `h3d-render` consumes only raw renderable primitives (`[f32; 3]` vertices, RGBA tiles, triangle indices) without domain awareness of projects or registration graphs.

---

## Coordinate Conventions & Physical Units

To avoid spatial errors common in multi-slice reconstruction software, HISTO3D enforces strict mathematical invariants across all crates:

| Space | Symbol | Coordinates & Units | Origin / Definition |
|---|---|---|---|
| **Slide Space** | $U$ | Micrometers ($\mu\text{m}$), `f64` | Top-left of slide Level 0. Continuous physical coordinates. |
| **Stack Space** | $W$ | Micrometers ($\mu\text{m}$), `f64` | Common reference frame aligned across all registered slices. |
| **Voxel Grid** | $V$ | Discrete indices `(i, j, k)`, `usize` | Aligned with Stack Space bounding box at chosen reconstruction resolution ($s_{xy}, s_z$). |
| **Render Space**| $R$ | Millimeters ($\text{mm}$), `f32` | Centered at mesh centroid in the 3D viewport (scale: $1\text{ mm} = 1000\,\mu\text{m}$). |

### Core Rules

- **Pixel Centers**: Defined at `(col + 0.5, row + 0.5)` for all pixel-to-micrometer conversions.
- **Transformation Direction**: Forward transform $T_i$ maps points from Slide Space $U$ into Stack Space $W$:
  $$p_W = T_i(p_U)$$
- **Resampling Direction**: Backward sampling via inverse transform $T_i^{-1}$ pulls pixel values from Slide Space into Stack Space, avoiding sampling holes.
- **Storage**: No pixel coordinates are stored in project files (`project.json`); all spatial references are stored in physical micrometers ($\mu\text{m}$).

---

## Prerequisites & Installation

### Toolchain

HISTO3D targets Rust **1.96+** (Stable). The required components and cross-compilation targets are configured in `rust-toolchain.toml`:

```bash
rustup toolchain install stable
rustup component add rustfmt clippy
rustup target add aarch64-apple-ios aarch64-apple-ios-sim
```

### Cloning the Repository

```bash
git clone https://github.com/cmjcode/histo3d.git
cd histo3d
```

---

## Running the GUI Application

### macOS

Run directly with Cargo:

```bash
cargo run -p h3d-app
```

Or build a production-bundled `.app` into `dist/macos/HISTO3D.app`:

```bash
./build_macos.sh app
open dist/macos/HISTO3D.app
```

### iPadOS & Apple Pencil

HISTO3D features native touch gestures, Apple Pencil pressure-sensitive brush strokes, and sandboxed iOS document storage (`Documents` / `Application Support`).

Use the included `build_ipad.sh` utility:

```bash
# Verify iOS compilation (CI gate)
./build_ipad.sh check

# Build and deploy to an active iPad Simulator
./build_ipad.sh run-sim

# Build for a connected physical iPad
./build_ipad.sh device

# Package an unsigned or signed .ipa for distribution
./build_ipad.sh ipa
```

---

## Headless CLI Guide (`h3d`)

HISTO3D includes a full-featured headless binary (`h3d`) suitable for automated pipelines, server jobs, and continuous integration.

Build the CLI executable:

```bash
./build_macos.sh cli
# Output is placed at dist/macos/h3d
```

### Command Reference

| Subcommand | Description |
|---|---|
| `info` | Inspect dimensions, pyramid levels, and `mpp` metadata of a slide file (JSON). |
| `thumb` | Generate a fast downscaled PNG preview of a slide. |
| `new` | Initialize a new empty `.h3d` project bundle. |
| `import` | Register slides into the project, configure physical spacing and `mpp`. |
| `suggest-order` | Propose optimal sequence and flip/mirror orientations via image similarity. |
| `register` | Execute rigid/affine registration across slices to the reference section. |
| `qc` | Output registration quality metrics and flags (JSON or terminal table). |
| `labels` | Label store operations, including importing QuPath GeoJSON annotations. |
| `segment` | Run local AI segmentation using an ONNX model bundle (`model.onnx` + `model.json`). |
| `interpolate` | Compute morphometric shape interpolation across unannotated slice gaps. |
| `reconstruct` | Extract 3D meshes from voxel SDFs with smoothing and decimation. |
| `measure` | Output Cavalieri volume, mesh volume, surface area, and bounding dimensions. |
| `analyze` | Run 3D spatial pathology analysis (margins, invasion, multifocality, vessels). |
| `export` | Export 3D meshes (STL, OBJ, GLB), NIfTI volumes, GeoJSON, or Markdown reports. |
| `clinical` | Read or update pseudonymized clinical attributes (`clinical.json`). |
| `attrs` | Inspect or attach per-slide scalar biomarker attributes. |
| `phantom` | Synthesize deterministic 3D ground-truth phantoms for benchmarking. |

### End-to-End Pipeline Example

```bash
# 1. Initialize project directory
h3d new ./colon_specimen.h3d

# 2. Import serial histology slices (0.5 um/px, 20 um section thickness)
h3d import ./colon_specimen.h3d ./raw_scans/*.tif --mpp 0.5 --spacing-um 20.0

# 3. Analyze and accept slice ordering & orientation
h3d suggest-order ./colon_specimen.h3d --apply

# 4. Run automated pairwise and chained registration with strict QC checks
h3d register ./colon_specimen.h3d --model affine --metric ncc --strict

# 5. Review registration Quality Control table
h3d qc ./colon_specimen.h3d

# 6. Run local ONNX segmentation for tumor regions
h3d segment ./colon_specimen.h3d --model ./models/colon_epithelium_v1

# 7. Interpolate tumor label between sparsely annotated levels
h3d interpolate ./colon_specimen.h3d --label Tumor

# 8. Reconstruct 3D surface mesh at 16 um voxel resolution with Taubin smoothing
h3d reconstruct ./colon_specimen.h3d --sxy 16.0 --smooth 10 --decimate 1.5

# 9. Perform quantitative spatial pathology analysis
h3d analyze ./colon_specimen.h3d

# 10. Query volumetric measurements in JSON format
h3d measure ./colon_specimen.h3d --json

# 11. Export watertight 3D models with colored resection margin distances
h3d export ./colon_specimen.h3d --format glb --out ./exports/ --scalar margin_distance

# 12. Export volumetric label mask to compressed NIfTI
h3d export ./colon_specimen.h3d --format nifti --out ./exports/
```

### Synthetic Phantoms & Benchmarks

HISTO3D provides built-in procedural phantom generation for reproducible validation:

```bash
# Generate a 20-slice synthetic ellipsoid tumor with surgical margins and vessels
h3d phantom ./phantom_run.h3d --shape ellipsoid --slices 20 --seed 5 --size 128 --margin 240
```

The CLI test suite continuously checks output consistency against seeded golden files stored in `testdata/golden/`.

---

## Spatial Analysis & Quantitative Pathology

HISTO3D implements structured quantitative spatial analysis directly on 3D voxel masks rather than surface approximations:

1. **Volume Estimation**:
   - **Cavalieri Method**: $V = s_z \sum_i A_i$, where $A_i$ is the 2D area on slice $i$ and $s_z$ is the physical slice spacing. Robust to slicing irregularities and stereologically validated.
   - **Watertight Mesh Volume**: Computed from surface nets triangles using the divergence theorem for cross-verification.
2. **Surgical Resection Margins**:
   - Computes minimum Euclidean distance from tumor boundary to specimen margin voxels.
   - Evaluates clearances across standardized anatomical directions (Superior, Inferior, Anterior, Posterior, Medial, Lateral / Deep / Superficial).
   - Projects distance fields onto GLB mesh vertices for interactive false-color clearance inspection.
3. **Depth of Invasion & Multifocality**:
   - Measures maximum perpendicular tumor penetration from reference baselines.
   - Identifies disconnected satellite nodules in 3D, calculating inter-focal distances and aggregate volumes.
4. **Structural Invasion Proxies**:
   - Evaluates spatial proximity and boundary intersections between tumor voxels and segmented vascular, lymphatic, or capsular structures.
5. **Biomarker Spatial Profiling**:
   - Integrates slide-level and region-level IHC panel scoring (e.g., Ki-67 proliferation index, PD-L1, ER/PR, HER2).
   - Enables cell density mapping and spatial tumor-infiltrating lymphocyte (TIL) distribution profiling.

---

## Project Directory Layout (`.h3d`)

A `.h3d` project is a self-contained directory with human-readable manifests and atomic data storage:

```
my_study.h3d/
├── project.json      # Primary manifest: slide index, transforms, labels, parameters
├── clinical.json     # Isolated pseudonymized clinical record (never leaked without flag)
├── slides/           # Embedded slide files (when imported with --embed)
├── labels/           # Sparse tiled binary label rasters (*.lbl) in Slide Space
│   ├── slide_000000.lbl
│   └── slide_000001.lbl
├── cache/            # Intermediate registration pyramids, thumbnails, distance fields
└── out/              # Generated outputs
    ├── mesh/         # Reconstructed surface meshes (*.glb, *.stl, *.obj)
    ├── volume/       # Volumetric NIfTI exports (*.nii.gz)
    ├── analysis.json # Spatial pathology metrics and uncertainty tables
    └── report.md     # Generated research summary report
```

---

## Quality Gates & Testing

To maintain stability across platforms and prevent regressions, all commits must pass the automated quality gates:

```bash
# 1. Code formatting
cargo fmt --all -- --check

# 2. Linter rules with zero warnings
cargo clippy --workspace --all-targets -- -D warnings

# 3. Complete workspace test suite
cargo test --workspace

# 4. iOS target cross-compilation check
cargo check --target aarch64-apple-ios -p h3d-app

# 5. Dependency security, bans, and license check
cargo deny check
```

Or run all gates in a single command using the build script:

```bash
./build_macos.sh check
```

---

## License

This project is dual-licensed under either:

- **Apache License, Version 2.0** ([LICENSE-APACHE](http://www.apache.org/licenses/LICENSE-2.0))
- **MIT License** ([LICENSE-MIT](http://opensource.org/licenses/MIT))

at your option.

---

## Disclaimer

HISTO3D is an experimental computational pathology research system. It is designed to assist biomedical research and scientific visualization. It has not been cleared or approved by any regulatory authority (such as the FDA or CE-IVD) and is **not intended for diagnostic, prognostic, or treatment decisions in humans or animals**.
