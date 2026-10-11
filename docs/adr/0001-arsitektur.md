# ADR 0001 — Arsitektur HISTO3D

Status: diterima (2026-10-10; hasil H0 diperbarui 2026-10-11). Sumber: histo3d.md bagian 1.2 (K1–K16), hasil H0 di
bawah.

## Keputusan

| No | Keputusan | Status setelah H0 |
|---|---|---|
| K1 | Repo terpisah dari DUCAD | Diterapkan: workspace `HISTO3D/` sendiri; hanya menyalin pola (i18n, tes glyph, kamera, apple.rs, skrip build). |
| K2 | Tanpa OCCT, ITK, VTK, OpenCV | Dijaga `deny.toml` (`[bans].deny`) dan tes `engine_has_no_gui_dependency`. |
| K3 | Rust di semua lapisan, macOS + iPadOS | Toolchain stable 1.96; target `aarch64-apple-ios` dan `-sim` terpasang; CI mengecek iOS tiap PR. |
| K4 | Registrasi ditulis sendiri di balik trait | `h3d-register::Registrar`; backend `sitk` **tidak** dipakai; model bawaan affine "rigid dulu" (lihat ADR 0002). |
| K5 | Pekerjaan 3D pada resolusi kerja 8–32 µm | `ReconConfig::s_xy_um` bawaan 16, penurunan otomatis bila melampaui `Budgets`. |
| K6 | Transformasi dalam µm | `Affine2`/`SlideTransform` di `h3d-core`; tidak ada piksel di `project.json`. |
| K7 | Label raster bertile di ruang slide | `LabelStore` 4 µm/px, tile 256, format `.lbl`. |
| K8 | Permukaan dari SDF + surface nets | `fast-surface-nets` 0.2 — lulus kompilasi iOS (H0.6); sisi/verteks tak-manifold dari muka kisi "papan catur" diperbaiki di `LabelMesh::repair_non_manifold`. |
| K9 | Volume Cavalieri utama, mesh pembanding | `h3d-volume::measure`. |
| K10 | `tract` bawaan, `ort` opsional | `tract-onnx` 0.23 — lulus kompilasi iOS (H0.5). `ort` belum ditambahkan (fitur `ort-coreml` dicadangkan). |
| K11 | Tanpa kuas sculpt | Tidak ada kode sculpt. |
| K12 | Deterministik, headless lewat CLI | `h3d-engine` + `h3d-cli`; reduksi paralel digabung berurutan. |
| K13 | Offline total | `deny.toml` melarang `ureq`/`reqwest`/`hyper`; tes arsitektur ikut menolak `tokio`. |
| K14 | Usulan urutan/orientasi dari kemiripan klasik | `h3d-register::suggest_order`. |
| K15 | Data medis tiga level, pengukuran ± ketidakpastian | `ClinicalMeta`/`PreanalyticMeta`/`IhcPanel`/`LabelKind`/`AttrStore`; `spatial` melaporkan `± u`. |
| K16 | `clinical.json` terpisah dan pseudonim | `SampleMeta::clinical` bertanda `#[serde(skip)]` sehingga **secara tipe** tidak bisa masuk `project.json`. |

## Hasil H0 (yang bisa dilakukan tanpa data nyata dan tanpa iPad fisik)

Diperbarui 2026-10-11 agar sesuai keadaan kode. "Lulus kompilasi iOS" berarti
`cargo check --target aarch64-apple-ios -p h3d-engine` hijau (gerbang CI);
belum berarti teruji di perangkat.

| Tugas | Hasil |
|---|---|
| H0.2 `wsi-rs` | Dipakai **0.6.0** (MIT OR Apache-2.0; 0.8.0 menuntut rustc 1.99 yang belum stable). **Lulus kompilasi iOS.** Tidak sepenuhnya Rust murni seperti klaimnya: menarik `zstd-sys` (libzstd C, BSD-3-Clause, dikompilasi `cc`) dan lewat `flate2` fitur `zlib` juga `libz-sys` — padahal 6.2 sengaja menghindari `zstd`. Keduanya terkompilasi untuk iOS. Pohonnya 128 crate, termasuk tumpukan `dicom-*` yang membawa `encoding` 0.2.33 (tidak dipelihara, RUSTSEC-2021-0153; diabaikan beralasan di `deny.toml`). `rust-version` crate ini 1.96, jadi MSRV workspace sebenarnya 1.96 walau `Cargo.toml` menulis 1.85. Belum dibandingkan dengan OpenSlide pada berkas SVS/NDPI nyata (butuh H0.1). |
| H0.3 `sitk` | 0.0.3, pembaruan terakhir 2026-07; terlalu dini. **Tidak dipakai** (ADR 0002). |
| H0.4 Registrasi bawaan | R1–R3 diimplementasikan. Model bawaan tetap `Affine`, tetapi "rigid dulu": model bebas hanya diterima per pasangan bila skornya membaik nyata (ADR 0002, "Rigid dulu"). Pada fantom berbenih 24 irisan: TRE rantai rata-rata 12,0 µm / terburuk 19,0 µm pada piksel registrasi 16 µm; volume mesh −0,7 %, voxel −0,8 % terhadap analitik. TRE pada jaringan nyata menunggu H0.1. |
| H0.5 `tract` | `tract-onnx` 0.23.8 **lulus kompilasi iOS**; model uji mungil dibuat oleh tes `h3d-segment`. Belum dijalankan di simulator/perangkat. |
| H0.6 Mesh | `fast-surface-nets` 0.2.1 lulus kompilasi iOS. **`baby_shark` tidak dipakai** (tidak ada di `Cargo.lock`): decimate dan olah mesh ditulis di dalam crate (`h3d-volume::mesh`: QEM Garland–Heckbert deterministik dengan syarat link, Taubin, las verteks, dan perbaikan sisi tak-manifold "potong lalu jahit") agar hasil byte-identik antar jalan dan tanpa dependensi tambahan. |
| H0.7 Memori iPad | **Belum**: butuh perangkat fisik. `Budgets::IPAD` memakai angka [TARGET] dari 9.2. |
| H0.1, H0.8 | **Belum**: butuh sampel nyata dari pemilik proyek dan jawaban bagian 13. |

## Konsekuensi

- Semua angka [TARGET] tetap bertanda target sampai H0.1/H0.7 selesai.
- Bila `wsi-rs` 0.6 gagal pada SVS/NDPI nyata: iPad hanya TIFF piramida + gambar
  biasa (fitur Cargo `wsi` di `h3d-slide` bisa dimatikan; itu juga membuang
  `zstd-sys`, `libz-sys` lewat jalur ini, dan `encoding`); desktop boleh
  memakai OpenSlide di balik fitur (saat ini dilarang `deny.toml`, harus
  dibuka sadar).
- `cargo deny check` hijau dengan `deny.toml` repo (advisories, bans,
  licenses, sources). Yang tersisa hanya peringatan: 18 crate berversi ganda
  (terutama dari pohon GUI/wgpu dan `thiserror` 1 vs 2) dan dua izin lisensi
  yang saat ini tidak terpakai (`BSL-1.0`, `MPL-2.0`).
