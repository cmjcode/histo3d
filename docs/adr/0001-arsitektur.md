# ADR 0001 — Arsitektur HISTO3D

Status: diterima (2026-10-10). Sumber: histo3d.md bagian 1.2 (K1–K16), hasil H0 di
bawah.

## Keputusan

| No | Keputusan | Status setelah H0 |
|---|---|---|
| K1 | Repo terpisah dari DUCAD | Diterapkan: workspace `HISTO3D/` sendiri; hanya menyalin pola (i18n, tes glyph, kamera, apple.rs, skrip build). |
| K2 | Tanpa OCCT, ITK, VTK, OpenCV | Dijaga `deny.toml` (`[bans].deny`) dan tes `engine_has_no_gui_dependency`. |
| K3 | Rust di semua lapisan, macOS + iPadOS | Toolchain stable 1.96; target `aarch64-apple-ios` dan `-sim` terpasang; CI mengecek iOS tiap PR. |
| K4 | Registrasi ditulis sendiri di balik trait | `h3d-register::Registrar`; backend `sitk` **tidak** dipakai (lihat ADR 0002). |
| K5 | Pekerjaan 3D pada resolusi kerja 8–32 µm | `ReconConfig::s_xy_um` bawaan 16, penurunan otomatis bila melampaui `Budgets`. |
| K6 | Transformasi dalam µm | `Affine2`/`SlideTransform` di `h3d-core`; tidak ada piksel di `project.json`. |
| K7 | Label raster bertile di ruang slide | `LabelStore` 4 µm/px, tile 256, format `.lbl`. |
| K8 | Permukaan dari SDF + surface nets | `fast-surface-nets` 0.2 — lulus kompilasi iOS (H0.6). |
| K9 | Volume Cavalieri utama, mesh pembanding | `h3d-volume::measure`. |
| K10 | `tract` bawaan, `ort` opsional | `tract-onnx` 0.23 — lulus kompilasi iOS (H0.5). `ort` belum ditambahkan (fitur `ort-coreml` dicadangkan). |
| K11 | Tanpa kuas sculpt | Tidak ada kode sculpt. |
| K12 | Deterministik, headless lewat CLI | `h3d-engine` + `h3d-cli`; reduksi paralel digabung berurutan. |
| K13 | Offline total | `deny.toml` melarang `ureq`/`reqwest`/`hyper`; tes arsitektur ikut menolak `tokio`. |
| K14 | Usulan urutan/orientasi dari kemiripan klasik | `h3d-register::suggest_order`. |
| K15 | Data medis tiga level, pengukuran ± ketidakpastian | `ClinicalMeta`/`PreanalyticMeta`/`IhcPanel`/`LabelKind`/`AttrStore`; `spatial` melaporkan `± u`. |
| K16 | `clinical.json` terpisah dan pseudonim | `SampleMeta::clinical` bertanda `#[serde(skip)]` sehingga **secara tipe** tidak bisa masuk `project.json`. |

## Hasil H0 (yang bisa dilakukan tanpa data nyata dan tanpa iPad fisik)

| Tugas | Hasil |
|---|---|
| H0.2 `wsi-rs` | Versi terbaru 0.8.0 menuntut rustc 1.99 (belum stable); dipakai **0.6.0** (rustc 1.96, MIT OR Apache-2.0). Hasil kompilasi iOS dan cakupan format dicatat di laporan crate `h3d-slide`. Belum dibandingkan dengan OpenSlide pada berkas SVS/NDPI nyata (butuh H0.1). |
| H0.3 `sitk` | 0.0.3, pembaruan terakhir 2026-07; terlalu dini. **Tidak dipakai** (ADR 0002). |
| H0.4 Registrasi bawaan | R1–R3 diimplementasikan dan diuji pada fantom berbenih (TRE pada fantom; lihat tes `h3d-register`). TRE pada jaringan nyata menunggu H0.1. |
| H0.5 `tract` | Lulus kompilasi iOS; model uji mungil dibuat oleh tes `h3d-segment`. |
| H0.6 Mesh | `fast-surface-nets` + `baby_shark` lulus kompilasi iOS (lihat laporan `h3d-volume`). |
| H0.7 Memori iPad | **Belum**: butuh perangkat fisik. `Budgets::IPAD` memakai angka [TARGET] dari 9.2. |
| H0.1, H0.8 | **Belum**: butuh sampel nyata dari pemilik proyek dan jawaban bagian 13. |

## Konsekuensi

- Semua angka [TARGET] tetap bertanda target sampai H0.1/H0.7 selesai.
- Bila `wsi-rs` 0.6 gagal pada SVS/NDPI nyata: iPad hanya TIFF piramida + gambar
  biasa; desktop boleh memakai OpenSlide di balik fitur (saat ini dilarang
  `deny.toml`, harus dibuka sadar).
