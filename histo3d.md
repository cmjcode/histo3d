# HISTO3D — rekonstruksi 3D tumor dari irisan histologi serial

## 0. Cara membaca dokumen ini


- Bagian 1–3: apa yang dibangun, kenapa, dan batasannya.
- Bagian 4–5: alur data dan sistem koordinat. **Baca bagian 5 sebelum menulis
  kode apa pun**; hampir semua bug di perangkat lunak sejenis berasal dari
  koordinat dan arah transformasi yang tertukar.
- Bagian 6–7: pustaka yang dipakai dan susunan crate.
- Bagian 8: rancangan teknis per crate (tipe data, algoritma, format berkas).
  8.12 khusus membahas atribut medis dan analisis spasial (margin, invasi,
  multifokalitas, skor IHC) yang menempel pada objek 3D.
- Bagian 9: rancangan khusus iPad.
- Bagian 10: fase implementasi dengan tugas, perintah validasi, dan kriteria
  selesai.
- Bagian 11–14: pengujian, risiko, pertanyaan terbuka, aturan keras.


## 1. Latar belakang dan keputusan

### 1.1 Masalah

Pemilik proyek punya gambar mikroskop dari banyak irisan (slice) jaringan
kanker. Dari tumpukan irisan itu ia ingin memodelkan **bentuk 3D tumor**.


### 1.2 Keputusan utama

| No | Keputusan | Alasan |
|---|---|---|
| K1 | Aplikasi **terpisah** dari DUCAD, repo sendiri | Lihat 1.2. DUCAD tetap CAD parametrik murni. |
| K2 | **Tanpa OCCT, tanpa ITK, tanpa VTK** | OCCT tidak relevan. VTK di iOS bergantung pada OpenGL ES yang sudah usang dan ingin memiliki jendela render sendiri. ITK adalah build C++ besar tanpa binding Rust resmi. |
| K3 | **Rust di semua lapisan**, target macOS dan iPadOS dari satu codebase | Permintaan pemilik proyek; mengikuti pola DUCAD (egui + wgpu) yang sudah terbukti jalan di iPad. |
| K4 | Registrasi **ditulis sendiri** di balik trait, dengan crate `sitk` sebagai backend opsional | Tidak ada pustaka registrasi Rust yang sudah terbukti. Trait membuat backend bisa diganti tanpa menyentuh sisanya. |
| K5 | Semua pekerjaan 3D dilakukan pada **resolusi kerja rendah** (8–32 µm/piksel), bukan resolusi penuh | Jarak antar irisan (puluhan µm) jauh lebih kasar daripada piksel (0,25 µm). Resolusi penuh di bidang XY tidak menambah informasi 3D dan tidak muat di memori. |
| K6 | Transformasi disimpan dalam **mikrometer**, bukan piksel | Satu transformasi berlaku di semua level piramida. |
| K7 | Label segmentasi disimpan sebagai **raster bertile di ruang slide** | Kuas dan keluaran model AI sama-sama raster; menyimpan di ruang slide membuat label tetap sah saat registrasi diulang. |
| K8 | Permukaan diekstrak dari **medan jarak bertanda (SDF)** dengan surface nets, bukan dari voxel biner | Hasil jauh lebih halus dan memungkinkan interpolasi bentuk antar irisan. |
| K9 | Volume dilaporkan dengan **estimator Cavalieri** (luas × jarak irisan); volume mesh hanya pembanding | Cavalieri adalah metode baku stereologi dan tidak bergantung pada penghalusan mesh. |
| K10 | Inferensi AI lewat `tract` (Rust murni) sebagai bawaan; `ort` + CoreML sebagai fitur opsional | Menjaga jalur Rust murni tetap utuh di iPad; `ort` hanya untuk percepatan. |
| K11 | **Tidak ada kuas sculpt** untuk mengubah bentuk hasil | Mengubah bentuk hasil ukur dengan tangan merusak nilai ilmiahnya. Penghalusan hanya lewat parameter yang tercatat. |
| K12 | Semua langkah **deterministik** dan bisa dijalankan headless lewat CLI | Hasil riset harus bisa diulang; GUI dan CLI memakai mesin yang sama. |
| K13 | Aplikasi **sepenuhnya offline**, tanpa jaringan | Data medis; tidak ada alasan mengirimnya keluar perangkat. |
| K14 | Usulan otomatis **urutan dan orientasi irisan** memakai kemiripan gambar klasik (R1–R2), bukan model terlatih | Tidak butuh data latih, tetap deterministik (K12) dan offline (K13). Hasilnya selalu **usulan**; pengguna yang memutuskan. Lihat 8.4.7. |
| K15 | Data medis disimpan **terstruktur di tiga level** (sampel, irisan, region/voxel), dan analisis spasial 3D menghasilkan **pengukuran dengan ketidakpastian**, bukan rekomendasi | Angka seperti jarak margin dan skor IHC per region baru berguna untuk keputusan bila bisa dihitung dan divisualisasikan di 3D; kalimat rekomendasi akan mengubah status alat menjadi alat diagnosis (2.2). Lihat 8.12. |
| K16 | Data klinis pasien **pseudonim** dan disimpan di berkas terpisah `clinical.json`, tidak pernah ikut ekspor tanpa flag eksplisit | Data klinis dibutuhkan untuk konteks laporan, tetapi bundel proyek sering dibagikan antar peneliti; memisahkannya membuat pembuangan data identitas menjadi satu operasi yang bisa diuji. Lihat 8.12.9. |

---

## 2. Lingkup

### 2.1 Yang dibangun

- Membuka gambar irisan: slide utuh (SVS, NDPI, TIFF piramida) dan gambar
  biasa dari kamera mikroskop (JPEG, PNG, TIFF).
- Mengurutkan irisan dan mengisi jarak antar irisan, dibantu usulan otomatis
  urutan dan orientasi (putar/cermin) irisan (8.4.7).
- Menyejajarkan irisan (registrasi) otomatis, dengan pemeriksaan mutu dan
  koreksi manual.
- Menandai area (segmentasi): kuas, laso, poligon; lalu metode klasik dan
  model AI.
- Mengisi irisan yang tidak ditandai dengan interpolasi bentuk.
- Merekonstruksi volume dan permukaan 3D per label.
- Mengukur: volume, luas permukaan, dimensi.
- Menyimpan data medis terstruktur (klinis, pra-analitik, panel IHC, jenis
  label) dan menghitung atribut per region (skor IHC, densitas inti, TIL).
- Analisis spasial 3D: jarak ke tepi reseksi per arah anatomis, kontak dan
  penetrasi struktur (pembuluh, saraf, kapsul), multifokalitas, kedalaman
  invasi, proksi hipoksia; hasilnya diwarnakan di permukaan mesh (8.12).
- Mencatat asal dan status tinjauan tiap anotasi (provenance).
- Menampilkan 2D (slide + label) dan 3D (mesh + irisan di ruang 3D).
- Mengekspor: mesh (STL/OBJ/GLB), volume label (NIfTI), anotasi (GeoJSON),
  laporan (CSV/Markdown), hasil analisis spasial (JSON), atribut (NIfTI,
  warna verteks GLB).

### 2.2 Yang tidak dibangun

- Diagnosis atau keputusan klinis. Aplikasi ini **alat riset**; semua laporan
  dan layar "Tentang" memuat tulisan "Hanya untuk riset, bukan untuk
  diagnosis". Analisis spasial (8.12) melaporkan angka dan
  ketidakpastiannya; tidak ada aturan yang menghasilkan kalimat tindakan
  ("reseksi ulang", "terapi X").
- Pelatihan model AI. Pengguna membawa model ONNX sendiri.
- Kuas sculpt (K11).
- Sinkronisasi cloud, akun, kolaborasi.
- Viewer slide serba bisa pengganti QuPath. Viewer 2D hanya secukupnya untuk
  registrasi dan segmentasi.
- Android, Windows, Linux pada tahap awal. Rancangan tidak menghalanginya,
  tetapi tidak diuji.

### 2.3 Asumsi tentang data dan pengguna

Belum dikonfirmasi pemilik proyek; lihat bagian 13.

- **[ASUMSI] A1** — Irisan adalah **potongan fisik** jaringan (mikrotom),
  bukan z-stack optik. Akibatnya registrasi wajib ada. Kalau ternyata
  z-stack konfokal, fase H3 dan H8 hampir seluruhnya bisa dilewati.
- **[ASUMSI] A2** — Pewarnaan H&E, mungkin sebagian IHC.
- **[ASUMSI] A3** — 20 sampai 300 irisan per sampel, jarak antar irisan
  5–100 µm.
- **[ASUMSI] A4** — Tujuan: visualisasi dan pengukuran volume untuk riset.
- **[ASUMSI] A5** — Tidak ada foto blockface atau penanda fidusial.

---

## 3. Istilah

| Istilah | Arti |
|---|---|
| Irisan / slide | Satu potongan jaringan tipis di kaca objek, dipindai jadi satu gambar. |
| WSI | *Whole-slide image*: gambar slide utuh, bisa > 100.000 × 100.000 piksel. |
| Piramida | WSI disimpan dalam beberapa level resolusi; level 0 paling tajam. |
| mpp | *Microns per pixel*: ukuran fisik satu piksel level 0, mis. 0,25 µm. |
| Registrasi | Mencari transformasi yang menyejajarkan satu irisan dengan irisan lain. |
| Rigid / affine / non-rigid | Geser+putar / ditambah skala dan geser miring / deformasi lokal. |
| Segmentasi | Menetapkan label (tumor, stroma, nekrosis, …) ke tiap piksel. |
| Label | Bilangan 0–255 per piksel; 0 = latar. |
| Ruang slide | Koordinat milik satu irisan sebelum disejajarkan. |
| Ruang tumpukan | Koordinat bersama setelah semua irisan disejajarkan. |
| Voxel | Elemen volume 3D; di sini anisotropik (Z lebih kasar dari XY). |
| SDF | *Signed distance field*: jarak ke tepi objek, negatif di dalam. |
| Surface nets | Algoritma ekstraksi permukaan dari SDF, saudara marching cubes. |
| Cavalieri | Estimator volume: jumlah (luas penampang × jarak antar irisan). |
| TRE | *Target registration error*: jarak titik acuan setelah registrasi. |
| OD | *Optical density*: −log10 intensitas; ruang tempat pewarna bersifat aditif. |
| IHC | Imunohistokimia: pewarnaan antibodi (Ki-67, ER, HER2, PD-L1, CD8) dengan kromogen, biasanya DAB cokelat. |
| H-score | Skor IHC 0–300: Σ (intensitas 0–3 × persen sel pada intensitas itu). |
| TPS / CPS | *Tumor / Combined Proportion Score*, skor PD-L1 berbasis sel tumor saja / tumor + sel imun. |
| TIL | *Tumor-infiltrating lymphocytes*: limfosit di dalam (intratumoral) atau di sekitar (peritumoral) tumor. |
| Margin | Tepi reseksi: permukaan luar spesimen bedah, biasanya ditandai tinta; jarak tumor ke margin menentukan status bebas/dekat/positif. |
| LVI / PNI | Invasi limfovaskular / perineural: tumor menyentuh atau masuk pembuluh / saraf. |
| Atribut | Besaran skalar per lokasi (persen positif, densitas sel) yang disimpan sejajar label (8.12.5). |
| Provenance | Catatan asal sebuah anotasi: manual, klasik, AI, interpolasi, impor, dan status tinjauan. |

---

## 4. Alur data

```
 gambar irisan (SVS/NDPI/TIFF/JPEG/PNG)
        │  h3d-slide: baca metadata, piramida, tile
        ▼
 [Proyek] daftar irisan + urutan + z_um + mpp
        │  h3d-imgproc: masker jaringan, kanal hematoksilin, turunkan ke 16 µm/px
        ▼
 gambar registrasi (abu-abu f32 + masker), satu per irisan
        │  h3d-register: berpasangan → rantai ke irisan acuan
        ▼
 transformasi T_i per irisan (µm slide → µm tumpukan) + skor QC
        │                                        ┌─ kuas/laso/poligon
        │  h3d-segment ◄──────────────────────────┼─ ambang klasik
        ▼                                        └─ model ONNX
 label raster per irisan (ruang slide, 4 µm/px, bertile)
        │  h3d-volume: resampling ke kisi tumpukan lewat T_i⁻¹
        ▼
 masker per label per irisan (ruang tumpukan, 16 µm/px)
        │  SDF 2D bertanda → interpolasi sepanjang Z
        ▼
 SDF 3D per label (diproses per chunk)
        │  surface nets → las verteks → Taubin → decimate
        ▼
 mesh per label (µm) ──► ukur (Cavalieri, luas, dimensi) ──► laporan
        │
        ▼
 h3d-render (2D + 3D) · h3d-io (STL/OBJ/GLB, NIfTI, GeoJSON, CSV)
```

Produk antara yang **disimpan** di proyek: metadata irisan, transformasi,
label. Produk yang **bisa dibuat ulang** (cache): piramida gambar biasa,
gambar registrasi, volume, mesh.

---

## 5. Sistem koordinat dan satuan

Aturan di bagian ini mengikat semua crate.

### 5.1 Satuan

- Panjang disimpan dalam **mikrometer (µm), `f64`**.
- Sudut dalam **radian** di kode, **derajat** di antarmuka.
- Render memakai **milimeter, `f32`**, relatif terhadap pusat volume
  (konversi di batas `h3d-render`), sumbu **Z ke atas** = arah tumpukan.
  Ini sama dengan konvensi `ducad-render` **[TERVERIFIKASI]**.

### 5.2 Ruang

| Ruang | Satuan | Asal dan arah | Dipakai untuk |
|---|---|---|---|
| S0: piksel level 0 | px | Kiri-atas gambar; x ke kanan, y ke bawah | Membaca tile, GeoJSON |
| Sl: piksel level l | px | Sama; `p_l = p_0 / d_l`, `d_l` = faktor downsample | Membaca tile level l |
| U: slide fisik | µm | Sama; `u = p_0 × mpp` (mpp bisa beda di x dan y) | Label, transformasi |
| W: tumpukan | µm | Ruang U milik irisan acuan | Volume, mesh |
| G: kisi volume | indeks | `w = origin + (i + 0,5) × s_xy`; `z = z_k` | Voxel |
| R: render | mm | `r = (w − pusat) / 1000`, y dibalik agar tangan-kanan | wgpu |

Pusat piksel berada di `+0,5`. Semua konversi indeks ↔ koordinat memakai
konvensi ini, tanpa kecuali.

### 5.3 Arah transformasi

Untuk irisan `i`, **`T_i` memetakan ruang U irisan itu ke ruang W**:

```
w = T_i(u)        (maju: slide → tumpukan)
u = T_i⁻¹(w)      (mundur: dipakai saat resampling)
```

Resampling selalu **mundur**: untuk tiap piksel tujuan di W, hitung asalnya di
U lalu ambil nilainya. Tidak pernah "menyebar" piksel sumber ke tujuan.

Registrasi berpasangan menghasilkan `P_{m→f}` yang memetakan **irisan bergerak
(moving) ke irisan tetap (fixed)**, keduanya di ruang U masing-masing:

```
u_f = P_{m→f}(u_m)
```

Komposisi rantai menuju irisan acuan `r` (dengan `T_r = identitas`):

```
T_i = T_{tetangga(i)} ∘ P_{i→tetangga(i)}
```

`tetangga(i)` adalah irisan sah terdekat ke arah `r`.

### 5.4 Bentuk affine

```rust
/// [ a  b  tx ]
/// [ c  d  ty ]   memetakan (x, y) → (a·x + b·y + tx, c·x + d·y + ty), satuan µm.
pub struct Affine2 { pub m: [f64; 6] } // urutan: a, b, tx, c, d, ty
```

- Determinan negatif **sah**: irisan sering terpasang terbalik (cermin).
- `Affine2` wajib punya `compose`, `inverse` (mengembalikan `Option`;
  determinan mendekati nol → `None`), `apply`, `determinant`,
  `decompose` (rotasi, skala, shear, cermin; untuk tampilan dan QC).

### 5.5 Sumbu Z

- Tiap irisan punya `z_um` eksplisit. Bawaan: `urutan × jarak_irisan`.
- Irisan hilang atau rusak **tidak** menggeser `z_um` irisan lain.
- Ketebalan irisan (mis. 4 µm) dicatat sebagai metadata tetapi tidak dipakai
  di geometri; yang menentukan geometri adalah **jarak** antar irisan.

---

## 6. Teknologi

Versi yang dicantumkan mengikuti workspace DUCAD **[TERVERIFIKASI]** agar pola
integrasi egui/wgpu/iOS-nya bisa disalin apa adanya. Crate tanpa versi: pakai
rilis terbaru saat H0, lalu kunci di `Cargo.lock`.

### 6.1 Daftar pustaka

| Peran | Crate | Lisensi | Rust murni | Status |
|---|---|---|---|---|
| GUI | `eframe` 0.36, `egui` 0.36 (fitur `wgpu`, **tanpa** `glow`) | MIT/Apache | Ya | [TERVERIFIKASI] jalan di iPad pada DUCAD |
| Render | `egui-wgpu` 0.36 dan `wgpu` yang di-re-export-nya | MIT/Apache | Ya | [TERVERIFIKASI] |
| Ikon | `egui_icons` | MIT/Apache | Ya | [TERVERIFIKASI] dipakai DUCAD |
| Matematika | `glam` 0.33 | MIT/Apache | Ya | [TERVERIFIKASI] |
| Larik N-dimensi | `ndarray` | MIT/Apache | Ya | Mapan |
| Paralel | `rayon` | MIT/Apache | Ya | Mapan; jalan di iOS |
| Gambar biasa | `image` 0.25 | MIT/Apache | Ya | [TERVERIFIKASI] dipakai DUCAD |
| TIFF / BigTIFF | `tiff` | MIT | Ya | Mapan; dukungan TIFF piramida bertile perlu diuji di H0 |
| WSI (SVS, NDPI, …) | `wsi-rs` 0.6.0 | MIT/Apache | **Tidak sepenuhnya**: menarik `zstd-sys` dan `libz-sys` (C) | Lulus kompilasi iOS (ADR 0001); belum diuji pada SVS/NDPI nyata |
| FFT | `rustfft` | MIT/Apache | Ya | Mapan |
| Registrasi (backend opsional) | `sitk` | perlu dicek | Ya (klaim) | **[BELUM DIVERIFIKASI]** |
| Inferensi ONNX | `tract-onnx` | MIT/Apache | Ya | Mapan; cakupan operator per model perlu diuji |
| Inferensi ONNX dipercepat | `ort` (fitur `coreml`) | MIT/Apache | Tidak (ONNX Runtime C++) | Opsional; build iOS perlu `ORT_LIB_PROFILE` |
| Ekstraksi permukaan | `fast-surface-nets` (+ `ndshape`) | MIT/Apache | Ya | Mapan; API perlu dicek saat H5 |
| Olah mesh | di dalam crate (`h3d-volume::mesh`), menggantikan `baby_shark` | — | Ya | Decimate QEM deterministik + perbaikan tak-manifold (ADR 0001) |
| BVH, picking | `parry3d` | Apache-2.0 | Ya | Mapan |
| Kompresi tile label | `lz4_flex` | MIT | Ya | Mapan |
| Ekspor NIfTI | `nifti` | MIT/Apache | Ya | Mapan |
| Serialisasi | `serde`, `serde_json` | MIT/Apache | Ya | [TERVERIFIKASI] |
| Error | `thiserror` 2 (pustaka), `anyhow` (hanya CLI/app) | MIT/Apache | Ya | [TERVERIFIKASI] |
| Log | `log` 0.4 | MIT/Apache | Ya | [TERVERIFIKASI] |
| Terjemahan | `fluent-bundle`, `unic-langid` | MIT/Apache | Ya | [TERVERIFIKASI] |
| Dialog berkas (desktop) | `rfd` 0.17 | MIT | Ya | [TERVERIFIKASI] |
| Apple | `objc2` 0.6, `objc2-foundation`, `objc2-ui-kit` | MIT | Ya | [TERVERIFIKASI] |
| Uji | `proptest`, `criterion` | MIT/Apache | Ya | [TERVERIFIKASI] |

Satu-satunya komponen non-Rust yang **direncanakan** adalah `ort`, dan itu di
balik fitur Cargo yang mati secara bawaan. Kenyataan setelah H0: `wsi-rs` 0.6
membawa `zstd-sys` (libzstd) dan `libz-sys` (zlib) yang dikompilasi dari C;
keduanya hilang bila fitur `wsi` di `h3d-slide` dimatikan.

### 6.2 Yang sengaja tidak dipakai

| Pustaka | Alasan |
|---|---|
| OCCT / `opencascade` | Tidak relevan; kompilasi 10–15 menit. |
| ITK, SimpleITK asli, elastix | C++ besar, tanpa binding Rust resmi. Dipakai **hanya** sebagai pembanding di H0 (lewat Python di desktop). |
| VTK / `vtk-rs` | iOS bergantung pada OpenGL ES; `vtk-rs` menyatakan dirinya belum siap pakai. |
| OpenSlide (`openslide-rs`) | LGPL-2.1 + pustaka C; tautan statis di iOS bermasalah. Cadangan **desktop saja** bila `wsi-rs` gagal. |
| OpenCV | Dependensi C++ besar; yang dibutuhkan darinya (ECC, korelasi fase) cukup kecil untuk ditulis sendiri. |
| `zstd` | Pembungkus pustaka C; `lz4_flex` cukup. Tidak dipakai langsung, tetapi ikut masuk secara transitif lewat `wsi-rs` (ADR 0001). |

### 6.3 Sumber

- `wsi-rs`: https://github.com/frames-sg/wsi-rs
- `sitk`: https://docs.rs/sitk/latest/sitk/
- `vtk-rs` (ditolak): https://github.com/jonaspleyer/vtk-rs
- `baby_shark`: https://github.com/dima634/baby_shark
- `ort`, penautan: https://ort.pyke.io/setup/linking
- VTK di perangkat bergerak: https://www.kitware.com/building-vtk-for-mobile-architectures/

---

## 7. Arsitektur workspace

### 7.1 Tata letak

```
HISTO3D/
├── CLAUDE.md                  # panduan agent (salin aturan bagian 14)
├── Cargo.toml                 # workspace + [workspace.dependencies]
├── deny.toml                  # salin dari DUCAD
├── rust-toolchain.toml
├── build_ipad.sh              # adaptasi dari DUCAD
├── build_macos.sh
├── docs/adr/                  # keputusan arsitektur
├── testdata/                  # fantom sintetis kecil (dibuat oleh tes)
└── crates/
    ├── h3d-core/
    ├── h3d-slide/
    ├── h3d-imgproc/
    ├── h3d-register/
    ├── h3d-segment/
    ├── h3d-volume/
    ├── h3d-io/
    ├── h3d-engine/
    ├── h3d-cli/
    ├── h3d-render/
    ├── h3d-ui/
    ├── h3d-i18n/
    └── h3d-app/
```

### 7.2 Peta crate

Panah = "bergantung pada".

| Crate | Isi | Bergantung pada |
|---|---|---|
| `h3d-core` | Tipe dasar: id, `Affine2`, `SlideTransform`, model `Project`, tabel label, tumpukan undo, trait `Progress` | — |
| `h3d-slide` | Trait `SlideSource`, backend WSI/TIFF/gambar biasa, cache tile CPU | core |
| `h3d-imgproc` | Dekonvolusi warna, masker jaringan, piramida, transformasi jarak, morfologi, komponen terhubung, resampling | core |
| `h3d-register` | Trait `Registrar`, backend bawaan, backend `sitk` (fitur), rantai, QC, landmark manual | core, imgproc |
| `h3d-segment` | `LabelStore` bertile, operasi kuas/laso, ambang klasik, trait `Segmenter` + backend ONNX, interpolasi bentuk | core, imgproc |
| `h3d-volume` | Kisi tumpukan, SDF 3D per chunk, surface nets, olah mesh, pengukuran | core, imgproc |
| `h3d-io` | Format proyek, berkas label, ekspor mesh/NIfTI/GeoJSON/laporan | core, slide, segment, volume |
| `h3d-engine` | Orkestrasi headless: langkah pipeline, sistem job (progres, batal) | semua di atas |
| `h3d-cli` | Binary `h3d` | engine |
| `h3d-render` | Pipeline wgpu: tile 2D, overlay label, pembanding registrasi, mesh 3D, irisan di 3D | core |
| `h3d-i18n` | Terjemahan Fluent (id, en) | — |
| `h3d-ui` | Widget egui, cakupan glyph | core, i18n |
| `h3d-app` | Binary GUI, macOS + iPadOS | semua |

### 7.3 Aturan ketergantungan

- `h3d-engine` dan semua crate di bawahnya **tidak boleh** bergantung pada
  `egui`, `eframe`, `wgpu`, `h3d-render`, `h3d-ui`, `rfd`. Dijaga tes
  `engine_has_no_gui_dependency` (membaca `cargo metadata`), meniru tes
  bernama sama di DUCAD.
- `h3d-render` hanya menerima data siap gambar (`[f32; 3]`, tile RGBA,
  indeks); tidak tahu apa itu proyek atau registrasi.
- Tidak ada crate yang bergantung pada `ducad-kernel` atau `opencascade`,
  langsung maupun tidak langsung.

### 7.4 Yang dipakai ulang dari DUCAD

Lokasi relatif terhadap `DUCAD/ducad-editor/` **[TERVERIFIKASI]**.

| Bagian DUCAD | Ketergantungannya | Cara pakai |
|---|---|---|
| `crates/ducad-i18n` | Hanya `fluent-*`, `serde`, `log` | **Salin** jadi `h3d-i18n`; ganti berkas terjemahan. |
| `crates/ducad-ui` | Menarik `ducad-kernel` (OCCT) | **Jangan dijadikan dependensi.** Salin widget yang dibutuhkan dan tes `tests/ui_glyph_coverage.rs` ke `h3d-ui`. |
| `crates/ducad-render` | `ducad-sketch`, `lyon`, `kurbo` | **Jangan dijadikan dependensi.** Salin `camera.rs` (`OrbitCamera`) dan pola re-export `pub use egui_wgpu::wgpu;` agar seluruh workspace memakai satu versi wgpu. |
| `crates/ducad-ink` | `ducad-core`, `ducad-sketch` | Acuan pola: goresan bertekanan, laso (`lasso.rs`), command undo (`commands.rs`). Tulis ulang yang perlu; jangan tarik `ducad-sketch`. |
| `crates/ducad-app/src/input/sketch.rs` sekitar baris 1943 | — | Pola membaca tekanan Pencil: `egui::Event::Touch { force, .. }`. |
| `crates/ducad-app/src/apple.rs` | `objc2` | Salin: folder `Documents` dan `Application Support` tersandbox. |
| `crates/ducad-app/src/import_worker.rs` | — | Acuan pola pekerja latar + kanal hasil. |
| `build_ipad.sh`, `DUCAD.xcodeproj`, `apple/` | — | Adaptasi: aksi `device`, `sim`, `run-sim`, `archive`, `ipa`. |
| `Cargo.toml` (blok `eframe`) | — | Salin daftar fitur `eframe` **tanpa `glow`**; `glutin` tidak bisa dikompilasi untuk iOS. |
| `deny.toml` | — | Salin daftar lisensi yang diizinkan. |
| `crates/ducad-chat` | `ureq`, `serde` | **Tidak dipakai** (K13: offline). |

---

## 8. Rancangan per crate

Potongan kode di bawah adalah **kontrak rancangan**, bukan kode final. Nama
dan bentuk tipe boleh disesuaikan asal semantiknya tetap.

### 8.1 `h3d-core`

```rust
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SlideId(pub u32);

/// 0 = latar. 1..=255 = label pengguna.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LabelId(pub u8);

pub struct LabelDef {
    pub id: LabelId,
    pub name: String,          // "Tumor", "Stroma", "Nekrosis"
    pub color: [u8; 4],        // RGBA
    pub visible: bool,
    pub reconstruct: bool,     // ikut dibuat mesh-nya
    pub kind: LabelKind,       // Tumor, Margin, Vessel, … (8.12.4)
    pub group: Option<String>, // agregasi di laporan (8.12.4)
}

pub enum SlideTransform {
    Identity,
    Affine(Affine2),
    /// Affine global diikuti perpindahan lokal B-spline (fase H8).
    AffineBSpline { affine: Affine2, grid: DisplacementGrid },
}

/// Kisi titik kendali B-spline kubik di ruang W. Perpindahan dalam µm.
pub struct DisplacementGrid {
    pub origin_um: [f64; 2],
    pub spacing_um: [f64; 2],
    pub size: [u32; 2],
    pub dx: Vec<f32>,
    pub dy: Vec<f32>,
}

pub enum SlideStatus { Ok, Excluded, Damaged }

pub struct RegRecord {
    pub fixed: SlideId,          // irisan tempat ia didaftarkan
    pub method: String,          // "builtin-ecc-affine", "manual-landmarks", …
    pub pairwise: Affine2,       // P_{i→fixed}
    pub ncc: f32,                // metrik setelah registrasi
    pub mask_dice: f32,
    pub area_ratio: f32,         // |det| bagian linear
    pub flagged: bool,           // gagal ambang QC
    pub manual: bool,
}

pub struct SlideEntry {
    pub id: SlideId,
    pub source: SlideSourceRef,  // lihat 8.2
    pub order: u32,
    pub order_source: OrderSource, // FileName, Manual, Suggested (8.4.7)
    pub orient_hint: Option<Affine2>, // usulan orientasi yang diterima (8.4.7)
    pub z_um: f64,
    pub mpp: [f64; 2],           // WAJIB terisi; tidak boleh ditebak
    pub size_l0: [u32; 2],
    pub stain: Stain,            // He, Ihc, Other
    pub status: SlideStatus,
    pub transform: SlideTransform, // T_i (U → W)
    pub reg: Option<RegRecord>,
    pub preanalytic: PreanalyticMeta, // blok, ketebalan, fiksasi (8.12.3)
    pub ihc: Option<IhcPanel>,     // antibodi, klon, skema skor; wajib bila Ihc (8.12.3)
}

pub struct Project {
    pub format_version: u32,
    pub sample: SampleMeta,      // nama, catatan, susut, orientasi, klinis (8.6.6, 8.12.2)
    pub slides: Vec<SlideEntry>,
    pub reference: Option<SlideId>,
    pub labels: Vec<LabelDef>,
    pub attrs: Vec<AttrDef>,     // atribut skalar (8.12.5)
    pub annotations: Vec<AnnotationRecord>, // provenance per (irisan, label) (8.12.7)
    pub recon: ReconConfig,      // s_xy, s_z maksimum, penghalusan, decimate
    pub spatial: SpatialConfig,  // margin_close_um, peri_band_um, hypoxia_um, … (8.12.6)
}
```

**Undo.** Pola command seperti `ducad-ink/src/commands.rs`: tiap perubahan
adalah objek dengan `apply` dan `revert`. Untuk suntingan label, command
menyimpan **salinan terkompresi tile yang berubah** (sebelum dan sesudah),
bukan seluruh label.

**Progres dan pembatalan.**

```rust
pub trait Progress: Sync {
    fn set(&self, fraction: f32, stage: &str);
    fn is_cancelled(&self) -> bool;
}
```

Semua fungsi yang bisa berjalan lebih dari ~100 ms menerima `&dyn Progress`
dan memeriksa `is_cancelled()` minimal sekali per irisan atau per chunk.

**Error.** Tiap crate pustaka punya satu enum error `thiserror`. Pesan yang
tampil ke pengguna berbahasa Indonesia dan lewat `h3d-i18n`; `Display` enum
berisi teks teknis untuk log.

### 8.2 `h3d-slide`

```rust
pub struct LevelInfo { pub size: [u32; 2], pub downsample: f64 }

pub struct SlideInfo {
    pub levels: Vec<LevelInfo>,      // indeks 0 = resolusi penuh
    pub mpp: Option<[f64; 2]>,       // None bila metadata tidak ada
    pub tile_size: u32,
    pub vendor: Option<String>,
}

pub trait SlideSource: Send + Sync {
    fn info(&self) -> &SlideInfo;
    /// Koordinat (x, y) dalam piksel LEVEL yang diminta. Area di luar gambar
    /// diisi putih (latar slide), bukan error.
    fn read_region(&self, level: u32, x: i64, y: i64, w: u32, h: u32)
        -> Result<RgbaImage, SlideError>;
}

pub fn open_slide(path: &Path) -> Result<Box<dyn SlideSource>, SlideError>;
```

Backend, dicoba berurutan oleh `open_slide`:

1. `WsiBackend` — `wsi-rs` (SVS, NDPI, dan lainnya). **[BELUM DIVERIFIKASI]**
2. `TiffBackend` — crate `tiff`: TIFF piramida bertile dan TIFF datar.
3. `FlatBackend` — crate `image`: JPEG/PNG. Membangun piramida sendiri
   dengan penurunan 2× (rata-rata kotak) sampai sisi terpanjang ≤ 512 px,
   lalu menyimpannya di cache proyek.

**mpp.** Bila `info().mpp` kosong (umum pada foto kamera mikroskop), impor
**berhenti dan meminta pengguna mengisi** mpp. Tidak ada nilai tebakan,
karena mpp yang salah merusak semua ukuran. Antarmuka menyediakan alat bantu:
ukur garis skala di gambar lalu masukkan panjang sebenarnya.

**`SlideSourceRef`.**

```rust
pub enum SlideSourceRef {
    /// Berkas di luar proyek (desktop). Path relatif terhadap folder proyek
    /// bila memungkinkan.
    External { path: PathBuf, size_bytes: u64, fingerprint: u64 },
    /// Berkas disalin ke dalam bundel proyek (wajib di iPad).
    Embedded { rel_path: PathBuf },
}
```

`fingerprint` adalah hash dari ukuran berkas + 1 MB pertama + 1 MB terakhir,
untuk mendeteksi berkas tertukar tanpa membaca gigabyte.

**Cache tile CPU.**

- Kunci: `(SlideId, level, tx, ty)`; nilai: RGBA8 `tile_size²`.
- LRU berbatas **byte** (anggaran di bagian 9.2).
- Permintaan masuk antrean berprioritas (tile di tengah layar lebih dulu);
  2–4 thread pekerja mendekode.
- Tiap permintaan membawa nomor generasi viewport; pekerja membuang
  permintaan yang generasinya sudah lewat.

### 8.3 `h3d-imgproc`

Semua fungsi murni, bekerja pada `ndarray` atau irisan `&[f32]`/`&[u8]`
dengan dimensi eksplisit. Tanpa I/O.

**Dekonvolusi warna (Ruifrok & Johnston).**

1. `OD_c = −log10((I_c + 1) / 256)` untuk tiap kanal R, G, B.
2. Vektor pewarna (dinormalkan ke panjang 1 sebelum dipakai):
   - Hematoksilin: `[0,65, 0,70, 0,29]`
   - Eosin: `[0,07, 0,99, 0,11]`
   - DAB: `[0,27, 0,57, 0,78]`
3. Konsentrasi = `OD × M⁻¹`, `M` = matriks 3×3 dari ketiga vektor.

Kanal **hematoksilin** dipakai sebagai gambar registrasi karena inti sel
tampak di H&E maupun IHC, sehingga tahan terhadap beda pewarnaan.

**Masker jaringan.**

1. Hitung `OD_total = OD_R + OD_G + OD_B` pada resolusi rendah.
2. Ambang Otsu.
3. Penutupan morfologi (radius ~50 µm), isi lubang, buang komponen yang
   luasnya < 0,5% dari komponen terbesar.

**Piramida.** Penurunan 2× dengan rata-rata kotak; untuk label memakai
**modus** (label terbanyak), bukan rata-rata.

**Transformasi jarak Euclid eksak (Felzenszwalb & Huttenlocher).** Dua
lintasan 1D (kolom lalu baris) dengan amplop bawah parabola; O(n) per piksel.
Mengembalikan jarak **kuadrat**; ambil akar lalu kalikan `s_xy` untuk µm.

**SDF bertanda dari masker.**
`sdf = EDT(luar) − EDT(dalam)`, dikoreksi setengah piksel, **negatif di
dalam**. Dijepit ke `±band` (bawaan 8 piksel) untuk menghemat presisi.

**Morfologi.** Dilasi dan erosi cakram lewat ambang EDT (lebih cepat dan
isotropik dibanding elemen struktur).

**Komponen terhubung.** Dua lintasan dengan union-find, konektivitas 8.

**Resampling.** `warp_affine(src, T_inv, out_size, interp)` dengan `interp` ∈
{`Nearest` (label), `Bilinear` (gambar)}. Memakai aturan pusat piksel `+0,5`
dari bagian 5.2.

### 8.4 `h3d-register`

#### 8.4.1 Kontrak

```rust
pub struct RegImage {
    pub gray: Array2<f32>,     // kanal hematoksilin, dinormalkan 0..1
    pub mask: Array2<u8>,      // 1 = jaringan
    pub mpp: f64,              // µm per piksel gambar ini (isotropik)
    pub origin_um: [f64; 2],   // posisi piksel (0,0) di ruang U
}

pub enum RegModel { Rigid, Similarity, Affine }
pub enum RegMetric { Ncc, MutualInformation }

pub struct RegOptions {
    pub model: RegModel,             // bawaan Affine ("rigid dulu", lihat R3)
    pub metric: RegMetric,           // bawaan Ncc
    pub allow_flip: bool,            // bawaan true
    pub levels_um: Vec<f64>,         // bawaan [64.0, 32.0, 16.0]
    pub max_iterations: u32,         // per level, bawaan 100
    pub convergence_eps: f64,        // bawaan 1e-5
}

pub struct RegResult {
    pub transform: Affine2,          // P_{moving→fixed}, µm → µm
    pub ncc: f32,
    pub mask_dice: f32,
    pub area_ratio: f32,
    pub iterations: u32,
}

pub trait Registrar: Send + Sync {
    fn name(&self) -> &'static str;
    fn register(&self, fixed: &RegImage, moving: &RegImage,
                init: Option<Affine2>, opts: &RegOptions,
                progress: &dyn Progress) -> Result<RegResult, RegError>;
}
```

#### 8.4.2 Backend bawaan (`BuiltinRegistrar`)

**Langkah R1 — kandidat awal dari momen.**

1. Hitung pusat massa dan momen orde dua tiap masker; sumbu utama memberi
   sudut `θ_f` dan `θ_m`.
2. Kandidat rotasi: `θ_f − θ_m` dan `θ_f − θ_m + 180°`.
3. Bila `allow_flip`: gandakan dengan versi cermin → 4 kandidat.
4. Bila jaringan nyaris bundar (rasio nilai eigen < 1,2), sumbu utama tidak
   bisa dipercaya: ganti dengan sapuan tiap 10° pada resolusi 64 µm
   (36 sudut × 2 cermin).

**Langkah R2 — translasi dengan korelasi fase.** Untuk tiap kandidat:

1. Terapkan rotasi/cermin pada gambar bergerak di resolusi 64 µm.
2. Kalikan kedua gambar dengan jendela Hann.
3. `R = F_f · conj(F_m) / |F_f · conj(F_m)|`; puncak `IFFT(R)` adalah
   geseran. Sub-piksel dengan pencocokan parabola di sekitar puncak.
4. Skor kandidat = NCC bermasker setelah geseran. Pilih yang terbaik.

**Langkah R3 — penghalusan dengan ECC.** Algoritma *enhanced correlation
coefficient* (Evangelidis & Psarakis, 2008), versi *forward additive*; acuan
implementasinya `findTransformECC` di OpenCV.

- Berjenjang: 64 → 32 → 16 µm; hasil tiap level jadi awal level berikutnya.
- Parameter dioptimasi dalam koordinat **ternormalisasi** (asal di pusat
  gambar tetap, skala = setengah diagonal) agar rotasi dan translasi
  sebanding besarnya.
- Hanya piksel dengan masker 1 pada **kedua** gambar yang dihitung.
- Berhenti bila perubahan parameter < `convergence_eps` atau iterasi habis.
- Bila koefisien korelasi turun dibanding awal → kembalikan hasil R2.
- **Rigid dulu.** Dengan metrik NCC, model `Similarity`/`Affine` dijalankan
  setelah penghalusan rigid (mulai dari hasil rigid) dan hanya diterima bila
  skor = NCC bermasker × Dice masker naik ≥ 0,01 **dan** ≥ 25 % dari
  `1 − skor_rigid`; bila tidak, hasil rigid yang dipakai. Tanpa aturan ini
  perubahan bentuk penampang antar irisan diserap sebagai skala yang menumpuk
  di sepanjang rantai (fantom: `|det T_i|` sampai 1,11, volume mesh +2,8 %).
  Alasan, pengukuran, dan batasannya ada di `docs/adr/0002-registrasi.md`.

**Beda pewarnaan.** Bila `metric = MutualInformation`: R1 dan R2 sama, R3
diganti Nelder–Mead atas informasi mutual Mattes (32 bin histogram).

**Keluaran.** `P` dikonversi dari piksel-registrasi ke µm memakai
`mpp` dan `origin_um` kedua gambar, sehingga bebas dari level piramida.

#### 8.4.3 Backend `sitk` (fitur Cargo `sitk`)

Implementasi `Registrar` di atas crate `sitk`. **[BELUM DIVERIFIKASI]** —
hanya diaktifkan bila uji H0 menunjukkan hasilnya setara SimpleITK asli.

#### 8.4.4 Rantai

```rust
pub fn register_stack(slides: &[RegInput], reference: SlideId,
                      registrar: &dyn Registrar, opts: &RegOptions,
                      progress: &dyn Progress) -> Result<Vec<RegRecord>, RegError>;
```

1. Irisan acuan `r`: pilihan pengguna, bawaan irisan `Ok` di tengah
   tumpukan. Memilih tengah memperpendek rantai sehingga galat menumpuk
   separuhnya.
2. Dari `r` berjalan ke dua arah. Tiap irisan didaftarkan ke irisan `Ok`
   terdekat ke arah `r`.
3. Pasangan **independen** → dijalankan paralel dengan `rayon`; komposisi
   `T_i` dilakukan sesudahnya secara berurutan.
4. Irisan `Excluded`/`Damaged` dilompati.

#### 8.4.5 Pemeriksaan mutu (QC)

Pasangan diberi tanda `flagged` bila salah satu terpenuhi **[TARGET]**:

- `ncc < 0,5`
- `mask_dice < 0,8`
- `area_ratio` di luar `0,85–1,15`

Irisan bertanda tampil di daftar "perlu diperiksa" dan pengguna bisa:

- **Landmark manual**: menandai ≥ 3 pasang titik → affine kuadrat terkecil;
  2 pasang → similarity.
- **Geser/putar manual** dengan pegangan di layar.
- **Tandai rusak** agar dilompati.

Setelah satu irisan dikoreksi, `T` semua irisan **setelahnya di rantai**
dihitung ulang dari `pairwise` yang tersimpan, tanpa registrasi ulang.

**Sisa galat terukur.** Tiap `RegRecord` juga menyimpan dua angka yang diukur
setelah penjajaran dari geseran sisa blok lokal 32 px (korelasi
ternormalisasi ± 4 px, sub-piksel parabola) pada resolusi registrasi:
`residual_um` (median besar geseran; galat lokal) dan `residual_fit_um` (RMS
medan gerak rigid yang dicocokkan ke geseran itu; galat transformasi, yang
menumpuk di rantai). Keduanya opsional di `project.json` (proyek lama tetap
terbuka) dan menjadi dasar `TRE_est` di 8.12.6. Laporan mencatat bila skala
luas kumulatif rantai `|det T_i|` menyimpang > 5 % dari 1.

#### 8.4.6 Batasan yang harus ditulis di dokumentasi pengguna

Registrasi berpasangan tanpa acuan luar cenderung **meluruskan** struktur
yang sebenarnya melengkung atau miring terhadap sumbu potong (dikenal sebagai
masalah "pisang" atau *z-shift*). Tidak ada algoritma yang bisa
menghilangkannya tanpa foto blockface atau penanda fidusial. Akibatnya:
volume dan luas penampang cukup andal, tetapi **kelengkungan global** bentuk
tidak bisa dijamin.

#### 8.4.7 Usulan urutan dan orientasi irisan

Membantu saat nama berkas tidak mencerminkan urutan potong, atau saat irisan
terpasang terputar/terbalik di kaca objek. Memakai kemiripan gambar klasik
(K14); tidak ada model yang perlu dibawa pengguna.

```rust
pub struct OrderSuggestion {
    pub order: Vec<SlideId>,          // urutan usulan
    pub orient: Vec<(SlideId, Affine2)>, // rigid + cermin, U irisan → U irisan acuan
    pub confidence: Vec<(SlideId, f32)>, // 0..1 per irisan
    pub uncertain: Vec<SlideId>,      // keyakinan di bawah ambang
}

pub fn suggest_order(slides: &[RegInput], current: &[SlideId],
                     progress: &dyn Progress) -> Result<OrderSuggestion, RegError>;
```

**Langkah U1 — matriks kemiripan.** Untuk tiap pasangan irisan `Ok`,
jalankan R1 dan R2 saja (tanpa ECC) pada resolusi 64 µm; simpan skor NCC
bermasker `S[i][j]` dan transformasi rigid+cermin terbaiknya. Paralel dengan
`rayon`. Bila jumlah irisan > 150, pasangan disaring dulu dengan deskriptor
bentuk masker (luas, rasio nilai eigen momen) dan hanya 20 kandidat terdekat
per irisan yang dihitung penuh **[TARGET]**.

**Langkah U2 — urutan.** Jarak `d = 1 − S`. Urutan usulan = lintasan
Hamilton terbuka berbiaya minimum: awal dari tetangga-terdekat rakus (dicoba
dari tiap irisan sebagai titik awal), lalu diperbaiki dengan 2-opt sampai
tidak ada perbaikan. Seri dipecah dengan `SlideId` agar deterministik.

**Langkah U3 — arah.** Gambar tidak bisa membedakan urutan naik dari urutan
turun. Arah dipilih yang paling sedikit inversinya terhadap urutan saat ini;
antarmuka menyediakan tombol "Balik urutan".

**Langkah U4 — orientasi.** Sepanjang urutan usulan, transformasi rigid+cermin
berpasangan dari U1 dikomposisikan menuju irisan acuan (aturan 5.3). Hasilnya
`orient` per irisan: sudut putar dan status cermin.

**Langkah U5 — keyakinan.** Per irisan: selisih antara skor terhadap tetangga
terpilih dan skor alternatif terbaik berikutnya, dinormalkan ke 0..1. Di
bawah ambang (bawaan 0,1 **[TARGET]**) → masuk `uncertain`.

**Saat diterima.**

- Hanya irisan yang posisinya berubah yang ditampilkan sebagai perubahan;
  pengguna bisa menerima semua, sebagian, atau menolak.
- `order` diperbarui dan `order_source = Suggested`. `z_um` dihitung ulang
  **hanya** untuk irisan yang `z_um`-nya masih nilai bawaan; `z_um` yang
  diisi tangan tidak disentuh dan pengguna diperingatkan bila jadi tidak
  monoton.
- `orient_hint` dipakai sebagai `init` registrasi berpasangan
  (`init = orient_fixed⁻¹ ∘ orient_moving`) dan untuk menampilkan thumbnail
  searah di mode `Slides`. Ia **bukan** `T_i`; geometri tetap ditentukan
  registrasi.
- `RegRecord` irisan yang tetangganya berubah dibuang dan ditandai perlu
  registrasi ulang. Label tidak terpengaruh karena berada di ruang slide (K7).
- Laporan mencatat bahwa urutan berasal dari usulan otomatis.

**Batasan yang harus ditulis di dokumentasi pengguna.**

- Arah tumpukan (atas–bawah) tidak bisa ditentukan dari gambar.
- Irisan berdekatan yang nyaris sama bisa tertukar posisinya; pengaruhnya ke
  volume kecil, tetapi urutan pasti tetap harus dari catatan lab.
- Jaringan nyaris bundar tanpa struktur dalam membuat sudut putar tidak andal.
- Irisan rusak atau terlipat memberi kemiripan rendah ke semua irisan dan
  cenderung terlempar ke ujung urutan; ia masuk `uncertain`.

Urutan dari nama berkas atau catatan lab tetap sumber utama. Fitur ini paling
berguna sebagai **pemeriksa**: menyorot irisan yang posisinya tampak janggal.

**Pengembangan kemudian (opsional).** Mengganti NCC dengan jarak antar
embedding dari model ONNX yang dibawa pengguna, lewat jalur inferensi 8.5.4.
Hanya dikerjakan bila kemiripan klasik terbukti kurang pada data nyata.

### 8.5 `h3d-segment`

#### 8.5.1 `LabelStore`

Satu per irisan, di **ruang slide (U)**.

```rust
pub struct LabelStore {
    pub mpp_label: f64,                      // bawaan 4.0 µm
    pub size_px: [u32; 2],                   // ukuran slide pada mpp_label
    pub tile_size: u32,                      // 256
    tiles: HashMap<(u32, u32), Box<[u8]>>,   // hanya tile tak kosong
}
```

- Tile yang seluruhnya 0 tidak disimpan.
- Operasi dasar: `get(x, y)`, `paint_disc(center_um, radius_um, label, mode)`,
  `fill_polygon(points_um, label, mode)`, `dirty_tiles()`.
- `mode`: `Replace`, `OnlyBackground` (tidak menimpa label lain), `Erase`.

#### 8.5.2 Alat manual

| Alat | Cara kerja |
|---|---|
| Kuas | Cakram di sepanjang polyline goresan; jari-jari = `r × f(tekanan)`. Jarak antar cakram ≤ `r/2`. |
| Penghapus | Kuas dengan mode `Erase`. |
| Laso | Poligon tertutup → isi scanline aturan genap-ganjil. |
| Poligon | Klik titik demi titik → sama dengan laso. |
| Isi lubang | Komponen terhubung latar yang tidak menyentuh tepi → label aktif. |
| Salin ke irisan berikut | Menyalin label lewat `T_{i+1}⁻¹ ∘ T_i` sebagai titik awal. |

Posisi sentuhan dikonversi layar → W → U sebelum masuk ke `LabelStore`.

#### 8.5.3 Segmentasi klasik

Ambang pada kanal dekonvolusi (mis. hematoksilin tinggi = padat inti), lalu
pembukaan/penutupan morfologi dan buang komponen kecil. Hasilnya **usulan**:
ditampilkan sebagai lapisan sementara, baru masuk `LabelStore` setelah
pengguna menekan "Terima".

#### 8.5.4 Segmentasi AI

```rust
pub trait Segmenter: Send + Sync {
    fn spec(&self) -> &ModelSpec;
    /// Masukan: RGB f32 NCHW [1, 3, S, S]. Keluaran: logit [1, C, S, S].
    fn infer(&self, input: &[f32]) -> Result<Vec<f32>, SegError>;
}
```

Model dibawa pengguna sebagai `model.onnx` + `model.json`:

```json
{
  "name": "tumor-he-v1",
  "input_size": 512,
  "mpp": 2.0,
  "mean": [0.485, 0.456, 0.406],
  "std": [0.229, 0.224, 0.225],
  "classes": [
    { "index": 0, "label": null },
    { "index": 1, "label": "Tumor" },
    { "index": 2, "label": "Stroma" }
  ]
}
```

Alur inferensi:

1. Baca slide pada level terdekat ke `mpp` model, lalu skala ke `mpp` persis.
2. Lewati tile yang masker jaringannya kosong.
3. Tile `S × S` dengan tumpang tindih 25%.
4. Logit tiap tile dikalikan jendela Hann 2D lalu dijumlahkan ke akumulator
   (menghilangkan garis sambungan).
5. `argmax` → peta kelas → peta ke `LabelId` lewat nama label.
6. Skala ke `mpp_label` (modus), simpan sebagai **usulan**.

Backend: `TractSegmenter` (bawaan, Rust murni, CPU) dan `OrtSegmenter` (fitur
`ort-coreml`). Inferensi diproses tile demi tile agar memori tetap.

#### 8.5.5 Interpolasi bentuk antar irisan

Pengguna cukup menandai tiap irisan ke-N. Untuk irisan `k` di antara irisan
bertanda `a` dan `b`, di **ruang W**:

```
t        = (z_k − z_a) / (z_b − z_a)
sdf_k    = (1 − t) · sdf_a + t · sdf_b
masker_k = sdf_k < 0
```

Hasilnya dipetakan balik ke ruang U irisan `k` dan disimpan sebagai
**usulan**. Saat diterima, dibuat `AnnotationRecord` dengan sumber
`Interpolated { from: [a, b] }` (8.12.7) supaya laporan bisa membedakannya
dari anotasi manual.

#### 8.5.6 Pertukaran anotasi

- **Impor GeoJSON** (format QuPath): poligon dalam piksel level 0 →
  dirasterkan ke `LabelStore`; `classification.name` → nama label.
- **Ekspor GeoJSON**: kontur tiap komponen terhubung (penelusuran tepi) →
  poligon piksel level 0.

### 8.6 `h3d-volume`

#### 8.6.1 Kisi tumpukan

```rust
pub struct StackGrid {
    pub origin_um: [f64; 2],   // sudut kiri-atas kisi di ruang W
    pub s_xy: f64,             // bawaan 16 µm
    pub size: [u32; 2],
    pub z_um: Vec<f64>,        // z tiap lapis keluaran
}
```

Batas kisi = kotak pembatas gabungan semua label yang direkonstruksi (setelah
`T_i`) + margin `2 × band`. **Bukan** seluruh slide; ini yang membuat
memorinya terjangkau.

#### 8.6.2 Langkah rekonstruksi

Per label yang `reconstruct = true`:

1. **Resampling.** Untuk tiap irisan `Ok` yang punya label: masker
   `size.x × size.y` di kisi, diisi dengan pemetaan mundur
   `u = T_i⁻¹(w)` → `LabelStore::get`. Paralel per irisan.
2. **SDF 2D.** `sdf_i` dalam µm, dijepit `±band`.
3. **Lapis Z keluaran.** Target `s_z ≤ k × s_xy` (bawaan `k = 2`). Di antara
   dua irisan berurutan disisipkan lapis dengan interpolasi linear SDF
   (rumus 8.5.5).
4. **Celah besar.** Bila jarak dua irisan berlabel > `max_gap_um` (bawaan
   5 × jarak irisan median), objek **diputus** di situ dan laporan memuat
   peringatan.
5. **Tutup ujung.** Satu lapis tambahan di tiap ujung dengan SDF = `+band`
   pada jarak setengah jarak irisan, agar permukaan tertutup.
6. **Ekstraksi per chunk.** Volume SDF dibagi chunk 64³ dengan lapisan
   tumpang 1 voxel; `fast-surface-nets` dijalankan per chunk. Lapis Z dibuat
   **sesuai kebutuhan** (jendela geser), sehingga SDF 3D penuh tidak pernah
   ada di memori sekaligus.
7. **Las verteks.** Verteks kembar di batas chunk disatukan lewat kunci
   koordinat voxel global.
8. **Ke µm.** Posisi × (`s_xy`, `s_xy`, `s_z` setempat) + `origin`.
9. **Penghalusan Taubin.** `λ = 0,5`, `μ = −0,53`, 10 iterasi. Taubin
   dipilih karena, tidak seperti Laplacian biasa, ia tidak menyusutkan
   volume secara berarti.
10. **Decimate.** `baby_shark`, galat maksimum `0,5 × s_xy`.
11. **Normal** per verteks dihitung ulang (rata-rata berbobot luas).

Konvensi SDF: **negatif di dalam**. Orientasi segitiga harus menghasilkan
volume bertanda positif; ini diuji (bagian 11).

#### 8.6.3 Tipe mesh

```rust
pub struct LabelMesh {
    pub label: LabelId,
    pub positions_um: Vec<[f64; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
    pub scalars: Vec<MeshScalars>,   // jarak margin, hipoksia, atribut per verteks (8.12.6)
}
```

#### 8.6.4 Pengukuran

| Besaran | Cara hitung | Peran |
|---|---|---|
| Volume (Cavalieri) | `V = Σ A_i × h_i`; `A_i` = luas label di irisan `i` (jumlah piksel × `mpp_label²`), `h_i = (z_{i+1} − z_{i−1}) / 2`, di ujung = jarak ke satu-satunya tetangga | **Angka utama laporan** |
| Volume mesh | `V = (1/6) Σ a · (b × c)` atas semua segitiga | Pembanding; selisih > 5% dari Cavalieri → peringatan |
| Luas permukaan | Jumlah luas segitiga | |
| Kotak pembatas | Min/maks posisi | |
| Diameter maksimum | Jarak terjauh antar verteks pada selubung cembung | |
| Jumlah komponen | Komponen terhubung mesh | |
| Margin, invasi, multifokalitas, kedalaman, hipoksia, statistik atribut | Lihat 8.12.6; dihitung pada masker voxel kisi tumpukan, bukan mesh | Bagian "Analisis" laporan |

Cavalieri hanya memakai irisan yang **benar-benar dianotasi** (manual atau
AI yang diterima), bukan hasil interpolasi.

#### 8.6.5 Anggaran memori

Contoh **[TARGET]**: jaringan 20 × 20 mm, 100 irisan berjarak 50 µm,
tumor 8 × 8 mm, `s_xy = 16 µm`.

| Besaran | Hitungan | Ukuran |
|---|---|---|
| Kisi tumor | 500 × 500 | — |
| Lapis Z keluaran (`s_z` = 32 µm) | ~156 | — |
| SDF 3D penuh `f32` (tidak dialokasikan) | 500 × 500 × 156 × 4 | 156 MB |
| Jendela geser 66 lapis | 500 × 500 × 66 × 4 | 66 MB |
| SDF 2D semua irisan sumber | 500 × 500 × 100 × 4 | 100 MB |

Bila perkiraan melampaui anggaran perangkat (9.2), mesin **menaikkan `s_xy`
otomatis** (16 → 24 → 32 µm) dan mencatatnya di laporan.

#### 8.6.6 Susut jaringan

Fiksasi dan pemrosesan menyusutkan jaringan. `SampleMeta` punya
`shrinkage_linear: Option<f64>` yang diisi pengguna. Laporan memuat **dua**
angka: terukur, dan terkoreksi (`V × f³`). Koreksi tidak pernah diterapkan
diam-diam ke geometri.

### 8.7 `h3d-io`

#### 8.7.1 Bundel proyek

Folder berakhiran `.h3d`:

```
sampel-01.h3d/
├── project.json            # manifes (struct Project)
├── slides/                 # hanya bila Embedded
│   └── 0007.svs
├── clinical.json           # ClinicalMeta pseudonim; opsional; terpisah (8.12.9)
├── labels/
│   └── 0007.lbl            # LabelStore per irisan
├── attrs/
│   └── 0007.ki67.atr       # AttrStore per (irisan, atribut) (8.12.5, 8.12.9)
├── models/                 # model ONNX + model.json (opsional)
├── cache/                  # bisa dihapus kapan saja
│   ├── pyramid/            # piramida untuk gambar biasa
│   ├── reg/                # gambar registrasi
│   └── thumbs/
└── out/
    ├── mesh/tumor.glb
    ├── volume/labels.nii.gz
    ├── volume/ki67.nii.gz  # AttrVolume f32
    ├── figures/            # gambar irisan untuk laporan
    ├── analysis.json       # keluaran analisis spasial (8.12.6)
    └── report.md, report.csv
```

Aturan tulis:

- **Atomik**: tulis ke `nama.tmp` lalu `rename`. Berlaku untuk
  `project.json` dan tiap `.lbl`.
- `format_version` dinaikkan pada perubahan tak kompatibel; pembaca menolak
  versi lebih baru dengan pesan jelas dan memigrasikan versi lebih lama.
- Path `External` disimpan relatif terhadap folder proyek bila bisa.
- **Sidik jari `External` ditegakkan.** Saat proyek dibuka dan tiap kali
  piksel dibaca, ukuran + sidik jari berkas dibandingkan dengan yang tersimpan
  saat impor. Ketidakcocokan tidak menggagalkan pembukaan proyek: irisannya
  masuk daftar "berubah" dan pikselnya ditolak dibaca sampai pengguna
  menautkan ulang berkas aslinya atau menerima berkas baru (ukuran piksel
  level 0 harus sama). Berkas yang hilang dilaporkan terpisah.

#### 8.7.2 Format `.lbl` (little-endian)

```
offset  ukuran  isi
0       4       magic "H3DL"
4       2       versi (u16) = 1
6       2       tile_size (u16) = 256
8       8       mpp_label (f64)
16      4       lebar_px (u32)
20      4       tinggi_px (u32)
24      4       n_tile (u32)
28      20·n    indeks: { tx: u32, ty: u32, offset: u64, len: u32 }
…       …       payload: tiap tile = lz4_flex (size-prepended) dari u8 × 256²
```

Pembaca wajib memvalidasi: magic, versi, `offset + len` ≤ ukuran berkas,
ukuran hasil dekompresi = `tile_size²`. Pelanggaran → error, tidak panik.

#### 8.7.3 Ekspor

| Format | Isi | Catatan |
|---|---|---|
| GLB | Mesh per label, warna label sebagai material; `COLOR_0` dari colormap skalar terpilih dan nilai mentah di atribut verteks `_H3D_<NAMA>` | Satuan meter (standar glTF): µm × 1e−6 |
| STL biner | Satu label per berkas | Satuan mm; STL tidak menyimpan satuan, tulis di nama berkas |
| OBJ | Semua label, satu grup per label | mm |
| NIfTI (`.nii.gz`) | Volume label `u8` pada kisi tumpukan; `pixdim` = (`s_xy`, `s_xy`, `s_z`) dalam mm; satu berkas `f32` per `AttrVolume` dan per medan jarak | Untuk dibuka di 3D Slicer sebagai validasi silang |
| GeoJSON | Kontur per irisan; `properties.classification` = nama label, `properties.measurements` = luas dan rata-rata atribut | Kompatibel QuPath |
| JSON analisis | `analysis.json`: semua keluaran 8.12.6 dengan ketidakpastian dan parameter | Sumber tunggal untuk CSV dan laporan |
| CSV + Markdown | Tabel pengukuran per label + bagian analisis (8.12.8) + parameter + peringatan + versi aplikasi | Memuat "Hanya untuk riset"; `clinical.json` tidak ikut tanpa `--with-clinical` |

NIfTI butuh `s_z` seragam; bila jarak irisan tidak seragam, volume
di-resample ke `s_z` median dan ini dicatat di laporan.

### 8.8 `h3d-engine`

Lapisan headless yang dipakai CLI dan GUI.

```rust
pub struct Session { /* Project + path + cache + undo */ }

impl Session {
    pub fn create(path: &Path) -> Result<Self, EngineError>;
    pub fn open(path: &Path) -> Result<Self, EngineError>;
    pub fn save(&mut self) -> Result<(), EngineError>;

    pub fn import_slides(&mut self, paths: &[PathBuf], mpp: Option<[f64; 2]>,
                         p: &dyn Progress) -> Result<Vec<SlideId>, EngineError>;
    pub fn register_all(&mut self, opts: &RegOptions, p: &dyn Progress)
        -> Result<RegSummary, EngineError>;
    pub fn suggest_order(&self, p: &dyn Progress)
        -> Result<OrderSuggestion, EngineError>;
    pub fn apply_order_suggestion(&mut self, s: &OrderSuggestion,
                                  accept: &[SlideId]) -> Result<(), EngineError>;
    pub fn segment_ai(&mut self, model: &Path, slides: &[SlideId],
                      p: &dyn Progress) -> Result<(), EngineError>;
    pub fn interpolate_labels(&mut self, label: LabelId, p: &dyn Progress)
        -> Result<(), EngineError>;
    pub fn reconstruct(&mut self, p: &dyn Progress)
        -> Result<Vec<LabelMesh>, EngineError>;   // juga menulis out/volume/labels.nii.gz
    pub fn measure(&self) -> Result<Report, EngineError>;
    pub fn export(&self, what: ExportKind, out: &Path) -> Result<(), EngineError>;

    // 8.12
    pub fn set_clinical(&mut self, c: Option<ClinicalMeta>) -> Result<(), EngineError>;
    pub fn compute_attr(&mut self, def: AttrDef, slides: &[SlideId],
                        p: &dyn Progress) -> Result<AttrId, EngineError>;
    pub fn analyze_spatial(&mut self, p: &dyn Progress)
        -> Result<SpatialAnalysis, EngineError>;   // juga mengisi LabelMesh.scalars
    pub fn mark_reviewed(&mut self, slide: SlideId, label: LabelId, by: &str)
        -> Result<(), EngineError>;

    // berkas slide External (8.7.1)
    pub fn missing_slides(&self) -> Vec<SlideId>;
    pub fn changed_slides(&self) -> Vec<SlideId>;
    pub fn recheck_slides(&self) -> Vec<SlideId>;
    pub fn relink_slide(&mut self, slide: SlideId, path: &Path) -> Result<bool, EngineError>;
    pub fn accept_changed_slide(&mut self, slide: SlideId) -> Result<(), EngineError>;

    // gambar laporan (8.12.8 butir 3)
    pub fn margin_figures(&self) -> Result<Vec<MarginFigure>, EngineError>;
}
```

`reconstruct` menyimpan masker voxel per label di cache (`cache/vox/`) agar
`analyze_spatial` tidak perlu meresampel ulang; cache dibuang bila label,
transformasi, atau `ReconConfig` berubah (kunci = hash atas ketiganya).

**Job.** GUI menjalankan metode panjang di thread latar:

```rust
pub struct JobHandle<T> {
    pub progress: Arc<JobProgress>,   // AtomicU32 fraksi + Mutex<String> tahap + AtomicBool batal
    pub result: mpsc::Receiver<Result<T, EngineError>>,
}
```

GUI memeriksa `result.try_recv()` tiap frame dan meminta repaint selama job
berjalan. Pola ini meniru `ducad-app/src/import_worker.rs`.

**Determinisme.** Tanpa bilangan acak tak berbenih, tanpa ketergantungan pada
jam, dan urutan penjumlahan paralel dibuat tetap (reduksi per irisan lalu
gabung berurutan). Masukan sama → keluaran identik byte demi byte.

### 8.9 `h3d-cli`

Binary `h3d`. Kode keluar: 0 sukses, 1 kesalahan pemakaian, 2 kesalahan
data/berkas, 3 QC gagal (ada irisan bertanda dan `--strict` aktif).

```
h3d info SLIDE                          # level, mpp, ukuran (JSON)
h3d thumb SLIDE --out thumb.png --max 1024
h3d new PROYEK.h3d
h3d import PROYEK.h3d SLIDE... [--mpp 0.5] [--spacing-um 50] [--embed]
h3d suggest-order PROYEK.h3d [--json] [--apply]   # usulan urutan + orientasi (8.4.7)
h3d register PROYEK.h3d [--model affine] [--metric ncc] [--reference N] [--strict]
h3d qc PROYEK.h3d                       # tabel NCC/Dice/rasio luas per irisan
h3d labels import PROYEK.h3d --slide N --geojson ANOT.geojson
h3d segment PROYEK.h3d --model DIR [--slides 1,5,9]
h3d interpolate PROYEK.h3d --label Tumor
h3d reconstruct PROYEK.h3d [--sxy 16] [--smooth 10] [--decimate 0.5]   # + out/mesh/mesh.glb, out/volume/labels.nii.gz
h3d measure PROYEK.h3d [--json]
h3d export PROYEK.h3d --format glb|stl|obj|nifti|geojson|report|analysis --out DIR [--scalar NAMA] [--with-clinical]
h3d clinical set PROYEK.h3d --json KLINIS.json      # tulis clinical.json (8.12.2)
h3d clinical strip PROYEK.h3d                       # hapus clinical.json
h3d attrs compute PROYEK.h3d --kind ihc|nuclei|til --slides 3,4 [--t-d 0.15] [--name "Ki-67 % positif"]
h3d analyze PROYEK.h3d [--json]                     # analisis spasial 8.12.6 → out/analysis.json
h3d review PROYEK.h3d --slide N --label Tumor --by "YJ"   # tandai sudah ditinjau (8.12.7)
h3d relink PROYEK.h3d --slide N [--path BERKAS] [--accept]  # tautkan ulang / terima berkas slide yang berubah (8.7.1)
h3d phantom --shape ellipsoid|torus|branch --out DIR [--slices 60] [--seed 1] [--shuffle]
```

Saat membuka proyek, tiap perintah mencetak satu baris `peringatan:` di stderr
per irisan yang berkas slidenya berubah sejak impor atau tidak ditemukan;
perintah yang tidak membaca piksel irisan itu tetap berjalan (kode keluar 0).
Ekspor `report` juga menulis `figures/margin_<label>.png` di folder keluaran.

`h3d phantom` membuat data uji sintetis (bagian 11.1) dan ikut dirilis
karena berguna untuk demo dan validasi oleh pengguna. Opsi `--margin D`,
`--foci 2 --gap G`, `--vessel`, `--capsule [--breach]`, `--surface H`, dan
`--ihc P` menghasilkan fantom untuk 8.12.11.

### 8.10 `h3d-render`

Satu `SceneRenderer` dengan beberapa pipeline wgpu, digambar lewat
*paint callback* `egui_wgpu` seperti DUCAD.

| Pipeline | Masukan | Catatan |
|---|---|---|
| Tile 2D | Daftar `(rect_w, tekstur)` | Satu quad per tile. Tekstur `Rgba8UnormSrgb` 256/512 px. |
| Overlay label | Tile `R8Uint` + palet `256 × 1` RGBA | Warna dari palet di shader; opasitas global; mode garis tepi (bandingkan dengan 4 tetangga). |
| Pembanding registrasi | Dua lapis tile + dua matriks | Mode: tembus pandang, papan catur, selisih warna (tetap = magenta, bergerak = hijau; sejajar = abu-abu). |
| Mesh 3D | `LabelMesh` → `f32` mm | Pencahayaan sederhana 2 lampu + ambient; alfa per label; bidang potong. |
| Skalar di mesh | Buffer `f32` per verteks + colormap `256 × 1` + `[min, max]` | Warna dari colormap di shader; `NaN` → abu-abu; legenda digambar egui di atas viewport (8.12.6). Picking: ray–segitiga di CPU atas mesh yang sudah didecimate, cukup untuk klik. |
| Irisan di 3D | Quad bertekstur thumbnail di `z_i` dengan `T_i` | Untuk melihat hubungan mesh dengan gambar asli. |
| Volume (H9) | Tekstur 3D `R8Unorm` | Raymarching; ditunda. |

**Pemilihan level tile.** Pilih level `l` terkecil dengan
`downsample_l ≥ µm_per_piksel_layar / mpp`. Sambil menunggu tile dimuat,
gambar tile induk (resolusi lebih rendah) yang sudah ada, supaya layar tidak
pernah kosong.

**Cache GPU.** LRU berbatas byte; tile yang tidak terlihat selama N frame
dibuang lebih dulu. Unggah dibatasi (mis. 8 tile per frame) agar frame tidak
tersendat.

**Transparansi mesh.** Label tak tembus pandang digambar dulu; label tembus
pandang diurutkan per objek dari belakang ke depan dengan tulis-kedalaman
mati. Cukup untuk beberapa label; OIT tidak diperlukan.

**Batas perangkat.** Jangan mengandaikan dimensi tekstur maksimum; baca
`device.limits()` dan pecah bila perlu.

### 8.11 `h3d-ui` dan `h3d-app`

**Mode aplikasi.**

```rust
pub enum AppMode { Slides, Register, Segment, Reconstruct, Report }
```

| Mode | Isi layar |
|---|---|
| `Slides` | Kisi thumbnail; impor; seret untuk mengurutkan; isi mpp dan jarak irisan; tandai rusak; tombol "Usulkan urutan dan orientasi" dengan pratinjau perubahan, tanda irisan ragu, "Balik urutan", dan Terima/Tolak (8.4.7). Panel "Klinis" dan orientasi anatomis; per irisan: pra-analitik dan panel IHC (8.12.10). |
| `Register` | Viewer pembanding (irisan `i` vs tetangga); daftar QC; tombol "Sejajarkan semua"; alat landmark dan geser/putar manual. |
| `Segment` | Viewer slide + overlay label; panel label (dengan `kind` dan grup); kuas/laso/poligon/penghapus; "Salin ke irisan berikut"; usulan klasik/AI/interpolasi dengan Terima/Tolak; overlay inti positif/negatif untuk kalibrasi ambang IHC; "Tandai sudah ditinjau" (8.12.10). |
| `Reconstruct` | Viewport 3D; parameter (`s_xy`, penghalusan, decimate); visibilitas dan opasitas label; bidang potong; irisan di 3D; pemilih lapisan skalar dengan legenda; klik mesh → nilai dan irisan terdekat (8.12.10). |
| `Report` | Tabel pengukuran; tab analisis (komponen, margin, invasi, ukuran, atribut, provenance; 8.12.8); peringatan; ekspor. |

Navigasi antar irisan (tombol, geser dua jari, tombol panah) tersedia di
mode `Register` dan `Segment`.

**Masukan.**

- Desktop: tetikus + papan ketik; roda = zoom, seret tombol tengah = geser.
- iPad: Pencil menggambar, jari menggeser/zoom (cubit). Tekanan dari
  `egui::Event::Touch { force, .. }`. Bila tidak ada Pencil, mode "jari
  menggambar" bisa dinyalakan.

**Antarmuka.**

- Teks lewat `h3d-i18n` (Indonesia bawaan, Inggris).
- Ikon **hanya** dari `egui_icons::icons::ICON_*`; dilarang memakai glyph
  Unicode atau emoji sebagai ikon (font aplikasi sering tidak memilikinya
  sehingga tampil kotak kosong). Tes `ui_glyph_coverage` disalin dari DUCAD
  dan wajib lulus.
- Usulan (AI, klasik, interpolasi) selalu lewat Terima/Tolak; tidak pernah
  langsung menimpa label.

### 8.12 Atribut medis dan analisis spasial

Bagian ini menambahkan **data medis terstruktur** pada objek 3D dan
**analisis spasial** yang mengubahnya menjadi angka yang dipakai patolog,
ahli bedah, dan peneliti saat mengambil keputusan. Aturan dasarnya tetap
(2.2, K15, aturan 16 bagian 14): aplikasi menghasilkan **pengukuran beserta ketidakpastiannya**,
bukan rekomendasi. Tidak ada kalimat "perlu reseksi ulang" di mana pun;
yang ada adalah "jarak tumor ke tepi reseksi anterior = 0,6 mm ± 0,1 mm".

#### 8.12.1 Tiga level penyimpanan

| Level | Tempat | Contoh | Dipakai untuk |
|---|---|---|---|
| Sampel | `SampleMeta.clinical` (8.12.2) | diagnosis, grade, pTNM, terapi, orientasi anatomis | konteks laporan; nama arah anatomis |
| Irisan | `SlideEntry.preanalytic`, `SlideEntry.ihc` (8.12.3) | antibodi, klon, ketebalan potong, lama fiksasi | kelayakan membandingkan antar kasus; parameter skor IHC |
| Region / voxel | `LabelDef.kind`, `AttrStore` (8.12.4–8.12.5) | subtipe tumor, tepi reseksi, pembuluh, skor Ki-67 per tile | analisis spasial 3D (8.12.6); pewarnaan mesh |

Level ketiga yang membawa nilai terbesar karena bisa **dihitung secara
spasial** (jarak, kontak, komponen) dan **ditampilkan di permukaan mesh**.

#### 8.12.2 Level sampel: `ClinicalMeta`

```rust
pub struct SampleMeta {
    pub name: String,
    pub notes: String,
    pub shrinkage_linear: Option<f64>,       // 8.6.6
    pub orientation: Option<SpecimenOrientation>,
    /// Dimuat dari `clinical.json`; `None` bila berkas tidak ada (8.12.9).
    pub clinical: Option<ClinicalMeta>,
}

/// Nama anatomis untuk arah +X, +Y, +Z ruang W. Arah negatif = lawannya.
pub struct SpecimenOrientation {
    pub pos_x: AnatDir,
    pub pos_y: AnatDir,
    pub pos_z: AnatDir,
}

pub enum AnatDir {
    Anterior, Posterior, Superior, Inferior, Medial, Lateral,
    Proximal, Distal, Left, Right, Unknown,
}

pub struct ClinicalMeta {
    pub subject_pseudo_id: String,          // bukan nama, bukan nomor RM
    pub age_years: Option<u8>,
    pub sex: Option<Sex>,                   // Female, Male, Other, Unknown
    pub topography_icdo: Option<String>,    // "C50.4"
    pub morphology_icdo: Option<String>,    // "8500/3"
    pub laterality: Option<Laterality>,     // Left, Right, Bilateral, NotApplicable
    pub procedure: Option<Procedure>,       // Biopsy, Excision, Resection, Other(String)
    pub grade: Option<Grade>,               // { system: String, value: String }
    pub ptnm: Option<Ptnm>,                 // { t, n, m: String; edition: u8; prefix: "p"|"yp" }
    pub neoadjuvant: Option<Therapy>,       // { given: bool, regimen: Option<String> }
    pub molecular: Vec<MolecularResult>,    // { gene: "KRAS", result: "G12D", method: Option<String> }
    pub specimen_month: Option<String>,     // "2025-11"; sengaja tanpa hari
}
```

Aturan:

- `orientation` divalidasi: tiga sumbu harus dari **pasangan berbeda**
  (Anterior/Posterior, Superior/Inferior, Medial/Lateral, Proximal/Distal,
  Left/Right). `Unknown` boleh; arah lalu dilaporkan sebagai `+X`.
- `grade.system` dan `ptnm.edition` adalah teks bebas yang dipilih dari
  daftar saran (WHO, Nottingham, Gleason, Fuhrman/WHO-ISUP; AJCC 8), bukan
  enum, supaya tidak perlu rilis baru untuk sistem lain.
- `ClinicalMeta` **tidak pernah** dipakai dalam perhitungan. Ia hanya tampil
  di laporan dan memberi nama arah. Ini menjaga agar pengukuran tidak
  bergantung pada data klinis yang mungkin tidak diisi.

#### 8.12.3 Level irisan: pra-analitik dan panel IHC

```rust
pub struct SlideEntry {
    // … (8.1)
    pub preanalytic: PreanalyticMeta,
    pub ihc: Option<IhcPanel>,              // wajib bila stain == Ihc
}

pub struct PreanalyticMeta {
    pub block_id: Option<String>,           // blok parafin asal
    pub section_thickness_um: Option<f64>,  // biasanya 3–5
    pub fixation_hours: Option<f64>,
    pub fixative: Option<String>,           // "NBF 10%"
    pub stained_month: Option<String>,      // "2025-11"
    pub scanner: Option<String>,            // dari metadata slide bila ada
    pub objective: Option<String>,          // "20x"
}

pub struct IhcPanel {
    pub antibody: String,                   // "Ki-67", "ER", "HER2", "PD-L1", "CD8"
    pub clone: Option<String>,              // "MIB-1", "SP263"
    pub chromogen: Chromogen,               // Dab, Aec, FastRed, Other
    pub pattern: StainPattern,              // Nuclear, Membranous, Cytoplasmic
    pub scoring: IhcScoring,
    /// Vektor OD pewarna hasil kalibrasi di irisan ini (8.3); None = bawaan.
    pub od_vectors: Option<[[f32; 3]; 3]>,
}

pub enum IhcScoring {
    PercentPositive,                        // Ki-67, ER/PR, CD8
    HScore,                                 // Σ i × %_i, 0–300
    Allred,                                 // proporsi 0–5 + intensitas 0–3
    Tps,                                    // PD-L1 sel tumor
    Cps,                                    // PD-L1 (tumor + imun) / tumor × 100
    Intensity03,                            // HER2 gaya 0/1+/2+/3+
}
```

Irisan IHC ikut rantai registrasi seperti irisan H&E (8.4.4, metrik MI
untuk beda pewarnaan). Ia **tidak** diberi label tumor sendiri; masker tumor
diambil dari irisan H&E tetangga terdekat lewat `T_{he}` → W → `T_{ihc}⁻¹`
(sama seperti "Salin ke irisan berikut", 8.5.2). Bila pengguna menandai
tumor langsung di IHC, label itu yang dipakai.

#### 8.12.4 Level region: jenis label dan grup

```rust
pub struct LabelDef {
    pub id: LabelId,
    pub name: String,
    pub color: [u8; 4],
    pub visible: bool,
    pub reconstruct: bool,
    pub kind: LabelKind,
    /// Nama grup untuk agregasi, mis. "Tumor" untuk "Tumor invasif" + "In situ".
    pub group: Option<String>,
}

pub enum LabelKind {
    Tumor,              // komponen tumor apa pun (subtipe lewat `name` + `group`)
    Necrosis,
    Stroma,
    Normal,
    Vessel(VesselKind), // Blood, Lymph, Unknown
    Nerve,
    Capsule,            // kapsul/serosa/fasia: permukaan tertutup atau sebagian
    Margin,             // tepi reseksi (permukaan luar spesimen, biasanya bertinta)
    Surface,            // permukaan epitel/mukosa, acuan kedalaman invasi
    Reference,          // garis acuan lain (mis. lapisan granular untuk Breslow)
    Other,
}
```

`kind` menentukan analisis mana yang berlaku (8.12.6); `group` menentukan
agregasi di laporan (volume grup = jumlah volume anggota; fraksi anggota =
volume anggota / volume grup). Label dengan `kind` `Margin`, `Surface`,
`Reference` **tidak** direkonstruksi sebagai mesh tertutup; mereka dipakai
sebagai sumber medan jarak. Pengguna menggambar tepi reseksi sebagai garis
tebal (kuas) di sepanjang tinta; label raster cukup, tidak perlu kurva.

#### 8.12.5 `AttrStore`: atribut skalar per irisan

Atribut adalah besaran kontinu per lokasi (persen positif Ki-67, densitas
inti, densitas limfosit). Ia disimpan **sejajar `LabelStore`**, di ruang
slide (U), bertile, tetapi lebih kasar karena merupakan agregat luas.

```rust
pub struct AttrId(pub u16);

pub struct AttrDef {
    pub id: AttrId,
    pub name: String,                       // "Ki-67 % positif"
    pub unit: String,                       // "%", "sel/mm²", "µm"
    pub kind: AttrKind,
    pub source: AttrSource,                 // Computed { method, params_hash } | Imported { file }
    pub colormap: Colormap,                 // Viridis, Magma, Coolwarm, Grey
    pub range: Option<[f32; 2]>,            // skala tampilan; None = p2–p98 data
}

pub enum AttrKind {
    IhcScore { slide: SlideId, scoring: IhcScoring },
    NucleiDensity,                          // sel/mm²
    TilDensity,                             // limfosit/mm² (perlu kelas Limfosit atau IHC CD3/CD8)
    Custom,
}

pub struct AttrStore {
    pub mpp_attr: f64,                      // bawaan 16.0 µm (= s_xy bawaan)
    pub size_px: [u32; 2],
    pub tile_size: u32,                     // 64
    pub scale: f32,                         // nilai = raw × scale + offset
    pub offset: f32,
    tiles: HashMap<(u32, u32), Box<[u16]>>, // u16::MAX = NODATA
}
```

- `u16` + `scale`/`offset` cukup untuk semua besaran di atas (persen pada
  resolusi 0,01; densitas hingga 65.534 sel/mm²) dan 2× lebih kecil dari
  `f32`. NODATA wajib dibedakan dari nol: "tidak ada jaringan" bukan "0 %".
- **Penurunan resolusi** memakai **rata-rata berbobot jumlah sel** (bukan
  modus seperti label), dengan NODATA diabaikan.
- **Resampling ke kisi tumpukan** (8.6.2 langkah 1) memakai pemetaan mundur
  yang sama, interpolasi bilinear dengan bobot NODATA = 0. Hasilnya
  `AttrVolume { grid: StackGrid, data: Vec<f32> }` per atribut, hanya pada
  irisan yang punya atribut; lapis Z di antaranya diisi interpolasi linear
  seperti SDF. Atribut dari irisan IHC yang berjarak > `max_gap_um` dari
  irisan lain tidak diinterpolasi (NODATA).

**Metode penghitungan atribut** (`h3d-imgproc`, modul `nuclei` dan `ihc`).
Semua berjalan pada level piramida terdekat ke `mpp_work = 1,0 µm` (bukan
level 0), per tile 1024 px dengan tumpang tindih 64 px, hanya di dalam
masker jaringan dan, bila ada, masker label yang diminta.

1. **Dekonvolusi warna** (8.3) → kanal OD hematoksilin `H`, kanal OD
   kromogen `D` (DAB/AEC/Fast Red; vektor dari `IhcPanel.od_vectors` atau
   bawaan Ruifrok). Pada H&E, `D` = eosin dan tidak dipakai untuk skor.
2. **Deteksi inti** (klasik): `H` dihaluskan Gauss σ = 1 µm; deteksi blob
   **LoG multiskala** σ ∈ {2, 3, 4} µm (Al-Kofahi 2010); kandidat = minimum
   lokal respons LoG dengan `H_smooth > t_H` (Otsu pada `H` di dalam masker
   jaringan). Supresi non-maksimum jarak 4 µm. Keluaran: daftar pusat inti
   `(x, y)` dalam µm, ukuran, `H_mean`, `D_mean` (rata-rata dalam cakram
   jari-jari σ). Ini **bukan** segmentasi inti penuh; cukup untuk hitungan
   dan skor. Bila model AI menyediakan head inti (8.5.4, `"task":
   "nuclei"`), daftarnya dipakai menggantikan langkah ini.
3. **Skor IHC inti** (`pattern = Nuclear`): inti positif bila
   `D_mean > t_D`; `t_D` bawaan 0,15 OD, dapat diubah per irisan dan
   ditampilkan sebagai overlay agar pengguna bisa mengkalibrasi terhadap
   kontrol positif/negatif. Intensitas 0/1/2/3 lewat tiga ambang OD bawaan
   `[0,15; 0,35; 0,60]`. Per tile `AttrStore`:
   `%pos = n_pos / n_all`, `H-score = Σ_i i × %_i`, Allred = skor proporsi
   (0–5 dari %pos) + skor intensitas (0–3 dari modus intensitas).
4. **Skor IHC membran/sitoplasma** (`Membranous`, `Cytoplasmic`): versi
   pertama memakai **fraksi luas**: piksel positif = `D > t_D` di dalam
   masker jaringan; `%pos = luas positif / luas jaringan` per tile; intensitas
   dari kuantil OD piksel positif. Ini dicatat di laporan sebagai "skor
   berbasis luas, bukan per sel". Skor HER2 per sel lengkap (cincin membran
   utuh vs sebagian) ditunda; butuh segmentasi sel.
5. **TPS / CPS** (PD-L1): `TPS = sel tumor positif / semua sel tumor`;
   `CPS = (sel tumor positif + sel imun positif) / semua sel tumor × 100`,
   dijepit ≤ 100. Butuh pemisahan sel tumor vs imun: lewat masker label
   (inti di dalam `Tumor` = sel tumor; di `Stroma` = imun) atau kelas model.
   Tanpa keduanya, hanya `%pos` yang dihitung dan TPS/CPS ditandai
   "tidak tersedia".
6. **Densitas inti**: `n_inti / luas_jaringan_tile` (sel/mm²).
7. **Densitas limfosit (TIL)**: sumber (a) kelas `Limfosit` dari model AI,
   atau (b) irisan IHC CD3/CD8 → inti positif. Dilaporkan terpisah untuk
   **intratumoral** (di dalam masker `Tumor`) dan **peritumoral** (jarak
   ≤ `peri_band_um`, bawaan 500 µm, di luar tumor), mengikuti pembagian
   rekomendasi International TIL Working Group (Salgado 2015), dengan catatan
   bahwa metode kerja kelompok itu memakai persen luas stroma, bukan
   densitas; keduanya dilaporkan bila masker `Stroma` ada.
8. **Tumor budding** (opsional, H11): komponen terhubung label `Tumor`
   dengan luas ≤ `bud_max_area_um2` (bawaan 500 µm² ≈ ≤ 4 sel) dalam pita
   1 mm dari front invasif; hitungan hotspot 0,785 mm² (ITBCC, Lugli 2017).
   Hanya diaktifkan bila `mpp_label ≤ 2 µm`; pada 4 µm sebuah tunas 20 µm
   hanya 5 piksel dan tidak dapat diandalkan.

**Agregasi per kompartemen.** Nilai tiap piksel `AttrStore` diagregasi dari
jendela 5 × 5 piksel (80 µm). Bila masker `Tumor` tersedia, skor per inti dan
densitas hanya mengagregasi sel (dan luas jaringan) dari kompartemen piksel
itu sendiri — di dalam atau di luar tumor — agar jendela tidak mencampur
kedua sisi batas tumor (tanpa ini densitas TIL intratumoral fantom terukur
+38 %, peritumoral −11 %). TPS/CPS sudah per kompartemen menurut definisinya.

Semua parameter (`t_D`, ambang intensitas, σ, `peri_band_um`) tersimpan di
`AttrDef.source.params_hash` dan dicetak di laporan. Mengubah parameter =
atribut baru, bukan menimpa diam-diam.

#### 8.12.6 Analisis spasial 3D (`h3d-volume::spatial`)

Semua analisis bekerja pada **kisi tumpukan** (`StackGrid`, 8.6.1) dengan
masker voxel per label hasil langkah 8.6.2 (sebelum surface nets), sehingga
tidak bergantung pada penghalusan mesh. Mesh hanya dipakai untuk tampilan.

**Medan jarak 3D anisotropik.** `edt3(mask, s_xy, s_xy, s_z)`: algoritma
Felzenszwalb–Huttenlocher dijalankan per sumbu dengan bobot jarak kuadrat
`s_axis²`; algoritma ini separabel sehingga anisotropi ditangani persis.
Bila `s_z` tidak seragam (jarak irisan tak rata) kisi diresampel ke `s_z`
median terlebih dahulu, seperti NIfTI (8.7.3), dan laporan mencatatnya.
Diproses per chunk kolom Z penuh × 64 × 64 XY (sumbu Z dijalankan terakhir,
butuh seluruh kolom) dengan jendela geser; memori `O(size_xy × n_z × 4)`
untuk satu medan, sudah masuk anggaran 8.6.5.

**Ketidakpastian jarak.** Tiap jarak dilaporkan dengan
`± (½ ukuran voxel pada sumbu dominan + TRE_est)`, `TRE_est` = median TRE
QC registrasi (8.4.5) irisan yang terlibat; bila QC tidak ada, dipakai
`s_xy`. TRE per irisan = `sqrt(Σ_k residual_fit_k² + residual_i²)` atas
pasangan di jalur irisan itu ke acuan (sisa galat terukur, 8.4.5); catatan
lama tanpa sisa galat memakai proksi dari Dice masker. Jarak pada arah Z juga ditandai "resolusi Z = jarak irisan".

| Analisis | Prasyarat `kind` | Metode | Keluaran |
|---|---|---|---|
| **Margin** | `Tumor` + `Margin` | `d_M = edt3(Margin)`; `d_min = min d_M` atas voxel tumor; arah = vektor voxel tumor → voxel margin terdekat (diambil dari argmin EDT yang disimpan sebagai indeks), diklasifikasi ke sumbu dominan ±X/±Y/±Z lalu dinamai lewat `SpecimenOrientation`; dihitung per arah (6 nilai) dan global | `d_min` ± u, per arah; status `Involved` (d_min ≤ ½ s_xy), `Close` (< `margin_close_um`, bawaan 1000), `Clear`; titik terdekat (µm); **peta jarak per verteks** mesh tumor |
| **Kontak / invasi struktur** | `Tumor` + `Vessel`/`Nerve`/`Capsule` | Pasangan voxel bertetangga muka (6-konektivitas) dengan satu tumor dan satu struktur; luas kontak = Σ luas muka (`s_xy × s_xy` atau `s_xy × s_z`), dilaporkan mentah dan terkoreksi (tiap muka dibobot `\|n·e\|`, `n` = gradien indikator struktur terpulus Gauss 1,5 voxel di muka itu; menghapus bias tangga +27 %); tambalan kontak = komponen terhubung pasangan tersebut | per struktur: ada/tidak, luas kontak mm², daftar tambalan (centroid µm, luas, irisan yang terlibat) |
| **Penetrasi kapsul** | `Tumor` + `Capsule` | `inside` = flood-fill 6-konektivitas dari batas kisi atas voxel bukan-kapsul → `outside`; sisanya `inside`. Komponen tumor yang memuat voxel `inside` **dan** `outside` = menembus | ya/tidak per komponen; volume tumor di luar kapsul |
| **Multifokalitas** | `Tumor` | Komponen terhubung 26-konektivitas atas gabungan grup `Tumor`; buang komponen < `min_focus_um3` (bawaan (3 × s_xy)³); per komponen: Cavalieri per irisan (8.6.4), centroid, kotak pembatas, diameter maks; jarak antar fokus = min `edt3(fokus A)` atas voxel fokus B | jumlah fokus, tabel per fokus terurut volume, matriks jarak |
| **Kedalaman invasi** | `Tumor` + `Surface` | `d_S = edt3(Surface)`; kedalaman = max `d_S` atas voxel tumor; titik terdalam | kedalaman ± u, titik terdalam, irisan |
| **Ketebalan Breslow** | `Tumor` + `Reference` (lapisan granular) | sama dengan kedalaman, acuan `Reference` | ketebalan ± u |
| **Ukuran staging** | `Tumor` | diameter maks selubung cembung (sudah ada), ditambah ukuran pada tiga sumbu utama PCA masker tumor (bukan kotak pembatas, agar tidak bergantung orientasi) | 3 ukuran ± u |
| **Fraksi komponen** | grup | volume anggota / volume grup; nekrosis % = `Necrosis` ∩ pita tumor / (tumor + nekrosis) | tabel fraksi |
| **Proksi hipoksia** | `Tumor` + `Vessel(Blood)` | `d_V = edt3(Vessel)`; histogram `d_V` atas voxel tumor (bin 25 µm); fraksi `d_V > hypoxia_um` (bawaan 150 µm, Thomlinson & Gray 1955) | fraksi, histogram, peta per verteks |
| **Statistik atribut per region** | `AttrVolume` + label | atas voxel label: mean, median, p10, p90; heterogenitas = koefisien variasi rata-rata kubus 1 mm³; tabel per irisan | tabel; peta per verteks |

**Proyeksi ke mesh.** Untuk tiap verteks `LabelMesh`, nilai medan (jarak
atau atribut) diambil dengan interpolasi trilinear pada kisi; verteks yang
jatuh di NODATA diberi `NaN` dan digambar abu-abu. Disimpan sebagai

```rust
pub struct MeshScalars { pub name: String, pub unit: String, pub values: Vec<f32> }
// LabelMesh.scalars: Vec<MeshScalars>
```

**Determinisme.** Flood-fill dan komponen terhubung memakai urutan
pemindaian tetap (z, y, x); argmin EDT memilih indeks terkecil saat seri.

#### 8.12.7 Provenance anotasi

Menggantikan penanda `interpolated` di 8.5.5 dengan catatan per
(irisan, label):

```rust
pub enum AnnotationSource {
    Manual { by: String },                  // nama/inisial pengguna, diisi di Pengaturan
    Classic { params_hash: u64 },
    Ai { model: String, version: String },
    Interpolated { from: [SlideId; 2] },
    Imported { file: String },
}

pub enum ReviewStatus { Draft, Reviewed { by: String, at: String } }

pub struct AnnotationRecord {
    pub slide: SlideId,
    pub label: LabelId,
    pub source: AnnotationSource,
    pub at: String,                         // ISO-8601, zona lokal
    pub review: ReviewStatus,
    pub confidence: Option<f32>,            // mean prob. argmax untuk AI; None untuk manual
}
```

- Tiap "Terima" atas usulan dan tiap goresan pertama pada (irisan, label)
  membuat/memperbarui `AnnotationRecord`. Goresan manual atas label hasil AI
  mengubah `source` menjadi `Manual` dengan catatan `edited_from: Ai`.
- Tombol "Tandai sudah ditinjau" di mode `Segment` mengisi `Reviewed`.
- Cavalieri (8.6.4) tetap hanya memakai irisan non-interpolasi. Laporan
  menambah: volume dari irisan **`Reviewed` saja**, fraksi irisan per
  `source`, dan daftar irisan `Draft`.

#### 8.12.8 Laporan dan peringatan

Laporan (8.7.3) mendapat bagian baru, urutannya tetap:

1. Konteks sampel (dari `ClinicalMeta`, bila ada) dan orientasi.
2. Komponen tumor: tabel grup/anggota, volume, fraksi, jumlah fokus.
3. Margin: tabel per arah, status, titik terdekat; gambar irisan terdekat
   dengan panah (PNG di `out/figures/`): `figures/margin_<label>.png` di
   samping `report.md`, berisi thumbnail irisan berlabel yang `z`-nya terdekat
   ke titik tumor terdekat (pada ruangnya sendiri), garis luar label, kedua
   titik terdekat, dan panah tumor → tepi. Gambar tidak memuat teks; angkanya
   ada di keterangan Markdown.
4. Invasi: per struktur ada/tidak, luas kontak, daftar tambalan.
5. Ukuran staging: diameter maks, tiga sumbu PCA, kedalaman/Breslow.
6. Atribut: per atribut, statistik per region, heterogenitas, tabel per
   irisan, parameter skor.
7. Provenance: fraksi irisan per sumber, irisan belum ditinjau.
8. Parameter, versi, peringatan, dan kalimat "Hanya untuk riset".

Peringatan otomatis (tampil di mode `Report` dan di laporan):

| Kondisi | Pesan |
|---|---|
| `d_min < margin_close_um` | Tumor dekat tepi reseksi arah … |
| Label `Margin` tidak ada di irisan berlabel pertama/terakhir | Jarak ke tepi pada arah Z tidak dapat dinilai |
| Rentang skor IHC antar irisan > `ihc_hetero_warn` (bawaan 20 poin) | Heterogenitas IHC tinggi; satu irisan mungkin tidak representatif |
| Fraksi irisan `Interpolated` > 50 % | Sebagian besar bentuk berasal dari interpolasi |
| Ada `Draft` | Angka memuat anotasi yang belum ditinjau |
| `s_z` > 2 × `margin_close_um` | Resolusi Z lebih kasar dari ambang margin |
| Skor membran berbasis luas | Skor … dihitung per luas, bukan per sel |

#### 8.12.9 Berkas, privasi, dan ekspor

Bundel (8.7.1) bertambah:

```
sampel-01.h3d/
├── project.json            # + labels.kind/group, attrs, annotations, spatial
├── clinical.json           # ClinicalMeta; OPSIONAL; terpisah agar mudah dibuang
├── attrs/
│   └── 0007.ki67.atr       # AttrStore per (irisan, atribut)
└── out/
    ├── figures/
    └── analysis.json       # keluaran 8.12.6 lengkap (dibuat ulang)
```

- `clinical.json` **dipisah** dari `project.json` sehingga
  `h3d clinical strip` cukup menghapus satu berkas dan proyek tetap sah.
  `project.json` tidak boleh memuat apa pun dari `ClinicalMeta`.
- Isi `clinical.json` sudah pseudonim; pemetaan `subject_pseudo_id` ke
  identitas **tidak pernah** ada di dalam bundel. Tanggal disimpan
  bulan-tahun saja.
- iPad: berkas proyek ditulis dengan kelas proteksi
  `NSFileProtectionComplete` (diset di `h3d-app` saat membuat berkas),
  sehingga terenkripsi saat perangkat terkunci. Mac: tidak ada enkripsi
  tambahan; dokumen pengguna menyarankan FileVault.
- Ekspor (8.7.3) **tidak** menyertakan `clinical.json` kecuali
  `--with-clinical`; laporan hanya memuat `subject_pseudo_id`, organ, dan
  diagnosis, tidak memuat usia/jenis kelamin kecuali flag yang sama.
- Format `.atr`: sama dengan `.lbl` (8.7.2) dengan magic `"H3DA"`, tile
  `u16`, dan header tambahan `scale: f32`, `offset: f32`, `attr_id: u16`.
- Ekspor tambahan: GLB membawa `COLOR_0` dari colormap atribut terpilih dan
  nilai mentah sebagai atribut verteks khusus `_H3D_<NAMA>` (`f32`, diizinkan
  glTF untuk awalan `_`); NIfTI atribut `f32` per `AttrVolume`
  (`<nama>.nii.gz`); GeoJSON kontur mendapat `properties.measurements`
  (luas, skor IHC rata-rata) agar terbaca QuPath; `analysis.json` adalah
  sumber untuk CSV.

#### 8.12.10 Antarmuka

- Mode `Slides`: panel "Klinis" (form `ClinicalMeta`, orientasi dengan
  gambar tiga sumbu), panel "Irisan" bertambah pra-analitik dan panel IHC.
- Mode `Segment`: pemilih `kind` saat membuat label (ikon dari
  `egui_icons`); overlay inti positif/negatif saat mengkalibrasi `t_D`;
  tombol "Tandai sudah ditinjau".
- Mode `Reconstruct`: pemilih lapisan skalar (jarak margin, hipoksia,
  atribut) dengan legenda colormap dan rentang; klik pada mesh menampilkan
  nilai di titik itu dan irisan terdekat.
- Mode `Report`: bagian 8.12.8 sebagai tab; peringatan di atas.

#### 8.12.11 Validasi

Semua lewat fantom (11.1) yang diperluas:

| Fantom | Kebenaran dasar | Kriteria **[TARGET]** |
|---|---|---|
| Ellipsoid + bidang `Margin` pada jarak `d` dari permukaan | `d_min = d` analitik; arah = normal bidang | galat ≤ `s_xy`; arah benar |
| Dua bola berjarak `g` | 2 fokus; jarak `g` | jumlah fokus benar; galat ≤ `s_xy` |
| Silinder `Vessel` menembus ellipsoid | luas kontak = keliling × panjang di dalam | galat ≤ 10 % |
| Bola `Tumor` di dalam kulit bola `Capsule`, satu varian tembus | tembus / tidak | benar pada kedua varian |
| Lapisan datar `Surface` di `z = 0`, tumor hingga kedalaman `h` | kedalaman = `h` | galat ≤ `s_z/2` |
| Gambar IHC sintetis: inti dicat dengan fraksi positif `p` per tile | `%pos = p` | galat ≤ 2 poin; deteksi inti F1 ≥ 0,9 |
| Fantom H&E dengan 2 kelas | provenance | `AnnotationRecord` benar untuk manual/AI/interpolasi |
| Acak (`proptest`) | — | bolak-balik `.atr`; `clinical.json` tidak pernah bocor ke `project.json` atau ekspor tanpa flag |

Pada sampel nyata: `d_min` dibandingkan dengan margin yang dilaporkan
patolog di 2D pada irisan yang sama (harus ≤ nilai 2D, karena 3D mencari
minimum global); skor Ki-67 per irisan dibandingkan dengan QuPath pada 10
tile acak (Bland–Altman, bias ≤ 5 poin).


---

## 9. Rancangan khusus iPad

### 9.1 Build

- Target: `aarch64-apple-ios` (perangkat), `aarch64-apple-ios-sim`
  (simulator).
- `eframe` tanpa fitur `glow`; backend Metal lewat wgpu.
- Skrip `build_ipad.sh` diadaptasi dari DUCAD (aksi `device`, `sim`,
  `run-sim`, `archive`, `ipa`, `check`).
- CI minimum: `cargo check --target aarch64-apple-ios -p h3d-app` di tiap
  PR, supaya dependensi yang tidak bisa dikompilasi untuk iOS tertangkap
  seketika.

### 9.2 Anggaran memori **[TARGET]**

Batas memori per aplikasi di iPadOS bergantung perangkat dan tidak
didokumentasikan pasti; angka ini titik awal yang harus diukur di H0.

| Anggaran | macOS | iPad |
|---|---|---|
| Cache tile CPU | 2 GB | 384 MB |
| Cache tile GPU | 1 GB | 256 MB |
| Set kerja rekonstruksi | 4 GB | 1 GB |
| Thread dekode | 4 | 2 |

Anggaran berada di satu struct `Budgets` yang dipilih saat mulai menurut
platform dan memori fisik; tidak ada angka tertanam di tempat lain.

### 9.3 Berkas

- Impor di iPad **menyalin** slide ke dalam bundel proyek
  (`SlideSourceRef::Embedded`) di folder Documents aplikasi. Lebih boros
  ruang, tetapi menghindari akses berkas luar-sandbox yang bisa hilang.
- Sebelum menyalin, periksa ruang kosong dan tampilkan ukuran yang
  dibutuhkan.
- Folder Documents dibuka ke aplikasi Files agar proyek bisa dipindah lewat
  AirDrop atau kabel.

### 9.4 Siklus hidup

- **Simpan otomatis** tiap suntingan label selesai (per goresan) dan saat
  aplikasi masuk latar belakang; iPadOS boleh mematikan aplikasi latar kapan
  saja.
- Job panjang (registrasi, inferensi, rekonstruksi) bersifat **bisa
  dilanjutkan**: hasil per irisan disimpan begitu selesai, sehingga job yang
  terputus melanjutkan dari irisan berikutnya.
- Saat menerima peringatan memori, kosongkan cache tile CPU dan GPU.

### 9.5 Kinerja **[TARGET]**

| Operasi | macOS (Apple Silicon) | iPad |
|---|---|---|
| Geser/zoom viewer 2D | 60 fps | 60 fps |
| Goresan kuas → layar | < 16 ms | < 16 ms |
| Registrasi 1 pasang (affine, 16 µm) | < 2 dtk | < 5 dtk |
| Registrasi 100 irisan | < 1 mnt | < 5 mnt |
| Rekonstruksi contoh 8.6.5 | < 10 dtk | < 30 dtk |
| Putar mesh 500 ribu segitiga | 60 fps | 60 fps |

---

## 10. Rencana implementasi

Urutan fase mengikuti ketergantungan. **MVP = H0 sampai H5.** Tiap fase
berakhir dengan semua gerbang berikut hijau:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo check --target aarch64-apple-ios -p h3d-app     # sejak H2
```

### H0 — Pembuktian dan keputusan (gerbang lanjut/berhenti)

**Tujuan:** menghilangkan semua tanda [BELUM DIVERIFIKASI] dan mendapat data
nyata.

| Tugas | Cara | Keluaran |
|---|---|---|
| H0.1 Data | Minta 1 sampel nyata dari pemilik proyek (≥ 20 irisan) + jawaban bagian 13 | Folder data + catatan |
| H0.2 `wsi-rs` | Program kecil: buka SVS dan NDPI, baca wilayah di tiap level, bandingkan dengan OpenSlide (Python); kompilasi untuk `aarch64-apple-ios`; cek lisensi | Lulus/gagal per format |
| H0.3 `sitk` | Registrasi affine 5 pasang irisan; bandingkan TRE dengan SimpleITK asli (Python) pada landmark yang sama; cek lisensi dan pemeliharaan | Tabel TRE |
| H0.4 Registrasi bawaan | Prototipe R1–R3 (boleh kasar) pada pasangan yang sama | Tabel TRE |
| H0.5 `tract` | Muat satu model segmentasi ONNX (mis. U-Net kecil), jalankan satu tile di macOS dan simulator iOS | Lulus/gagal + waktu |
| H0.6 Mesh | `fast-surface-nets` + `baby_shark` pada SDF bola; kompilasi untuk iOS | Lulus/gagal |
| H0.7 Memori iPad | Aplikasi uji yang mengalokasikan bertahap sampai dimatikan sistem, di perangkat target | Angka anggaran nyata |
| H0.8 Alur acuan | Jalankan sampel nyata lewat Fiji/QuPath/3D Slicer sekali | Hasil pembanding untuk H5 |

Data berlandmark publik untuk H0.3/H0.4: set tantangan ANHIR (registrasi
histologi); periksa lisensinya sebelum dipakai.

**Keluaran dokumen:** `docs/adr/0001-arsitektur.md` (salinan keputusan K1–K13
yang sudah disesuaikan hasil H0) dan `docs/adr/0002-registrasi.md` (backend
mana yang jadi bawaan).

**Kriteria selesai:**

- Tiap baris [BELUM DIVERIFIKASI] di 6.1 berganti jadi lulus atau diganti
  alternatifnya.
- Keputusan tertulis: backend registrasi bawaan; pembaca WSI di iPad.
- Bila `wsi-rs` gagal: iPad hanya mendukung TIFF piramida + gambar biasa,
  dan desktop boleh memakai OpenSlide di balik fitur. Bila registrasi bawaan
  **dan** `sitk` gagal mencapai TRE yang layak: **berhenti dan tinjau ulang**
  (pilihan: registrasi sebagai pra-proses desktop dengan elastix).

### H1 — Kerangka, inti, pembacaan slide

**Tugas:**

1. Workspace, `deny.toml`, `rust-toolchain.toml`, `CLAUDE.md`, CI.
2. `h3d-core`: semua tipe 8.1; `Affine2` lengkap; undo; `Progress`.
3. `h3d-slide`: trait, tiga backend, cache tile CPU.
4. `h3d-io`: baca/tulis `project.json` (atomik, berversi).
5. `h3d-engine`: `Session::{create, open, save, import_slides}`; tes
   `engine_has_no_gui_dependency`.
6. `h3d-cli`: `info`, `thumb`, `new`, `import`.

**Tes wajib:**

- proptest `Affine2`: `inverse(inverse(A)) ≈ A`;
  `compose(A, inverse(A)) ≈ I`; `apply(compose(A, B), p) ≈ apply(A, apply(B, p))`.
- `FlatBackend`: tiap level piramida berukuran `ceil(sebelumnya / 2)`.
- Bolak-balik `project.json`: simpan → muat → sama persis.
- Berkas rusak/terpotong → error, bukan panik.

**Kriteria selesai:** `h3d import` atas sampel nyata membuat proyek yang
bisa dibuka lagi; `h3d thumb` menghasilkan gambar benar untuk tiap format.

### H2 — Viewer 2D (macOS + iPad)

**Tugas:**

1. `h3d-render`: pipeline tile, cache GPU, pemilihan level, tile induk
   sebagai pengganti sementara.
2. `h3d-i18n`, `h3d-ui` (salin tes cakupan glyph).
3. `h3d-app`: kerangka eframe, `AppMode::Slides`, viewer geser/zoom,
   navigasi irisan, impor (rfd di desktop; folder Documents di iPad).
4. `build_macos.sh`, `build_ipad.sh`.

**Kriteria selesai:** WSI terbesar di sampel bisa digeser dan di-zoom mulus
di Mac dan iPad fisik; memori tetap dalam anggaran 9.2 selama 5 menit
pemakaian (diukur dengan Instruments).

### H3 — Registrasi rigid/affine

**Tugas:**

1. `h3d-imgproc`: dekonvolusi, masker jaringan, piramida, `warp_affine`.
2. `h3d-register`: `BuiltinRegistrar` (R1–R3), `register_stack`, QC,
   affine dari landmark.
3. Engine + CLI: `register`, `qc`; cache gambar registrasi.
4. Render: pipeline pembanding. App: mode `Register` lengkap.
5. `h3d phantom` (bentuk ellipsoid) agar tes punya kebenaran dasar.

**Tes wajib:**

- Fantom dengan transformasi acak yang diketahui (rotasi 0–360°, geser
  ± 20% lebar, cermin acak, skala 0,95–1,05): TRE rata-rata ≤ 2 piksel
  pada resolusi registrasi **[TARGET]**.
- Korelasi fase: geseran bulat yang diketahui ditemukan persis.
- Landmark: 3 pasang titik dari affine yang diketahui → affine itu kembali.
- Rantai: `T_i` hasil komposisi sama dengan transformasi kebenaran dasar.

**Kriteria selesai:** pada sampel nyata, ≥ 90% pasangan lolos QC tanpa
campur tangan **[TARGET]**; sisanya bisa dibetulkan dengan landmark dalam
< 1 menit per irisan.

### H4 — Label dan segmentasi manual

**Tugas:**

1. `h3d-segment`: `LabelStore`, cakram, isi poligon, isi lubang.
2. `h3d-io`: format `.lbl`; impor/ekspor GeoJSON.
3. Command undo berbasis tile.
4. Render: overlay label. App: mode `Segment`, panel label, alat, tekanan
   Pencil, simpan otomatis.

**Tes wajib:**

- Bolak-balik `.lbl` (proptest atas tile acak).
- `.lbl` rusak (magic salah, offset di luar berkas, payload terpotong) →
  error.
- Luas cakram berjari-jari `r` ≈ `πr²` (galat < 2% untuk `r` ≥ 10 px).
- Undo setelah N goresan mengembalikan keadaan persis.
- GeoJSON: ekspor → impor → Dice ≥ 0,99.

**Kriteria selesai:** satu irisan bisa dianotasi nyaman dengan Pencil di
iPad (latensi sesuai 9.5); label tetap di tempatnya setelah registrasi
diulang.

### H5 — Rekonstruksi, 3D, pengukuran, ekspor (akhir MVP)

**Tugas:**

1. `h3d-imgproc`: EDT, SDF bertanda.
2. `h3d-volume`: kisi, langkah 8.6.2, pengukuran 8.6.4, penurunan resolusi
   otomatis.
3. `h3d-io`: GLB, STL, OBJ, NIfTI, laporan.
4. Render: mesh 3D, irisan di 3D, bidang potong. App: mode `Reconstruct`
   dan `Report`.
5. CLI: `reconstruct`, `measure`, `export`.

**Tes wajib:**

- EDT terhadap perhitungan brute-force pada gambar kecil acak (sama persis).
- Fantom ellipsoid: galat volume Cavalieri ≤ 2% dan volume mesh ≤ 3% saat
  jarak irisan ≤ 1/10 diameter **[TARGET]**.
- Volume mesh bertanda **positif** (orientasi segitiga benar).
- Mesh tertutup: tiap sisi dipakai tepat dua segitiga.
- Fantom torus: genus 1 (karakteristik Euler 0). Fantom bercabang: satu
  komponen.
- Dua kali rekonstruksi atas masukan sama → keluaran identik byte demi byte.

**Kriteria selesai:** sampel nyata menghasilkan mesh dan laporan; volume
label NIfTI yang dibuka di 3D Slicer memberi volume yang sama dalam 1%;
hasil dibandingkan dengan alur acuan H0.8.

### H6 — Interpolasi dan segmentasi klasik

Tugas: 8.5.5 dan 8.5.3; lapisan usulan dengan Terima/Tolak; penanda asal
label; CLI `interpolate`.

Tes: fantom dengan anotasi tiap irisan ke-5 → Dice hasil interpolasi vs
kebenaran dasar ≥ 0,95 **[TARGET]**.

Tugas tambahan: usulan urutan dan orientasi irisan (8.4.7); CLI
`suggest-order`; `h3d phantom --shuffle`; tombol dan pratinjau di mode
`Slides`.

Tes usulan urutan:

- Fantom `branch` 60 irisan, diacak urutannya dan diberi putar/cermin acak
  berbenih → ≥ 95% pasangan bertetangga benar (urutan asli atau
  kebalikannya), dan cermin terdeteksi benar pada ≥ 95% irisan **[TARGET]**.
  Ellipsoid **tidak** dipakai: simetris terhadap bidang tengah sehingga
  urutannya memang tidak bisa ditentukan. Tekstur fantom harus berkorelasi
  antar irisan bertetangga, bukan acak lepas per irisan.
- Masukan yang sudah berurutan → usulan tidak mengubah apa pun.
- Dua kali dijalankan atas masukan sama → usulan identik.
- Menerima usulan tidak mengubah `z_um` yang diisi tangan dan tidak mengubah
  label.

### H7 — Segmentasi AI

Tugas: `Segmenter`, `TractSegmenter`, alur 8.5.4, fitur `ort-coreml`, CLI
`segment`, layar pengelola model.

Tes: model uji mungil (dibuat untuk tes, beberapa KB) → keluaran sama dengan
ONNX Runtime acuan; penggabungan tile tanpa garis sambungan (masukan konstan
→ keluaran konstan).

Kriteria selesai: satu WSI tersegmentasi di iPad tanpa melampaui anggaran
memori.

### H8 — Registrasi non-rigid

Tugas: `DisplacementGrid`, B-spline FFD di atas hasil affine (jarak titik
kendali ~1 mm, penalti energi lentur, berjenjang), batas perpindahan
maksimum, QC tambahan (determinan Jacobian > 0 di semua titik).

Hanya dikerjakan bila hasil affine di sampel nyata terbukti tidak cukup.

### H9 — Penyempurnaan

Volume rendering (raymarching), relaksasi rantai (registrasi ke `i ± 2` lalu
kuadrat terkecil untuk mengurangi hanyut), TestFlight, dokumen pengguna
(termasuk batasan 8.4.6), laporan validasi.

### H10 — Atribut medis dan analisis spasial

Dikerjakan setelah H5 dan H6 (butuh masker voxel dan provenance). Dibagi
tiga sub-fase agar tiap bagian bisa dipakai lebih dulu.

**H10a — Metadata dan provenance.** `ClinicalMeta`, `SpecimenOrientation`,
`PreanalyticMeta`, `IhcPanel`, `LabelKind`, `group`, `AnnotationRecord`;
`clinical.json` terpisah; migrasi `format_version`; panel UI 8.12.10
(Slides, Segment); CLI `clinical`, `review`; laporan bagian 1, 2, 7.

Tes: bolak-balik serde semua tipe; proyek lama tanpa field baru terbuka
dengan nilai bawaan; `clinical.json` tidak ada → `clinical = None` tanpa
error; `proptest`: tidak ada field `ClinicalMeta` yang muncul di
`project.json` atau di keluaran `export` tanpa `--with-clinical`; validasi
orientasi menolak dua sumbu dari pasangan yang sama.

**H10b — Analisis spasial.** `edt3` anisotropik per chunk; margin, kontak,
penetrasi kapsul, multifokalitas, kedalaman, PCA ukuran, hipoksia;
`MeshScalars`; pipeline skalar di render; legenda; `analysis.json`; laporan
bagian 3–5; peringatan 8.12.8; CLI `analyze`; fantom baru di `h3d phantom`.

Tes: `edt3` terhadap brute-force pada volume kecil acak anisotropik (sama
persis); semua baris 8.12.11 kecuali IHC; dua kali `analyze` → `analysis.json`
identik byte demi byte; memori `edt3` pada contoh 8.6.5 ≤ anggaran.

**H10c — Atribut.** `AttrStore`, `.atr`, deteksi inti LoG, dekonvolusi
kromogen, skor %pos / H-score / Allred / berbasis luas / TPS-CPS, densitas
inti, TIL intra/peri, `AttrVolume`, statistik per region, NIfTI dan GLB
`COLOR_0`, overlay kalibrasi, CLI `attrs compute`, laporan bagian 6.

Tes: fantom IHC 8.12.11; penurunan resolusi mengabaikan NODATA (tile
setengah kosong → nilai bagian berisi, bukan setengahnya); bolak-balik
`.atr`; berkas `.atr` rusak → error.

**Kriteria selesai H10:** pada sampel nyata dengan tinta margin dan satu
irisan Ki-67, laporan memuat `d_min` per arah anatomis yang cocok dengan
pembacaan patolog 2D (≤ nilai 2D), mesh tumor berwarna jarak margin tampil
di iPad, dan Ki-67 per irisan berbeda ≤ 5 poin dari QuPath.

### H11 — Lanjutan atribut (opsional)

Tumor budding (8.12.5 butir 8, perlu `mpp_label ≤ 2 µm`), skor HER2 per sel
(segmentasi membran), head inti dari model AI, DICOM WSI Annotation untuk
interoperabilitas. Dikerjakan hanya bila ada permintaan dari pemilik proyek.

### Ketergantungan antar fase

```
H0 ─► H1 ─► H2 ─► H3 ─► H5 ─► H6 ─► H7
              └──► H4 ──┘      ├──► H8 ─► H9
                               └──► H10a ─► H10b ─► H10c ─► H11
```

H3 dan H4 bisa dikerjakan paralel setelah H2. H10a hanya butuh H4 (label)
dan bisa maju lebih awal; H10b butuh H5; H10c butuh H10b dan H7 bila head
inti AI dipakai.

---

## 11. Strategi pengujian

### 11.1 Fantom sintetis

Sumber kebenaran utama, karena data nyata tidak punya jawaban benar.

1. Bentuk 3D analitik: ellipsoid, torus, cabang Y (gabungan kapsul); varian
   8.12.11: bidang margin, dua fokus, silinder pembuluh, kulit kapsul,
   lapisan permukaan, dan gambar IHC sintetis dengan fraksi positif diketahui.
2. Diiris pada `z` tertentu → masker 2D.
3. Dirender jadi "gambar histologi" sintetis: tekstur bintik (inti) di dalam,
   latar putih, derau.
4. Tiap irisan diberi transformasi acak **yang dicatat** (berbenih).
5. Pipeline dijalankan; hasil dibandingkan dengan: transformasi tercatat
   (TRE), masker asli (Dice), volume analitik.

### 11.2 Jenis tes

| Jenis | Isi |
|---|---|
| Unit | Tiap fungsi `imgproc`, `Affine2`, `LabelStore`, format berkas. |
| Properti (`proptest`) | Aljabar transformasi, bolak-balik berkas, undo. |
| Emas | Keluaran CLI atas fantom berbenih dibandingkan dengan berkas tersimpan (`testdata/golden/`, tes `golden_files_match_cli_output_of_seeded_phantoms`; segarkan dengan `H3D_UPDATE_GOLDEN=1`, lihat `testdata/README.md`). |
| Ketahanan | Berkas rusak/terpotong/kosong → error, tidak pernah panik. |
| Bench (`criterion`) | EDT 2D dan 3D, registrasi satu pasang, surface nets, cat kuas, deteksi inti satu tile. |
| Arsitektur | `engine_has_no_gui_dependency`; tidak ada `opencascade` di pohon dependensi. |
| UI | `ui_glyph_coverage`. |
| Validasi silang | NIfTI → 3D Slicer; GeoJSON → QuPath (manual, tiap rilis). |

### 11.3 Perangkat

Tiap fase sejak H2 diuji di iPad fisik, bukan hanya simulator; simulator
tidak mencerminkan batas memori maupun kinerja GPU.

---

## 12. Risiko

| Risiko | Peluang | Dampak | Mitigasi |
|---|---|---|---|
| Registrasi bawaan tidak cukup baik pada jaringan nyata (robek, lipatan, beda pewarnaan) | Tinggi | Tinggi | H0.4 lebih dulu; QC + koreksi landmark; backend bisa diganti lewat trait; cadangan pra-proses elastix di desktop. |
| `wsi-rs` belum matang atau tidak jalan di iOS | Sedang | Tinggi | H0.2; cadangan: TIFF piramida saja di iPad, konversi di desktop. |
| `sitk` tidak setara SimpleITK | Sedang | Rendah | Hanya backend opsional; bawaan tidak bergantung padanya. |
| Efek "pisang": bentuk global terluruskan | Tinggi | Sedang | Tidak bisa dihilangkan tanpa acuan luar; didokumentasikan (8.4.6); relaksasi rantai di H9. |
| Memori iPad tidak cukup untuk sampel besar | Sedang | Sedang | Anggaran terpusat; penurunan resolusi otomatis; pemrosesan per chunk; ukur di H0.7. |
| Operator ONNX model pengguna tidak didukung `tract` | Sedang | Sedang | Fitur `ort-coreml`; dokumentasikan operator yang didukung. |
| Usulan urutan salah dan diterima tanpa diperiksa → bentuk 3D salah | Sedang | Tinggi | Selalu lewat Terima/Tolak dengan pratinjau; irisan ragu disorot; `order_source` tercatat di laporan; nama berkas/catatan lab tetap sumber utama (8.4.7). |
| mpp atau jarak irisan salah isi → semua ukuran salah | Sedang | Tinggi | Tidak ada nilai tebakan; alat ukur garis skala; nilai tampil jelas di laporan. |
| Hasil dipakai untuk keputusan klinis | Rendah | Tinggi | Label "hanya untuk riset" di aplikasi dan laporan; tinjau regulasi sebelum distribusi publik. |
| Data pasien bocor | Rendah | Tinggi | K13: tanpa jaringan sama sekali; metadata slide yang memuat identitas tidak ikut diekspor; K16: `clinical.json` pseudonim dan terpisah, tes properti menjamin tidak bocor ke `project.json`/ekspor; `NSFileProtectionComplete` di iPad (8.12.9). |
| Angka margin/invasi 3D dipercaya melebihi ketelitiannya (anisotropi Z, galat registrasi) | Sedang | Tinggi | Setiap jarak membawa ± ketidakpastian (8.12.6); peringatan bila `s_z` kasar atau margin tidak ada di irisan ujung; tanpa kalimat rekomendasi (K15). |
| Skor IHC klasik meleset dari patolog (kromogen lemah, latar, kontra-pewarnaan kuat) | Tinggi | Sedang | Ambang per irisan dengan overlay kalibrasi; parameter tercetak di laporan; validasi Bland–Altman vs QuPath; skor membran ditandai "berbasis luas". |
| Orientasi anatomis salah diisi → nama arah margin salah | Sedang | Tinggi | Gambar tiga sumbu saat mengisi; laporan selalu menulis arah mentah (±X/±Y/±Z) di samping nama anatomis. |
| Lisensi crate tidak cocok | Rendah | Sedang | `cargo deny` di CI; cek lisensi `wsi-rs` dan `sitk` di H0. |

---

## 13. Pertanyaan terbuka untuk pemilik proyek

Jawaban mengubah rancangan; tanyakan sebelum H1.

1. **Jenis irisan**: potongan fisik (mikrotom) atau z-stack optik
   (konfokal)? Bila z-stack, H3 dan H8 hampir tidak diperlukan.
2. **Format berkas** dari mikroskop atau pemindai: SVS, NDPI, TIFF, atau
   JPEG dari kamera? Adakah metadata mpp?
3. **Skala**: berapa irisan per sampel, berapa jarak antar irisan, berapa
   ukuran berkas?
4. **Irisan berurutan penuh** atau hanya tiap irisan ke-N yang dipindai?
5. **Pewarnaan**: H&E saja, atau bercampur IHC?
6. **Tujuan akhir**: gambar untuk publikasi, angka volume, cetak 3D, atau
   lainnya?
7. **Segmentasi**: siapa yang menandai tumor (ahli patologi secara manual,
   atau sudah ada model AI)?
8. **Acuan luar**: adakah foto blockface atau penanda fidusial?
9. **Peran iPad**: alur penuh di iPad, atau iPad hanya untuk anotasi dan
   tampilan sementara komputasi berat di Mac?
10. **Jenis kanker** sasaran pertama (payudara, kolorektal, prostat, kulit,
    …)? Menentukan daftar label bawaan, sistem grade, dan ukuran staging
    mana yang relevan (kedalaman invasi vs Breslow vs diameter) di 8.12.
11. **Panel IHC** yang tersedia (antibodi, klon, kromogen) dan siapa yang
    menetapkan ambang positif? Apakah skor patolog 2D tersedia untuk
    validasi?
12. **Praktik margin**: apakah spesimen reseksi diberi tinta per sisi (warna
    berbeda per arah) dan apakah orientasi spesimen tercatat?
13. **Data klinis**: field mana dari 8.12.2 yang benar-benar tersedia dan
    boleh disimpan di perangkat menurut etika/IRB setempat?

---

## 14. Aturan keras untuk pelaksana

Salin ke `CLAUDE.md` repo baru.

1. **Tanpa OCCT, ITK, VTK, OpenCV.** Tidak ada crate yang menariknya.
2. **`h3d-engine` dan crate di bawahnya bebas GUI** (dijaga tes).
3. **Satuan**: µm `f64` di data; mm `f32` hanya di `h3d-render`. Tidak ada
   koordinat piksel yang disimpan di proyek selain di GeoJSON.
4. **Arah transformasi** mengikuti 5.3. Tiap fungsi yang menerima
   transformasi menyebut arahnya di doc comment.
5. **Pusat piksel di +0,5** di semua konversi.
6. **Tidak ada `unwrap()`/`expect()` pada masukan luar** (berkas, argumen
   CLI, model, metadata slide). Berkas rusak menghasilkan error.
7. **mpp tidak pernah ditebak.**
8. **Usulan tidak pernah menimpa label, urutan, atau orientasi irisan**
   tanpa "Terima" dari pengguna.
9. **Deterministik**: tanpa acak tak berbenih; hasil paralel digabung dalam
   urutan tetap.
10. **Tulis atomik** untuk semua berkas proyek.
11. **Tanpa jaringan.** Tidak ada crate HTTP di pohon dependensi `h3d-app`.
12. **Komentar dan pesan pengguna berbahasa Indonesia; identifier Inggris.**
13. **Ikon dari `egui_icons`**, bukan glyph Unicode/emoji; jalankan
    `cargo test -p h3d-ui --test ui_glyph_coverage` setelah menambah teks UI.
14. **Dependensi baru** harus lolos `cargo deny` dan `cargo check --target
    aarch64-apple-ios`.
15. **Angka anggaran** hanya di struct `Budgets`.
16. **Semua laporan dan layar Tentang** memuat "Hanya untuk riset, bukan
    untuk diagnosis".
17. **Data klinis hanya di `clinical.json`**, pseudonim, tanggal bulan-tahun;
    tidak pernah masuk `project.json`, log, atau ekspor tanpa `--with-clinical`
    (dijaga tes properti).
18. **Setiap jarak/ukuran 3D dilaporkan dengan ketidakpastian**; tidak ada
    teks rekomendasi tindakan di kode, laporan, atau UI. `ClinicalMeta`
    tidak pernah menjadi masukan perhitungan.
19. **Analisis spasial dihitung pada masker voxel**, bukan mesh; parameter
    atribut tersimpan sebagai hash dan tercetak di laporan.

---

## 15. Rujukan algoritma

- Ruifrok & Johnston (2001) — dekonvolusi warna histologi.
- Otsu (1979) — ambang otomatis.
- Felzenszwalb & Huttenlocher (2012) — transformasi jarak Euclid eksak.
- Evangelidis & Psarakis (2008) — penyejajaran gambar dengan ECC.
- Mattes dkk. (2003) — informasi mutual untuk registrasi.
- Rueckert dkk. (1999) — registrasi non-rigid B-spline (FFD).
- Raya & Udupa (1990) — interpolasi berbasis bentuk.
- Gibson (1998) — surface nets.
- Taubin (1995) — penghalusan mesh tanpa penyusutan.
- Estimator Cavalieri — stereologi klasik untuk volume dari penampang.
- Al-Kofahi dkk. (2010) — deteksi inti LoG multiskala.
- McCarty dkk. (1985) — H-score; Allred dkk. (1998) — skor Allred ER/PR.
- Salgado dkk. (2015) — rekomendasi International TIL Working Group.
- Lugli dkk. (2017) — konsensus ITBCC untuk tumor budding.
- Thomlinson & Gray (1955) — jarak difusi oksigen ~150 µm (proksi hipoksia).
- Amin dkk. (2017) — AJCC Cancer Staging Manual edisi 8 (ukuran, kedalaman invasi, margin).
- Breslow (1970) — ketebalan melanoma.
