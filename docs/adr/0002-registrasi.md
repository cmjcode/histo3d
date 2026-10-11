# ADR 0002 — Backend registrasi bawaan

Status: diterima (2026-10-10); bagian "Rigid dulu" ditambahkan 2026-10-11.

## Konteks

histo3d.md K4 dan 8.4 mewajibkan registrasi di balik trait `Registrar`, dengan
`BuiltinRegistrar` (momen → korelasi fase → ECC) dan `sitk` sebagai backend
opsional yang hanya diaktifkan bila H0.3 membuktikannya setara SimpleITK.

## Temuan

- `sitk` di crates.io berada di versi 0.0.3 (Juli 2026), mengaku "pure-Rust
  port of SimpleITK". Tidak ada rilis stabil, tidak ada bukti kesetaraan, dan
  tidak ada data nyata untuk mengujinya (H0.1 belum ada).
- `BuiltinRegistrar` hanya butuh `rustfft`, `ndarray`, `rayon`; semuanya
  sudah lulus kompilasi iOS.

## Keputusan

1. **Bawaan: `BuiltinRegistrar`** (R1 momen, R2 korelasi fase, R3 ECC
   berjenjang 64→32→16 µm; MI Nelder–Mead untuk beda pewarnaan).
2. **`sitk` tidak ditambahkan** ke pohon dependensi. Fitur Cargo `sitk`
   dicadangkan namanya, tetapi kodenya tidak ditulis sampai ada data nyata
   dan versi `sitk` ≥ 0.1 yang bisa diuji terhadap SimpleITK.
3. Kriteria membuka kembali keputusan: TRE `BuiltinRegistrar` pada sampel
   nyata (H0.1) gagal mencapai ≤ 2 px pada resolusi registrasi, **dan**
   koreksi landmark manual tidak cukup dalam < 1 menit/irisan (H3).
4. Cadangan bila keduanya gagal: pra-proses desktop dengan elastix (di luar
   aplikasi), sesuai 10.H0.

## Rigid dulu: model bebas hanya bila terbukti perlu (2026-10-11)

### Masalah yang terukur

Pada pipeline fantom ellipsoid berbenih (24 irisan, piksel registrasi 16 µm)
model `Affine` tanpa pengaman memberi TRE rantai rata-rata 26,9 µm / terburuk
58,6 µm dan volume mesh +2,8 % (voxel +2,6 %) terhadap analitik; `Rigid`
memberi 12,0 / 19,0 µm dan −0,7 %. `Similarity` berperilaku seperti affine
(+2,5 %).

Diagnosis per pasangan terhadap kebenaran dasar (transformasi sebenarnya
rigid, `|det| = 1`):

- `|det P|` pasangan menyimpang **searah**: +0,2 … +2,7 % (nilai singular
  hingga 1,022) pada irisan yang penampang tumornya mengecil menjauhi acuan,
  dengan geser kecil yang juga searah.
- Karena `T_i = T_tetangga ∘ P`, simpangan itu **berlipat** di sepanjang
  rantai: `|det T_i|` mencapai 1,11 (nilai singular 1,09 × 1,01) di ujung.
- NCC tiap pasangan nyaris tidak berubah (affine vs rigid: +0,000 … +0,009),
  sehingga QC per pasangan tidak melihat apa-apa.

Akar masalah: penampang tumor memang berubah dari irisan ke irisan. ECC hanya
melihat piksel di irisan kedua masker, jadi buta terhadap garis luar jaringan,
dan "menjelaskan" perubahan bentuk itu sebagai skala/geser. Ini bukan galat
numerik; registrasi berpasangan tanpa acuan luar memang tidak bisa membedakan
deformasi irisan dari perubahan anatomi (bandingkan 8.4.6).

### Pilihan yang ditimbang

- (a) **Rigid dulu, model bebas hanya bila skornya membaik nyata** — dipilih.
- (b) Koreksi tingkat rantai yang membuang hanyutan skala/geser (membuang
  komponen frekuensi rendah regangan sepanjang z). Tidak dipilih sekarang:
  `T_i` tidak lagi sama dengan komposisi `pairwise` yang tersimpan, sehingga
  koreksi landmark/manual (8.4.5: "dihitung ulang dari `pairwise` tanpa
  registrasi ulang") dan kebenaran rantai harus diberi pengecualian; perlu
  data nyata untuk memilih lebar tapisnya.
- (c) `Rigid` sebagai bawaan, affine opt-in. Tidak dipilih: jaringan nyata
  memang terdeformasi antar irisan (kompresi pisau beberapa persen), dan
  rigid murni meninggalkan galat yang sebanding dengan jarak dari pusat
  jaringan. Dengan (a) bawaan affine sudah berperilaku rigid bila affine tidak
  dibutuhkan.

### Keputusan

1. Model bawaan tetap `Affine` (CLI `--model affine`), tetapi dengan metrik
   NCC `Similarity`/`Affine` dijalankan **setelah** penghalusan rigid dan
   mulai dari hasil rigid.
2. Hasil model bebas diterima hanya bila skor = NCC bermasker × Dice masker
   (skor kandidat R2) naik setidaknya 0,01 **dan** setidaknya 25 % dari
   `1 − skor_rigid` (`AFFINE_MIN_SCORE_GAIN`, `AFFINE_MIN_RESIDUAL_GAIN` di
   `h3d-register::builtin`). Bila tidak, hasil rigid yang dipakai. Dice ikut
   dihitung karena garis luar jaringan adalah saksi terkuat untuk skala.
3. Ambang berasal dari pengukuran: perubahan bentuk murni pada fantom memberi
   fraksi ≤ 0,20 (dan pada fantom "tumor mengecil 7 %" skornya malah turun),
   sedangkan deformasi sungguhan ≥ 2,5 % linear memberi ≥ 0,63. Konsekuensi
   yang diterima: deformasi nyata di bawah kira-kira 1 % linear tidak
   dikoreksi, karena memang tidak terbedakan dari perubahan bentuk.
4. Metrik informasi mutual tidak memakai pengaman (NCC tidak bermakna antar
   pewarnaan); model bebas dengan MI tetap atas tanggungan pengguna.

### Hasil

| Fantom berbenih | Model | TRE rantai rata-rata / terburuk | Volume mesh / voxel vs analitik |
|---|---|---|---|
| 24 irisan, 192 px | affine, sebelum | 26,9 / 58,6 µm | +2,8 % / +2,6 % |
| 24 irisan, 192 px | affine (bawaan), sesudah | 12,0 / 19,0 µm | −0,7 % / −0,8 % |
| 60 irisan, 256 px (bawaan `h3d phantom`) | affine (bawaan), sesudah | 5,5 / 14,0 µm | −0,2 % / −0,1 % |

Dijaga tes: `h3d-register` `shape_change_between_sections_is_not_absorbed_as_scale`
(tanpa pengaman affine memberi `|det|` = 1,16), `recovers_random_affine_transform_on_phantom`
(skala 0,95–1,05 tetap ditemukan; TRE rata-rata 0,07 px), dan pipeline mesin
`full_pipeline_volume_and_two_runs_byte_identical` (TRE rata-rata ≤ 1 px,
`|det T_i| − 1| ≤ 2 %`, volume mesh dan voxel ≤ 3 %).

### Batasan yang tersisa

- Pada data nyata banyak pasangan akan lolos pengaman (deformasinya memang
  nyata), dan tiap affine yang diterima tetap memuat sedikit bias perubahan
  bentuk; bias itu masih bisa menumpuk di rantai panjang. Laporan mencatat
  bila `|det T_i|` menyimpang > 5 % dari 1; volume Cavalieri (ruang irisan)
  tidak terpengaruh. Pilihan (b) dibuka kembali bila H0.1 menunjukkan
  hanyutan itu nyata.
- Kriteria membuka kembali butir 3 di atas (TRE ≤ 2 px pada sampel nyata)
  tetap berlaku.

## Taksiran galat registrasi (TRE) untuk ketidakpastian (2026-10-11)

`RegRecord` kini menyimpan dua sisa galat yang **diukur setelah penjajaran**
(`h3d-register::pair_residual`): geseran sisa per blok 32 px (langkah 16 px)
dicari dengan korelasi ternormalisasi ± 4 px + parabola sub-piksel.

- `residual_um`: median besar geseran blok (galat lokal; termasuk perubahan
  tekstur antar irisan).
- `residual_fit_um`: RMS medan gerak rigid yang dicocokkan ke geseran blok
  (galat transformasi; bagian yang menumpuk di rantai).

`tre_est` irisan `i` = `sqrt(Σ_k fit_k² + lokal_i²)` atas pasangan di jalur
ke acuan; `Session::tre_estimate_um` = median atas irisan. Proksi Dice lama
hanya cadangan untuk catatan tanpa field itu (proyek lama tetap terbuka).

Catatan jujur: korelasi fase per blok (usulan awal) memberi 15–60 µm pada
fantom yang galat pasangannya 1–4 µm (spektrum yang diputihkan didominasi
derau), dan medan affine ikut menghitung perubahan bentuk; karena itu dipakai
korelasi ternormalisasi dan medan rigid. Pada fantom 24 irisan taksirannya
7,7 µm terhadap TRE rantai sebenarnya 12,0 µm (60 irisan: 5,4 vs 5,5 µm) —
dalam faktor 2, tetapi cenderung **kurang** dari sebenarnya pada rantai
pendek, karena bias yang tidak terlihat di citra (ECC sudah konvergen) tidak
bisa diukur dari citra. Sisa galat lokal per pasangan (4–17 µm) justru lebih
besar dari galat pasangan sebenarnya (1–4 µm) karena memuat perubahan tekstur.
