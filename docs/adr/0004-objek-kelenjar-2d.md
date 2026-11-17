# ADR 0004 — Objek 2D kelenjar per irisan untuk rekonstruksi 3D

Status: diterima (2026-11-08).

## Konteks

Tujuan utama HISTO3D adalah merekonstruksi struktur dari irisan serial:
(1) tiap irisan menjadi kumpulan **objek 2D**, (2) tiap objek diekstrusi
setebal jarak irisan, (3) objek yang sama pada irisan berurutan ditautkan.
Langkah 2–3 sudah ada (`h3d-volume::recon` lewat SDF, `h3d-volume::track`
lewat IoU di kisi W, termasuk melompati irisan hilang). Langkah 1 gagal:

- Deteksi **inti sel** (ADR 0003) menghasilkan ribuan poligon sel per
  irisan. Satuan ini salah untuk 3D: satu inti tidak pernah muncul lagi
  50 µm kemudian, sehingga tidak ada yang bisa ditautkan.
- Segmentasi klasik satu kanal (ambang H → label "Kelenjar") lalu
  pembelahan watershed EDT menghasilkan **lembaran epitel menyambung**,
  bukan satu wilayah per kelenjar; EDT tidak punya lembah di antara
  kelenjar yang berbagi dinding.

Satuan objek yang benar untuk prostat (data Tampere, tikus) adalah
**kelenjar** = cincin epitel **beserta lumennya**, ditambah sarang tumor
padat dan pembuluh. Pengguna meminta bentuk "cembung yang mendekati bentuk
nyata".

## Keputusan

### Metode: segmentasi kelenjar berpandu lumen (`h3d_imgproc::glands`)

Mengikuti pendekatan baku kelenjar prostat (Naik dkk. 2008; Nguyen dkk.
2012): lumen → cincin epitel → batas stroma, pada skala **1–2 µm/px**
(bawaan 2) sehingga inti melebur menjadi tekstur "epitel padat".

1. **Pewarna per slide**: estimasi Macenko dkk. (2009) atas OD piksel
   jaringan (eigen Jacobi 3×3, persentil sudut 1 %); jatuh ke Ruifrok bila
   jaringan < 1000 piksel atau sudut H–E < 10°. Ini menyelesaikan masalah
   epitel merah tua tikus2 yang kanal H Ruifrok-nya bocor ke sitoplasma.
2. **Peta epitel**: kanal H Gauss σ 4 µm, Otsu di dalam masker jaringan.
3. **Lumen** dari dua sumber: (a) lubang cincin epitel tertutup (closing
   5 µm) berluas 50–250 000 µm², apa pun isinya (sekresi, sel lepas);
   (b) piksel terang (OD total < 0,15) yang lolos **uji rasio cincin**:
   ≥ 50 % cincin selebar 10 µm di sekelilingnya berupa epitel. Vakuola
   lemak dan celah stroma gugur di (b).
4. **Masker kelenjar** = epitel ∪ lumen, buka 2 µm, tutup 5 µm, lubang
   kecil diisi, dipotong masker jaringan.
5. **Pemisahan**: komponen berlumen dibagi watershed berpenanda di atas
   `−jarak ke lumen terdekat` (tiap lumen = penanda; lembah < 4 µm =
   kribriform, tetap satu); komponen tanpa lumen dibelah watershed EDT
   (jarak pusat 40 µm, lembah 8 µm). Keping < 300 µm² digabung/dibuang;
   objek tanpa lumen lebih tipis dari 8 µm (serpihan tepi) dibuang.
6. **Bentuk star-convex**: tiap objek diganti *star hull*-nya dari piksel
   EDT maksimum (jari-jari terjauh per 2°, bin kosong diisi minimum
   tetangga, Gauss melingkar σ 6°), dirasterkan tanpa menimpa objek lain
   dan tanpa keluar jaringan; bila hull > 1,5 × luas mentah, bentuk mentah
   dipertahankan (objek tipis melengkung tidak menjadi kipas). Soliditas
   bentuk mentah tetap dilaporkan.
7. **Artefak kaca**: objek yang > 50 % piksel non-lumennya abu-abu/hitam
   tak berwarna (kroma < 30, intensitas < 180: serat debu, tepi gelembung
   udara, tinta) dibuang; uji rasio cincin berlaku juga untuk lubang cincin
   epitel sehingga lingkaran serat tipis tidak menjadi "lumen".

**Convex hull murni ditolak**: kelenjar berbentuk U, bercabang, dan
kribriform akan menelan stroma dan menyatukan tetangga. Star hull mengisi
cekungan radial dan lekuk U menjadi satu badan, tetapi dua kelenjar
bertetangga tidak pernah disatukan.

Alternatif yang ditolak untuk saat ini: U-Net semantik + watershed
(DCAN/GlaS) butuh label piksel penuh yang tidak tersedia; Cellpose/SAM
terlalu berat untuk iPad.

### Integrasi

- `h3d_segment::glands::gland_proposal` membungkus peta instance menjadi
  **usulan biner dengan pemisah** antar objek; lebar pemisah
  `ceil(mpp_label / mpp) + 1` piksel (modus "≥ setengah" pada
  `accept_proposal` akan menyambungkan celah yang lebih sempit). Dengan
  demikian setelah "Terima", `track.rs` melihat tiap kelenjar sebagai
  komponen tersendiri tanpa format `.lbl` instance baru.
- Engine: `Session::segment_glands` (job latar, dapat dibatalkan) →
  `GlandProposal { proposal, summary }`; asal `Classic { params_hash }`.
- CLI `h3d glands` (hitung saja; `--accept` = persetujuan eksplisit,
  `--replace` mengosongkan label dulu; `--overlay-dir` PNG garis tepi;
  `--json`). UI: bagian "Objek 2D kelenjar" di laci Label dan usulan,
  memakai jalur usulan klasik yang sama (Terima/Tolak).
- Jumlah dilaporkan dengan rentang (aturan keras 18): batas bawah dari
  saringan luas × 1,25 dan rasio cincin + 0,15; batas atas = jumlah +
  lumen yang gugur tipis (dalam 0,15 di bawah ambang). Tanpa jalan ulang.

### Sumbu Z

Ketebalan irisan laboratorium **tidak** mengubah geometri (histo3d.md
5.5): yang dipakai adalah **jarak antar irisan** `default_spacing_um`
(slider di laci Rekonstruksi, `--spacing-um` saat impor). Bila setiap
irisan serial diambil, jarak = ketebalan; bila tiap irisan ke-k, jarak =
k × ketebalan. `section_thickness_um` tetap metadata pra-analitik.

### Hasil pada tikus2 (Tampere PTENX036-011, 0,46 µm/px, 29 irisan)

Pada irisan 10: 470 objek (rentang 401–635), 297 berlumen, median luas
≈ 2500 µm², 4,7 s per irisan pada skala 2 µm/px. Ringkasan seluruh
tumpukan dan hasil penautan ada di README bagian "Objek 2D kelenjar".

## Konsekuensi

- Hasil adalah **usulan**; label berubah hanya setelah Terima (aturan 8).
  Di laci Label, usulan digambar sebagai **arsir diagonal** warna label;
  setelah Terima baru tergambar penuh.
- Semua parameter berhash (`h3d-glands-v1`) dan tercetak di JSON/laporan.
- Pekerjaan lanjutan (Tingkat B): StarDist pada skala kelenjar (mpp 2 µm,
  `n_rays` 64) dilatih dari usulan yang dikoreksi pengguna, lewat kontrak
  `task = "glands"` pada jalur `StarDistDetector` yang ada; poligon
  star-convex-nya identik secara bentuk dengan keluaran langkah 6.
- Batas yang diketahui: epitel tipis sangat pucat dengan lumen besar dapat
  memberi cincin terputus (hanya sebagian cincin menjadi objek); turunkan
  `min_ring_fraction`/`epithelium_sigma_um` atau koreksi manual.
