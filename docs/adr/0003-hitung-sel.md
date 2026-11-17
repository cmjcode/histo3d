# ADR 0003 — Penghitungan objek dan sel pada penampang 2D

Status: diterima (2026-11-06).

## Konteks

Pengguna membutuhkan jumlah objek per penampang dalam dua bentuk:

1. **Objek jaringan** (kelenjar/asinus berlumen, pulau tumor, pembuluh) —
   bentuk besar, bervariasi ukuran, sering bersentuhan.
2. **Sel/inti** — ribuan per irisan, harus dideteksi langsung dari citra
   tanpa label manual, seperti metode deteksi (Mask R-CNN, U-Net, StarDist).

Batasan proyek (histo3d.md 14): tanpa OpenCV/ITK, tanpa jaringan, Rust
murni, berjalan di CPU dan iPad, model dibawa pengguna sebagai `model.onnx`
+ `model.json` lewat `tract`.

## Keputusan

### Objek jaringan: `h3d_imgproc::count` atas masker label

Isi lumen → komponen terhubung per kotak pembatas → **watershed EDT
berpenanda h-maxima** (`split_touching`, sudah dipakai penautan objek 3D) →
gabung wilayah kecil → buang objek < luas minimum → **bingkai hitung tak bias
Gundersen** (objek yang menyentuh tepi kiri/bawah tidak dihitung).
Ketidakpastian = rentang jumlah dari varian `h_um × {0,5; 2}` dan
`min_seed × {0,75; 1,25}`. Alternatif yang ditolak: komponen terhubung saja
(undercount objek bersentuhan), deteksi blob LoG (tidak cocok objek berlumen
dengan ukuran 5–20× berbeda).

### Sel/inti: StarDist (ONNX) + cadangan LoG klasik — `h3d_segment::cells`

| Metode | Alasan |
|---|---|
| Mask R-CNN / U-Net semantik | Lama; U-Net menggandakan inti rapat, Mask R-CNN undercount |
| **StarDist** (Schmidt dkk. 2018) | **Dipilih.** U-Net kecil (~1 M parameter), poligon star-convex cocok untuk inti rapat, cepat di CPU, ekspor ONNX mudah (`script/export_stardist_onnx.py`). Pasca-proses NMS ditulis ulang di Rust dalam versi raster yang sekaligus menghasilkan masker instance |
| Cellpose / Cellpose-SAM | Integrasi aliran berat; SAM terlalu besar untuk iPad |
| HoVer-Net / CellViT / CellViT++ | Akurasi tertinggi + tipe sel, tetapi backbone ViT berat di CPU; dicadangkan sebagai opsi lanjutan lewat jalur `task = "nuclei"` yang sama |

Satu trait `NucleiDetector` menaungi tiga pendeteksi sehingga aplikasi
tetap bisa menghitung tanpa model dan tanpa Python:

- `WatershedNucleiDetector` (**bawaan**): segmentasi klasik ala "cell
  detection" QuPath — kanal H → Gauss σ 1 µm → ambang maks(lantai 0,15 OD,
  Otsu tile) → isi lubang/buka → **watershed berpenanda di atas kanal H
  yang dihaluskan** (`split_on_field`, h-maxima 0,03 OD, jarak puncak
  3 µm) → saring luas 8–300 µm², sirkularitas ≥ 0,25, H ≥ E. Watershed di
  atas intensitas (bukan EDT) dipilih karena inti berdempetan di epitel
  tidak punya lembah EDT. Pada tikus2 (0,46 µm/px) inti di stroma dan
  epitel berwarna normal terpisah satu per satu dan kaca bersih; lihat
  batas yang diketahui di bawah untuk epitel merah tua pekat.
  Kontur inti ditelusuri di sepanjang retakan tepi wilayah watershed
  (`h3d_imgproc::label_contours`), disederhanakan Douglas–Peucker
  (`contour_tolerance_um`, bawaan 0,3 µm) dan dihaluskan Chaikin satu kali,
  sehingga overlay menampilkan bentuk inti sebenarnya (termasuk cekung),
  bukan lingkaran/sinar. Kontur hanya untuk tampilan; luas dan jumlah
  tetap dari piksel.
  Keping cembung satu sel yang terbelah beberapa benih disatukan
  (`h3d_imgproc::merge_convex_regions`): dua wilayah bertetangga digabung
  rakus bila soliditas gabungan ≥ `merge_solidity` (bawaan 0,93) dan luas
  ≤ `max_area_um2`. Dua inti bulat yang bersentuhan (soliditas ≈ 0,88)
  tetap terpisah.
  **Batas yang diketahui** (tikus2, Tampere, 0,46 µm/px): pada epitel
  berwarna merah tua pekat, kanal H Ruifrok bernilai tinggi juga di
  sitoplasma (Otsu tile ≥ 0,45: ambang 0,15–0,45 memberi hasil
  sama), sehingga masker menutup seluruh lembaran dan watershed memotongnya menjadi kepingan yang bukan sel. Menaikkan
  ambang H (0,6) atau menurunkan `merge_solidity` (0,85, luas 800 µm²)
  tidak memperbaikinya: inti kecil di stroma hilang, atau kepingan gabungan
  memuat beberapa inti. Untuk jaringan seperti ini gunakan StarDist, atau
  normalisasi pewarna per slide (Macenko) sebagai pekerjaan lanjutan.
- `ClassicNucleiDetector`: puncak LoG multiskala + saringan OD/kontras/H≥E;
  lebih cepat, lebih banyak salah deteksi pada jaringan bertekstur.
- `StarDistDetector`: model ONNX (akurasi tertinggi).

Tile bertumpang tindih;
tiap inti dimiliki tepat satu tile (batas = tengah daerah tumpang tindih),
hasil digabung dalam urutan tile (deterministik). Ketidakpastian = jumlah
pada ambang `utama ± delta`, dihitung sekali karena NMS berurutan
probabilitas membuat himpunan pada ambang ketat ⊂ himpunan pada ambang
longgar.

### Kontrak `model.json` untuk `task = "nuclei"`

```json
{
  "name": "stardist-he", "version": "2D_versatile_he", "task": "nuclei",
  "input_size": 512, "mpp": 0.5, "mean": [0,0,0], "std": [1,1,1], "classes": [],
  "nuclei": { "n_rays": 32, "grid": [2,2], "prob_threshold": 0.69,
              "nms_threshold": 0.3, "layout": "nhwc",
              "percentile_norm": [1.0, 99.8], "threshold_delta": 0.1, "margin_px": 32 }
}
```

Model `task = "nuclei"` tidak muncul di combo segmentasi (laci Deteksi AI)
dan `TractSegmenter` menolaknya; sebaliknya `StarDistDetector` menolak model
segmentasi.

## Konsekuensi

- Semua jumlah dilaporkan bersama rentang, hash parameter, dan densitas per
  mm² jaringan (aturan 18–19); tidak ada teks rekomendasi tindakan.
- Hasil deteksi sel **tidak** masuk `LabelStore`/proyek; ia tampilan dan
  keluaran CLI/JSON. Menyimpannya sebagai anotasi instance adalah pekerjaan
  lanjutan (butuh format `.lbl` instance).
- Densitas inti sebagai atribut bertile (`AttrKind::NucleiDensity`) masih
  memakai LoG klasik; mengganti sumbernya dengan head `nuclei` (histo3d.md
  8.12.5 butir 2) dapat dilakukan tanpa mengubah antarmuka `AttrStore`.

## Adendum (2026-11-17): mode "objek utuh"

**Masalah.** Pada epitel yang sitoplasmanya ikut gelap di kanal H (tikus2,
epitel merah tua), masker inti menutupi seluruh pita epitel dan watershed
`split_on_field` memotongnya menjadi keping bersambung yang bukan sel;
gabung cembung (`merge_solidity`) dan ambang H tidak menolong (lihat
catatan watershed epitel gelap). Pengguna ingin pita/cincin itu tampil
sebagai **satu objek utuh**.

**Keputusan.** Metode ketiga di balik `CellMethod`:
`CellMethod::WholeRegions(WholeRegionDetector)` (`h3d_segment::regions`,
`regions-classic-v1`). Tidak memakai trait `NucleiDetector` karena
kepemilikan objek per tile (pusat di daerah milik tile) memotong objek yang
lebih besar dari tile: raster wilayah dirakit dulu dari tile bertumpang
tindih (H halus, H−E), lalu satu ambang `maks(lantai, Otsu jaringan)`,
buka → tutup → isi lubang kecil (≤ `max_hole_area_um2`, lumen tetap
lubang) → buang komponen kecil → komponen tersambung (konektivitas 4) =
objek. Kontur luar dari `label_contours`, lubang dari
`label_hole_contours` (pemilik lubang = label di atas piksel pertama lubang,
sehingga serpihan di lumen tidak merebut lubang); disimpan di
`DetectedNucleus::holes_um` (serde bawaan kosong, JSON lama tetap terbaca).
Objek dibuang bila < separuh luasnya di masker jaringan registrasi.

**Memori.** Raster wilayah ≈ `REGION_BYTES_PER_PX` = 24 B/px. Bila wilayah
melebihi `Budgets::cell_region_bytes` (macOS 768 MB, iPad 192 MB), mpp
kerja dinaikkan sebanding akar rasio luas; mpp yang dipakai ada di
`CellDetection::mpp` (masuk hash) dan ditampilkan di panel.

**Ketidakpastian.** `n = n_max` = semua objek; `n_min` = objek dengan OD H
rata-rata ≥ 1,2 × ambang.

**Konsekuensi.** Angkanya jumlah objek, bukan jumlah sel; UI dan CLI
menyebut "objek". Pita yang saling bersentuhan menjadi satu objek. Jumlah
sel yang benar pada epitel gelap tetap butuh StarDist. Bawaan panel dan
`h3d cells` tetap watershed.
