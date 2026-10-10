# HISTO3D - Bahasa Indonesia (id-ID), bahasa bawaan.
# Kunci harus identik dengan en-US/main.ftl (dijaga tes key_parity).
# Hanya tanda baca ASCII; karakter non-ASCII yang dipakai (µ ² ³) dijaga tes
# ui_glyph_coverage di h3d-ui.

# --- Aplikasi dan Tentang ---
app-name = HISTO3D
app-title = { $name } - HISTO3D
app-tagline = Rekonstruksi 3D tumor dari irisan histologi serial
about-title = Tentang HISTO3D
about-research-only = Hanya untuk riset, bukan untuk diagnosis
about-version = Versi { $version }
about-license = Lisensi MIT atau Apache-2.0

# --- Bahasa dan tema ---
lang-current = Bahasa
lang-id = Bahasa Indonesia
lang-en = English
theme-light = Terang
theme-dark = Gelap

# --- Mode aplikasi ---
mode-slides = Irisan
mode-register = Registrasi
mode-segment = Segmentasi
mode-reconstruct = Rekonstruksi
mode-report = Laporan

# --- Aksi umum ---
action-new = Proyek baru
action-open = Buka
action-save = Simpan
action-save-as = Simpan sebagai
action-import = Impor
action-export = Ekspor
action-accept = Terima
action-reject = Tolak
action-undo = Urungkan
action-redo = Ulangi
action-cancel = Batal
action-apply = Terapkan
action-close = Tutup
action-ok = OK
action-delete = Hapus
action-add = Tambah
action-edit = Sunting
action-reset = Atur ulang
action-run = Jalankan
action-stop = Hentikan
action-help = Bantuan
action-settings = Pengaturan
action-prev-slice = Irisan sebelumnya
action-next-slice = Irisan berikutnya
action-zoom-in = Perbesar
action-zoom-out = Perkecil
action-zoom-fit = Pas ke layar

# --- Umum ---
common-yes = Ya
common-no = Tidak
common-none = Tidak ada
common-unknown = Tidak diketahui
common-loading = Memuat...
common-working = Sedang diproses...
common-done = Selesai
common-slice-n = Irisan { $n }
common-slices-count = { $n } irisan
common-value-with-uncertainty = { $value } +/- { $uncertainty } { $unit }
unit-um = µm
unit-mm = mm
unit-mm2 = mm²
unit-mm3 = mm³
unit-percent = %

# --- Mode Irisan ---
slides-import = Impor irisan
slides-import-hint = Pilih berkas slide (SVS, TIFF, NDPI, PNG, JPEG)
slides-grid = Kisi thumbnail
slides-mpp = Mikrometer per piksel (mpp)
slides-mpp-x = mpp X
slides-mpp-y = mpp Y
mpp-required = mpp tidak ditemukan di metadata; isi nilai mpp secara manual (tidak pernah ditebak)
slides-spacing = Jarak antar irisan (µm)
slides-z-position = Posisi Z (µm)
slides-order = Urutan
slides-order-source-filename = Dari nama berkas
slides-order-source-manual = Diatur tangan
slides-order-source-suggested = Usulan diterima
slides-drag-to-reorder = Seret untuk mengurutkan
slides-mark-damaged = Tandai rusak
slides-unmark-damaged = Batalkan tanda rusak
slides-exclude = Kecualikan
slides-include = Sertakan
slides-status-ok = Dipakai
slides-status-excluded = Dikecualikan
slides-status-damaged = Rusak
slides-stain = Pewarnaan
slides-stain-he = H&E
slides-stain-ihc = IHC
slides-stain-other = Lainnya
slides-suggest-order = Usulkan urutan dan orientasi
slides-suggest-preview = Pratinjau usulan
slides-flip-order = Balik urutan
slides-uncertain-slice = Irisan ragu
slides-uncertain-hint = Irisan ini ditandai ragu oleh pengusul; periksa sebelum menerima
slides-orientation = Orientasi
slides-orientation-flip-h = Balik horizontal
slides-orientation-flip-v = Balik vertikal
slides-orientation-rotate = Putar 90 derajat
slides-file-name = Nama berkas
slides-size-px = Ukuran (piksel)
slides-size-um = Ukuran (µm)
slides-sample-name = Nama sampel
slides-sample-notes = Catatan
slides-shrinkage = Faktor susut linear

# --- Panel klinis (ClinicalMeta) ---
clinical-panel = Klinis
clinical-hint = Data pseudonim; disimpan terpisah di clinical.json dan tidak pernah menjadi masukan perhitungan
clinical-subject-pseudo-id = ID pseudonim subjek
clinical-age-years = Usia (tahun)
clinical-sex = Jenis kelamin
clinical-sex-female = Perempuan
clinical-sex-male = Laki-laki
clinical-sex-other = Lainnya
clinical-sex-unknown = Tidak diketahui
clinical-topography-icdo = Topografi ICD-O
clinical-morphology-icdo = Morfologi ICD-O
clinical-laterality = Lateralitas
clinical-laterality-left = Kiri
clinical-laterality-right = Kanan
clinical-laterality-bilateral = Bilateral
clinical-laterality-not-applicable = Tidak berlaku
clinical-procedure = Prosedur
clinical-procedure-biopsy = Biopsi
clinical-procedure-excision = Eksisi
clinical-procedure-resection = Reseksi
clinical-procedure-other = Lainnya
clinical-grade = Grade
clinical-grade-system = Sistem grade
clinical-grade-value = Nilai grade
clinical-ptnm = pTNM
clinical-ptnm-t = T
clinical-ptnm-n = N
clinical-ptnm-m = M
clinical-ptnm-edition = Edisi AJCC
clinical-ptnm-prefix = Awalan (p/yp)
clinical-neoadjuvant = Terapi neoadjuvan
clinical-neoadjuvant-given = Diberikan
clinical-neoadjuvant-regimen = Regimen
clinical-molecular = Hasil molekuler
clinical-molecular-gene = Gen
clinical-molecular-result = Hasil
clinical-molecular-method = Metode
clinical-specimen-month = Bulan spesimen (YYYY-MM)
clinical-strip = Buang data klinis
clinical-export-with = Sertakan data klinis saat ekspor

# --- Orientasi anatomis ---
anat-panel = Orientasi anatomis
anat-pos-x = Arah +X
anat-pos-y = Arah +Y
anat-pos-z = Arah +Z (tumpukan)
anat-anterior = Anterior
anat-posterior = Posterior
anat-superior = Superior
anat-inferior = Inferior
anat-medial = Medial
anat-lateral = Lateral
anat-proximal = Proksimal
anat-distal = Distal
anat-left = Kiri
anat-right = Kanan
anat-unknown = Tidak diketahui

# --- Pra-analitik (PreanalyticMeta) ---
preanalytic-panel = Pra-analitik
preanalytic-block-id = ID blok parafin
preanalytic-section-thickness-um = Ketebalan potong (µm)
preanalytic-fixation-hours = Lama fiksasi (jam)
preanalytic-fixative = Fiksatif
preanalytic-stained-month = Bulan pewarnaan (YYYY-MM)
preanalytic-scanner = Pemindai
preanalytic-objective = Objektif

# --- Panel IHC (IhcPanel) ---
ihc-panel = Panel IHC
ihc-required = Panel IHC wajib diisi untuk irisan berpewarnaan IHC
ihc-antibody = Antibodi
ihc-clone = Klon
ihc-chromogen = Kromogen
ihc-chromogen-dab = DAB (cokelat)
ihc-chromogen-aec = AEC (merah)
ihc-chromogen-fast-red = Fast Red
ihc-chromogen-other = Lainnya
ihc-pattern = Pola pewarnaan
ihc-pattern-nuclear = Inti
ihc-pattern-membranous = Membran
ihc-pattern-cytoplasmic = Sitoplasma
ihc-scoring = Skema skor
ihc-scoring-percent-positive = Persen sel positif
ihc-scoring-h-score = H-score (0-300)
ihc-scoring-allred = Allred
ihc-scoring-tps = TPS (PD-L1 sel tumor)
ihc-scoring-cps = CPS (PD-L1 gabungan)
ihc-scoring-intensity03 = Intensitas 0/1+/2+/3+
ihc-od-vectors = Vektor OD pewarna
ihc-od-default = Bawaan (Ruifrok)
ihc-od-calibrated = Hasil kalibrasi

# --- Jenis label (LabelKind::i18n_key) ---
label-panel = Label
label-name = Nama
label-color = Warna
label-visible = Tampil
label-reconstruct = Dibuat mesh
label-kind = Jenis
label-group = Grup
label-kind-tumor = Tumor
label-kind-necrosis = Nekrosis
label-kind-stroma = Stroma
label-kind-normal = Jaringan normal
label-kind-vessel = Pembuluh
label-kind-vessel-blood = Pembuluh darah
label-kind-vessel-lymph = Pembuluh limfe
label-kind-vessel-unknown = Pembuluh (tidak diketahui)
label-kind-nerve = Saraf
label-kind-capsule = Kapsul / serosa / fasia
label-kind-margin = Tepi reseksi
label-kind-surface = Permukaan
label-kind-reference = Garis acuan
label-kind-other = Lainnya

# --- Mode Registrasi ---
register-all = Sejajarkan semua
register-pair = Sejajarkan irisan ini
register-qc-list = Daftar QC
register-ncc = NCC
register-dice = Dice masker
register-area-ratio = Rasio luas
register-method = Metode
register-flagged = Gagal QC
register-flagged-count = { $n } irisan gagal QC
register-manual = Koreksi manual
register-landmarks = Landmark
register-landmark-add = Tambah landmark
register-landmark-clear = Hapus semua landmark
register-landmark-apply = Terapkan landmark
register-landmark-hint = Klik titik yang sama di kedua irisan (minimal 3 pasang)
register-translate = Geser
register-rotate = Putar
register-reset-transform = Atur ulang transformasi
register-mark-damaged = Tandai rusak
register-neighbor = Tetangga
register-compare-mode = Mode banding
register-compare-blend = Campur
register-compare-checkerboard = Papan catur
register-compare-difference = Selisih
register-blend-alpha = Opasitas campuran
register-checker-size = Ukuran kotak
register-progress = Registrasi { $done } dari { $total }

# --- Mode Segmentasi ---
segment-tool-brush = Kuas
segment-tool-eraser = Penghapus
segment-tool-lasso = Laso
segment-tool-polygon = Poligon
segment-brush-size = Ukuran kuas
segment-fill-holes = Isi lubang
segment-copy-to-next = Salin ke irisan berikut
segment-copy-to-prev = Salin ke irisan sebelumnya
segment-overlay-opacity = Opasitas overlay
segment-proposals = Usulan
segment-proposal-classic = Usulan klasik
segment-proposal-ai = Usulan AI
segment-proposal-interpolate = Usulan interpolasi
segment-proposal-hint = Usulan tidak pernah menimpa label tanpa Terima
segment-proposal-accept = Terima usulan
segment-proposal-reject = Tolak usulan
segment-mark-reviewed = Tandai sudah ditinjau
segment-review-draft = Draf
segment-review-reviewed = Sudah ditinjau
segment-source-manual = Manual
segment-source-classic = Klasik
segment-source-ai = AI
segment-source-interpolated = Interpolasi
segment-calibrate-threshold = Kalibrasi ambang IHC
segment-threshold = Ambang
segment-positive-nuclei = Inti positif
segment-negative-nuclei = Inti negatif
segment-finger-draw = Jari menggambar
segment-pressure = Tekanan

# --- Mode Rekonstruksi ---
recon-run = Bangun rekonstruksi
recon-params = Parameter
recon-s-xy = Ukuran voxel XY (s_xy, µm)
recon-s-z = Ukuran voxel Z (s_z, µm)
recon-smoothing = Penghalusan
recon-decimate = Decimate
recon-decimate-ratio = Rasio decimate
recon-opacity = Opasitas
recon-visibility = Visibilitas
recon-clipping-plane = Bidang potong
recon-clipping-enable = Aktifkan bidang potong
recon-clipping-axis = Sumbu potong
recon-clipping-offset = Posisi potong
recon-slices-in-3d = Irisan di 3D
recon-show-slice = Tampilkan irisan
recon-scalar-layer = Lapisan skalar
recon-scalar-none = Tanpa lapisan skalar
recon-legend = Legenda
recon-colormap = Colormap
recon-colormap-viridis = Viridis
recon-colormap-magma = Magma
recon-colormap-coolwarm = Coolwarm
recon-colormap-grey = Abu-abu
recon-range-min = Minimum
recon-range-max = Maksimum
recon-range-auto = Rentang otomatis (p2-p98)
recon-pick-value = Nilai: { $value } { $unit }
recon-pick-nearest-slice = Irisan terdekat: { $n }
recon-reset-camera = Atur ulang kamera
recon-mesh-stats = { $vertices } titik, { $triangles } segitiga

# --- Mode Laporan ---
report-title = Laporan
report-export = Ekspor laporan
report-export-csv = CSV
report-export-markdown = Markdown
report-export-mesh = Mesh (OBJ/PLY/STL)
report-export-nifti = NIfTI
report-section-1-context = 1. Konteks sampel dan orientasi
report-section-2-components = 2. Komponen tumor
report-section-3-margin = 3. Margin
report-section-4-invasion = 4. Invasi
report-section-5-staging = 5. Ukuran staging
report-section-6-attributes = 6. Atribut
report-section-7-provenance = 7. Provenance
report-section-8-params = 8. Parameter, versi, dan peringatan
report-measurements = Tabel pengukuran
report-warnings = Peringatan
report-no-warnings = Tidak ada peringatan
report-generated-at = Dibuat { $time }
report-app-version = Versi aplikasi { $version }
report-params-hash = Hash parameter { $hash }

measure-volume-cavalieri = Volume (Cavalieri)
measure-volume-mesh = Volume (mesh)
measure-volume-corrected = Volume terkoreksi susut
measure-surface-area = Luas permukaan
measure-bbox = Kotak pembatas
measure-max-diameter = Diameter maksimum
measure-pca-axes = Tiga sumbu PCA
measure-depth = Kedalaman invasi
measure-breslow = Ketebalan Breslow
measure-components = Jumlah komponen
measure-foci = Jumlah fokus
measure-fraction = Fraksi
measure-contact-area = Luas kontak
measure-patches = Tambalan
measure-uncertainty = Ketidakpastian
measure-margin-dmin = Jarak minimum ke tepi
measure-margin-direction = Arah
measure-margin-nearest = Titik terdekat
measure-margin-status = Status margin
margin-status-involved = Terlibat
margin-status-close = Dekat
margin-status-clear = Bebas
margin-status-unknown = Tidak dapat dinilai
measure-heterogeneity = Heterogenitas
measure-per-slice = Per irisan
measure-per-region = Per region
measure-score-params = Parameter skor
provenance-source-fraction = Fraksi irisan per sumber
provenance-unreviewed = Irisan belum ditinjau
provenance-reviewed-only-volume = Volume dari irisan yang sudah ditinjau saja

# --- Peringatan otomatis (8.12.8) ---
warn-margin-close = Tumor dekat tepi reseksi arah { $dir }
warn-margin-z-unassessable = Jarak ke tepi pada arah Z tidak dapat dinilai
warn-ihc-heterogeneity = Heterogenitas IHC tinggi; satu irisan mungkin tidak representatif
warn-mostly-interpolated = Sebagian besar bentuk berasal dari interpolasi
warn-has-draft = Angka memuat anotasi yang belum ditinjau
warn-z-resolution-coarse = Resolusi Z lebih kasar dari ambang margin
warn-membrane-area-score = Skor { $name } dihitung per luas, bukan per sel

# --- Galat ---
err-mpp-missing = mpp tidak ada pada irisan { $name }; isi secara manual
err-file-corrupt = Berkas rusak atau tidak dapat dibaca: { $name }
err-format-too-new = Format proyek lebih baru ({ $version }) daripada yang didukung aplikasi ini
err-singular = Transformasi singular; registrasi dibatalkan
err-io = Galat baca/tulis: { $message }
err-unsupported-format = Format berkas tidak didukung: { $name }
err-ihc-panel-missing = Panel IHC belum diisi untuk irisan { $name }
err-no-labels = Belum ada label untuk direkonstruksi
err-generic = Terjadi galat: { $message }
