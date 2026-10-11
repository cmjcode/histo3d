//! Ikon bernama HISTO3D, semuanya dari `egui_icons::icons::ICON_*`
//! (aturan keras 13). Nama di sini menyatakan *fungsi* di aplikasi, sehingga
//! mengganti glyph cukup di satu tempat.
//!
//! Font ikon dipasang oleh [`crate::theme::apply`] /
//! [`crate::theme::install_fonts`]; tanpa itu ikon tampil sebagai kotak.

use egui_icons::icons;
pub use egui_icons::MaterialIcon;

/// Tutup panel/dialog.
pub const CLOSE: MaterialIcon = icons::ICON_CLOSE;
/// Proyek baru.
pub const NEW: MaterialIcon = icons::ICON_NOTE_ADD;
/// Buka proyek/folder.
pub const OPEN_FOLDER: MaterialIcon = icons::ICON_FOLDER_OPEN;
/// Simpan.
pub const SAVE: MaterialIcon = icons::ICON_SAVE;
/// Impor irisan.
pub const IMPORT: MaterialIcon = icons::ICON_UPLOAD;
/// Ekspor laporan/mesh.
pub const EXPORT: MaterialIcon = icons::ICON_DOWNLOAD;
/// Urungkan.
pub const UNDO: MaterialIcon = icons::ICON_UNDO;
/// Ulangi.
pub const REDO: MaterialIcon = icons::ICON_REDO;
/// Jalankan proses (registrasi, rekonstruksi).
pub const PLAY: MaterialIcon = icons::ICON_PLAY_ARROW;
/// Pengaturan.
pub const SETTINGS: MaterialIcon = icons::ICON_SETTINGS;
/// Peringatan.
pub const WARNING: MaterialIcon = icons::ICON_WARNING;
/// Galat.
pub const ERROR: MaterialIcon = icons::ICON_CANCEL;
/// Centang / Terima.
pub const CHECK: MaterialIcon = icons::ICON_CHECK;
/// Informasi / Tentang.
pub const INFO: MaterialIcon = icons::ICON_INFO;
/// Bantuan.
pub const HELP: MaterialIcon = icons::ICON_HELP;
/// Bahasa.
pub const LANGUAGE: MaterialIcon = icons::ICON_LANGUAGE;

/// Alat kuas.
pub const BRUSH: MaterialIcon = icons::ICON_BRUSH;
/// Alat penghapus.
pub const ERASER: MaterialIcon = icons::ICON_INK_ERASER;
/// Alat laso.
pub const LASSO: MaterialIcon = icons::ICON_LASSO_SELECT;
/// Alat poligon.
pub const POLYGON: MaterialIcon = icons::ICON_PENTAGON;
/// Isi lubang.
pub const FILL: MaterialIcon = icons::ICON_FORMAT_PAINT;
/// Salin ke irisan berikut.
pub const COPY: MaterialIcon = icons::ICON_CONTENT_COPY;
/// Hapus.
pub const DELETE: MaterialIcon = icons::ICON_DELETE;
/// Tambah.
pub const ADD: MaterialIcon = icons::ICON_ADD;
/// Kurangi.
pub const REMOVE: MaterialIcon = icons::ICON_REMOVE;

/// Perbesar.
pub const ZOOM_IN: MaterialIcon = icons::ICON_ZOOM_IN;
/// Perkecil.
pub const ZOOM_OUT: MaterialIcon = icons::ICON_ZOOM_OUT;
/// Terlihat.
pub const VISIBILITY_ON: MaterialIcon = icons::ICON_VISIBILITY;
/// Tersembunyi.
pub const VISIBILITY_OFF: MaterialIcon = icons::ICON_VISIBILITY_OFF;
/// Lapisan / label.
pub const LAYERS: MaterialIcon = icons::ICON_LAYERS;
/// Objek 3D (mode Rekonstruksi).
pub const CUBE: MaterialIcon = icons::ICON_VIEW_IN_AR;
/// Tabel laporan (mode Laporan).
pub const REPORT: MaterialIcon = icons::ICON_TABLE_CHART;
/// Kisi thumbnail (mode Irisan).
pub const GRID: MaterialIcon = icons::ICON_GRID_VIEW;
/// Pembanding (mode Registrasi).
pub const COMPARE: MaterialIcon = icons::ICON_COMPARE;
/// Menggambar (mode Segmentasi).
pub const DRAW: MaterialIcon = icons::ICON_DRAW;
/// Balik (orientasi / urutan).
pub const FLIP: MaterialIcon = icons::ICON_FLIP;
/// Balik urutan (atas-bawah).
pub const SWAP_VERT: MaterialIcon = icons::ICON_SWAP_VERT;
/// Putar.
pub const ROTATE: MaterialIcon = icons::ICON_ROTATE_RIGHT;
/// Tandai (QC gagal / rusak).
pub const FLAG: MaterialIcon = icons::ICON_FLAG;
/// Irisan rusak.
pub const DAMAGED: MaterialIcon = icons::ICON_BROKEN_IMAGE;
/// Landmark.
pub const LANDMARK: MaterialIcon = icons::ICON_MY_LOCATION;
/// Irisan sebelumnya.
pub const PREV: MaterialIcon = icons::ICON_CHEVRON_LEFT;
/// Irisan berikutnya.
pub const NEXT: MaterialIcon = icons::ICON_CHEVRON_RIGHT;
/// Parameter / penyetelan.
pub const TUNE: MaterialIcon = icons::ICON_TUNE;
/// Pengukuran.
pub const MEASURE: MaterialIcon = icons::ICON_STRAIGHTEN;
/// Palet / colormap.
pub const PALETTE: MaterialIcon = icons::ICON_PALETTE;
/// Bidang potong.
pub const CLIP: MaterialIcon = icons::ICON_CROP;

// --- Ikon tambahan untuk h3d-app ---

/// Alat geser pandangan.
pub const PAN: MaterialIcon = icons::ICON_PAN_TOOL;
/// Tema terang.
pub const LIGHT_MODE: MaterialIcon = icons::ICON_LIGHT_MODE;
/// Tema gelap.
pub const DARK_MODE: MaterialIcon = icons::ICON_DARK_MODE;
/// Muat ulang / hitung ulang.
pub const REFRESH: MaterialIcon = icons::ICON_REFRESH;
/// Pindah ke posisi lebih awal.
pub const ARROW_UP: MaterialIcon = icons::ICON_ARROW_UPWARD;
/// Pindah ke posisi lebih akhir.
pub const ARROW_DOWN: MaterialIcon = icons::ICON_ARROW_DOWNWARD;
/// Irisan acuan.
pub const REFERENCE: MaterialIcon = icons::ICON_PUSH_PIN;
/// Dikecualikan.
pub const EXCLUDE: MaterialIcon = icons::ICON_BLOCK;
/// Pegangan geser (translasi manual).
pub const MOVE: MaterialIcon = icons::ICON_OPEN_WITH;
/// Pengguna / inisial.
pub const PERSON: MaterialIcon = icons::ICON_PERSON;
/// Folder / bundel proyek.
pub const FOLDER: MaterialIcon = icons::ICON_FOLDER;
/// Berkas gambar.
pub const IMAGE: MaterialIcon = icons::ICON_IMAGE;
/// Segmentasi klasik.
pub const CLASSIC: MaterialIcon = icons::ICON_SCIENCE;
/// Segmentasi AI.
pub const AI: MaterialIcon = icons::ICON_AUTO_AWESOME;
/// Interpolasi antar irisan.
pub const INTERPOLATE: MaterialIcon = icons::ICON_INSIGHTS;
/// Sudah ditinjau.
pub const REVIEWED: MaterialIcon = icons::ICON_VERIFIED;
/// Kalibrasi ambang IHC.
pub const CALIBRATE: MaterialIcon = icons::ICON_BIOTECH;
/// Opasitas.
pub const OPACITY: MaterialIcon = icons::ICON_OPACITY;
/// Mode garis tepi overlay.
pub const OUTLINE: MaterialIcon = icons::ICON_BORDER_STYLE;
/// Jari menggambar.
pub const TOUCH: MaterialIcon = icons::ICON_TOUCH_APP;
/// Pusatkan kamera.
pub const CENTER_VIEW: MaterialIcon = icons::ICON_CENTER_FOCUS_STRONG;
/// Analisis spasial.
pub const ANALYZE: MaterialIcon = icons::ICON_ANALYTICS;
/// Pegangan seret.
pub const DRAG_HANDLE: MaterialIcon = icons::ICON_DRAG_INDICATOR;
/// Jenis label: tumor.
pub const KIND_TUMOR: MaterialIcon = icons::ICON_CORONAVIRUS;
/// Jenis label: nekrosis.
pub const KIND_NECROSIS: MaterialIcon = icons::ICON_BLUR_ON;
/// Jenis label: stroma.
pub const KIND_STROMA: MaterialIcon = icons::ICON_GRAIN;
/// Jenis label: jaringan normal.
pub const KIND_NORMAL: MaterialIcon = icons::ICON_CHECK_CIRCLE;
/// Jenis label: pembuluh.
pub const KIND_VESSEL: MaterialIcon = icons::ICON_BLOODTYPE;
/// Jenis label: saraf.
pub const KIND_NERVE: MaterialIcon = icons::ICON_HUB;
/// Jenis label: kapsul.
pub const KIND_CAPSULE: MaterialIcon = icons::ICON_SHIELD;
/// Jenis label: tepi reseksi.
pub const KIND_MARGIN: MaterialIcon = icons::ICON_BORDER_OUTER;
/// Jenis label: permukaan.
pub const KIND_SURFACE: MaterialIcon = icons::ICON_HORIZONTAL_RULE;
/// Jenis label: lainnya.
pub const KIND_OTHER: MaterialIcon = icons::ICON_LABEL;

/// Semua ikon bernama (dipakai tes `named_icons_render`).
pub const ALL: [MaterialIcon; 80] = [
    CLOSE,
    NEW,
    OPEN_FOLDER,
    SAVE,
    IMPORT,
    EXPORT,
    UNDO,
    REDO,
    PLAY,
    SETTINGS,
    WARNING,
    ERROR,
    CHECK,
    INFO,
    HELP,
    LANGUAGE,
    BRUSH,
    ERASER,
    LASSO,
    POLYGON,
    FILL,
    COPY,
    DELETE,
    ADD,
    REMOVE,
    ZOOM_IN,
    ZOOM_OUT,
    VISIBILITY_ON,
    VISIBILITY_OFF,
    LAYERS,
    CUBE,
    REPORT,
    GRID,
    COMPARE,
    DRAW,
    FLIP,
    SWAP_VERT,
    ROTATE,
    FLAG,
    DAMAGED,
    LANDMARK,
    PREV,
    NEXT,
    TUNE,
    MEASURE,
    PALETTE,
    CLIP,
    PAN,
    LIGHT_MODE,
    DARK_MODE,
    REFRESH,
    ARROW_UP,
    ARROW_DOWN,
    REFERENCE,
    EXCLUDE,
    MOVE,
    PERSON,
    FOLDER,
    IMAGE,
    CLASSIC,
    AI,
    INTERPOLATE,
    REVIEWED,
    CALIBRATE,
    OPACITY,
    OUTLINE,
    TOUCH,
    CENTER_VIEW,
    ANALYZE,
    DRAG_HANDLE,
    KIND_TUMOR,
    KIND_NECROSIS,
    KIND_STROMA,
    KIND_NORMAL,
    KIND_VESSEL,
    KIND_NERVE,
    KIND_CAPSULE,
    KIND_MARGIN,
    KIND_SURFACE,
    KIND_OTHER,
];
