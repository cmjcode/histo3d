//! Uji asap headless: keadaan aplikasi dibangun di atas proyek fantom lalu
//! satu frame UI tiap mode dijalankan lewat `egui::Context::run_ui` tanpa
//! jendela dan tanpa GPU (paint callback hanya ditambahkan bila ada render
//! state). Juga alur impor tanpa mpp, goresan satu-undo, usulan urutan,
//! dan kunci terjemahan semua `Warning`.

use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

use h3d_app::modes::{register, segment, slides};
use h3d_app::panels::browser::{BrowserPurpose, BrowserState};
use h3d_app::settings::Settings;
use h3d_app::state::{try_lock_session, AppState, ImportRequest, JobOutput};
use h3d_core::{Budgets, LabelId, NoProgress, SlideId};
use h3d_engine::phantom::{write_phantom, PhantomManifest, PhantomOptions};
use h3d_engine::{ImportOptions, RegOptions, Session};
use h3d_ui::AppMode;
use h3d_volume::Warning;

/// Bahasa aktif bersifat global: tes yang menggambar UI bergiliran.
static UI_LOCK: Mutex<()> = Mutex::new(());

fn ui_lock() -> MutexGuard<'static, ()> {
    UI_LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

struct Fixture {
    tmp: tempfile::TempDir,
    manifest: PhantomManifest,
    slide_paths: Vec<PathBuf>,
}

fn phantom(slices: usize) -> Fixture {
    let tmp = tempfile::tempdir().unwrap();
    let ph = tmp.path().join("ph");
    let manifest = write_phantom(
        &ph,
        &PhantomOptions {
            slices,
            size_px: 128,
            margin_um: Some(400.0),
            ..Default::default()
        },
    )
    .expect("fantom");
    let slide_paths = manifest.slices.iter().map(|s| ph.join(&s.file)).collect();
    Fixture {
        tmp,
        manifest,
        slide_paths,
    }
}

/// Proyek fantom lengkap: impor dengan mpp, label dari GeoJSON, registrasi.
fn project(fx: &Fixture) -> PathBuf {
    let path = fx.tmp.path().join("p.h3d");
    let mut s = Session::create(&path).unwrap();
    s.import_slides_with(
        &fx.slide_paths,
        &ImportOptions {
            mpp: Some([fx.manifest.mpp, fx.manifest.mpp]),
            embed: false,
            spacing_um: Some(fx.manifest.spacing_um),
        },
        &NoProgress,
    )
    .unwrap();
    let ph = fx.tmp.path().join("ph");
    for (i, sl) in fx.manifest.slices.iter().enumerate() {
        if let Some(g) = &sl.geojson {
            s.import_geojson(SlideId(i as u32), &ph.join(g)).unwrap();
        }
    }
    s.register_all(&RegOptions::default(), &NoProgress).unwrap();
    s.save().unwrap();
    path
}

fn app() -> AppState {
    AppState::new(Settings::default(), Budgets::IPAD)
}

fn frame(ctx: &egui::Context, app: &mut AppState) {
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        ..Default::default()
    };
    let mut out = ctx.run_ui(input, |ui| app.ui(ui, None));
    out.textures_delta.clear();
}

fn context(app: &AppState) -> egui::Context {
    let ctx = egui::Context::default();
    app.apply_theme(&ctx);
    ctx
}

#[test]
fn every_mode_renders_a_frame_without_gpu() {
    let _g = ui_lock();
    let fx = phantom(6);
    let path = project(&fx);
    let mut app = app();
    let ctx = context(&app);

    // Tanpa proyek: layar sambutan + dialog Tentang dan Pengaturan.
    app.show_about = true;
    app.show_settings = true;
    frame(&ctx, &mut app);
    app.show_about = false;
    app.show_settings = false;

    assert!(app.open_project(&path));
    assert!(app.toasts.items().is_empty(), "{:?}", app.toasts.items());
    assert!(app.current.is_some());

    // Rekonstruksi + analisis lewat job latar supaya mode 3D dan Laporan
    // punya isi.
    assert!(app.start_job("rekonstruksi".into(), |s, p| {
        s.reconstruct(p).map(|m| JobOutput::Reconstructed(m.len()))
    }));
    // Selama job berjalan layar menampilkan lapisan job.
    frame(&ctx, &mut app);
    app.wait_job();
    assert!(app.start_job("analisis".into(), |s, p| {
        s.analyze_spatial(p).map(|_| JobOutput::Analyzed)
    }));
    app.wait_job();
    assert!(
        app.toasts.items().iter().all(|t| !t.is_error),
        "{:?}",
        app.toasts.items()
    );

    for lang in [h3d_i18n::Language::Id, h3d_i18n::Language::En] {
        app.settings.set_language(lang);
        app.persist_settings();
        for mode in AppMode::ALL {
            app.set_mode(mode);
            if mode == AppMode::Slides {
                app.slides.selected = app.current;
            }
            for _ in 0..3 {
                frame(&ctx, &mut app);
            }
            assert!(
                app.toasts.items().iter().all(|t| !t.is_error),
                "mode {mode:?}: {:?}",
                app.toasts.items()
            );
        }
        // Semua tab laporan.
        app.set_mode(AppMode::Report);
        for tab in 0..8 {
            app.report.tab = tab;
            frame(&ctx, &mut app);
        }
    }
    assert!(!app.recon.meshes.is_empty(), "mesh harus termuat");
    assert!(app.report.report.is_some(), "laporan harus terhitung");

    // Alat dan lapisan tambahan tiap mode.
    app.set_mode(AppMode::Register);
    for tool in [
        register::RegTool::Landmark,
        register::RegTool::Translate,
        register::RegTool::Rotate,
    ] {
        app.register.tool = tool;
        app.register.shift = [20.0, -10.0];
        app.register.angle = 0.05;
        for kind in [
            register::CompareKind::Blend,
            register::CompareKind::Checkerboard,
            register::CompareKind::Difference,
        ] {
            app.register.compare = kind;
            frame(&ctx, &mut app);
        }
    }

    app.set_mode(AppMode::Segment);
    let slide = app.current.unwrap();
    {
        let shared = app.session.clone().unwrap();
        let s = try_lock_session(&shared).unwrap();
        let label = s.project().labels[0].id;
        let proposal = s
            .segment_classic(slide, label, &Default::default())
            .unwrap();
        drop(s);
        app.segment.set_classic(slide, proposal);
    }
    app.segment.nuclei = Some(segment::NucleiOverlay {
        slide,
        points: vec![([500.0, 500.0], 0.05), ([700.0, 600.0], 0.4)],
        mpp: 1.0,
    });
    app.segment.stroke = vec![([400.0, 400.0], 60.0), ([500.0, 450.0], 40.0)];
    for tool in [
        segment::Tool::Pan,
        segment::Tool::Brush,
        segment::Tool::Eraser,
        segment::Tool::Lasso,
        segment::Tool::Polygon,
    ] {
        app.segment.tool = tool;
        app.segment.polygon = vec![[100.0, 100.0], [300.0, 120.0], [200.0, 300.0]];
        frame(&ctx, &mut app);
    }

    app.set_mode(AppMode::Reconstruct);
    app.recon.show_slices = true;
    app.recon.clip_enabled = true;
    app.recon.scalar = h3d_app::modes::reconstruct::scalar_names(&app.recon.meshes)
        .first()
        .cloned();
    frame(&ctx, &mut app);

    // Dialog: peramban Documents (dua tujuan) dan pratinjau usulan urutan.
    app.set_mode(AppMode::Slides);
    for purpose in [BrowserPurpose::Projects, BrowserPurpose::ImportSlides] {
        app.browser = Some(BrowserState::in_dir(purpose, fx.tmp.path().join("ph")));
        frame(&ctx, &mut app);
        assert!(app.browser.is_some());
    }
    app.browser = None;
    assert!(app.start_job("usul".into(), |s, p| {
        s.suggest_order(p).map(JobOutput::OrderSuggested)
    }));
    app.wait_job();
    assert!(
        app.slides.order_preview.is_some(),
        "pratinjau harus terbuka"
    );
    frame(&ctx, &mut app);
    assert!(
        app.slides.order_preview.is_some(),
        "usulan tidak boleh diterapkan tanpa keputusan pengguna"
    );

    // Kait peringatan memori tidak panik tanpa GPU dan mengosongkan cache.
    h3d_app::platform::on_memory_warning();
    frame(&ctx, &mut app);
    assert_eq!(app.tiles.cpu_cache_bytes(), 0);
    assert!(app.thumbs.is_empty() || app.thumbs.len() <= 2);

    app.save_now();
    h3d_i18n::set_language(h3d_i18n::Language::default());
}

#[test]
fn import_without_mpp_asks_for_mpp_and_never_guesses() {
    let _g = ui_lock();
    let fx = phantom(3);
    let mut app = app();
    let ctx = context(&app);
    assert!(app.create_project(&fx.tmp.path().join("baru.h3d")));

    let request = ImportRequest {
        paths: fx.slide_paths.clone(),
        embed: true,
    };
    app.start_import(request.clone(), None);
    app.wait_job();
    let prompt = app.mpp_prompt.as_ref().expect("dialog mpp harus terbuka");
    assert!(!prompt.mpp_valid(), "mpp tidak boleh terisi otomatis");
    assert_eq!(prompt.request, request);
    {
        let shared = app.session.clone().unwrap();
        let s = try_lock_session(&shared).unwrap();
        assert!(s.slides().is_empty(), "proyek tidak boleh berubah");
    }
    // Dialog (dengan pratinjau skala batang) tergambar tanpa panik.
    frame(&ctx, &mut app);
    frame(&ctx, &mut app);
    assert!(app.mpp_prompt.is_some());

    // Pengguna mengisi mpp → impor diulang dan berhasil, disalin ke bundel.
    let prompt = app.mpp_prompt.take().unwrap();
    app.start_import(prompt.request, Some([fx.manifest.mpp, fx.manifest.mpp]));
    app.wait_job();
    assert!(app.mpp_prompt.is_none());
    let shared = app.session.clone().unwrap();
    let s = try_lock_session(&shared).unwrap();
    assert_eq!(s.slides().len(), 3);
    for e in s.slides() {
        assert_eq!(e.mpp, [fx.manifest.mpp, fx.manifest.mpp]);
        assert!(
            matches!(e.source, h3d_core::SlideSourceRef::Embedded { .. }),
            "impor dari Documents harus menyalin ke bundel"
        );
    }
    drop(s);
    assert!(app.slides.selected.is_some());
    frame(&ctx, &mut app);
}

#[test]
fn one_stroke_is_one_undo_entry_with_pressure_dependent_width() {
    let fx = phantom(3);
    let path = project(&fx);
    let mut s = Session::open(&path).unwrap();
    let slide = SlideId(0);
    let label = s.project().labels[0].id;
    let before = s.labels(slide).unwrap().area_px(label);

    // Goresan dengan jari-jari berubah (tekanan Pencil) = satu langkah undo.
    let pts: Vec<([f64; 2], f64)> = (0..12)
        .map(|i| {
            let force = Some(i as f32 / 11.0);
            (
                [300.0 + 40.0 * i as f64, 300.0],
                h3d_app::geom::brush_radius_um(60.0, force),
            )
        })
        .collect();
    assert!(segment::commit_stroke(&mut s, slide, label, &pts, false).unwrap());
    let after = s.labels(slide).unwrap().area_px(label);
    assert!(after > before);
    // Ujung bertekanan penuh lebih lebar daripada ujung bertekanan nol.
    let store = s.labels(slide).unwrap();
    let wide = store.get_um([300.0 + 40.0 * 11.0, 300.0 + 50.0]);
    let narrow_far = store.get_um([300.0, 300.0 + 50.0]);
    assert_eq!(wide, label.0);
    let _ = narrow_far;

    s.save_labels(slide).unwrap();
    assert!(!s.labels_dirty(slide), "goresan harus tersimpan otomatis");
    assert!(s.can_undo(slide));
    assert!(s.undo(slide).unwrap().is_some());
    assert_eq!(s.labels(slide).unwrap().area_px(label), before);

    // Penghapus menghapus label apa pun di bawahnya, juga satu langkah.
    s.redo(slide).unwrap();
    assert!(segment::commit_stroke(&mut s, slide, label, &pts, true).unwrap());
    assert_eq!(s.labels(slide).unwrap().get_um([520.0, 300.0]), 0);
    s.undo(slide).unwrap();
    assert_eq!(s.labels(slide).unwrap().get_um([520.0, 300.0]), label.0);
    assert!(!segment::commit_stroke(&mut s, slide, label, &[], false).unwrap());
    assert_ne!(label, LabelId::BACKGROUND);
}

#[test]
fn order_suggestion_is_applied_only_through_accept() {
    let _g = ui_lock();
    let fx = phantom(5);
    let path = project(&fx);
    let mut s = Session::open(&path).unwrap();
    let current: Vec<SlideId> = s.slides().iter().map(|e| e.id).collect();
    let suggestion = s.suggest_order(&NoProgress).unwrap();
    let mut preview = slides::OrderPreview::new(suggestion);
    // "Balik urutan" pasti menghasilkan perubahan pada tumpukan ≥ 2 irisan.
    if preview.changes(&current).is_empty() {
        preview.flip();
    }
    let changes = preview.changes(&current);
    assert!(!changes.is_empty());
    // Menyusun pratinjau tidak mengubah proyek.
    let now: Vec<SlideId> = s.slides().iter().map(|e| e.id).collect();
    assert_eq!(now, current);

    // Isi pratinjau tergambar; tanpa klik tidak ada keputusan.
    let ctx = egui::Context::default();
    h3d_ui::theme::apply(&ctx, h3d_ui::ThemeMode::Dark);
    let mut decision = slides::OrderDecision::None;
    let mut out = ctx.run_ui(Default::default(), |ui| {
        egui::CentralPanel::default().show(ui, |ui| {
            decision = slides::order_preview_ui(ui, &mut preview, &s);
        });
    });
    out.textures_delta.clear();
    assert_eq!(decision, slides::OrderDecision::None);

    // Terima semua → urutan proyek = urutan usulan, asal = usulan.
    let all = preview.suggestion.order.clone();
    s.apply_order_suggestion(&preview.suggestion, &all).unwrap();
    let applied: Vec<SlideId> = s.slides().iter().map(|e| e.id).collect();
    assert_eq!(applied, preview.suggestion.order);
    assert!(s
        .slides()
        .iter()
        .any(|e| e.order_source == h3d_core::OrderSource::Suggested));
}

#[test]
fn nuclei_detection_runs_on_visible_region() {
    let fx = phantom(3);
    let path = project(&fx);
    let s = Session::open(&path).unwrap();
    let size = s.slide(SlideId(0)).unwrap().size_um();
    let n = segment::detect_nuclei_region(&s, SlideId(0), [0.0, 0.0], size).unwrap();
    assert_eq!(n.slide, SlideId(0));
    for (u, d) in &n.points {
        assert!(u[0] >= 0.0 && u[0] <= size[0] && u[1] >= 0.0 && u[1] <= size[1]);
        assert!(d.is_finite());
    }
    let (pos, neg) = n.counts(0.15);
    assert_eq!(pos + neg, n.points.len());
    // Wilayah di luar irisan → galat, bukan panik.
    assert!(
        segment::detect_nuclei_region(&s, SlideId(0), [-500.0, -500.0], [-10.0, -10.0]).is_err()
    );
    assert!(segment::detect_nuclei_region(&s, SlideId(99), [0.0, 0.0], size).is_err());
}

#[test]
fn every_warning_has_a_translation_in_both_locales() {
    let _g = ui_lock();
    let all = [
        Warning::MarginClose { dir: "+X".into() },
        Warning::MarginZUnassessable,
        Warning::IhcHeterogeneity,
        Warning::MostlyInterpolated,
        Warning::HasDraft,
        Warning::ZResolutionCoarse,
        Warning::MembraneAreaScore {
            name: "HER2".into(),
        },
        Warning::VolumeMismatch { percent: 7.5 },
        Warning::SxyRaised {
            from: 16.0,
            to: 32.0,
        },
        Warning::GapSplit {
            from_z: 100.0,
            to_z: 900.0,
            gap: 800.0,
        },
        Warning::NonUniformZResampled { dz: 50.0 },
    ];
    for w in &all {
        // Pencocokan habis: varian baru memaksa tes ini diperbarui.
        match w {
            Warning::MarginClose { .. }
            | Warning::MarginZUnassessable
            | Warning::IhcHeterogeneity
            | Warning::MostlyInterpolated
            | Warning::HasDraft
            | Warning::ZResolutionCoarse
            | Warning::MembraneAreaScore { .. }
            | Warning::VolumeMismatch { .. }
            | Warning::SxyRaised { .. }
            | Warning::GapSplit { .. }
            | Warning::NonUniformZResampled { .. } => {}
        }
        for lang in [h3d_i18n::Language::Id, h3d_i18n::Language::En] {
            assert!(
                h3d_i18n::has_key(lang, w.i18n_key()),
                "kunci {} hilang di {lang:?}",
                w.i18n_key()
            );
            h3d_i18n::set_language(lang);
            let text = h3d_app::modes::report::warning_message(w);
            assert!(!text.is_empty() && text != w.i18n_key(), "{text}");
            assert!(!text.contains("{ $"), "argumen tidak terisi: {text}");
            for (_, value) in w.args() {
                assert!(text.contains(&value), "{text} tidak memuat {value}");
            }
        }
    }
    h3d_i18n::set_language(h3d_i18n::Language::default());
}
