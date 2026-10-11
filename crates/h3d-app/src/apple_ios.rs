//! Integrasi iPadOS lewat UIKit (`objc2`), dipanggil hanya dari thread
//! utama. Pola disalin dari DUCAD `apple_ios.rs`, dipangkas ke dua hal yang
//! dibutuhkan HISTO3D:
//!
//! - **Peringatan memori** (`UIApplicationDidReceiveMemoryWarning`) →
//!   [`crate::platform::on_memory_warning`]; eframe sendiri membuang
//!   `Event::MemoryWarning` winit.
//! - **Siklus hidup UIScene** (wajib untuk aplikasi yang dibangun dengan
//!   iOS SDK terbaru; tanpa itu UIKit menolak launch). winit 0.30 membuat
//!   `UIWindow` tanpa `windowScene`, padahal jendela tanpa scene tidak
//!   pernah tampil. Kelas [`SCENE_DELEGATE_CLASS_NAME`] harus dirujuk
//!   `UISceneDelegateClassName` di `Info.plist`, dan jendela winit
//!   ditempelkan ke `UIWindowScene` yang terhubung.
//!
//! Objek ObjC yang harus tetap hidup disimpan di `thread_local` thread
//! utama; `Retained<MainThreadOnly>` tidak `Send`.

use std::cell::RefCell;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::{define_class, ClassType, MainThreadMarker, MainThreadOnly, Message};
use objc2_foundation::{NSNotification, NSNotificationCenter, NSObject, NSObjectProtocol};
use objc2_ui_kit::{
    UIApplication, UIApplicationDidReceiveMemoryWarningNotification, UICoordinateSpace, UIScene,
    UISceneConnectionOptions, UISceneDelegate, UISceneSession, UIView, UIWindow, UIWindowScene,
    UIWindowSceneDelegate,
};

type Observer = Retained<ProtocolObject<dyn NSObjectProtocol>>;

thread_local! {
    static OBSERVERS: RefCell<Vec<Observer>> = const { RefCell::new(Vec::new()) };
}

/// Daftarkan observer peringatan memori. Aman dipanggil berkali-kali
/// (observer tambahan hanya menyalakan flag yang sama).
pub fn install_memory_warning_observer() {
    if MainThreadMarker::new().is_none() {
        return;
    }
    let center = NSNotificationCenter::defaultCenter();
    let block = block2::RcBlock::new(|_: NonNull<NSNotification>| {
        crate::platform::on_memory_warning();
    });
    // SAFETY: nama notifikasi adalah konstanta UIKit yang valid; block hidup
    // selama observer disimpan di `OBSERVERS`.
    let token = unsafe {
        center.addObserverForName_object_queue_usingBlock(
            Some(UIApplicationDidReceiveMemoryWarningNotification),
            None,
            None,
            &block,
        )
    };
    OBSERVERS.with(|o| o.borrow_mut().push(token));
}

/// Nama kelas delegate scene; harus sama dengan `UISceneDelegateClassName`
/// di `apple/ios/Info.plist.template`.
pub const SCENE_DELEGATE_CLASS_NAME: &str = "Histo3dWindowSceneDelegate";

define_class!(
    // SAFETY: NSObject tidak punya syarat subclass; tidak mengimplementasikan
    // Drop. UIKit sendiri yang meng-alloc/init kelas ini saat scene pertama
    // terhubung, jadi tidak boleh punya ivar yang butuh inisialisasi.
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "Histo3dWindowSceneDelegate"]
    struct SceneDelegate;

    unsafe impl NSObjectProtocol for SceneDelegate {}

    unsafe impl UISceneDelegate for SceneDelegate {
        #[unsafe(method(scene:willConnectToSession:options:))]
        fn will_connect(
            &self,
            scene: &UIScene,
            _session: &UISceneSession,
            _options: &UISceneConnectionOptions,
        ) {
            let Some(window_scene) = scene.downcast_ref::<UIWindowScene>() else {
                return;
            };
            // Jendela yang sudah dibuat winit sebelum scene terhubung (bila
            // ada) ditempelkan di sini; yang dibuat sesudahnya lewat
            // `attach_window_to_scene`.
            #[allow(deprecated)]
            let windows = UIApplication::sharedApplication(self.mtm()).windows();
            for window in windows.iter() {
                if !window.isHidden() {
                    attach(&window, window_scene);
                }
            }
        }
    }

    unsafe impl UIWindowSceneDelegate for SceneDelegate {}
);

/// Daftarkan kelas [`SceneDelegate`] ke runtime ObjC. `define_class!`
/// mendaftar malas pada akses pertama, sedangkan UIKit mencari kelas lewat
/// `NSClassFromString` saat scene terhubung — jadi panggil ini sebelum
/// `eframe::run_native`.
pub fn register_scene_delegate() {
    let _ = SceneDelegate::class();
}

fn connected_window_scene(mtm: MainThreadMarker) -> Option<Retained<UIWindowScene>> {
    UIApplication::sharedApplication(mtm)
        .connectedScenes()
        .iter()
        .find_map(|scene| scene.downcast::<UIWindowScene>().ok())
}

/// Tempelkan `window` ke `scene` dan tampilkan. Frame diselaraskan dengan
/// bounds scene (bisa lebih kecil dari layar saat Split View).
#[allow(deprecated)] // `coordinateSpace`: penggantinya baru ada di iOS 26.
fn attach(window: &UIWindow, scene: &UIWindowScene) {
    if window.windowScene().as_deref() == Some(scene) {
        return;
    }
    window.setWindowScene(Some(scene));
    window.setFrame(scene.coordinateSpace().bounds());
    window.makeKeyAndVisible();
}

fn ui_view_from(cc: &eframe::CreationContext<'_>) -> Option<Retained<UIView>> {
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    let handle = cc.window_handle().ok()?;
    match handle.as_raw() {
        RawWindowHandle::UiKit(h) => {
            let ptr = h.ui_view.as_ptr().cast::<UIView>();
            // SAFETY: winit menjamin pointer ini `UIView` yang hidup selama
            // jendela ada; di-retain supaya aman dipakai di sini.
            Some(unsafe { &*ptr }.retain())
        }
        _ => None,
    }
}

/// Tempelkan jendela winit milik `cc` ke scene yang terhubung. eframe
/// membuat jendela saat `Event::Resumed`, yang terjadi SETELAH
/// `scene:willConnectToSession:`, sehingga jalur delegate saja tidak cukup.
/// Mengembalikan `false` bila belum ada scene atau handle jendela.
pub fn attach_window_to_scene(cc: &eframe::CreationContext<'_>) -> bool {
    let Some(mtm) = MainThreadMarker::new() else {
        return false;
    };
    let Some(view) = ui_view_from(cc) else {
        return false;
    };
    let Some(window) = view.window() else {
        return false;
    };
    let Some(scene) = connected_window_scene(mtm) else {
        return false;
    };
    attach(&window, &scene);
    true
}
