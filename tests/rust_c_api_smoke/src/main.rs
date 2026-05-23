use std::env;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::PathBuf;
use std::ptr;

#[cfg(windows)]
#[link(name = "kernel32")]
extern "system" {
    fn LoadLibraryA(name: *const c_char) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
    fn FreeLibrary(module: *mut c_void) -> c_int;
}

type AbiVersion = unsafe extern "C" fn() -> *const c_char;
type StringFree = unsafe extern "C" fn(*mut c_char);
type Json0 = unsafe extern "C" fn() -> *mut c_char;
type Json1 = unsafe extern "C" fn(*const c_char) -> *mut c_char;
type CreateSession =
    unsafe extern "C" fn(*const c_char, *const c_char, u64, c_int, *mut *mut c_char) -> *mut c_void;
type DestroySession = unsafe extern "C" fn(*mut c_void);
type DecideJson =
    unsafe extern "C" fn(*mut c_void, c_int, f64, c_int, *mut *mut c_char) -> *mut c_char;

struct Api {
    module: *mut c_void,
    abi_version: AbiVersion,
    string_free: StringFree,
    available_games_json: Json0,
    game_metadata_json: Json1,
    session_create: CreateSession,
    session_destroy: DestroySession,
    session_decide_json: DecideJson,
}

impl Drop for Api {
    fn drop(&mut self) {
        unsafe {
            FreeLibrary(self.module);
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!(
            "usage: {} <path-to-dinoboard_c_api.dll> <path-to-splendor_2p.onnx>",
            args[0]
        );
        std::process::exit(2);
    }

    let dll = PathBuf::from(&args[1]);
    let model = PathBuf::from(&args[2]);
    let api = unsafe { load_api(&dll) };
    unsafe {
        println!("abi={}", cstr_const((api.abi_version)()));
        println!(
            "games={}",
            owned_json((api.available_games_json)(), api.string_free)
        );

        let game = CString::new("splendor_2p").unwrap();
        println!(
            "metadata={}",
            owned_json((api.game_metadata_json)(game.as_ptr()), api.string_free)
        );

        let model_c = CString::new(model.to_string_lossy().as_bytes()).unwrap();
        let mut err: *mut c_char = ptr::null_mut();
        let session = (api.session_create)(game.as_ptr(), model_c.as_ptr(), 20260524, 0, &mut err);
        if session.is_null() {
            let message = take_error(err, api.string_free);
            panic!("session_create failed: {}", message);
        }
        let result = (api.session_decide_json)(session, 32, 0.0, 1, &mut err);
        if result.is_null() {
            let message = take_error(err, api.string_free);
            (api.session_destroy)(session);
            panic!("session_decide_json failed: {}", message);
        }
        let json = owned_json(result, api.string_free);
        (api.session_destroy)(session);
        println!("decide={}", json);
        if !json.contains("\"action_id\"") || !json.contains("\"root_actions\"") {
            panic!("decide JSON did not contain expected fields");
        }
    }
}

unsafe fn load_api(path: &PathBuf) -> Api {
    let dll_c = CString::new(path.to_string_lossy().as_bytes()).unwrap();
    let module = LoadLibraryA(dll_c.as_ptr());
    if module.is_null() {
        panic!("LoadLibraryA failed for {}", path.display());
    }
    Api {
        module,
        abi_version: load_sym(module, "dinoboard_abi_version"),
        string_free: load_sym(module, "dinoboard_string_free"),
        available_games_json: load_sym(module, "dinoboard_available_games_json"),
        game_metadata_json: load_sym(module, "dinoboard_game_metadata_json"),
        session_create: load_sym(module, "dinoboard_session_create"),
        session_destroy: load_sym(module, "dinoboard_session_destroy"),
        session_decide_json: load_sym(module, "dinoboard_session_decide_json"),
    }
}

unsafe fn load_sym<T>(module: *mut c_void, name: &str) -> T {
    let cname = CString::new(name).unwrap();
    let ptr = GetProcAddress(module, cname.as_ptr());
    if ptr.is_null() {
        panic!("GetProcAddress failed for {}", name);
    }
    std::mem::transmute_copy(&ptr)
}

unsafe fn cstr_const(ptr: *const c_char) -> String {
    CStr::from_ptr(ptr).to_string_lossy().into_owned()
}

unsafe fn owned_json(ptr: *mut c_char, free_fn: StringFree) -> String {
    if ptr.is_null() {
        return String::new();
    }
    let out = CStr::from_ptr(ptr).to_string_lossy().into_owned();
    free_fn(ptr);
    out
}

unsafe fn take_error(ptr: *mut c_char, free_fn: StringFree) -> String {
    if ptr.is_null() {
        return "unknown error".to_string();
    }
    owned_json(ptr, free_fn)
}
