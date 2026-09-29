//! Builds the vendored Box3D C sources with `cc` and (optionally) regenerates
//! the Rust bindings with `bindgen`.
//!
//! Flags mirror Box3D's CMakeLists.txt / src/CMakeLists.txt:
//! - C17 with GNU extensions (`C_STANDARD 17`, `C_EXTENSIONS YES`)
//! - `-ffp-contract=off` on GCC/Clang (cross-platform determinism)
//! - SIMD: Box3D picks SSE2 on x86_64 and NEON on aarch64 from compiler
//!   predefines (see src/core.h). There is no AVX2 path in this version, so
//!   no extra `-m` flags are needed. `disable-simd` defines BOX3D_DISABLE_SIMD.
//! - BOX3D_VALIDATE only with the `validate` feature (CMake enables it by
//!   default for top-level debug builds; it is heavy, so opt-in here).
//! - NDEBUG when Rust debug assertions are off (release), so Box3D asserts
//!   compile out like a CMake Release build. `enable-assert` keeps them.

use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let box3d_dir = manifest_dir.join("box3d");
    let include_dir = box3d_dir.join("include");
    let src_dir = box3d_dir.join("src");

    if !src_dir.join("physics_world.c").exists() {
        panic!("Box3D sources not found at {}. Did you run `git submodule update --init`?", src_dir.display());
    }

    let double = env::var_os("CARGO_FEATURE_DOUBLE_PRECISION").is_some();
    let validate = env::var_os("CARGO_FEATURE_VALIDATE").is_some();
    let disable_simd = env::var_os("CARGO_FEATURE_DISABLE_SIMD").is_some();
    let enable_assert = env::var_os("CARGO_FEATURE_ENABLE_ASSERT").is_some();
    let debug_assertions = env::var_os("CARGO_CFG_DEBUG_ASSERTIONS").is_some();

    let mut sources: Vec<PathBuf> = std::fs::read_dir(&src_dir)
        .expect("read box3d/src")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "c"))
        .collect();
    sources.sort();

    let mut build = cc::Build::new();
    build.files(&sources).include(&include_dir).include(&src_dir).warnings(false).extra_warnings(false);

    let compiler = build.get_compiler();
    if compiler.is_like_msvc() {
        build.flag("/std:c17");
        if compiler.is_like_clang_cl() {
            build.flag("/clang:-ffp-contract=off");
        }
    } else {
        build.flag("-std=gnu17");
        build.flag("-ffp-contract=off");
        // Box3D's CMake hides internal symbols; for a static lib this only
        // matters if this crate ends up inside a cdylib.
        build.flag_if_supported("-fvisibility=hidden");
    }

    if double {
        build.define("BOX3D_DOUBLE_PRECISION", None);
    }
    if validate {
        build.define("BOX3D_VALIDATE", None);
    }
    if disable_simd {
        build.define("BOX3D_DISABLE_SIMD", None);
    }
    if !debug_assertions {
        build.define("NDEBUG", None);
    }
    if enable_assert {
        build.define("B3_ENABLE_ASSERT", None);
    }

    build.compile("box3d");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "linux" || target_os == "android" {
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=pthread");
    }

    println!("cargo:include={}", include_dir.display());
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed={}", include_dir.display());
    println!("cargo:rerun-if-changed={}", src_dir.display());

    #[cfg(feature = "bindgen")]
    generate_bindings(&manifest_dir, &include_dir, double);
    #[cfg(not(feature = "bindgen"))]
    let _ = (&manifest_dir, double);
}

#[cfg(feature = "bindgen")]
fn generate_bindings(manifest_dir: &std::path::Path, include_dir: &std::path::Path, double: bool) {
    let mut builder = bindgen::Builder::default()
        .header(manifest_dir.join("wrapper.h").to_string_lossy())
        .clang_arg(format!("-I{}", include_dir.display()))
        .clang_arg("-std=c17")
        .allowlist_function("b3.*")
        .allowlist_type("b3.*")
        .allowlist_var("B3_.*")
        // `static const` header constants have internal linkage: no symbol to link.
        .blocklist_item("b3Vec3_.*|b3Quat_identity|b3Transform_identity|b3Mat3_.*|b3Pos_zero|b3WorldTransform_identity|b3_emptyDistanceCache|b3_null.*Id")
        .default_enum_style(bindgen::EnumVariation::Consts)
        .derive_default(true)
        .derive_debug(true)
        .derive_copy(true)
        .layout_tests(true)
        .generate_comments(true)
        .size_t_is_usize(true)
        .use_core()
        .merge_extern_blocks(true)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));
    if double {
        builder = builder.clang_arg("-DBOX3D_DOUBLE_PRECISION");
    }
    let bindings = builder.generate().expect("bindgen failed");

    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
    bindings.write_to_file(&out).expect("write bindings");

    // `BOX3D_SYS_UPDATE_BINDINGS=1 cargo build -p box3d-sys --features bindgen`
    // refreshes the committed pregenerated file for the current precision.
    if env::var_os("BOX3D_SYS_UPDATE_BINDINGS").is_some() {
        let name = if double { "bindings_double.rs" } else { "bindings.rs" };
        let dst = manifest_dir.join("src").join(name);
        std::fs::copy(&out, &dst).expect("copy bindings into src/");
        println!("cargo:warning=updated {}", dst.display());
    }
    println!("cargo:rerun-if-env-changed=BOX3D_SYS_UPDATE_BINDINGS");
}
