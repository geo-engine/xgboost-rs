extern crate bindgen;
extern crate cmake;

use cmake::Config;
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let target = env::var("TARGET").unwrap();
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let xgb_root = Path::new(&out_path).join("xgboost");

    // copy source code into OUT_DIR for compilation if it doesn't exist
    if !xgb_root.exists() {
        Command::new("cp")
            .args(&["-r", "xgboost", xgb_root.to_str().unwrap()])
            .status()
            .unwrap_or_else(|e| {
                panic!("Failed to copy ./xgboost to {}: {}", xgb_root.display(), e);
            });
    }

    // CMake
    let dst = Config::new(&xgb_root)
        .uses_cxx11()
        .define("BUILD_STATIC_LIB", "ON")
        .build();

    // CONFIG BINDGEN
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_args(&["-x", "c++", "-std=c++11"])
        .clang_arg(format!("-I{}", xgb_root.join("include").display()))
        .clang_arg(format!("-I{}", xgb_root.join("rabit/include").display()))
        .clang_arg(format!(
            "-I{}",
            xgb_root.join("dmlc-core/include").display()
        ))
        .generate_comments(false)
        .generate()
        .expect("Unable to generate bindings.");

    // GENERATE THE BINDINGS

    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings.");

    println!("cargo:rustc-link-search={}", dst.join("lib").display());
    println!("cargo:rustc-link-lib=xgboost");
    println!("cargo:rustc-link-lib=dmlc");

    if target.contains("apple") {
        println!("cargo:rustc-link-lib=dylib=c++");
        println!("cargo:rustc-link-lib=dylib=omp");
    } else {
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-link-lib=dylib=gomp");
    }
}
