extern crate bindgen;

use bindgen::callbacks::ParseCallbacks;
use regex::Regex;
use std::env;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;
use std::path::PathBuf;

#[derive(Debug)]
struct DeriveEnumPrimitive;
impl ParseCallbacks for DeriveEnumPrimitive {
    fn add_derives(&self, info: &bindgen::callbacks::DeriveInfo<'_>) -> Vec<String> {
        if info.name == "ElementType_t" {
            vec!["TryFromPrimitive".to_string(), "IntoPrimitive".to_string()]
        } else {
            Vec::default()
        }
    }
}

fn replace_string<P: AsRef<Path>>(filename: P, oldre: &Regex, newstr: &str) -> String {
    let file = File::open(filename).expect("Unable to open file");
    let reader = BufReader::new(file);
    let mut res = String::new();
    for line in reader.lines() {
        let current_line = line.unwrap();
        let modified_line = oldre.replace_all(&current_line, newstr);
        res.push_str(&modified_line);
        res.push('\n');
    }
    res
}

fn main() {
    println!("cargo:rustc-link-lib=cgns");
    println!("cargo:rerun-if-env-changed=CGNS_INLCUDE_DIR");
    println!("cargo:rerun-if-env-changed=CGNS_DIR");
    let include_dir = env::var("CGNS_DIR").map_or_else(
        |_| {
            env::var("CGNS_INLCUDE_DIR")
                .map_or_else(|_| PathBuf::from("/usr/include"), PathBuf::from)
        },
        |prefix| {
            let prefix = PathBuf::from(prefix);
            let lib_dir = prefix.join("lib");
            #[cfg(any(target_os = "macos", target_os = "linux"))]
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
            // non standard key
            // see https://doc.rust-lang.org/cargo/reference/build-script-examples.html#linking-to-system-libraries
            // and https://github.com/rust-lang/cargo/issues/5077
            println!("cargo:rpath={}", lib_dir.display());
            prefix.join("include")
        },
    );
    let type_path = include_dir.join("cgnstypes.h");
    let main_path = include_dir.join("cgnslib.h");
    println!("cargo:rerun-if-changed={}", type_path.display());
    println!("cargo:rerun-if-changed={}", main_path.display());
    // We want typedef and not #define to get a strongly type wrapper so
    // we disable CG_BUILD_LEGACY
    let type_header = replace_string(
        type_path,
        &Regex::new("#define CG_BUILD_LEGACY .*").unwrap(),
        "",
    );
    let main_header = replace_string(
        main_path,
        &Regex::new("#include \"cgnstypes.h\"").unwrap(),
        &type_header,
    );
    let bindings = bindgen::Builder::default()
        .header_contents("cgnslib.h", &main_header)
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .parse_callbacks(Box::new(DeriveEnumPrimitive))
        .generate()
        .expect("generate bindings");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("write bindings.rs");
}
