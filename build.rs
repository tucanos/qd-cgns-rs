extern crate bindgen;

use regex::Regex;
use std::env;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;
use std::path::PathBuf;

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
    println!("cargo:rerun-if-changed=cgnslib.h");
    let include_dir = PathBuf::from(match env::var("CGNS_INLCUDE_DIR") {
        Ok(x) => x,
        Err(_) => "/usr/include".to_owned(),
    });
    // We want typedef and not #define to get a strongly type wrapper so
    // we disable CG_BUILD_LEGACY
    let type_header = replace_string(
        include_dir.join("cgnstypes.h"),
        &Regex::new("#define CG_BUILD_LEGACY .*").unwrap(),
        "",
    );
    let main_header = replace_string(
        include_dir.join("cgnslib.h"),
        &Regex::new("#include \"cgnstypes.h\"").unwrap(),
        &type_header,
    );
    dbg!(&main_header);
    let bindings = bindgen::Builder::default()
        .header_contents("cgnslib.h", &main_header)
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        //.size_t_is_usize(true)
        //.type_alias("cgsize_t")
        .generate()
        .expect("generate bindings");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("write bindings.rs");
}
