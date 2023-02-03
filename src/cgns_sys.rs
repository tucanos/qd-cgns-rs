#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

impl Default for ElementType_t {
    fn default() -> Self {
        ElementType_t::ElementTypeNull
    }
}
