use core::ffi::CStr;
use std::ffi::{c_void, CString};
use std::fmt::Debug;
use std::sync::{Mutex, MutexGuard};

use cgns_sys::DataType_t::RealDouble;
use cgns_sys::ZoneType_t::Unstructured;
use cgns_sys::*;

pub use cgns_sys::ElementType_t;
pub struct Error(i32);
type Result<T> = std::result::Result<T, Error>;

pub enum Mode {
    Read,
    Write,
    Modify,
}

pub trait CgnsDataType {
    const SYS: DataType_t::Type;
}

pub struct GotoContext<'a>(MutexGuard<'a, ()>);

impl<'a> GotoContext<'a> {
    pub fn array_write<T: CgnsDataType>(
        &self,
        arrayname: &str,
        dimensions: &[i32],
        data: &[T],
    ) -> Result<()> {
        let arrayname = CString::new(arrayname).unwrap();
        assert_eq!(
            dimensions.iter().cloned().reduce(|a, v| a * v).unwrap(),
            data.len() as i32
        );
        let e = unsafe {
            cg_array_write(
                arrayname.as_ptr(),
                T::SYS,
                dimensions.len() as i32,
                dimensions.as_ptr(),
                data.as_ptr() as *const c_void,
            )
        };
        if e == 0 {
            Ok(())
        } else {
            Err(e.into())
        }
    }
}

impl CgnsDataType for i32 {
    const SYS: DataType_t::Type = DataType_t::Integer;
}

impl From<Mode> for i32 {
    fn from(m: Mode) -> i32 {
        match m {
            Mode::Read => CG_MODE_READ as i32,
            Mode::Write => CG_MODE_WRITE as i32,
            Mode::Modify => CG_MODE_MODIFY as i32,
        }
    }
}

impl From<i32> for Error {
    fn from(code: i32) -> Self {
        Error(code)
    }
}

impl Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = unsafe { CStr::from_ptr(cg_get_error()) };
        write!(f, "{} (error {})", msg.to_str().unwrap(), self.0)
    }
}

static CGNS_MUTEX: Mutex<()> = Mutex::new(());

pub fn open(path: &str, mode: Mode) -> Result<File> {
    let _l = CGNS_MUTEX.lock().unwrap();
    let mut fd: i32 = 0;
    let path = CString::new(path).unwrap();
    let f = unsafe { cg_open(path.as_ptr(), mode.into(), &mut fd) };
    if f == 0 {
        Ok(File(fd))
    } else {
        Err(f.into())
    }
}

pub struct File(i32);
#[derive(Copy, Clone)]
pub struct Base(i32);
impl Base {
    pub fn new(arg: i32) -> Base {
        Base(arg)
    }
}
#[derive(Copy, Clone)]
pub struct Zone(i32);

pub struct SectionDef<'a> {
    base: Base,
    zone: Zone,
    pub section_name: &'a str,
    typ: ElementType_t::Type,
    pub start: isize,
    end: isize,
    pub nbndry: i32,
    elements: &'a [i32],
}

impl<'a> SectionDef<'a> {
    pub fn new(
        base: Base,
        zone: Zone,
        typ: ElementType_t::Type,
        end: isize,
        elements: &'a [i32],
    ) -> Self {
        Self {
            base,
            zone,
            section_name: "Elem",
            typ,
            start: 1,
            end,
            nbndry: 0,
            elements,
        }
    }
}

impl File {
    pub fn close(&mut self) -> Result<()> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let f = unsafe { cg_close(self.0) };
        if f == 0 {
            Ok(())
        } else {
            Err(f.into())
        }
    }

    pub fn biter_write(&mut self, base: Base, base_iter_name: &str, n_steps: i32) -> Result<()> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let base_iter_name = CString::new(base_iter_name).unwrap();
        let e = unsafe { cg_biter_write(self.0, base.0, base_iter_name.as_ptr(), n_steps) };
        if e == 0 {
            Ok(())
        } else {
            Err(e.into())
        }
    }

    pub fn biter_read(&mut self, base: Base) -> Result<(String, i32)> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let mut n_steps = 0;
        let mut name = [0_u8; 33];
        let e =
            unsafe { cg_biter_read(self.0, base.0, name.as_mut_ptr() as *mut i8, &mut n_steps) };
        if e == 0 {
            let nulpos = name.iter().position(|&r| r == 0).unwrap();
            Ok((
                CStr::from_bytes_with_nul(&name[0..=nulpos])
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_string(),
                n_steps,
            ))
        } else {
            Err(e.into())
        }
    }

    pub fn golist(&self, base: Base, labels: &[&str], index: &[i32]) -> Result<GotoContext> {
        let l = CGNS_MUTEX.lock().unwrap();
        let labels: Vec<_> = labels.iter().map(|&s| CString::new(s).unwrap()).collect();
        let mut labels_ptr: Vec<_> = labels.iter().map(|s| s.as_ptr() as *mut i8).collect();
        let e = unsafe {
            cg_golist(
                self.0,
                base.0,
                labels.len() as i32,
                labels_ptr.as_mut_ptr(),
                index.as_ptr() as *mut i32,
            )
        };
        if e == 0 {
            Ok(GotoContext(l))
        } else {
            Err(e.into())
        }
    }

    // https://cgns.github.io/CGNS_docs_current/sids/timedep.html
    pub fn ziter_write(&mut self, base: Base, zone: Zone, zone_iter_name: &str) -> Result<()> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let zone_iter_name = CString::new(zone_iter_name).unwrap();
        let e = unsafe { cg_ziter_write(self.0, base.0, zone.0, zone_iter_name.as_ptr()) };
        if e == 0 {
            Ok(())
        } else {
            Err(e.into())
        }
    }

    // https://cgns.github.io/CGNS_docs_current/midlevel/structural.html
    pub fn base_write(&mut self, basename: &str, cell_dim: i32, phys_dim: i32) -> Result<Base> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let basename = CString::new(basename).unwrap();
        let mut b: i32 = 0;
        let e = unsafe { cg_base_write(self.0, basename.as_ptr(), cell_dim, phys_dim, &mut b) };
        if e == 0 {
            Ok(Base(b))
        } else {
            Err(e.into())
        }
    }
    pub fn zone_write(
        &mut self,
        base: Base,
        zonename: &str,
        vertex_size: i32,
        cell_size: i32,
        boundary_size: i32,
    ) -> Result<Zone> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let zonename = CString::new(zonename).unwrap();
        let mut z: i32 = 0;
        let size = [vertex_size, cell_size, boundary_size];
        let e = unsafe {
            cg_zone_write(
                self.0,
                base.0,
                zonename.as_ptr(),
                size.as_ptr(),
                Unstructured,
                &mut z,
            )
        };
        if e == 0 {
            Ok(Zone(z))
        } else {
            Err(e.into())
        }
    }

    // https://cgns.github.io/CGNS_docs_current/midlevel/grid.html
    pub fn coord_write(
        &mut self,
        base: Base,
        zone: Zone,
        coordname: &str,
        coord: &[f64],
    ) -> Result<()> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let coordname = CString::new(coordname).unwrap();
        let mut c = 0;
        let e = unsafe {
            cg_coord_write(
                self.0,
                base.0,
                zone.0,
                RealDouble,
                coordname.as_ptr(),
                coord.as_ptr() as *const c_void,
                &mut c,
            )
        };
        if e == 0 {
            Ok(())
        } else {
            Err(e.into())
        }
    }

    pub fn section_write(&mut self, args: SectionDef) -> Result<()> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let section_name = CString::new(args.section_name).unwrap();
        let mut c = 0;
        let e = unsafe {
            cg_section_write(
                self.0,
                args.base.0,
                args.zone.0,
                section_name.as_ptr(),
                args.typ,
                args.start as i32,
                args.end as i32,
                args.nbndry,
                args.elements.as_ptr(),
                &mut c,
            )
        };
        if e == 0 {
            Ok(())
        } else {
            Err(e.into())
        }
    }
}

impl Drop for File {
    fn drop(&mut self) {
        self.close().unwrap();
    }
}
