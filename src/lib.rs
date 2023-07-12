use core::ffi::CStr;
use std::array;
use std::ffi::{c_void, CString};
use std::fmt::Debug;
use std::sync::{Mutex, MutexGuard};

mod cgns_sys;
pub mod tools;
pub use cgns_sys::DataType_t;
use cgns_sys::DataType_t::RealDouble;
use cgns_sys::ZoneType_t::Unstructured;
use cgns_sys::{
    cg_array_write, cg_base_write, cg_biter_read, cg_biter_write, cg_close, cg_coord_info,
    cg_coord_read, cg_coord_write, cg_elements_read, cg_get_error, cg_golist, cg_nsections,
    cg_open, cg_save_as, cg_section_read, cg_section_write, cg_ziter_write, cg_zone_read,
    cg_zone_write, CG_FILE_ADF, CG_FILE_ADF2, CG_FILE_HDF5, CG_FILE_NONE, CG_MODE_MODIFY,
    CG_MODE_READ, CG_MODE_WRITE,
};

pub use cgns_sys::{cgsize_t, ElementType_t};
pub struct Error(i32);
type Result<T> = std::result::Result<T, Error>;

pub enum Mode {
    Read,
    Write,
    Modify,
}

pub enum FileType {
    NONE,
    ADF,
    HDF5,
    ADF2,
}

pub enum GotoQueryItem {
    Name(String),
    LabelIndex(String, i32),
}

impl GotoQueryItem {
    fn string(&self) -> &str {
        match self {
            GotoQueryItem::Name(n) => n,
            GotoQueryItem::LabelIndex(n, _) => n,
        }
    }
    fn index(&self) -> i32 {
        match self {
            GotoQueryItem::Name(_) => 0,
            GotoQueryItem::LabelIndex(_, i) => *i,
        }
    }
}

impl From<String> for GotoQueryItem {
    fn from(value: String) -> Self {
        Self::Name(value)
    }
}

impl From<&str> for GotoQueryItem {
    fn from(value: &str) -> Self {
        Self::Name(value.into())
    }
}

impl From<(&str, i32)> for GotoQueryItem {
    fn from(value: (&str, i32)) -> Self {
        Self::LabelIndex(value.0.into(), value.1)
    }
}

pub trait CgnsDataType {
    const SYS: DataType_t;
}

pub struct GotoContext<'a> {
    _mutex: MutexGuard<'a, ()>,
    file: &'a File,
}

type Where = (Base, Vec<(String, i32)>);

impl<'a> GotoContext<'a> {
    pub fn array_write<T: CgnsDataType>(
        &self,
        arrayname: &str,
        dimensions: &[i32],
        data: &[T],
    ) -> Result<()> {
        let arrayname = CString::new(arrayname).unwrap();
        assert_eq!(
            dimensions.iter().copied().reduce(|a, v| a * v).unwrap(),
            i32::try_from(data.len()).unwrap()
        );
        let e = unsafe {
            cg_array_write(
                arrayname.as_ptr(),
                T::SYS,
                i32::try_from(dimensions.len()).unwrap(),
                dimensions.as_ptr(),
                data.as_ptr().cast::<std::ffi::c_void>(),
            )
        };
        if e == 0 {
            Ok(())
        } else {
            Err(e.into())
        }
    }

    pub fn narrays(&self) -> Result<i32> {
        let mut r = 0;
        let e = unsafe { cgns_sys::cg_narrays(&mut r) };
        if e == 0 {
            Ok(r)
        } else {
            Err(e.into())
        }
    }

    pub fn array_info(&self, array_id: i32) -> Result<(String, DataType_t, Vec<i32>)> {
        let mut raw_name = [0_u8; 64];
        let mut dimensions = [0; 12];
        let mut rank = 0;
        let mut datatype = DataType_t::DataTypeNull;
        let e = unsafe {
            cgns_sys::cg_array_info(
                array_id,
                raw_name.as_mut_ptr().cast(),
                &mut datatype,
                &mut rank,
                dimensions.as_mut_ptr(),
            )
        };
        if e == 0 {
            let rank = rank.try_into().unwrap();
            Ok((
                raw_to_string(&raw_name),
                datatype,
                dimensions[0..rank].to_vec(),
            ))
        } else {
            Err(e.into())
        }
    }

    pub fn array_read<T: CgnsDataType>(&self, array_id: i32, data: &mut [T]) -> Result<()> {
        let e = unsafe { cgns_sys::cg_array_read(array_id, data.as_mut_ptr().cast()) };
        if e == 0 {
            Ok(())
        } else {
            Err(e.into())
        }
    }

    pub fn gorel(&self, query: &[GotoQueryItem]) -> Result<()> {
        // varargs is not yet available in stable Rust so we recurse
        if let Some((head, queue)) = query.split_first() {
            let end = "end\0".as_bytes().as_ptr();
            let (s, i) = (head.string(), head.index());
            let s = CString::new(s).unwrap();
            let e = unsafe { cgns_sys::cg_gorel(self.file.0, s.as_ptr(), i, end) };
            if e == 0 {
                self.gorel(queue)
            } else {
                Err(e.into())
            }
        } else {
            Ok(())
        }
    }

    pub fn famname_write(&self, family_name: &str) -> Result<()> {
        let n = CString::new(family_name).unwrap();
        let e = unsafe { cgns_sys::cg_famname_write(n.as_ptr()) };
        if e == 0 {
            Ok(())
        } else {
            Err(e.into())
        }
    }

    pub fn r#where(&self) -> Result<Where> {
        const MAX_DEPTH: usize = cgns_sys::CG_MAX_GOTO_DEPTH as usize;
        let mut index = [0_i32; MAX_DEPTH];
        let mut base = 0;
        let mut depth = 0;
        let mut file = 0;
        let mut label = [[0_u8; 34]; MAX_DEPTH];
        let mut labelp: [_; MAX_DEPTH] = array::from_fn(|i| label[i].as_mut_ptr());
        let e = unsafe {
            cgns_sys::cg_where(
                &mut file,
                &mut base,
                &mut depth,
                labelp.as_mut_ptr().cast(),
                index.as_mut_ptr(),
            )
        };
        if e == 0 {
            Ok((
                Base(base),
                (0..depth as usize)
                    .map(|i| (raw_to_string(&label[i]), index[i]))
                    .collect(),
            ))
        } else {
            Err(e.into())
        }
    }
}

impl CgnsDataType for i32 {
    const SYS: DataType_t = DataType_t::Integer;
}

impl CgnsDataType for u8 {
    const SYS: DataType_t = DataType_t::Character;
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

impl From<FileType> for i32 {
    fn from(t: FileType) -> i32 {
        match t {
            FileType::NONE => CG_FILE_NONE as i32,
            FileType::ADF => CG_FILE_ADF as i32,
            FileType::ADF2 => CG_FILE_ADF2 as i32,
            FileType::HDF5 => CG_FILE_HDF5 as i32,
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

#[derive(Debug)]
pub struct File(std::os::raw::c_int);
#[derive(Copy, Clone, Debug)]
pub struct Base(std::os::raw::c_int);
impl From<i32> for Base {
    fn from(value: i32) -> Self {
        assert!(value >= 1);
        Base(value)
    }
}
#[derive(Copy, Clone)]
pub struct Zone(std::os::raw::c_int);
impl From<i32> for Zone {
    fn from(value: i32) -> Self {
        assert!(value >= 1);
        Zone(value)
    }
}

fn raw_to_string(buf: &[u8]) -> String {
    let nulpos = buf.iter().position(|&r| r == 0).unwrap();
    CStr::from_bytes_with_nul(&buf[0..=nulpos])
        .unwrap()
        .to_str()
        .unwrap()
        .to_string()
}

#[derive(Default)]
pub struct SectionInfo {
    pub section_name: String,
    pub typ: ElementType_t,
    pub start: usize,
    pub end: usize,
    pub nbndry: i32,
}

impl SectionInfo {
    #[must_use]
    pub fn new(typ: ElementType_t, end: usize) -> Self {
        Self {
            section_name: "Elem".to_owned(),
            typ,
            start: 1,
            end,
            nbndry: 0,
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

    pub fn save_as(&self, filename: &str, file_type: FileType, follow_links: bool) -> Result<()> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let filename = CString::new(filename).unwrap();
        let follow_links = i32::from(follow_links);
        let file_type = file_type.into();
        let e = unsafe { cg_save_as(self.0, filename.as_ptr(), file_type, follow_links) };
        if e == 0 {
            Ok(())
        } else {
            Err(e.into())
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
        let e = unsafe { cg_biter_read(self.0, base.0, name.as_mut_ptr().cast(), &mut n_steps) };
        if e == 0 {
            Ok((raw_to_string(&name), n_steps))
        } else {
            Err(e.into())
        }
    }

    pub fn goto(&self, base: Base, query: &[GotoQueryItem]) -> Result<GotoContext> {
        let _mutex = CGNS_MUTEX.lock().unwrap();
        let end = "end\0".as_bytes().as_ptr();
        let e = unsafe { cgns_sys::cg_goto(self.0, base.0, end) };
        if e == 0 {
            // varargs is not yet available in stable Rust so we rely on cg_gorel
            let l = GotoContext { _mutex, file: self };
            l.gorel(query)?;
            Ok(l)
        } else {
            Err(e.into())
        }
    }

    pub fn golist(&self, base: Base, labels: &[&str], index: &[i32]) -> Result<GotoContext> {
        let _mutex = CGNS_MUTEX.lock().unwrap();
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
            Ok(GotoContext { _mutex, file: self })
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
        vertex_size: usize,
        cell_size: usize,
        boundary_size: usize,
    ) -> Result<Zone> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let zonename = CString::new(zonename).unwrap();
        let mut z: i32 = 0;
        let size = [
            vertex_size as cgsize_t,
            cell_size as cgsize_t,
            boundary_size as cgsize_t,
        ];
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
                coord.as_ptr().cast::<c_void>(),
                &mut c,
            )
        };
        if e == 0 {
            Ok(())
        } else {
            Err(e.into())
        }
    }

    pub fn zone_read(&self, base: Base, zone: Zone) -> Result<(String, [usize; 9])> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let mut r = [0 as cgsize_t; 9];
        let mut buf = [0_u8; 64];
        let err = unsafe {
            cg_zone_read(
                self.0,
                base.0,
                zone.0,
                buf.as_mut_ptr().cast(),
                r.as_mut_ptr(),
            )
        };
        if err == 0 {
            Ok((raw_to_string(&buf), r.map(|s| s as usize)))
        } else {
            Err(err.into())
        }
    }

    pub fn coord_info(&self, base: Base, zone: Zone, c: i32) -> Result<(DataType_t, String)> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let mut datatype = DataType_t::Integer;
        let mut raw_name = [0_u8; 64];
        let err = unsafe {
            cg_coord_info(
                self.0,
                base.0,
                zone.0,
                c,
                &mut datatype,
                raw_name.as_mut_ptr().cast(),
            )
        };
        if err == 0 {
            Ok((datatype, raw_to_string(&raw_name)))
        } else {
            Err(err.into())
        }
    }

    pub fn coord_read(
        &self,
        base: Base,
        zone: Zone,
        coordname: &str,
        range_min: usize,
        range_max: usize,
        coord_array: &mut [f64],
    ) -> Result<()> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let range_min = range_min as cgsize_t;
        let range_max = range_max as cgsize_t;
        let p = CString::new(coordname).unwrap();
        let err = unsafe {
            cg_coord_read(
                self.0,
                base.0,
                zone.0,
                p.as_ptr(),
                RealDouble,
                &range_min,
                &range_max,
                coord_array.as_mut_ptr().cast(),
            )
        };
        if err == 0 {
            Ok(())
        } else {
            Err(err.into())
        }
    }

    pub fn section_write(
        &mut self,
        base: Base,
        zone: Zone,
        args: &SectionInfo,
        elements: &[cgsize_t],
    ) -> Result<()> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let section_name = CString::new(args.section_name.clone()).unwrap();
        let mut c = 0;
        let e = unsafe {
            cg_section_write(
                self.0,
                base.0,
                zone.0,
                section_name.as_ptr(),
                args.typ,
                args.start as cgsize_t,
                args.end as cgsize_t,
                args.nbndry,
                elements.as_ptr(),
                &mut c,
            )
        };
        if e == 0 {
            Ok(())
        } else {
            Err(e.into())
        }
    }

    pub fn poly_section_write(
        &mut self,
        base: Base,
        zone: Zone,
        args: &SectionInfo,
        elements: &[cgsize_t],
        offsets: &[cgsize_t],
    ) -> Result<()>{
        let _l = CGNS_MUTEX.lock().unwrap();
        let section_name = CString::new(args.section_name.clone()).unwrap();
        let mut c = 0;
        let e = unsafe {
            cgns_sys::cg_poly_section_write(
                self.0,
                base.0,
                zone.0,
                section_name.as_ptr(),
                args.typ,
                args.start as cgsize_t,
                args.end as cgsize_t,
                args.nbndry,
                elements.as_ptr(),
                offsets.as_ptr(),
                &mut c,
            )
        };
        if e == 0 {
            Ok(())
        } else {
            Err(e.into())
        }
    }

    pub fn elements_read(
        &self,
        base: Base,
        zone: Zone,
        section: i32,
        elements: &mut [i32],
        parent_data: &mut [i32],
    ) -> Result<()> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let ptr = if parent_data.is_empty() {
            std::ptr::null_mut()
        } else {
            parent_data.as_mut_ptr()
        };
        let e = unsafe {
            cg_elements_read(self.0, base.0, zone.0, section, elements.as_mut_ptr(), ptr)
        };
        if e == 0 {
            Ok(())
        } else {
            Err(e.into())
        }
    }

    pub fn nzones(&self, base: Base) -> Result<i32> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let mut r = 0;
        let e = unsafe { cgns_sys::cg_nzones(self.0, base.0, &mut r) };
        if e == 0 {
            Ok(r)
        } else {
            Err(e.into())
        }
    }

    pub fn nsections(&self, base: Base, zone: Zone) -> Result<i32> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let mut r = 0;
        let e = unsafe { cg_nsections(self.0, base.0, zone.0, &mut r) };
        if e == 0 {
            Ok(r)
        } else {
            Err(e.into())
        }
    }

    pub fn section_read(
        &self,
        base: Base,
        zone: Zone,
        section: i32,
    ) -> Result<(SectionInfo, bool)> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let mut info = SectionInfo::default();
        let mut parent_flag = 0_i32;
        let mut raw_name = [0_u8; 64];
        let mut start: cgsize_t = 0;
        let mut end: cgsize_t = 0;
        let e = unsafe {
            cg_section_read(
                self.0,
                base.0,
                zone.0,
                section,
                raw_name.as_mut_ptr().cast(),
                &mut info.typ,
                &mut start,
                &mut end,
                &mut info.nbndry,
                &mut parent_flag,
            )
        };
        if e == 0 {
            info.section_name = raw_to_string(&raw_name);
            info.start = start as usize;
            info.end = end as usize;
            Ok((info, parent_flag != 0))
        } else {
            Err(e.into())
        }
    }
}

impl Drop for File {
    fn drop(&mut self) {
        if let Err(e) = self.close() {
            eprintln!("{e:?}");
        }
    }
}
