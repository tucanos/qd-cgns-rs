use std::array;
use std::ffi::{CStr, CString, c_int, c_void};
use std::fmt::Debug;
use std::num::NonZero;
use std::ptr::null_mut;
use std::sync::{Mutex, MutexGuard};

mod cgns_sys;
pub mod tools;
use DataType::RealDouble;
pub use cgns_sys::DataType_t as DataType;
pub use cgns_sys::GridLocation_t as GridLocation;
use cgns_sys::ZoneType_t::Unstructured;
use cgns_sys::{
    CG_FILE_ADF, CG_FILE_ADF2, CG_FILE_HDF5, CG_FILE_NONE, CG_MODE_MODIFY, CG_MODE_READ,
    CG_MODE_WRITE, cg_array_write, cg_base_write, cg_biter_read, cg_biter_write, cg_close,
    cg_coord_info, cg_coord_read, cg_coord_write, cg_elements_read, cg_get_error, cg_golist,
    cg_nsections, cg_open, cg_save_as, cg_section_read, cg_section_write, cg_ziter_write,
    cg_zone_read, cg_zone_write,
};

pub use cgns_sys::{
    BCType_t as BCType, ElementType_t as ElementType, PointSetType_t as PointSetType,
    cgsize_t as cgsize,
};
use num_enum::TryFromPrimitive;
use num_enum::TryFromPrimitiveError;

#[derive(Debug)]
pub struct Error(i32);
pub type Result<T> = std::result::Result<T, Error>;

impl TryFrom<u8> for ElementType {
    type Error = TryFromPrimitiveError<Self>;

    fn try_from(value: u8) -> std::result::Result<Self, Self::Error> {
        Self::try_from_primitive(u32::from(value))
    }
}

impl ElementType {
    /// Get the number of nodes for an element type
    ///
    /// See `cg_npe` in CGNS documentation
    pub fn npe(self) -> Result<usize> {
        let mut result = 0;
        let e = unsafe { cgns_sys::cg_npe(self, &raw mut result) };
        if e == 0 {
            Ok(e.try_into().unwrap())
        } else {
            Err(e.into())
        }
    }
}

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
            Self::Name(n) => n,
            Self::LabelIndex(n, _) => n,
        }
    }
    const fn index(&self) -> i32 {
        match self {
            Self::Name(_) => 0,
            Self::LabelIndex(_, i) => *i,
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
    const SYS: DataType;
}

pub struct GotoContext<'a> {
    #[allow(dead_code)]
    mutex: MutexGuard<'a, ()>,
    file: &'a File,
}

type Where = (Base, Vec<(String, i32)>);

impl GotoContext<'_> {
    pub fn array_write<T: CgnsDataType>(
        &self,
        arrayname: &str,
        dimensions: &[usize],
        data: &[T],
    ) -> Result<()> {
        let arrayname = CString::new(arrayname).unwrap();
        assert_eq!(
            dimensions.iter().copied().reduce(|a, v| a * v).unwrap(),
            data.len()
        );
        let dimensions: Vec<_> = dimensions
            .iter()
            .map(|&x| cgsize::try_from(x).unwrap())
            .collect();
        let e = unsafe {
            cg_array_write(
                arrayname.as_ptr(),
                T::SYS,
                i32::try_from(dimensions.len()).unwrap(),
                dimensions.as_ptr(),
                data.as_ptr().cast::<c_void>(),
            )
        };
        if e == 0 { Ok(()) } else { Err(e.into()) }
    }

    pub fn narrays(&self) -> Result<i32> {
        let mut r = 0;
        let e = unsafe { cgns_sys::cg_narrays(&raw mut r) };
        if e == 0 { Ok(r) } else { Err(e.into()) }
    }

    pub fn array_info(&self, array_id: i32) -> Result<(String, DataType, Vec<usize>)> {
        let mut raw_name = [0_u8; 64];
        let mut dimensions = [0; 12];
        let mut rank = 0;
        let mut datatype = DataType::DataTypeNull;
        let e = unsafe {
            cgns_sys::cg_array_info(
                array_id,
                raw_name.as_mut_ptr().cast(),
                &raw mut datatype,
                &raw mut rank,
                dimensions.as_mut_ptr(),
            )
        };
        if e == 0 {
            let rank = rank.try_into().unwrap();
            Ok((
                raw_to_string(&raw_name),
                datatype,
                dimensions[0..rank]
                    .iter()
                    .map(|&x| usize::try_from(x).unwrap())
                    .collect(),
            ))
        } else {
            Err(e.into())
        }
    }

    pub fn array_read<T: CgnsDataType>(&self, array_id: i32, data: &mut [T]) -> Result<()> {
        let e = unsafe { cgns_sys::cg_array_read(array_id, data.as_mut_ptr().cast()) };
        if e == 0 { Ok(()) } else { Err(e.into()) }
    }

    pub fn gorel(&self, query: &[GotoQueryItem]) -> Result<()> {
        // varargs is not yet available in stable Rust so we recurse
        if let Some((head, queue)) = query.split_first() {
            let end = c"end".as_ptr();
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
        if e == 0 { Ok(()) } else { Err(e.into()) }
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
                &raw mut file,
                &raw mut base,
                &raw mut depth,
                labelp.as_mut_ptr().cast(),
                index.as_mut_ptr(),
            )
        };
        if e == 0 {
            Ok((
                base.try_into().unwrap(),
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
    const SYS: DataType = DataType::Integer;
}

impl CgnsDataType for u8 {
    const SYS: DataType = DataType::Character;
}

impl CgnsDataType for f64 {
    const SYS: DataType = RealDouble;
}

impl From<Mode> for i32 {
    fn from(m: Mode) -> Self {
        match m {
            Mode::Read => CG_MODE_READ as Self,
            Mode::Write => CG_MODE_WRITE as Self,
            Mode::Modify => CG_MODE_MODIFY as Self,
        }
    }
}

impl From<FileType> for i32 {
    fn from(t: FileType) -> Self {
        match t {
            FileType::NONE => CG_FILE_NONE as Self,
            FileType::ADF => CG_FILE_ADF as Self,
            FileType::ADF2 => CG_FILE_ADF2 as Self,
            FileType::HDF5 => CG_FILE_HDF5 as Self,
        }
    }
}

impl From<i32> for Error {
    fn from(code: i32) -> Self {
        Self(code)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = unsafe { CStr::from_ptr(cg_get_error()) };
        write!(f, "{} (error {})", msg.to_str().unwrap(), self.0)
    }
}
impl std::error::Error for Error {}

static CGNS_MUTEX: Mutex<()> = Mutex::new(());

pub fn open(path: &str, mode: Mode) -> Result<File> {
    let _l = CGNS_MUTEX.lock().unwrap();
    let mut fd: i32 = 0;
    let path = CString::new(path).unwrap();
    let f = unsafe { cg_open(path.as_ptr(), mode.into(), &raw mut fd) };
    if f == 0 { Ok(File(fd)) } else { Err(f.into()) }
}

#[derive(Debug, PartialEq, Eq)]
pub enum IndexError {
    TypeConversion(std::num::TryFromIntError),
    NotPositive,
}

#[derive(Debug)]
pub struct File(c_int);
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Base(NonZero<c_int>);

impl Default for Base {
    fn default() -> Self {
        Self(NonZero::new(1).unwrap())
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Zone(NonZero<c_int>);

impl Default for Zone {
    fn default() -> Self {
        Self(NonZero::new(1).unwrap())
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Section(NonZero<c_int>);

impl Default for Section {
    fn default() -> Self {
        Self(NonZero::new(1).unwrap())
    }
}

impl From<Base> for c_int {
    fn from(value: Base) -> Self {
        value.0.get()
    }
}

impl From<Zone> for c_int {
    fn from(value: Zone) -> Self {
        value.0.get()
    }
}

impl From<Section> for c_int {
    fn from(value: Section) -> Self {
        value.0.get()
    }
}

impl TryFrom<c_int> for Base {
    type Error = <NonZero<c_int> as TryFrom<c_int>>::Error;

    fn try_from(value: c_int) -> std::result::Result<Self, Self::Error> {
        let value = value.try_into()?;
        Ok(Self(value))
    }
}

impl TryFrom<c_int> for Zone {
    type Error = <NonZero<c_int> as TryFrom<c_int>>::Error;

    fn try_from(value: c_int) -> std::result::Result<Self, Self::Error> {
        let value = value.try_into()?;
        Ok(Self(value))
    }
}

impl TryFrom<c_int> for Section {
    type Error = <NonZero<c_int> as TryFrom<c_int>>::Error;

    fn try_from(value: c_int) -> std::result::Result<Self, Self::Error> {
        let value = value.try_into()?;
        Ok(Self(value))
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct FlowSolution(c_int);
impl From<i32> for FlowSolution {
    fn from(value: i32) -> Self {
        assert!(value >= 1);
        Self(value)
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

impl Default for ElementType {
    fn default() -> Self {
        Self::try_from_primitive(0).unwrap()
    }
}

impl Default for BCType {
    fn default() -> Self {
        Self::try_from_primitive(0).unwrap()
    }
}

impl Default for PointSetType {
    fn default() -> Self {
        Self::try_from_primitive(0).unwrap()
    }
}

impl Default for DataType {
    fn default() -> Self {
        Self::try_from_primitive(0).unwrap()
    }
}

#[derive(Default, Debug)]
pub struct SectionInfo {
    pub section_name: String,
    pub typ: ElementType,
    pub start: usize,
    pub end: usize,
    pub nbndry: i32,
}

impl SectionInfo {
    #[must_use]
    pub fn new(typ: ElementType, end: usize) -> Self {
        Self {
            section_name: "Elem".to_owned(),
            typ,
            start: 1,
            end,
            nbndry: 0,
        }
    }
}

#[derive(Default, Debug)]
pub struct BoCoInfo {
    pub name: String,
    pub r#type: BCType,
    pub ptset_type: PointSetType,
    pub npnts: usize,
    pub normal_index: usize,
    pub normal_list_size: usize,
    pub normal_data_type: DataType,
    pub ndataset: usize,
}

impl File {
    pub fn close(&mut self) -> Result<()> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let f = unsafe { cg_close(self.0) };
        if f == 0 { Ok(()) } else { Err(f.into()) }
    }

    pub fn save_as(&self, filename: &str, file_type: FileType, follow_links: bool) -> Result<()> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let filename = CString::new(filename).unwrap();
        let follow_links = i32::from(follow_links);
        let file_type = file_type.into();
        let e = unsafe { cg_save_as(self.0, filename.as_ptr(), file_type, follow_links) };
        if e == 0 { Ok(()) } else { Err(e.into()) }
    }

    pub fn biter_write(&mut self, base: Base, base_iter_name: &str, n_steps: i32) -> Result<()> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let base_iter_name = CString::new(base_iter_name).unwrap();
        let e = unsafe { cg_biter_write(self.0, base.into(), base_iter_name.as_ptr(), n_steps) };
        if e == 0 { Ok(()) } else { Err(e.into()) }
    }

    pub fn biter_read(&mut self, base: Base) -> Result<(String, i32)> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let mut n_steps = 0;
        let mut name = [0_u8; 33];
        let e =
            unsafe { cg_biter_read(self.0, base.into(), name.as_mut_ptr().cast(), &raw mut n_steps) };
        if e == 0 {
            Ok((raw_to_string(&name), n_steps))
        } else {
            Err(e.into())
        }
    }

    pub fn goto(&self, base: Base, query: &[GotoQueryItem]) -> Result<GotoContext<'_>> {
        let mutex = CGNS_MUTEX.lock().unwrap();
        let end = c"end".as_ptr();
        let e = unsafe { cgns_sys::cg_goto(self.0, base.into(), end) };
        if e == 0 {
            // varargs is not yet available in stable Rust so we rely on cg_gorel
            let l = GotoContext { mutex, file: self };
            l.gorel(query)?;
            Ok(l)
        } else {
            Err(e.into())
        }
    }

    pub fn golist(&self, base: Base, labels: &[&str], index: &[i32]) -> Result<GotoContext<'_>> {
        let mutex = CGNS_MUTEX.lock().unwrap();
        let labels: Vec<_> = labels.iter().map(|&s| CString::new(s).unwrap()).collect();
        let mut labels_ptr: Vec<_> = labels.iter().map(|s| s.as_ptr().cast_mut()).collect();
        let e = unsafe {
            cg_golist(
                self.0,
                base.into(),
                labels.len() as i32,
                labels_ptr.as_mut_ptr(),
                index.as_ptr().cast_mut(),
            )
        };
        if e == 0 {
            Ok(GotoContext { mutex, file: self })
        } else {
            Err(e.into())
        }
    }

    // https://cgns.github.io/CGNS_docs_current/sids/timedep.html
    pub fn ziter_write(&mut self, base: Base, zone: Zone, zone_iter_name: &str) -> Result<()> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let zone_iter_name = CString::new(zone_iter_name).unwrap();
        let e = unsafe { cg_ziter_write(self.0, base.into(), zone.into(), zone_iter_name.as_ptr()) };
        if e == 0 { Ok(()) } else { Err(e.into()) }
    }

    // https://cgns.github.io/CGNS_docs_current/midlevel/structural.html
    pub fn base_write(&mut self, basename: &str, cell_dim: i32, phys_dim: i32) -> Result<Base> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let basename = CString::new(basename).unwrap();
        let mut b: i32 = 0;
        let e = unsafe { cg_base_write(self.0, basename.as_ptr(), cell_dim, phys_dim, &raw mut b) };
        if e == 0 { Ok(b.try_into().unwrap()) } else { Err(e.into()) }
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
            vertex_size as cgsize,
            cell_size as cgsize,
            boundary_size as cgsize,
        ];
        let e = unsafe {
            cg_zone_write(
                self.0,
                base.into(),
                zonename.as_ptr(),
                size.as_ptr(),
                Unstructured,
                &raw mut z,
            )
        };
        if e == 0 { Ok(z.try_into().unwrap()) } else { Err(e.into()) }
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
                base.into(),
                zone.into(),
                RealDouble,
                coordname.as_ptr(),
                coord.as_ptr().cast::<c_void>(),
                &raw mut c,
            )
        };
        if e == 0 { Ok(()) } else { Err(e.into()) }
    }

    pub fn zone_read(&self, base: Base, zone: Zone) -> Result<(String, [usize; 9])> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let mut r = [0 as cgsize; 9];
        let mut buf = [0_u8; 64];
        let err = unsafe {
            cg_zone_read(
                self.0,
                base.into(),
                zone.into(),
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

    pub fn coord_info(&self, base: Base, zone: Zone, c: i32) -> Result<(DataType, String)> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let mut datatype = DataType::Integer;
        let mut raw_name = [0_u8; 64];
        let err = unsafe {
            cg_coord_info(
                self.0,
                base.into(),
                zone.into(),
                c,
                &raw mut datatype,
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
        let range_min = range_min as cgsize;
        let range_max = range_max as cgsize;
        let p = CString::new(coordname).unwrap();
        let err = unsafe {
            cg_coord_read(
                self.0,
                base.into(),
                zone.into(),
                p.as_ptr(),
                RealDouble,
                &raw const range_min,
                &raw const range_max,
                coord_array.as_mut_ptr().cast(),
            )
        };
        if err == 0 { Ok(()) } else { Err(err.into()) }
    }

    pub fn section_write(
        &mut self,
        base: Base,
        zone: Zone,
        args: &SectionInfo,
        elements: &[cgsize],
    ) -> Result<()> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let section_name = CString::new(args.section_name.clone()).unwrap();
        let mut c = 0;
        let e = unsafe {
            cg_section_write(
                self.0,
                base.into(),
                zone.into(),
                section_name.as_ptr(),
                args.typ,
                args.start as cgsize,
                args.end as cgsize,
                args.nbndry,
                elements.as_ptr(),
                &raw mut c,
            )
        };
        if e == 0 { Ok(()) } else { Err(e.into()) }
    }

    pub fn poly_section_write(
        &mut self,
        base: Base,
        zone: Zone,
        args: &SectionInfo,
        elements: &[cgsize],
        offsets: &[cgsize],
    ) -> Result<()> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let section_name = CString::new(args.section_name.clone()).unwrap();
        let mut c = 0;
        let e = unsafe {
            cgns_sys::cg_poly_section_write(
                self.0,
                base.into(),
                zone.into(),
                section_name.as_ptr(),
                args.typ,
                args.start as cgsize,
                args.end as cgsize,
                args.nbndry,
                elements.as_ptr(),
                offsets.as_ptr(),
                &raw mut c,
            )
        };
        if e == 0 { Ok(()) } else { Err(e.into()) }
    }

    pub fn elements_read(
        &self,
        base: Base,
        zone: Zone,
        section: Section,
        elements: &mut [cgsize],
        parent_data: &mut [cgsize],
    ) -> Result<()> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let ptr = if parent_data.is_empty() {
            null_mut()
        } else {
            parent_data.as_mut_ptr()
        };
        let e = unsafe {
            cg_elements_read(
                self.0,
                base.into(),
                zone.into(),
                section.into(),
                elements.as_mut_ptr(),
                ptr,
            )
        };
        if e == 0 { Ok(()) } else { Err(e.into()) }
    }

    #[allow(clippy::too_many_arguments)] // imposed by CGNS API
    pub fn elements_partial_read(
        &self,
        base: Base,
        zone: Zone,
        section: Section,
        start: usize,
        end: usize,
        elements: &mut [cgsize],
        parent_data: &mut [cgsize],
    ) -> Result<()> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let ptr = if parent_data.is_empty() {
            null_mut()
        } else {
            parent_data.as_mut_ptr()
        };
        let e = unsafe {
            cgns_sys::cg_elements_partial_read(
                self.0,
                base.into(),
                zone.into(),
                section.into(),
                start.try_into().unwrap(),
                end.try_into().unwrap(),
                elements.as_mut_ptr(),
                ptr,
            )
        };
        if e == 0 { Ok(()) } else { Err(e.into()) }
    }

    pub fn element_data_size(&self, base: Base, zone: Zone, section: Section) -> Result<usize> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let mut r = 0;
        let e =
            unsafe { cgns_sys::cg_ElementDataSize(self.0, base.into(), zone.into(), section.into(), &raw mut r) };
        if e == 0 {
            Ok(r as usize)
        } else {
            Err(e.into())
        }
    }

    pub fn poly_elements_read(
        &self,
        base: Base,
        zone: Zone,
        section: Section,
        elements: &mut [cgsize],
        connect_offset: &mut [cgsize],
        parent_data: &mut [cgsize],
    ) -> Result<()> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let ptr = if parent_data.is_empty() {
            null_mut()
        } else {
            parent_data.as_mut_ptr()
        };
        let e = unsafe {
            cgns_sys::cg_poly_elements_read(
                self.0,
                base.into(),
                zone.into(),
                section.into(),
                elements.as_mut_ptr(),
                connect_offset.as_mut_ptr(),
                ptr,
            )
        };
        if e == 0 { Ok(()) } else { Err(e.into()) }
    }

    #[allow(clippy::too_many_arguments)] // imposed by CGNS API
    pub fn poly_elements_partial_read(
        &self,
        base: Base,
        zone: Zone,
        section: Section,
        start: usize,
        end: usize,
        elements: &mut [cgsize],
        connect_offset: &mut [cgsize],
        parent_data: &mut [cgsize],
    ) -> Result<()> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let ptr = if parent_data.is_empty() {
            null_mut()
        } else {
            parent_data.as_mut_ptr()
        };
        let e = unsafe {
            cgns_sys::cg_poly_elements_partial_read(
                self.0,
                base.into(),
                zone.into(),
                section.into(),
                start.try_into().unwrap(),
                end.try_into().unwrap(),
                elements.as_mut_ptr(),
                connect_offset.as_mut_ptr(),
                ptr,
            )
        };
        if e == 0 { Ok(()) } else { Err(e.into()) }
    }

    /// Get size of element connectivity data array for partial read
    ///
    /// See `cg_ElementPartialSize` in CGNS documentation
    pub fn element_partial_size(
        &self,
        base: Base,
        zone: Zone,
        section: Section,
        start: usize,
        end: usize,
    ) -> Result<usize> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let mut result = 0;
        let e = unsafe {
            cgns_sys::cg_ElementPartialSize(
                self.0,
                base.into(),
                zone.into(),
                section.into(),
                start.try_into().unwrap(),
                end.try_into().unwrap(),
                &raw mut result,
            )
        };
        if e == 0 {
            Ok(result.try_into().unwrap())
        } else {
            Err(e.into())
        }
    }

    pub fn nzones(&self, base: Base) -> Result<i32> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let mut r = 0;
        let e = unsafe { cgns_sys::cg_nzones(self.0, base.into(), &raw mut r) };
        if e == 0 { Ok(r) } else { Err(e.into()) }
    }

    pub fn nsections(&self, base: Base, zone: Zone) -> Result<i32> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let mut r = 0;
        let e = unsafe { cg_nsections(self.0, base.into(), zone.into(), &raw mut r) };
        if e == 0 { Ok(r) } else { Err(e.into()) }
    }

    pub fn section_read(
        &self,
        base: Base,
        zone: Zone,
        section: Section,
    ) -> Result<(SectionInfo, bool)> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let mut info = SectionInfo::default();
        let mut parent_flag = 0_i32;
        let mut raw_name = [0_u8; 64];
        let mut start: cgsize = 0;
        let mut end: cgsize = 0;
        let e = unsafe {
            cg_section_read(
                self.0,
                base.into(),
                zone.into(),
                section.into(),
                raw_name.as_mut_ptr().cast(),
                &raw mut info.typ,
                &raw mut start,
                &raw mut end,
                &raw mut info.nbndry,
                &raw mut parent_flag,
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

    pub fn sol_write(
        &mut self,
        base: Base,
        zone: Zone,
        sol_name: &str,
        grid_location: GridLocation,
    ) -> Result<FlowSolution> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let sol_name = CString::new(sol_name).unwrap();
        let mut sol = 0;
        let e = unsafe {
            cgns_sys::cg_sol_write(
                self.0,
                base.into(),
                zone.into(),
                sol_name.as_ptr(),
                grid_location,
                &raw mut sol,
            )
        };
        if e == 0 {
            Ok(FlowSolution(sol))
        } else {
            Err(e.into())
        }
    }

    pub fn field_write<T: CgnsDataType>(
        &self,
        base: Base,
        zone: Zone,
        sol: FlowSolution,
        field_name: &str,
        data: &[T],
    ) -> Result<i32> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let field_name = CString::new(field_name).unwrap();
        let mut r = 0;
        let e = unsafe {
            cgns_sys::cg_field_write(
                self.0,
                base.into(),
                zone.into(),
                sol.0,
                T::SYS,
                field_name.as_ptr(),
                data.as_ptr().cast(),
                &raw mut r,
            )
        };
        if e == 0 { Ok(r) } else { Err(e.into()) }
    }

    pub fn nbocos(&self, base: Base, zone: Zone) -> Result<u32> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let mut r = 0;
        let e = unsafe { cgns_sys::cg_nbocos(self.0, base.into(), zone.into(), &raw mut r) };
        if e == 0 { Ok(r as u32) } else { Err(e.into()) }
    }

    pub fn boco_info(&self, base: Base, zone: Zone, bc: u32) -> Result<BoCoInfo> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let mut raw_name = [0_u8; 256];
        let mut r = BoCoInfo::default();
        let mut npnts = 0;
        let mut normal_index = 0;
        let mut normal_list_size = 0;
        let mut ndataset = 0;
        let err = unsafe {
            cgns_sys::cg_boco_info(
                self.0,
                base.into(),
                zone.into(),
                bc as c_int,
                raw_name.as_mut_ptr().cast(),
                &raw mut r.r#type,
                &raw mut r.ptset_type,
                &raw mut npnts,
                &raw mut normal_index,
                &raw mut normal_list_size,
                &raw mut r.normal_data_type,
                &raw mut ndataset,
            )
        };
        r.npnts = usize::try_from(npnts).unwrap();
        r.normal_index = usize::try_from(normal_index).unwrap();
        r.normal_list_size = usize::try_from(normal_list_size).unwrap();
        r.ndataset = usize::try_from(ndataset).unwrap();
        r.name = raw_to_string(&raw_name);
        if err == 0 { Ok(r) } else { Err(err.into()) }
    }

    /// Wrap `cg_boco_read`
    ///
    /// See <https://cgns.github.io/standard/MLL/api/c_api.html#_CPPv412cg_boco_readiiiiP8cgsize_tPv>
    pub fn boco_read(&self, base: Base, zone: Zone, bc: u32, pnts: &mut [cgsize]) -> Result<()> {
        let _l = CGNS_MUTEX.lock().unwrap();
        let err = unsafe {
            cgns_sys::cg_boco_read(
                self.0,
                base.into(),
                zone.into(),
                bc as c_int,
                pnts.as_mut_ptr(),
                null_mut(),
            )
        };
        if err == 0 { Ok(()) } else { Err(err.into()) }
    }

    /// Wrap `cg_boco_write`
    ///
    /// See <https://cgns.github.io/standard/MLL/api/c_api.html#_CPPv413cg_boco_writeiiiPKc8BCType_t14PointSetType_t8cgsize_tPK8cgsize_tPi>
    pub fn boco_write(
        &self,
        base: Base,
        zone: Zone,
        boconame: &str,
        bocotype: BCType,
        ptset_type: PointSetType,
        pnts: &[cgsize],
    ) -> Result<u32> {
        let boconame = CString::new(boconame).unwrap();
        let mut r = 0;
        let _l = CGNS_MUTEX.lock().unwrap();
        let err = unsafe {
            cgns_sys::cg_boco_write(
                self.0,
                base.into(),
                zone.into(),
                boconame.as_ptr(),
                bocotype,
                ptset_type,
                pnts.len().try_into().unwrap(),
                pnts.as_ptr(),
                &raw mut r,
            )
        };
        if err == 0 {
            Ok(r as u32)
        } else {
            Err(err.into())
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
