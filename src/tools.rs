use crate::{
    Base, DataType, File, GotoContext, PointSetType, Result, Zone, cgns_sys, cgsize,
    tools::buffered_iterator::{BufferedIterator, ChunkSource},
};
use std::{ffi::CStr, iter};

mod buffered_iterator {
    use std::cmp::min;

    /// A trait representing a source that can load data in chunks/blocks.
    pub trait ChunkSource {
        type Item;
        type Error;

        /// Load a chunk of data starting at `start` (0-based absolute index) with length `len`.
        fn load_chunk(&mut self, start: usize, len: usize) -> Result<(), Self::Error>;

        /// Retrieve an item from the loaded buffer at `offset`.
        /// `offset` is relative to the start of the current buffer (0 to len-1).
        fn get_from_buffer(&self, offset: usize) -> Self::Item;

        // Number of items in the current chunk
        fn chunk_size(&self) -> usize;

        // Set chunck size to zero
        fn clear_chunk(&mut self);
    }

    pub struct BufferedIterator<S: ChunkSource> {
        source: S,
        total_size: usize,
        current_idx: usize,
        buf_idx: usize, // Index within current chunk
    }

    impl<S: ChunkSource> BufferedIterator<S> {
        pub const fn new(source: S, total_size: usize) -> Self {
            Self {
                source,
                total_size,
                current_idx: 0,
                buf_idx: 0,
            }
        }

        fn refill(&mut self) -> Result<(), S::Error> {
            if self.current_idx >= self.total_size {
                return Ok(());
            }

            let remaining = self.total_size - self.current_idx;
            let read_count = min(remaining, 2048); // Fixed buffer size of 2048

            // Delegate the actual reading to the source
            self.source.load_chunk(self.current_idx, read_count)?;
            self.buf_idx = 0;
            Ok(())
        }
    }

    impl<S: ChunkSource> Iterator for BufferedIterator<S> {
        type Item = Result<S::Item, S::Error>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.current_idx >= self.total_size {
                return None;
            }

            // Check if we need to refill
            if self.buf_idx >= self.source.chunk_size() {
                if let Err(e) = self.refill() {
                    // Consume the index so we don't loop forever on error
                    self.current_idx += 1;
                    return Some(Err(e));
                }
                if self.source.chunk_size() == 0 {
                    return None;
                }
            }

            // Get value from source (cheap retrieval from memory)
            let item = self.source.get_from_buffer(self.buf_idx);
            self.buf_idx += 1;
            self.current_idx += 1;
            Some(Ok(item))
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            let rem = self.total_size - self.current_idx;
            (rem, Some(rem))
        }

        fn nth(&mut self, n: usize) -> Option<Self::Item> {
            if n == 0 {
                return self.next();
            }

            let new_idx = self.current_idx + n;

            // If jumping out of bounds
            if new_idx >= self.total_size {
                self.current_idx = self.total_size;
                self.source.clear_chunk();
                return None;
            }

            // If jumping WITHIN the current buffer
            let remaining_in_buf = self.source.chunk_size().saturating_sub(self.buf_idx);
            if n < remaining_in_buf {
                self.buf_idx += n;
                self.current_idx = new_idx;
            } else {
                // Jump OUTSIDE buffer: Update index, invalidate buffer
                self.current_idx = new_idx;
                self.source.clear_chunk();
                self.buf_idx = 0;
            }

            self.next()
        }
    }

    impl<S: ChunkSource> ExactSizeIterator for BufferedIterator<S> {}
}

struct CoordSource<'a> {
    file: &'a File,
    base: Base,
    zone: Zone,
    buffer: [Vec<f64>; 3],
}

impl ChunkSource for CoordSource<'_> {
    type Item = [f64; 3];
    type Error = crate::Error;
    fn load_chunk(&mut self, start: usize, len: usize) -> Result<()> {
        let range_min = start + 1;
        let range_max = start + len;
        for buf in &mut self.buffer {
            buf.resize(len, 0.0);
        }
        for (i, label) in ["CoordinateX", "CoordinateY", "CoordinateZ"]
            .iter()
            .enumerate()
        {
            self.file.coord_read(
                self.base,
                self.zone,
                label,
                range_min,
                range_max,
                &mut self.buffer[i][..len],
            )?;
        }
        Ok(())
    }

    fn get_from_buffer(&self, offset: usize) -> Self::Item {
        std::array::from_fn(|i| self.buffer[i][offset])
    }

    fn chunk_size(&self) -> usize {
        self.buffer[0].len()
    }

    fn clear_chunk(&mut self) {
        for buf in &mut self.buffer {
            buf.clear();
        }
    }
}

struct PolyElementsSource<'a, IM> {
    file: &'a File,
    base: Base,
    zone: Zone,
    section: crate::Section,
    elem_buffer: Vec<cgsize>,
    offset_buffer: Vec<cgsize>,
    elem_builder: IM,
}

impl<IM, IT> ChunkSource for PolyElementsSource<'_, IM>
where
    IM: Fn(&[cgsize]) -> IT,
{
    type Item = IT;
    type Error = crate::Error;

    fn load_chunk(&mut self, start: usize, len: usize) -> Result<()> {
        let range_min = start + 1;
        let range_max = start + len;
        let conn_size = self.file.element_partial_size(
            self.base,
            self.zone,
            self.section,
            range_min,
            range_max,
        )?;
        self.elem_buffer.resize(conn_size, 0);
        self.offset_buffer.resize(len + 1, 0);
        self.file.poly_elements_partial_read(
            self.base,
            self.zone,
            self.section,
            range_min,
            range_max,
            &mut self.elem_buffer,
            &mut self.offset_buffer,
            &mut [],
        )?;
        Ok(())
    }

    fn get_from_buffer(&self, offset: usize) -> Self::Item {
        let start = self.offset_buffer[offset] as usize;
        let end = self.offset_buffer[offset + 1] as usize;
        (self.elem_builder)(&self.elem_buffer[start..end])
    }

    fn chunk_size(&self) -> usize {
        self.offset_buffer.len() - 1
    }

    fn clear_chunk(&mut self) {
        self.offset_buffer.clear();
        self.elem_buffer.clear();
    }
}

impl GotoContext<'_> {
    pub fn array_info_from_name(&self, name: &str) -> Result<Option<(i32, DataType, Vec<usize>)>> {
        let na = self.narrays()?;
        for i in 1..=na {
            let (aname, typ, dims) = self.array_info(i)?;
            if aname == name {
                return Ok(Some((i, typ, dims)));
            }
        }
        Ok(None)
    }

    /// High level alternative to `array_read`
    pub fn read_char_array(&self, name: &str) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
        let r = if let Some((id, typ, dims)) = self.array_info_from_name(name)? {
            assert_eq!(typ, DataType::Character);
            let flatsize: usize = dims.iter().product();
            let mut raw_data = vec![0_u8; flatsize];
            self.array_read(id, &mut raw_data)?;
            Some((raw_data, dims))
        } else {
            None
        };
        Ok(r)
    }

    /// High level alternative to `array_read`
    pub fn read_string_array(&self, name: &str) -> Result<Option<(Vec<String>, Vec<usize>)>> {
        let Some((raw_data, dims)) = self.read_char_array(name)? else {
            return Ok(None);
        };
        let Some((string_size, asize)) = dims.split_first() else {
            return Ok(None);
        };
        let r = raw_data
            .chunks(*string_size)
            .map(|raw_string| {
                CStr::from_bytes_until_nul(raw_string)
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_string()
            })
            .collect();
        Ok(Some((r, asize.to_vec())))
    }

    pub fn write_string_array<S, I>(&self, name: S, values: I) -> Result<()>
    where
        S: AsRef<str>,
        I: ExactSizeIterator<Item = Option<S>>,
    {
        let nv = values.len();
        let mut buff: Vec<u8> = Vec::with_capacity(32 * nv);
        for v in values {
            let s = v.as_ref().map_or("Null", |v| v.as_ref());
            let n = s.len();
            assert!(n <= 32);
            buff.extend_from_slice(s.as_bytes());
            buff.extend((0..(32 - n)).map(|_| 0_u8));
        }
        self.array_write(name.as_ref(), &[32, nv], &buff)
    }
}

impl File {
    /// Returns an iterator over the coordinates of the given zone
    ///
    /// This is a high level wrapper for `cg_coord_read`.
    pub fn coord_iter(
        &self,
        base: Base,
        zone: Zone,
    ) -> Result<impl ExactSizeIterator<Item = Result<[f64; 3]>> + '_> {
        let size = self.zone_read(base, zone)?;
        let num_vertices = size.1[0];
        Ok(BufferedIterator::new(
            CoordSource {
                file: self,
                base,
                zone,
                buffer: [Vec::new(), Vec::new(), Vec::new()],
            },
            num_vertices,
        ))
    }

    /// Returns a buffered iterator over the elements of a zone section.
    ///
    /// This reads data in chunks to save memory, rather than loading all elements at once.
    /// This is a high level wrapper of `cg_poly_elements_partial_read`.
    ///
    /// # Arguments
    /// * `elem_builder`: A closure that converts the raw node indices slice (`&[cgsize]`)
    ///   into your desired type `IT`.
    pub fn poly_elements_iter<'a, IT>(
        &'a self,
        base: Base,
        zone: Zone,
        section: crate::Section,
        elem_builder: impl Fn(&[cgsize]) -> IT + 'a,
    ) -> Result<impl ExactSizeIterator<Item = Result<IT>> + 'a> {
        let (section_info, _) = self.section_read(base, zone, section)?;
        Ok(BufferedIterator::new(
            PolyElementsSource {
                file: self,
                base,
                zone,
                section,
                elem_buffer: Vec::new(),
                offset_buffer: vec![0],
                elem_builder,
            },
            section_info.end - section_info.start,
        ))
    }

    /// Returns an iterator over the sections of a zone.
    ///
    /// This is a high level wrapper for `cg_nsections`.
    pub fn section_iter(
        &self,
        base: Base,
        zone: Zone,
    ) -> Result<impl ExactSizeIterator<Item = crate::Section>> {
        let nsec = self.nsections(base, zone)?;
        Ok((0..nsec).map(|i| (i + 1).try_into().unwrap()))
    }

    pub fn zone_pointers_read(&self, base: Base) -> Result<Vec<Vec<String>>> {
        let gc = match self.goto(base, &[("BaseIterativeData_t", 1).into()]) {
            Ok(x) => x,
            Err(crate::Error(x)) if x == cgns_sys::CG_NODE_NOT_FOUND as i32 => {
                return Ok(Vec::new());
            }
            Err(x) => return Err(x),
        };
        let Some((mut zones, dims)) = gc.read_string_array("ZonePointers")? else {
            return Ok(Vec::new());
        };
        drop(gc);
        let r = zones
            .chunks_exact_mut(dims[0])
            .map(|it| {
                it.iter_mut()
                    .filter_map(|x| {
                        if x.is_empty() || x == "Null" {
                            None
                        } else {
                            Some(std::mem::take(x))
                        }
                    })
                    .collect()
            })
            .collect();
        Ok(r)
    }

    pub fn zone_pointers_write(&mut self, base: Base, data: &[Vec<String>]) -> Result<()> {
        let nz: Vec<_> = data
            .iter()
            .map(|x| i32::try_from(x.len()).unwrap())
            .collect();
        let num_iter = data.len();
        let num_zone_by_iter = data.iter().map(Vec::len).max().unwrap();
        let null_str = "Null".to_string();
        let raw_data: Vec<_> = data
            .iter()
            .flat_map(|iter| -> Vec<_> {
                iter.iter()
                    .chain(iter::repeat_n(&null_str, num_zone_by_iter - iter.len()))
                    .flat_map(|x| {
                        let mut v = x.as_bytes().to_vec();
                        v.extend(iter::repeat_n(0, 32 - x.len()));
                        v
                    })
                    .collect()
            })
            .collect();
        let gc = self.goto(base, &[("BaseIterativeData_t", 1).into()])?;
        gc.array_write("NumberOfZones", &[num_iter], &nz)?;
        gc.array_write("ZonePointers", &[32, num_zone_by_iter, num_iter], &raw_data)
    }

    /// Read a boundary condition as a list
    pub fn boco_read_as_vec(&self, base: Base, zone: Zone, bc: u32) -> Result<Vec<cgsize>> {
        let info = self.boco_info(base, zone, bc)?;
        match info.ptset_type {
            PointSetType::ElementRange | PointSetType::PointRange => {
                let mut range = [0; 2];
                self.boco_read(base, zone, bc, &mut range)?;
                let n = range[1] - range[0] + 1;
                Ok((0..n).map(|i| range[0] - 1 + i).collect())
            }
            PointSetType::ElementList | PointSetType::PointList => {
                let mut pnts = vec![0; info.npnts];
                self.boco_read(base, zone, bc, &mut pnts)?;
                Ok(pnts)
            }
            x => unimplemented!("Boundary conditions of type {x:?} are not supported"),
        }
    }
}
