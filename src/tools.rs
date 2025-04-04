use crate::{Base, DataType, File, GotoContext, PointSetType, Result, Zone, cgns_sys, cgsize};
use std::{ffi::CStr, iter};

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
                    .chain(iter::repeat(&null_str).take(num_zone_by_iter - iter.len()))
                    .flat_map(|x| {
                        let mut v = x.as_bytes().to_vec();
                        v.extend(iter::repeat(0).take(32 - x.len()));
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
