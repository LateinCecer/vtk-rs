use std::error::Error;
use std::fmt::{Display, Formatter};
use std::io::Write;
use nalgebra::{Const, Dyn, Matrix, SVector, VecStorage};
use crate::legacy::LegacyWriter;
use crate::mesh::UnstructuredMesh;

/// The VTK format describes the format of the VTK format. Currently, there are three major formats
/// implemented for Paraview: Legacy, XML and HDF. Since Legacy is the simplest to implement, this
/// will be the only format supported for now, however, other formats may be implemented in the
/// future.
#[derive(Copy, Clone, Debug)]
pub enum VTKFormat {
    /// legacy file format
    Legacy,
    /// xml file format
    Xml,
    /// hdf file format
    Hdf,
}

impl Display for VTKFormat {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Legacy => write!(f, "Legacy VTK file format"),
            Self::Xml => write!(f, "XML VTK file format"),
            Self::Hdf => write!(f, "HDF VTK file format"),
        }
    }
}

impl VTKFormat {
    /// Creates a writer for the file format
    pub fn make_writer<W: Write>(&self, write: W, options: VTKOptions) -> impl VTKWriter {
        match self {
            Self::Legacy => LegacyWriter::new(write, options),
            _ => panic!("")
        }
    }
}


/// VTK data format. The data format controls in with format raw data is written to the VTK format.
/// Available formats for now are ASCII and Binary, however, this may change in the future as more
/// formats are implemented.
///
/// **NOTE**: Not all VTK file formats support all datasets. The default data format is ASCII and
/// should work on all file formats.
#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub enum VTKDataFormat {
    Ascii,
    Binary,
}

impl Default for VTKDataFormat {
    fn default() -> Self {
        VTKDataFormat::Ascii
    }
}

impl VTKKeyword for VTKDataFormat {
    fn vtk(&self) -> &'static str {
        match self {
            Self::Ascii => "ASCII",
            Self::Binary => "BINARY",
        }
    }
}

#[derive(Default)]
pub struct VTKOptions {
    pub data_format: VTKDataFormat
}

pub enum MeshData<T, const DIM: usize> {
    StructuredGrid,
    UnstructuredGrid,
    UnstructuredPolygon(UnstructuredMesh<T, DIM>),
}

pub struct Data<T, const N: usize> {
    data: Matrix<T, Const<N>, Dyn, VecStorage<T, Const<N>, Dyn>>
}

pub trait VTKWriter {
    type Error: Error;


    fn write_header(&mut self, header: &impl Display) -> Result<(), Self::Error>;
    fn write_geometry_f32(&mut self, msh: MeshData<f32, 3>) -> Result<(), Self::Error>;
    fn write_data<T, const N: usize>(&mut self, msh: Data<T, N>) -> Result<(), Self::Error>;
}


/// The `VTKKeyword` trait can be implemented for any datatype that can be referred to as a VTK
/// keyword.
pub trait VTKKeyword {
    fn vtk(&self) -> &'static str;
}

/// A VTK write component is a component of a VTK format that is writable.
pub trait VTKWriteComp<Writer: VTKWriter> {
    fn write(&self, writer: &mut Writer) -> Result<(), Writer::Error>;
}
