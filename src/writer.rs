use crate::data::{FieldData, VectorField};
use crate::legacy::LegacyWriter;
use crate::mesh::UnstructuredMesh;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::io::Write;

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
    pub fn make_writer<W: Write>(&self, write: W, options: VTKOptions) -> impl VTKGeneralWriter {
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

pub trait VTKWriter {
    type Error: Error;

    /// Writes header data with the specified data.
    fn write_header(&mut self, header: &impl Display) -> Result<(), Self::Error>;
}

pub trait VTKGeometryWriter<M>: VTKWriter {
    /// Writes the specified geometry data to `self`
    fn write_geometry(&mut self, msh: M) -> Result<(), Self::Error>;
}

pub trait VTKDataWriter<D>: VTKWriter {
    /// Writes the data contained within `data` to `self`. Since different kinds of data may
    /// consist of differing data-structures with different representations in the different VTK
    /// file formats, a writer function has to be implemented specifically for each datatype-writer
    /// pair by implementing this trait.
    fn write(&mut self, data: D) -> Result<(), Self::Error>;
}

/// A general writer combines the `VTKWriter` trait, which is used to implement header and geometry
/// data, with the `VTKDataWriter` trait for common data types.
pub trait VTKGeneralWriter: VTKWriter
    + VTKGeometryWriter<MeshData<f32, 3>>
    + VTKGeometryWriter<MeshData<f64, 3>>
    + VTKDataWriter<FieldData>
    + VTKDataWriter<VectorField<f32>>
    + VTKDataWriter<VectorField<f64>> {}


/// The `VTKKeyword` trait can be implemented for any datatype that can be referred to as a VTK
/// keyword.
pub trait VTKKeyword {
    fn vtk(&self) -> &'static str;
}

/// A VTK write component is a component of a VTK format that is writable.
pub trait VTKWriteComp<Writer: VTKWriter> {
    fn write(&self, writer: &mut Writer) -> Result<(), Writer::Error>;
}
