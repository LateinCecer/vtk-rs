mod dataset;

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::io::Write;
use std::ops::{Deref, DerefMut};
use nalgebra::{Dim, Scalar};
use nalgebra::Const;
use nalgebra::Dyn;
use nalgebra::Matrix;
use nalgebra::RawStorage;
use num::Zero;
use crate::data::{FieldData, FieldType};
use crate::legacy::dataset::{Attrib, Cells, CellType, Dataset, Field, Points, TypedField};
use crate::writer::{MeshData, VTKDataFormat, VTKDataWriter, VTKGeneralWriter, VTKGeometryWriter, VTKKeyword, VTKOptions, VTKWriteComp, VTKWriter};

#[derive(Clone, Copy, PartialOrd, PartialEq, Debug)]
enum WriteState {
    Header,
    Geometry,
    Data,
}

impl WriteState {
    /// Consumes the write state and returns the next write state that precedes this write state.
    fn advance(self) -> Self {
        match self {
            WriteState::Header => WriteState::Geometry,
            WriteState::Geometry => WriteState::Data,
            WriteState::Data => WriteState::Data,
        }
    }
}

impl Display for WriteState {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Header => write!(f, "header"),
            Self::Geometry => write!(f, "geometry"),
            Self::Data => write!(f, "data"),
        }
    }
}

#[derive(Debug)]
pub enum LegacyError {
    IOError(std::io::Error),
    StateError(WriteState, String),
    HeaderToLong(usize),
}

impl Display for LegacyError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IOError(e) => write!(f, "{}", e),
            Self::StateError(state, op) => write!(
                f, "Cannot apply operation {op} to legacy VTK writer in state {state}"),
            Self::HeaderToLong(len) => write!(
                f, "Header len {len} \\ 256 is to long for legacy file format"),
        }
    }
}

impl Error for LegacyError {}

impl From<std::io::Error> for LegacyError {
    fn from(value: std::io::Error) -> Self {
        Self::IOError(value)
    }
}

trait LegacyDataType {
    /// can be used to write data to the writer
    fn write<W: Write>(&self, writer: &mut W, format: &VTKDataFormat) -> Result<(), LegacyError>;
    fn write_mapped<W: Write>(&self, writer: &mut W, format: &VTKDataFormat) -> Result<(), LegacyError>;
}

trait NamedLegacyDataType: LegacyDataType {
    fn dt_name() -> &'static str;
}

macro_rules! impl_datatype(
    (floating $name:ident $keyword:literal) => (
        impl LegacyDataType for $name {
            fn write<W: Write>(&self, writer: &mut W, format: &VTKDataFormat) -> Result<(), LegacyError> {
                match format {
                    VTKDataFormat::Binary => {
                        writer.write(&self.to_le_bytes())?;
                    }
                    VTKDataFormat::Ascii => {
                        write!(writer, "{}", self)?;
                    }
                }
                Ok(())
            }

            fn write_mapped<W: Write>(&self, writer: &mut W, format: &VTKDataFormat) -> Result<(), LegacyError> {
                match format {
                    VTKDataFormat::Binary => {
                        writer.write(&(Self::min(Self::max(*self, 0.0), 1.0)).to_le_bytes())?;
                    }
                    VTKDataFormat::Ascii => {
                        write!(writer, "{}", self)?;
                    }
                }
                Ok(())
            }
        }

        impl NamedLegacyDataType for $name {
            fn dt_name() -> &'static str {
                $keyword
            }
        }

        impl VTKKeyword for $name {
            fn vtk(&self) -> &'static str {
                $keyword
            }
        }
    );
    (integer $name:ident $keyword:literal) => (
        impl LegacyDataType for $name {
            fn write<W: Write>(&self, writer: &mut W, format: &VTKDataFormat) -> Result<(), LegacyError> {
                match format {
                    VTKDataFormat::Binary => {
                        writer.write(&self.to_le_bytes())?;
                    }
                    VTKDataFormat::Ascii => {
                        write!(writer, "{}", self)?;
                    }
                }
                Ok(())
            }

            fn write_mapped<W: Write>(&self, writer: &mut W, format: &VTKDataFormat) -> Result<(), LegacyError> {
                match format {
                    VTKDataFormat::Binary => {
                        writer.write(&(f32::min(f32::max(*self as f32 / (Self::MAX as f32), 0.0), 1.0)).to_le_bytes())?;
                    }
                    VTKDataFormat::Ascii => {
                        write!(writer, "{}", self)?;
                    }
                }
                Ok(())
            }
        }

        impl NamedLegacyDataType for $name {
            fn dt_name() -> &'static str {
                $keyword
            }
        }

        impl VTKKeyword for $name {
            fn vtk(&self) -> &'static str {
                $keyword
            }
        }
    );
);

impl_datatype!(floating f32 "float");
impl_datatype!(floating f64 "double");
impl_datatype!(integer u8 "unsigned_char");
impl_datatype!(integer i8 "char");
impl_datatype!(integer u16 "unsigned_short");
impl_datatype!(integer i16 "short");
impl_datatype!(integer u32 "unsigned_int");
impl_datatype!(integer i32 "int");
impl_datatype!(integer u64 "unsigned_long");
impl_datatype!(integer i64 "long");


impl LegacyDataType for bool {
    fn write<W: Write>(&self, writer: &mut W, format: &VTKDataFormat) -> Result<(), LegacyError> {
        match format {
            VTKDataFormat::Binary => {
                writer.write(&[*self as u8])?;
            }
            VTKDataFormat::Ascii => {
                write!(writer, "{}", *self as u8)?; // write as 1 or 0
            }
        }
        Ok(())
    }

    fn write_mapped<W: Write>(&self, writer: &mut W, format: &VTKDataFormat) -> Result<(), LegacyError> {
        self.write(writer, format)
    }
}

impl NamedLegacyDataType for bool {
    fn dt_name() -> &'static str {
        "bit"
    }
}

impl VTKKeyword for bool {
    fn vtk(&self) -> &'static str {
        Self::dt_name()
    }
}

impl<T, R, Storage> LegacyDataType for Matrix<T, R, Const<1>, Storage>
where T: LegacyDataType,
    R: Dim,
    Storage: RawStorage<T, R, Const<1>> {

    fn write<W: Write>(&self, writer: &mut W, format: &VTKDataFormat) -> Result<(), LegacyError> {
        match format {
            VTKDataFormat::Ascii => {
                self[0].write(writer, format)?;
                for i in 1..self.len() {
                    write!(writer, " ")?;
                    self[i].write(writer, format)?;
                }
            }
            VTKDataFormat::Binary => {
                for e in self.iter() {
                    e.write(writer, format)?;
                }
            }
        }
        Ok(())
    }

    fn write_mapped<W: Write>(&self, writer: &mut W, format: &VTKDataFormat) -> Result<(), LegacyError> {
        match format {
            VTKDataFormat::Ascii => {
                self[0].write(writer, format)?;
                for i in 1..self.len() {
                    write!(writer, " ")?;
                    self[i].write_mapped(writer, format)?;
                }
            }
            VTKDataFormat::Binary => {
                for e in self.iter() {
                    e.write_mapped(writer, format)?;
                }
            }
        }
        Ok(())
    }
}

impl<T, R, S> LegacyDataType for Matrix<T, R, Dyn, S>
where T: LegacyDataType,
    R: Dim,
    S: RawStorage<T, R, Dyn> {

    fn write<W: Write>(&self, writer: &mut W, format: &VTKDataFormat) -> Result<(), LegacyError> {
        match format {
            VTKDataFormat::Ascii => {
                for row in self.row_iter() {
                    row[0].write(writer, format)?;
                    for i in 1..row.len() {
                        write!(writer, " ")?;
                        row[i].write(writer, format)?;
                    }
                    writeln!(writer)?;
                }
            }
            VTKDataFormat::Binary => {
                for row in self.row_iter() {
                    for el in row.iter() {
                        el.write(writer, format)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn write_mapped<W: Write>(&self, writer: &mut W, format: &VTKDataFormat) -> Result<(), LegacyError> {
        match format {
            VTKDataFormat::Ascii => {
                for row in self.row_iter() {
                    row[0].write_mapped(writer, format)?;
                    for i in 1..row.len() {
                        write!(writer, " ")?;
                        row[i].write_mapped(writer, format)?;
                    }
                    writeln!(writer)?;
                }
            }
            VTKDataFormat::Binary => {
                for row in self.row_iter() {
                    for el in row.iter() {
                        el.write_mapped(writer, format)?;
                    }
                }
            }
        }
        Ok(())
    }
}


macro_rules! impl_matrix_datatype(
    (row $n:tt) => (
        impl<T, Storage> LegacyDataType for Matrix<T, Const<1>, Const<$n>, Storage>
            where T: LegacyDataType,
                  Storage: RawStorage<T, Const<1>, Const<$n>> {

            fn write<W: Write>(&self, writer: &mut W, format: &VTKDataFormat) -> Result<(), LegacyError> {
                match format {
                    VTKDataFormat::Ascii => {
                        self[0].write(writer, format)?;
                        for i in 1..self.len() {
                            write!(writer, " ")?;
                            self[i].write(writer, format)?;
                        }
                    }
                    VTKDataFormat::Binary => {
                        for e in self.iter() {
                            e.write(writer, format)?;
                        }
                    }
                }
                Ok(())
            }

            fn write_mapped<W: Write>(&self, writer: &mut W, format: &VTKDataFormat) -> Result<(), LegacyError> {
                match format {
                    VTKDataFormat::Ascii => {
                        self[0].write(writer, format)?;
                        for i in 1..self.len() {
                            write!(writer, " ")?;
                            self[i].write_mapped(writer, format)?;
                        }
                    }
                    VTKDataFormat::Binary => {
                        for e in self.iter() {
                            e.write_mapped(writer, format)?;
                        }
                    }
                }
                Ok(())
            }
        }
    );
    (mat ($n:tt)($m:tt)) => (
        impl<T, S> LegacyDataType for Matrix<T, Const<$n>, Const<$m>, S>
        where T: LegacyDataType,
              S: RawStorage<T, Const<$n>, Const<$m>> {

            fn write<W: Write>(&self, writer: &mut W, format: &VTKDataFormat) -> Result<(), LegacyError> {
                match format {
                    VTKDataFormat::Ascii => {
                        for row in self.row_iter() {
                            row[0].write(writer, format)?;
                            for i in 1..row.len() {
                                write!(writer, " ")?;
                                row[i].write(writer, format)?;
                            }
                            writeln!(writer)?;
                        }
                    }
                    VTKDataFormat::Binary => {
                        for row in self.row_iter() {
                            for el in row.iter() {
                                el.write(writer, format)?;
                            }
                        }
                    }
                }
                Ok(())
            }

            fn write_mapped<W: Write>(&self, writer: &mut W, format: &VTKDataFormat) -> Result<(), LegacyError> {
                match format {
                    VTKDataFormat::Ascii => {
                        for row in self.row_iter() {
                            row[0].write_mapped(writer, format)?;
                            for i in 1..row.len() {
                                write!(writer, " ")?;
                                row[i].write_mapped(writer, format)?;
                            }
                            writeln!(writer)?;
                        }
                    }
                    VTKDataFormat::Binary => {
                        for row in self.row_iter() {
                            for el in row.iter() {
                                el.write_mapped(writer, format)?;
                            }
                        }
                    }
                }
                Ok(())
            }
        }
    );
);
impl_matrix_datatype!(row 2);
impl_matrix_datatype!(row 3);
impl_matrix_datatype!(mat (3)(3));


/// Implements the legacy VTK format as a writer.
pub struct LegacyWriter<W: Write> {
    writer: W,
    state: WriteState,
    format: VTKDataFormat,
}

impl<W: Write> LegacyWriter<W> {
    pub fn new(writer: W, options: VTKOptions) -> Self {
        LegacyWriter {
            writer,
            state: WriteState::Header,
            format: options.data_format,
        }
    }

    fn write_str(&mut self, val: &str) -> Result<(), LegacyError> {
        write!(self.writer, "{val}")?;
        Ok(())
    }

    fn write_nl(&mut self) -> Result<(), LegacyError> {
        if self.format == VTKDataFormat::Ascii {
            writeln!(self.writer)?; // only write new line for ascii formats
        }
        Ok(())
    }

    fn write_separator(&mut self) -> Result<(), LegacyError> {
        if self.format == VTKDataFormat::Ascii {
            write!(self.writer, " ")?; // only write whitespace for ascii formats
        }
        Ok(())
    }

    fn write_data<T: LegacyDataType>(&mut self, val: &T) -> Result<(), LegacyError> {
        val.write(&mut self.writer, &self.format)
    }

    fn write_mapped_data<T: LegacyDataType>(&mut self, val: &T) -> Result<(), LegacyError> {
        val.write_mapped(&mut self.writer, &self.format)
    }

    fn write_component<T: VTKWriteComp<Self>>(&mut self, comp: &T) -> Result<(), LegacyError> {
        comp.write(self)
    }
}

impl<W: Write> VTKWriter for LegacyWriter<W> {
    type Error = LegacyError;

    fn write_header(&mut self, header: &impl Display) -> Result<(), Self::Error> {
        // check header length
        let fmt_header = format!("{}", header);
        if fmt_header.len() > 255 {
            return Err(LegacyError::HeaderToLong(fmt_header.len()));
        }
        if self.state != WriteState::Header {
            return Err(LegacyError::StateError(
                self.state, "write header".to_owned()));
        }
        // write file header
        writeln!(&mut self.writer, "# vtk DataFile Version 2.0")?;
        writeln!(&mut self.writer, "{}", header)?;
        writeln!(&mut self.writer, "{}", self.format.vtk())?;
        self.state = self.state.advance();
        Ok(())
    }
}

impl<T, W> VTKGeometryWriter<MeshData<T, 3>> for LegacyWriter<W>
where
    T: VTKKeyword + NamedLegacyDataType + Scalar + Zero,
    W: Write {

    fn write_geometry(&mut self, msh: MeshData<T, 3>) -> Result<(), Self::Error> {
        // check state
        if self.state != WriteState::Geometry {
            return Err(LegacyError::StateError(self.state, "write geometry".to_owned()));
        }

        // write vbo
        let mesh = if let MeshData::UnstructuredPolygon(mesh) = msh {
            mesh
        } else {
            unimplemented!()
        };

        let points = Points::from(mesh.vbo.as_slice());
        let mut cells = Cells::new();
        let mut cell_types = Vec::new();
        for region in mesh.regions.values().into_iter() {
            let cell_type: CellType = region.shape.clone().into();
            region.iter(&mesh).into_iter()
                .for_each(|cell| {
                    cells.push(cell.indices().iter().map(|idx| *idx as i32)
                        .collect::<Vec<_>>().into());
                    cell_types.push(cell_type);
                })
        }
        let num_cells = cells.len();
        let dataset = Dataset::UnstructuredGrid(points, cells, cell_types);
        self.write_component(&dataset)?;

        writeln!(self, "CELL_DATA {}", num_cells)?;
        self.state = self.state.advance();
        Ok(())
    }
}

impl<W: Write> VTKDataWriter<FieldData> for LegacyWriter<W> {
    fn write(&mut self, data: FieldData) -> Result<(), Self::Error> {
        // check state
        if self.state != WriteState::Data {
            return Err(LegacyError::StateError(self.state, "write data".to_owned()));
        }

        let array: Vec<_> = data.components.into_iter()
            .map(|(name, comp)| {
                match comp.data {
                    FieldType::U8(value) => TypedField::U8Field(
                        Field::new(name, value)),
                    FieldType::U16(value) => TypedField::U16Field(
                        Field::new(name, value)),
                    FieldType::U32(value) => TypedField::U32Field(
                        Field::new(name, value)),
                    FieldType::U64(value) => TypedField::U64Field(
                        Field::new(name, value)),
                    FieldType::I8(value) => TypedField::I8Field(
                        Field::new(name, value)),
                    FieldType::I16(value) => TypedField::I16Field(
                        Field::new(name, value)),
                    FieldType::I32(value) => TypedField::I32Field(
                        Field::new(name, value)),
                    FieldType::I64(value) => TypedField::I64Field(
                        Field::new(name, value)),
                    FieldType::F32(value) => TypedField::F32Field(
                        Field::new(name, value)),
                    FieldType::F64(value) => TypedField::F64Field(
                        Field::new(name, value)),
                }
            })
            .collect();
        let field_data = Attrib::<i32>::FieldData(data.name, array);
        self.write_component(&field_data)?;
        Ok(())
    }
}

impl<W: Write> VTKGeneralWriter for LegacyWriter<W> {}

impl<W: Write> Deref for LegacyWriter<W> {
    type Target = W;

    fn deref(&self) -> &Self::Target {
        &self.writer
    }
}

impl<W: Write> DerefMut for LegacyWriter<W> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.writer
    }
}



#[cfg(test)]
mod test {
    use std::error::Error;
    use std::fs::File;
    use std::io::BufWriter;
    use nalgebra::{Const, DMatrix, DVector, Dyn, Matrix, SVector, VecStorage, Vector3};
    use crate::data::{AddFieldComp, FieldData};
    use crate::legacy::{LegacyError, LegacyWriter};
    use crate::legacy::dataset::{Attrib, Cells, CellType, Dataset, Field, Points, TypedField};
    use crate::mesh::{CellShape, UnstructuredMeshBuilder};
    use crate::writer::{MeshData, VTKDataWriter, VTKFormat, VTKGeometryWriter, VTKOptions, VTKWriter};

    #[test]
    fn test_write() -> Result<(), Box<dyn Error>> {
        let file = File::create("test.vtk")?;
        let options = VTKOptions {
            .. Default::default()
        };
        let mut writer = VTKFormat::Legacy.make_writer(BufWriter::new(file), options);
        writer.write_header(&"Generated with vtk-rs (https://github.com/LateinCecer/vtk-rs)".to_owned())?;
        Ok(())
    }

    #[test]
    fn test_unstructured_grid() -> Result<(), LegacyError> {
        let file = File::create("test.vtk")?;
        let options = VTKOptions {
            .. Default::default()
        };
        let mut writer = LegacyWriter::new(BufWriter::new(file), options);
        writer.write_header(&"Generated with vtk-rs (https://github.com/LateinCecer/vtk-rs)".to_owned())?;

        // write some unstructured mesh
        let mesh = UnstructuredMeshBuilder::<f32, 3>::new()
            .add_region(
                "main".to_owned(),
                CellShape::triangle(),
                vec![
                    Vector3::new(0.0, 0.0, 0.0),
                    Vector3::new(1.0, 0.0, 0.0),
                    Vector3::new(1.0, 1.0, 0.0),
                    Vector3::new(0.0, 1.0, 0.0),
                    Vector3::new(1.0, 2.0, 0.0),
                ],
                vec![
                    0, 1, 2,
                    2, 3, 0,
                    2, 4, 3,
                ],
            ).unwrap()
            .build();
        writer.write_geometry(MeshData::UnstructuredPolygon(mesh))?;

        // attach data
        let mut cell_data = DMatrix::<i32>::zeros(3, 1);
        cell_data[(0, 0)] = 0;
        cell_data[(1, 0)] = 1;
        cell_data[(2, 0)] = 2;
        let mut pressure_data = DMatrix::<f32>::zeros(3, 1);
        pressure_data[(0, 0)] = 0.01;
        pressure_data[(1, 0)] = 0.3;
        pressure_data[(2, 0)] = 0.2;
        let mut velocity_data = Matrix::<f32, Const<3>, Dyn, VecStorage<f32, Const<3>, Dyn>>::zeros(3);
        velocity_data[(0, 0)] = 1.0;    velocity_data[(0, 1)] = 0.3;    velocity_data[(0, 2)] = 0.3;
        velocity_data[(1, 0)] = 1.1;    velocity_data[(1, 1)] = 0.2;    velocity_data[(1, 2)] = 0.2;
        velocity_data[(2, 0)] = 1.2;    velocity_data[(2, 1)] = 0.1;    velocity_data[(2, 2)] = 0.1;

        let mut field = FieldData::new("TimeStep".to_owned());
        field.add_field_component("cellIds".to_owned(), cell_data);
        field.add_field_component("pressure".to_owned(), pressure_data);
        writer.write(field)?;
        writer.write_component(&Attrib::Vectors("velocity".to_owned(), velocity_data))?;
        Ok(())
    }
}
