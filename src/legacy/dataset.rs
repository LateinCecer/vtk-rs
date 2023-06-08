use std::io::Write;
use std::marker::PhantomData;
use std::ops::{Deref, Index, IndexMut};
use nalgebra::{Const, DMatrix, DVector, Dyn, Matrix, MatrixView, MatrixViewMut, Scalar, SimdValue, SMatrix, SVector, U1, VecStorage, Vector3, Vector4};
use num::Zero;
use crate::legacy::{LegacyDataType, LegacyError, LegacyWriter, NamedLegacyDataType};
use crate::mesh::{CellShape, CellShapeName};
use crate::writer::{VTKDataFormat, VTKKeyword, VTKOptions, VTKWriteComp};


macro_rules! impl_veclike(
    ($imp:ty) => (
        impl<W: Write> VTKWriteComp<LegacyWriter<W>> for $imp {
            fn write(&self, writer: &mut LegacyWriter<W>) -> Result<(), LegacyError> {
                writer.write_str(self.vtk())?;
                match writer.format {
                    VTKDataFormat::Ascii => {
                        writer.write_str(" ")?;
                        writer.write_data(&self.0)?;
                    }
                    VTKDataFormat::Binary => {
                        writer.write_data(&self.0)?;
                    }
                }
                Ok(())
            }
        }
    );
);


pub struct Dims(SVector<i32, 3>);

impl_veclike!(Dims);
impl VTKKeyword for Dims {
    fn vtk(&self) -> &'static str {
        "DIMENSIONS"
    }
}

pub struct Origin(SVector<i32, 3>);

impl_veclike!(Origin);
impl VTKKeyword for Origin {
    fn vtk(&self) -> &'static str {
        "ORIGIN"
    }
}

pub struct Spacing(SVector<i32, 3>);

impl_veclike!(Spacing);
impl VTKKeyword for Spacing {
    fn vtk(&self) -> &'static str {
        "SPACING"
    }
}

pub struct Points<T: VTKKeyword>(Matrix<T, Const<3>, Dyn, VecStorage<T, Const<3>, Dyn>>);

impl<T: VTKKeyword + Scalar + Zero> Points<T> {
    /// Creates a new point list initialized with `zero` values.
    pub fn new(n: usize) -> Self {
        Points(Matrix::<T, Const<3>, Dyn, VecStorage<T, Const<3>, Dyn>>::zeros(n))
    }

    /// Returns the number of points in the point list.
    pub fn len(&self) -> usize {
        self.0.ncols()
    }

    /// Returns a view into the point vector at the specified index from the index list.
    pub fn vec(&self, idx: usize) -> MatrixView<T, Const<3>, U1, U1, Const<3>> {
        self.0.column(idx)
    }

    /// Returns a mutable view into the point vector at the specified index from the index list.
    pub fn vec_mut(&mut self, idx: usize) -> MatrixViewMut<T, Const<3>, U1, U1, Const<3>> {
        self.0.column_mut(idx)
    }
}

impl<T: VTKKeyword + Scalar> From<&[SVector<T, 3>]> for Points<T> {
    fn from(value: &[SVector<T, 3>]) -> Self {
        Points(Matrix::from_columns(value))
    }
}

impl<T: VTKKeyword> VTKKeyword for Points<T> {
    fn vtk(&self) -> &'static str {
        "POINTS"
    }
}

impl<T: VTKKeyword + NamedLegacyDataType + Scalar + Zero, W: Write> VTKWriteComp<LegacyWriter<W>> for Points<T> {
    fn write(&self, writer: &mut LegacyWriter<W>) -> Result<(), LegacyError> {
        writeln!(writer, "{} {} {}", self.vtk(), self.0.ncols(), T::dt_name())?;
        for col in self.0.column_iter() {
            writer.write_data(&col)?;
            writer.write_nl()?;
        }
        Ok(())
    }
}


pub struct XCoords<T: VTKKeyword>(DVector<T>);

impl<T: VTKKeyword> VTKKeyword for XCoords<T> {
    fn vtk(&self) -> &'static str {
        "X_COORDINATES"
    }
}

pub struct YCoords<T: VTKKeyword>(DVector<T>);

impl<T: VTKKeyword> VTKKeyword for YCoords<T> {
    fn vtk(&self) -> &'static str {
        "Y_COORDINATES"
    }
}

pub struct ZCoords<T: VTKKeyword>(DVector<T>);

impl<T: VTKKeyword> VTKKeyword for ZCoords<T> {
    fn vtk(&self) -> &'static str {
        "Z_COORDINATES"
    }
}

macro_rules! impl_coord (
    ($imp:ident) => (
        impl<T: NamedLegacyDataType + VTKKeyword, W: Write> VTKWriteComp<LegacyWriter<W>> for $imp<T> {
            fn write(&self, writer: &mut LegacyWriter<W>) -> Result<(), LegacyError> {
                // write header
                writeln!(writer, "{} {} {}", self.vtk(), self.0.len(), T::dt_name())?;
                // write data
                writer.write_data(&self.0)?;
                writer.write_nl()?;
                Ok(())
            }
        }
    );
);

impl_coord!(XCoords);
impl_coord!(YCoords);
impl_coord!(ZCoords);



struct PrimitiveData(Vec<DVector<i32>>);
pub enum Primitive {
    Vertices(PrimitiveData),
    Lines(PrimitiveData),
    Polygons(PrimitiveData),
    TriangleStrip(PrimitiveData),
}

impl VTKKeyword for Primitive {
    fn vtk(&self) -> &'static str {
        match self {
            Primitive::Vertices(_) => "VERTICES",
            Primitive::Lines(_) => "LINES",
            Primitive::Polygons(_) => "POLYGONS",
            Primitive::TriangleStrip(_) => "TRIANGLE_STRIPS",
        }
    }
}

impl<W: Write> VTKWriteComp<LegacyWriter<W>> for Primitive {
    fn write(&self, writer: &mut LegacyWriter<W>) -> Result<(), LegacyError> {
        // writer header
        let (name, points) = match self {
            Primitive::Vertices(cont) => ("VERTICES", &cont.0),
            Primitive::Lines(cont) => ("LINES", &cont.0),
            Primitive::Polygons(cont) => ("POLYGONS", &cont.0),
            Primitive::TriangleStrip(cont) => ("TRIANGLE_STRIPS", &cont.0),
        };
        let size: usize = points.iter()
            .map(|points| points.len() + 1)
            .sum();
        writeln!(writer, "{name} {} {}", points.len(), size)?;
        // write body
        for line in points {
            writer.write_data(&(line.len() as i32))?;
            writer.write_separator()?;
            writer.write_data(line)?;
            writer.write_nl()?;
        }
        Ok(())
    }
}

pub struct Cells {
    indices: Vec<DVector<i32>>,
}

#[derive(Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Debug)]
pub enum CellType {
    Vertex,
    PolyVertex,
    Line,
    PolyLine,
    Triangle,
    TriangleStrip,
    Polygon,
    Pixel,
    Quad,
    Tetra,
    Voxel,
    Hexahedron,
    Wedge,
    Pyramid,
    PentagonalPrism,
    HexagonalPrism,
}

impl From<CellShape> for CellType {
    fn from(value: CellShape) -> Self {
        match value.name {
            CellShapeName::Triangle => Self::Triangle,
            CellShapeName::Rectangle => Self::Quad,
            CellShapeName::Tetrahedron => Self::Tetra,
            CellShapeName::Prism => Self::Wedge,
            CellShapeName::Cube => Self::Voxel,
        }
    }
}

impl From<CellType> for usize {
    fn from(value: CellType) -> Self {
        match value {
            CellType::Vertex => 1,
            CellType::PolyVertex => 2,
            CellType::Line => 3,
            CellType::PolyLine => 4,
            CellType::Triangle => 5,
            CellType::TriangleStrip => 6,
            CellType::Polygon => 7,
            CellType::Pixel => 8,
            CellType::Quad => 9,
            CellType::Tetra => 10,
            CellType::Voxel => 11,
            CellType::Hexahedron => 12,
            CellType::Wedge => 13,
            CellType::Pyramid => 14,
            CellType::PentagonalPrism => 15,
            CellType::HexagonalPrism => 16,
        }
    }
}

impl VTKKeyword for CellType {
    fn vtk(&self) -> &'static str {
        match self {
            CellType::Vertex => "VTK_VERTEX",
            CellType::PolyVertex => "VTK_POLY_VERTEX",
            CellType::Line => "VTK_LINE",
            CellType::PolyLine => "VTK_POLY_LINE",
            CellType::Triangle => "VTK_TRIANGLE",
            CellType::TriangleStrip => "VTK_TRIANGLE_STRIP",
            CellType::Polygon => "VTK_POLYGON",
            CellType::Pixel => "VTK_PIXEL",
            CellType::Quad => "VTK_QUAD",
            CellType::Tetra => "VTK_TETRA",
            CellType::Voxel => "VTK_VOXEL",
            CellType::Hexahedron => "VTK_HEXAHEDRON",
            CellType::Wedge => "VTK_WEDGE",
            CellType::Pyramid => "VTK_PYRAMID",
            CellType::PentagonalPrism => "VTK_PENTAGONAL_PRISM",
            CellType::HexagonalPrism => "VTK_HEXAGONAL_PRISM",
        }
    }
}

impl Cells {
    /// Creates a new empty cells sections.
    pub fn new() -> Self {
        Cells {
            indices: Vec::new()
        }
    }

    /// Pushes a new cell to the `Cells` section. A single cell is defined by the indices of the
    /// cell corners. The type of the cell is specified in a separate types vector. The cell type
    /// is used to interpret how the cell indices should be interpreted to form a cell.
    pub fn push(&mut self, indices: DVector<i32>) {
        self.indices.push(indices);
    }

    /// Returns the number of cells in the `Cells` section.
    pub fn len(&self) -> usize {
        self.indices.len()
    }
}

impl Index<usize> for Cells {
    type Output = DVector<i32>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.indices[index]
    }
}

impl IndexMut<usize> for Cells {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.indices[index]
    }
}

impl VTKKeyword for Cells {
    fn vtk(&self) -> &'static str {
        "CELLS"
    }
}

impl<W: Write> VTKWriteComp<LegacyWriter<W>> for Cells {
    fn write(&self, writer: &mut LegacyWriter<W>) -> Result<(), LegacyError> {
        let size: usize = self.indices.iter()
            .map(|list| list.len() + 1)
            .sum();
        writeln!(writer, "CELLS {} {}", self.indices.len(), size)?;
        // write body
        for list in self.indices.iter() {
            writer.write_data(&(list.len() as i32))?;
            writer.write_separator()?;
            writer.write_data(list)?;
            writer.write_nl()?;
        }
        Ok(())
    }
}

impl<W: Write> VTKWriteComp<LegacyWriter<W>> for CellType {
    fn write(&self, writer: &mut LegacyWriter<W>) -> Result<(), LegacyError> {
        let idx: usize = self.clone().into();
        writer.write_data(&(idx as i32))?;
        Ok(())
    }
}

pub enum Dataset<T: VTKKeyword> {
    StructuredPoints(Dims, Origin, Spacing),
    StructuredGrid(Dims, Points<T>),
    RectilinearGrind(Dims, XCoords<T>, YCoords<T>, ZCoords<T>),
    /// polygon data contains:
    /// - data points for each vertex at the start
    /// - (optional) list of vertex indices for arbitrary primitives
    /// - (optional) list of line indices
    /// - (optional) list of polygon indices
    /// - (optional) list of triangle strips
    PolygonalData(Points<T>, Option<Primitive>, Option<Primitive>, Option<Primitive>, Option<Primitive>),
    UnstructuredGrid(Points<T>, Cells, Vec<CellType>),
    Field(),
}

impl<T: VTKKeyword> VTKKeyword for Dataset<T> {
    fn vtk(&self) -> &'static str {
        match self {
            Self::StructuredPoints(..) => "STRUCTURED_POINTS",
            Self::StructuredGrid(..) => "STRUCTURED_GRID",
            Self::RectilinearGrind(..) => "RECTILINEAR_GRID",
            Self::PolygonalData(..) => "POLYDATA",
            Self::UnstructuredGrid(..) => "UNSTRUCTURED_GRID",
            Self::Field(..) => "FIELD",
        }
    }
}

impl<T: VTKKeyword + NamedLegacyDataType + Scalar + Zero, W: Write> VTKWriteComp<LegacyWriter<W>> for Dataset<T> {
    fn write(&self, writer: &mut LegacyWriter<W>) -> Result<(), LegacyError> {
        match self {
            Dataset::StructuredPoints(dimensions, origin, spacing) => {
                writeln!(writer, "DATASET STRUCTURED_POINTS")?;
                writer.write_component(dimensions)?;
                writer.write_component(origin)?;
                writer.write_component(spacing)?;
            }
            Dataset::StructuredGrid(dimensions, points) => {
                writeln!(writer, "DATASET STRUCTURED_GRID")?;
                writer.write_component(dimensions)?;
                writer.write_component(points)?;
            }
            Dataset::RectilinearGrind(dimensions, x, y, z) => {
                writeln!(writer, "DATASET RECTILINEAR_GRID")?;
                writer.write_component(dimensions)?;
                writer.write_component(x)?;
                writer.write_component(y)?;
                writer.write_component(z)?;
            }
            Dataset::PolygonalData(points, vertices, lines, polygons, triangle_strips) => {
                writeln!(writer, "DATASET POLYDATA")?;
                writer.write_component(points)?;
                if let Some(vertices) = vertices {
                    writer.write_component(vertices)?;
                }
                if let Some(lines) = lines {
                    writer.write_component(lines)?;
                }
                if let Some(polygons) = polygons {
                    writer.write_component(polygons)?;
                }
                if let Some(triangle_strips) = triangle_strips {
                    writer.write_component(triangle_strips)?;
                }
            }
            Dataset::UnstructuredGrid(points, cells, cell_types) => {
                writeln!(writer, "DATASET UNSTRUCTURED_GRID")?;
                writer.write_component(points)?;
                writer.write_component(cells)?;
                assert_eq!(cells.indices.len(), cell_types.len(),
                           "number of cell types has to be equal to the number of cell polygons");
                writeln!(writer, "CELL_TYPES {}", cell_types.len())?;
                for ty in cell_types {
                    writer.write_component(ty)?;
                    writer.write_nl()?;
                }
            }
            Dataset::Field() => {}
        }
        Ok(())
    }
}


/// A color scalar is a u8, mapped in the range 0..1.
pub type ColorScalar = u8;
/// An rgba color value consists of 4 u8 values (one for each color channel), mapped in the range
/// 0..1.
pub type RgbaValue = Vector4<u8>;


/// # Written format
/// arrayName numComponents numTuples dataType
/// f_00 f_01 .. f_0(numComponents-1)
/// f_10 f_11 .. f_1(numComponents-1)
/// ..
/// f_(numTuples-1)0 f_(numTuples-1)1 .. f_(numTuples-1)(numComponents-1)
pub struct Field<T: VTKKeyword> {
    name: String,
    /// # Matrix dimensions:
    /// N = numComponents (dimensionality of the data)
    /// M = numTuples (number of field entries)
    mat: DMatrix<T>,
}

impl<T: VTKKeyword> Field<T> {
    pub fn new(name: String, mat: DMatrix<T>) -> Self {
        Field {
            name,
            mat,
        }
    }
}

impl<T: VTKKeyword + NamedLegacyDataType, W: Write> VTKWriteComp<LegacyWriter<W>> for Field<T> {
    fn write(&self, writer: &mut LegacyWriter<W>) -> Result<(), LegacyError> {
        writeln!(writer, "{} {} {} {}", self.name, self.mat.ncols(), self.mat.nrows(), T::dt_name())?;
        writer.write_data(&self.mat)?;
        Ok(())
    }
}

pub enum TypedField {
    U8Field(Field<u8>),
    I8Field(Field<i8>),
    U16Field(Field<u16>),
    I16Field(Field<i16>),
    U32Field(Field<u32>),
    I32Field(Field<i32>),
    U64Field(Field<u64>),
    I64Field(Field<i64>),
    F32Field(Field<f32>),
    F64Field(Field<f64>),
}

impl<W: Write> VTKWriteComp<LegacyWriter<W>> for TypedField {
    fn write(&self, writer: &mut LegacyWriter<W>) -> Result<(), LegacyError> {
        match self {
            TypedField::U8Field(val) => val.write(writer),
            TypedField::I8Field(val) => val.write(writer),
            TypedField::U16Field(val) => val.write(writer),
            TypedField::I16Field(val) => val.write(writer),
            TypedField::U32Field(val) => val.write(writer),
            TypedField::I32Field(val) => val.write(writer),
            TypedField::U64Field(val) => val.write(writer),
            TypedField::I64Field(val) => val.write(writer),
            TypedField::F32Field(val) => val.write(writer),
            TypedField::F64Field(val) => val.write(writer),
        }
    }
}

pub struct LookupTable<T> {
    name: String,
    data: DMatrix<T>,
}

impl<T: VTKKeyword + NamedLegacyDataType, W: Write> VTKWriteComp<LegacyWriter<W>> for LookupTable<T> {
    fn write(&self, writer: &mut LegacyWriter<W>) -> Result<(), LegacyError> {
        writeln!(writer, "LOOKUP_TABLE {} {}", self.name, self.data.len())?;
        writer.write_data(&self.data)?;
        Ok(())
    }
}


pub enum Attrib<T: VTKKeyword> {
    /// Scalar definition includes specification of a lookup table. The definition of a lookup table
    /// is optional. If not specified, the default VTK table will be used (and tableName should be
    /// “default”). Also note that the numComp variable is optional—by default the number of
    /// components is equal to one. (The parameter numComp must range between 1 and 4 inclusive;
    /// in versions of VTK prior to 2.3 this parameter was not supported.)
    ///
    /// # Parameters
    /// - dataName
    /// - dataType
    /// - numComp
    Scalars(String, LookupTable<T>),

    /// The definition of color scalars (i.e., unsigned char values directly mapped to color) varies
    /// depending upon the number of values (nValues) per scalar. If the file format is ASCII, the
    /// color scalars are defined using nValues float values between (0,1). If the file format is
    /// BINARY, the stream of data consists of nValues unsigned char values per scalar value.
    ///
    /// # Parameters
    /// - dataName
    /// - nValues
    ColorScalars(String, DMatrix<ColorScalar>),

    /// The tableName field is a character string (without embedded white space) used to identify
    /// the lookup table. This label is used by the VTK reader to extract a specific table. Each
    /// entry in the lookup table is a rgba[4] (red-green-blue-alpha) array (alpha is opacity where
    /// alpha=0 is transparent). If the file format is ASCII, the lookup table values must be float
    /// values between (0,1). If the file format is BINARY, the stream of data must be four unsigned
    /// char values per table entry.
    ///
    /// # Parameters
    /// - tableName
    /// - size
    LookupTable(LookupTable<T>),


    Vectors(String, Matrix<T, Const<3>, Dyn, VecStorage<T, Const<3>, Dyn>>),
    Normals(String, Matrix<T, Const<3>, Dyn, VecStorage<T, Const<3>, Dyn>>),
    TextureCoords(String, DMatrix<T>),
    Tensors(String, Vec<SMatrix<T, 3, 3>>),

    /// Field data is essentially an array of data arrays. Defining field data means giving a name
    /// to the field and specifying the number of arrays it contains. Then, for each array, the name
    /// of the array arrayName(i), the number of components of the array, numComponents, the number
    /// of tuples in the array, numTuples, and the data type, dataType, are defined.
    ///
    /// # Parameters
    /// - dataName
    /// - numArrays
    FieldData(String, Vec<TypedField>),
}

impl<T: VTKKeyword> VTKKeyword for Attrib<T> {
    fn vtk(&self) -> &'static str {
        match self {
            Self::Scalars(..) => "SCALARS",
            Self::ColorScalars(..) => "COLOR_SCALARS",
            Self::LookupTable(..) => "LOOKUP_TABLE",
            Self::Vectors(..) => "VECTORS",
            Self::Normals(..) => "NORMALS",
            Self::TextureCoords(..) => "TEXTURE_COORDINATES",
            Self::Tensors(..) => "TENSORS",
            Self::FieldData(..) => "FIELD",
        }
    }
}

impl<T: VTKKeyword + NamedLegacyDataType + Clone, W: Write> VTKWriteComp<LegacyWriter<W>> for Attrib<T> {
    fn write(&self, writer: &mut LegacyWriter<W>) -> Result<(), LegacyError> {
        match self {
            Attrib::Scalars(data_name, table) => {
                writeln!(writer, "SCALARS {} {} {}", data_name, T::dt_name(), table.data.ncols())?;
                writeln!(writer, "LOOKUP_TABLE {}", table.name)?;
                writer.write_data(&table.data)?;
            }
            Attrib::ColorScalars(data_name, values) => {
                writeln!(writer, "COLOR_SCALARS {} {}", data_name, values.ncols())?;
                writer.write_data(values)?;
            }
            Attrib::LookupTable(table) => {
                writer.write_component(table)?;
            }
            Attrib::Vectors(data_name, data) => {
                writeln!(writer, "VECTORS {} {}", data_name, T::dt_name())?;
                writer.write_data(data)?;
            }
            Attrib::Normals(data_name, data) => {
                writeln!(writer, "NORMALS {} {}", data_name, T::dt_name())?;
                writer.write_data(data)?;
            }
            Attrib::TextureCoords(data_name, data) => {
                writeln!(writer, "TEXTURE_COORDINATES {} {} {}", data_name, data.ncols(), T::dt_name())?;
                writer.write_data(data)?;
            }
            Attrib::Tensors(data_name, data) => {
                writeln!(writer, "TENSORS {} {}", data_name, T::dt_name())?;
                for tensor in data {
                    writer.write_data(tensor)?;
                }
            }
            Attrib::FieldData(data_name, arrays) => {
                writeln!(writer, "FIELD {} {}", data_name, arrays.len())?;
                for field in arrays {
                    writer.write_component(field)?;
                }
            }
        }
        Ok(())
    }
}
