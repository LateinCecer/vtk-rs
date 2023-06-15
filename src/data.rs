use std::collections::HashMap;
use nalgebra::{Const, DMatrix, Dyn, Matrix, VecStorage};

/// `FieldData` contains generalized field data that can be written (or read) from a VTK file. Each
/// `FieldData` instance has a name and a list of components, where each component represents one
/// property of the generalized field. The field components themselves consist of a name and an
/// `nalgebra::DMatrix` and can be added are removed from the `FieldData`.
///
/// # Example
///
/// For a lot of physical simulations there are different fields with physically relevant values.
/// These include, for example, the local pressure and velocity values. For a 3-dimensional fluid
/// domain, the velocity field has 3 components per cell and the pressure field only has 1 component
/// per cell. Both fields can be encapsulated into a single `FieldData` instance, which is
/// especially useful when dealing with multiple iterations of the same field, as can be field in
/// time-dependent simulations.
///
/// ````rust
/// use std::fs::File;
/// use std::io::BufWriter;
/// use nalgebra::DMatrix;
/// use vtk_rs::prelude::*;
///
/// let num_cells = 1000;
/// let mut timestep = FieldData::new("physical_fields_step_1".to_owned());
/// timestep.add_field_component("pressure".to_owned(), DMatrix::zeros(num_cells, 1));
/// timestep.add_field_component("velocity".to_owned(), DMatrix::zeros(num_cells, 2));
///
/// let file_writer = BufWriter::new(File::create("test.vtk").unwrap());
/// let mut writer = VTKFormat::Legacy.make_writer(file_writer, VTKOptions::default());
/// writer.write_header("field component example").unwrap();
/// writer.write_geometry(create_mesh()).unwrap();
/// writer.write(timestep).unwrap()
/// ````
pub struct FieldData {
    pub name: String,
    pub components: HashMap<String, FieldComponent>,
}

pub trait AddFieldComp<T> {
    /// Adds a new component with the specified `data` payload and the `name` to the set of named
    /// components of a `FieldData` instance.
    ///
    /// # Example
    /// ````rust
    /// use nalgebra::{DMatrix, DVector};
    /// use vtk_rs::prelude::*;
    ///
    /// let n = 100;
    /// let mut field_data = FieldData::new("some_data".to_owned());
    /// // add scalar pressure values
    /// field_data.add_field_component("pressure".to_owned(), DVector::<f32>::zeros(n));
    /// // add vectorized velocity field
    /// field_data.add_field_component("velocity".to_owned(), DMatrix::<f64>::zeros(n, 3))
    /// ````
    fn add_field_component(&mut self, name: String, data: T);
}

pub enum FieldType {
    U8(DMatrix<u8>),
    U16(DMatrix<u16>),
    U32(DMatrix<u32>),
    U64(DMatrix<u64>),

    I8(DMatrix<i8>),
    I16(DMatrix<i16>),
    I32(DMatrix<i32>),
    I64(DMatrix<i64>),

    F32(DMatrix<f32>),
    F64(DMatrix<f64>),
}

pub struct FieldComponent {
    pub name: String,
    pub data: FieldType,
}

macro_rules! impl_from_data(
    ($($base:ident -> $field:ident),+) => ($(
        impl From<DMatrix<$base>> for FieldType {
            fn from(value: DMatrix<$base>) -> Self {
                FieldType::$field(value)
            }
        }
    )+);
);

impl_from_data!(
    u8 -> U8,
    u16 -> U16,
    u32 -> U32,
    u64 -> U64,
    i8 -> I8,
    i16 -> I16,
    i32 -> I32,
    i64 -> I64,
    f32 -> F32,
    f64 -> F64
);

impl FieldData {
    /// Creates a new empty `FieldData` instance with the specified `name`. Field data is always
    /// initialized empty and filled by adding components after initialization. The field data can
    /// then be written to a writer instance.
    pub fn new(name: String) -> Self {
        FieldData {
            name,
            components: HashMap::new(),
        }
    }
}

impl<T> AddFieldComp<T> for FieldData
where FieldType: From<T> {
    fn add_field_component(&mut self, name: String, data: T) {
        let comp = FieldComponent {
            name: name.clone(),
            data: FieldType::from(data),
        };
        self.components.insert(name, comp);
    }
}

pub struct VectorField<T> {
    pub data: Matrix<T, Const<3>, Dyn, VecStorage<T, Const<3>, Dyn>>,
    pub name: String,
}

impl<T> VectorField<T> {
    pub fn new(name: String, mat: Matrix<T, Const<3>, Dyn, VecStorage<T, Const<3>, Dyn>>) -> Self {
        VectorField {
            data: mat,
            name
        }
    }
}
