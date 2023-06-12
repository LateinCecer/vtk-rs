pub mod mesh;
pub mod writer;
mod legacy;
pub mod data;
pub mod prelude;



#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::fs::File;
    use std::io::BufWriter;
    use nalgebra::{DMatrix, Vector3};
    use crate::prelude::*;

    #[test]
    fn legacy() -> Result<(), Box<dyn Error>> {
        let file = File::create("legacy.vtk")?;
        let options = VTKOptions {
            .. Default::default()
        };
        let mut writer = VTKFormat::Legacy.make_writer(BufWriter::new(file), options);
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
        let mut velocity_data = DMatrix::<f32>::zeros(3, 3);
        velocity_data[(0, 0)] = 1.0;    velocity_data[(0, 1)] = 0.3;    velocity_data[(0, 2)] = 0.3;
        velocity_data[(1, 0)] = 1.1;    velocity_data[(1, 1)] = 0.2;    velocity_data[(1, 2)] = 0.2;
        velocity_data[(2, 0)] = 1.2;    velocity_data[(2, 1)] = 0.1;    velocity_data[(2, 2)] = 0.1;

        let mut field = FieldData::new("TimeStep".to_owned());
        field.add_field_component("cellIds".to_owned(), cell_data);
        field.add_field_component("pressure".to_owned(), pressure_data);
        field.add_field_component("velocity".to_owned(), velocity_data);
        writer.write(field)?;
        Ok(())
    }
}
