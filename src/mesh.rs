use std::collections::HashMap;
use std::ops::Range;
use nalgebra::{Scalar, SVector};

#[derive(Clone, PartialOrd, PartialEq)]
pub enum CellShapeName {
    Line,
    Triangle,
    Rectangle,
    Tetrahedron,
    Prism,
    Cube,
}

#[derive(Clone, PartialOrd, PartialEq)]
pub struct CellShape {
    pub num_vertices: usize,
    faces: Vec<CellFace>,
    pub name: CellShapeName,
}

#[derive(Clone, PartialOrd, PartialEq)]
struct CellFace {
    num_vertices: usize,
    indices: Vec<usize>,
}

pub struct CellFaceRef<'a, T, const DIM: usize> {
    msh: &'a UnstructuredMesh<T, DIM>,
    shape: &'a CellShape,
    id: usize,
    mesh_id: usize,
}

impl<'a, T: 'static, const DIM: usize> CellFaceRef<'a, T, DIM> {
    /// Returns the indices for this face as a vector
    pub fn indices(&self) -> Vec<usize> {
        self.shape.faces[self.id].indices.iter()
            .map(|&i| self.msh.ibo[i + self.mesh_id])
            .collect()
    }

    /// Returns the vertices of this face as a vector
    pub fn vertices(&self) -> Vec<&'a SVector<T, DIM>> {
        self.indices().iter()
            .map(|&i| &self.msh.vbo[i])
            .collect()
    }
}

pub struct FaceIter<'a, T, const DIM: usize> {
    shape: &'a CellShape,
    mesh_id: usize,
    mesh: &'a UnstructuredMesh<T, DIM>,
    face_id: usize,
}

impl CellShape {
    /// Creates an iterator that can be used to iterate over the cells faces. This is especially
    /// useful for converting the mesh into other formats.
    pub fn face_iter<'a, T, const DIM: usize>(
        &'a self, msh: &'a UnstructuredMesh<T, DIM>, mesh_id: usize
    ) -> FaceIter<'a, T, DIM> {
        FaceIter {
            shape: self,
            mesh_id,
            mesh: msh,
            face_id: 0,
        }
    }

    /// Returns the number of faces for this cell shape
    pub fn num_faces(&self) -> usize {
        self.faces.len()
    }

    pub fn line() -> Self {
        CellShape {
            num_vertices: 2,
            faces: vec![
                CellFace { num_vertices: 1, indices: vec![0] },
                CellFace { num_vertices: 1, indices: vec![1] },
            ],
            name: CellShapeName::Line,
        }
    }

    pub fn triangle() -> Self {
        CellShape {
            num_vertices: 3,
            faces: vec![
                CellFace { num_vertices: 2, indices: vec![0, 1] },
                CellFace { num_vertices: 2, indices: vec![1, 2] },
                CellFace { num_vertices: 2, indices: vec![2, 0] },
            ],
            name: CellShapeName::Triangle,
        }
    }

    pub fn rectangle() -> Self {
        CellShape {
            num_vertices: 4,
            faces: vec![
                CellFace { num_vertices: 2, indices: vec![0, 1] },
                CellFace { num_vertices: 2, indices: vec![1, 2] },
                CellFace { num_vertices: 2, indices: vec![2, 3] },
                CellFace { num_vertices: 2, indices: vec![3, 0] },
            ],
            name: CellShapeName::Rectangle,
        }
    }

    pub fn tetrahedron() -> Self {
        CellShape {
            num_vertices: 4,
            faces: vec![
                CellFace { num_vertices: 3, indices: vec![2, 1, 0] },
                CellFace { num_vertices: 3, indices: vec![1, 2, 3] },
                CellFace { num_vertices: 3, indices: vec![0, 3, 2] },
                CellFace { num_vertices: 3, indices: vec![3, 0, 1] },
            ],
            name: CellShapeName::Tetrahedron,
        }
    }

    pub fn prism() -> Self {
        CellShape {
            num_vertices: 6,
            faces: vec![
                CellFace { num_vertices: 3, indices: vec![0, 1, 2] },
                CellFace { num_vertices: 3, indices: vec![5, 4, 3] },
                CellFace { num_vertices: 3, indices: vec![4, 5, 2, 1] },
                CellFace { num_vertices: 3, indices: vec![0, 2, 5, 3] },
                CellFace { num_vertices: 3, indices: vec![3, 4, 1, 0] },
            ],
            name: CellShapeName::Prism,
        }
    }

    pub fn cube() -> Self {
        CellShape {
            num_vertices: 8,
            faces: vec![
                CellFace { num_vertices: 3, indices: vec![3, 2, 1, 0] },
                CellFace { num_vertices: 3, indices: vec![4, 5, 6, 7] },
                CellFace { num_vertices: 3, indices: vec![2, 6, 5, 1] },
                CellFace { num_vertices: 3, indices: vec![4, 7, 3, 0] },
                CellFace { num_vertices: 3, indices: vec![2, 3, 7, 6] },
                CellFace { num_vertices: 3, indices: vec![0, 1, 5, 4] },
            ],
            name: CellShapeName::Cube,
        }
    }
}

impl<'a, T, const DIM: usize> Iterator for FaceIter<'a, T, DIM> {
    type Item = CellFaceRef<'a, T, DIM>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.face_id < self.shape.faces.len() {
            let id = self.face_id;
            self.face_id += 1;
            Some(CellFaceRef {
                id,
                msh: self.mesh,
                shape: self.shape,
                mesh_id: self.mesh_id,
            })
        } else {
          None
        }
    }
}

/// A mesh region consists of a cell shape and an index range. All cells that are defined within the
/// `index_range` in the ibo of the parent mesh can be interpreted as having the specified cell
/// type.
/// Mesh regions can be used to build polymorphic unstructured polygon meshes, among other things.
pub struct MeshRegion {
    pub index_region: Range<usize>,
    pub shape: CellShape,
}

impl MeshRegion {
    /// Returns the number of cells in the mesh region.
    pub fn num_cells(&self) -> usize {
        self.index_region.len() / self.shape.num_vertices
    }

    /// Creates an iterator over the mesh cells within this region of the mesh
    pub fn iter<'a, T, const DIM: usize>(&'a self, mesh: &'a UnstructuredMesh<T, DIM>) -> IntoCellIter<'a, T, DIM> {
        IntoCellIter {
            region: self,
            mesh,
        }
    }
}

pub struct IntoCellIter<'a, T, const DIM: usize> {
    region: &'a MeshRegion,
    mesh: &'a UnstructuredMesh<T, DIM>,
}

impl<'a, T, const DIM: usize> IntoIterator for IntoCellIter<'a, T, DIM> {
    type Item = MeshCell<'a, T, DIM>;
    type IntoIter = CellIter<'a, T, DIM>;

    fn into_iter(self) -> Self::IntoIter {
        CellIter {
            mesh: self.mesh,
            region: self.region,
            idx: 0
        }
    }
}

pub struct CellIter<'a, T, const DIM: usize> {
    mesh: &'a UnstructuredMesh<T, DIM>,
    region: &'a MeshRegion,
    idx: usize,
}

pub struct MeshCell<'a, T, const DIM: usize> {
    mesh_id: usize,
    mesh: &'a UnstructuredMesh<T, DIM>,
    shape: &'a CellShape,
}

impl<'a, T, const DIM: usize> MeshCell<'a, T, DIM> {
    pub fn indices(&self) -> &[usize] {
        &self.mesh.ibo[self.mesh_id..(self.mesh_id + self.shape.num_vertices)]
    }
}

impl<'a, T, const DIM: usize> Iterator for CellIter<'a, T, DIM> {
    type Item = MeshCell<'a, T, DIM>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.region.index_region.end {
            let i = self.idx;
            self.idx += self.region.shape.num_vertices;
            Some(MeshCell {
                mesh_id: i,
                mesh: self.mesh,
                shape: &self.region.shape,
            })
        } else {
            None
        }
    }
}

/// This mesh struct represents an unstructured polygon mesh.
pub struct UnstructuredMesh<T, const DIM: usize> {
    pub vbo: Vec<SVector<T, DIM>>,
    ibo: Vec<usize>,
    pub regions: HashMap<String, MeshRegion>
}

impl<T, const DIM: usize> UnstructuredMesh<T, DIM> {}

struct BuilderRegion {
    indices: Vec<usize>,
    shape: CellShape,
    name: String,
}

impl BuilderRegion {
    /// Replaces all ibo entries that point to `idx` with a reference to `rpl`.
    fn replace_idx(&mut self, idx: usize, rpl: usize) {
        self.indices.iter_mut()
            .filter(|i| **i == idx)
            .for_each(|e| *e = rpl);
    }
}

pub struct UnstructuredMeshBuilder<T, const DIM: usize> {
    vertices: Vec<SVector<T, DIM>>,
    regions: Vec<BuilderRegion>,
}

impl<T, const DIM: usize> UnstructuredMeshBuilder<T, DIM>
where T: Copy + Clone + Scalar + PartialOrd + PartialEq {
    pub fn new() -> Self {
        UnstructuredMeshBuilder {
            vertices: Vec::new(),
            regions: Vec::new(),
        }
    }

    fn get_region(&self, name: &str) -> Option<&BuilderRegion> {
        self.regions.iter().find(|r| r.name == name)
    }

    fn get_region_mut(&mut self, name: &str) -> Option<&mut BuilderRegion> {
        self.regions.iter_mut().find(|r| r.name == name)
    }

    pub fn add_region(
        mut self, name: String, shape: CellShape, vbo: Vec<SVector<T, DIM>>, mut ibo: Vec<usize>
    ) -> Result<Self, ()> {
        if self.get_region(&name).is_some() {
            return Err(());
        }

        ibo.iter_mut().for_each(|i| *i += self.vertices.len());
        vbo.iter().for_each(|&v| self.vertices.push(v));
        self.regions.push(BuilderRegion {
            indices: ibo,
            shape,
            name,
        });
        Ok(self)
    }

    pub fn add_cell(mut self, region: &str, vbo: &[SVector<T, DIM>], ibo: &[usize]) -> Result<Self, ()> {
        let offset = self.vertices.len();
        if let Some(region) = self.get_region_mut(region) {
            if region.shape.num_vertices != vbo.len() {
                return Err(());
            }
            (0..ibo.len()).for_each(|i| region.indices.push(i + offset));
        } else {
            return Err(());
        }
        ibo.iter().for_each(|&i| self.vertices.push(vbo[i]));
        Ok(self)
    }

    /// Replaces all references to index `idx` with reference to index `rpl` in all regions of this
    /// mesh builder.
    fn replace_idx(&mut self, idx: usize, rpl: usize) {
        self.regions.iter_mut()
            .for_each(|r| r.replace_idx(idx, rpl));
    }

    /// Reduces the size of the produced mesh by removing duplicates and unreferenced vertices.
    pub fn aggressive_reduce(mut self) -> Result<Self, ()> {
        let mut vertices = vec![];
        let mut transfer_indices = vec![0; self.vertices.len()];
        // find transfer indices and reduces vertex buffer
        for r in self.regions.iter_mut() {
            for &index in r.indices.iter() {
                let vertex = &self.vertices[index];
                if let Some(idx) = vertices.iter().position(|v| v == vertex) {
                    transfer_indices[index] = idx;
                } else {
                    transfer_indices[index] = vertices.len();
                    vertices.push(*vertex);
                }
            }
        };

        // apply transfer indices and gathered vertices
        transfer_indices.iter().enumerate()
            .for_each(|(idx, &rpl)| self.replace_idx(idx, rpl));
        self.vertices = vertices; // replace vertices
        Ok(self)
    }

    /// Builds the an unstructured mesh from the mesh data contained in this mesh builder.
    pub fn build(self) -> UnstructuredMesh<T, DIM> {
        let mut mesh_indices = Vec::new();
        let mut i = 0;
        let mut mesh_regions = HashMap::new();
        // format regions
        for r in self.regions {
            let BuilderRegion {
                shape,
                mut indices,
                name,
            } = r;

            let start = i;
            i += indices.len();
            mesh_indices.append(&mut indices);
            mesh_regions.insert(name, MeshRegion {
                shape,
                index_region: start..i
            });
        }

        UnstructuredMesh {
            regions: mesh_regions,
            vbo: self.vertices,
            ibo: mesh_indices,
        }
    }
}
