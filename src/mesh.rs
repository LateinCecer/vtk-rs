use std::collections::HashMap;
use std::ops::Range;
use nalgebra::SVector;

#[derive(Clone, PartialOrd, PartialEq)]
pub struct CellShape {
    pub num_vertices: usize,
    faces: Vec<CellFace>,
}

#[derive(Clone, PartialOrd, PartialEq)]
struct CellFace {
    num_vertices: usize,
    indices: Vec<usize>,
}

pub struct CellFaceRef<'a, const DIM: usize> {
    msh: &'a UnstructuredMesh<DIM>,
    shape: &'a CellShape,
    id: usize,
    mesh_id: usize,
}

impl<'a, const DIM: usize> CellFaceRef<'a, DIM> {
    /// Returns the indices for this face as a vector
    pub fn indices(&self) -> Vec<usize> {
        self.shape.faces[self.id].indices.iter()
            .map(|&i| self.msh.ibo[i + self.mesh_id])
            .collect()
    }

    /// Returns the vertices of this face as a vector
    pub fn vertices(&self) -> Vec<&'a SVector<f64, DIM>> {
        self.indices().iter()
            .map(|&i| &self.msh.vbo[i])
            .collect()
    }
}

pub struct FaceIter<'a, const DIM: usize> {
    shape: &'a CellShape,
    mesh_id: usize,
    mesh: &'a UnstructuredMesh<DIM>,
    face_id: usize,
}

impl CellShape {
    /// Creates an iterator that can be used to iterate over the cells faces. This is especially
    /// useful for converting the mesh into other formats.
    pub fn face_iter<'a, const DIM: usize>(
        &'a self, msh: &'a UnstructuredMesh<DIM>, mesh_id: usize
    ) -> FaceIter<'_, DIM> {
        FaceIter {
            shape: self,
            mesh_id,
            mesh: msh,
            face_id: 0,
        }
    }

    pub fn triangle() -> Self {
        CellShape {
            num_vertices: 3,
            faces: vec![
                CellFace { num_vertices: 2, indices: vec![0, 1] },
                CellFace { num_vertices: 2, indices: vec![1, 2] },
                CellFace { num_vertices: 2, indices: vec![2, 0] },
            ]
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
            ]
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
            ]
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
            ]
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
            ]
        }
    }
}

impl<'a, const DIM: usize> Iterator for FaceIter<'a, DIM> {
    type Item = CellFaceRef<'a, DIM>;

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
    pub fn iter<'a, const DIM: usize>(&'a self, mesh: &'a UnstructuredMesh<DIM>) -> IntoCellIter<'a, DIM> {
        IntoCellIter {
            region: self,
            mesh,
        }
    }
}

pub struct IntoCellIter<'a, const DIM: usize> {
    region: &'a MeshRegion,
    mesh: &'a UnstructuredMesh<DIM>,
}

impl<'a, const DIM: usize> IntoIterator for IntoCellIter<'a, DIM> {
    type Item = MeshCell<'a, DIM>;
    type IntoIter = CellIter<'a, DIM>;

    fn into_iter(self) -> Self::IntoIter {
        CellIter {
            mesh: self.mesh,
            region: self.region,
            idx: 0
        }
    }
}

pub struct CellIter<'a, const DIM: usize> {
    mesh: &'a UnstructuredMesh<DIM>,
    region: &'a MeshRegion,
    idx: usize,
}

pub struct MeshCell<'a, const DIM: usize> {
    mesh_id: usize,
    mesh: &'a UnstructuredMesh<DIM>,
    shape: &'a CellShape,
}

impl<'a, const DIM: usize> Iterator for CellIter<'a, DIM> {
    type Item = MeshCell<'a, DIM>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.region.num_cells() {
            let i = self.idx;
            self.idx += 1;
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
pub struct UnstructuredMesh<const DIM: usize> {
    vbo: Vec<SVector<f64, DIM>>,
    ibo: Vec<usize>,
    pub regions: HashMap<String, MeshRegion>
}

impl<const DIM: usize> UnstructuredMesh<DIM> {}

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

pub struct UnstructuredMeshBuilder<const DIM: usize> {
    vertices: Vec<SVector<f64, DIM>>,
    regions: Vec<BuilderRegion>,
}

impl<const DIM: usize> UnstructuredMeshBuilder<DIM> {
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
        mut self, name: String, shape: CellShape, vbo: Vec<SVector<f64, DIM>>, mut ibo: Vec<usize>
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

    pub fn add_cell(mut self, region: &str, vbo: &[SVector<f64, DIM>], ibo: &[usize]) -> Result<Self, ()> {
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
    pub fn build(self) -> UnstructuredMesh<DIM> {
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
