# Cell Cluster Implementation Overview

This module implements deformable cell clusters for ESPResSo/object-in-fluid simulations. A cluster is represented as a collection of `OifCell` objects, where each cell is an elastic mesh-based object created from an `OifCellType`. The cluster layer adds initialization of multi-cell geometries, cell-cell interactions, deformation/relaxation procedures, contact-area analysis, VTK output, and JSON-based save/load support. :contentReference[oaicite:0]{index=0}

## Acknowledgement
This code is part of the work funded by the EU NextGenerationEU through the Recovery and Resilience Plan for Slovakia under the project 09I03-03-V04-00705.

## Core design

The central class is `OifCluster`. It stores a cluster name, a list of cells, the initial cell positions, the expected number of contact areas, and parameters for the currently active cell-cell interactions. When initialized with existing cells, the base class checks that all cells are spherical and have the same radius. Empty clusters are also supported and can be populated later.

Cluster-level geometry is handled through cell origins rather than through a single merged mesh. Methods such as `get_origin()`, `set_origin()`, `set_velocity()`, `get_mean_velocity()`, `rotate()`, and `pos_bounds()` operate by looping over the member cells. The total number of mesh nodes is obtained by summing the node counts of all cells. :contentReference[oaicite:1]{index=1}

## Built-in cluster geometries

Several subclasses create predefined cluster shapes by computing initial cell centers and then calling the base `_create_cells()` method:

- `OifBiCluster`: two-cell cluster with one contact area.
- `OifL3Cluster`: three-cell L-shaped cluster.
- `OifL4Cluster`: four-cell L-shaped cluster.
- `OifSquareCluster`: four cells arranged in a square.
- `OifTetraCluster`: four cells arranged as a tetrahedron.
- `OifStarCluster`: seven-cell star-like cluster.
- `OifDiamondCluster`: five-cell diamond-like cluster.
- `OifChainCluster`: a random chain with fixed radius cells.
- `OifVarChainCluster`: a chain with variable cell radii/cell types.

All predefined fixed-radius clusters require spherical cell meshes. Most geometries use the cell radius from `cell_type.resize[0]` and a user-specified `space` parameter to place cells initially separated by approximately `2 * radius + space`. :contentReference[oaicite:2]{index=2}

## Solid cluster mesh files

The `solid-clusters.zip` archive contains pre-generated mesh files for several cell-cluster geometries. Each mesh is stored as a pair of plain text files:

- `*nodes.dat`: one 3D point per line, stored as `x y z`.
- `*triangles.dat`: one triangular face per line, stored as three node indices.

Despite the “solid cluster” naming, these files are surface triangular meshes rather than volumetric tetrahedral-element meshes: the connectivity files contain triples of node IDs, not quadruples. They are therefore suitable for representing the external surface of a compact cluster shape in the same mesh format used by `OifCellType`/`Mesh`, where nodes and triangular facets define an elastic object.

The archive includes mesh resolutions for several predefined cluster shapes:

| Shape family | Available variants |
|---|---|
| Two-cell cluster | thicker: 142, 325 nodes; thinner: 151, 284 nodes |
| Three-in-line cluster | thicker: 210/215-named, 437 nodes; thinner: 222/227-named, 335/340-named nodes |
| Four-in-line cluster | thicker: 386, 680 nodes; thinner: 287, 517 nodes |
| Tetrahedral/perpendicular cluster | 389 and 561 nodes |

The tetrahedral/perpendicular cluster files are:

- `tetra-perp-solid-cluster-389nodes.dat`
- `tetra-perp-solid-cluster-389triangles.dat`
- `tetra-perp-solid-cluster-561nodes.dat`
- `tetra-perp-solid-cluster-561triangles.dat`

These meshes describe a compact four-cell-like cluster with three outer lobes arranged approximately around one central/perpendicular direction. The first four nodes in the 389-node version define extremal anchor-like positions near `(0, 0.6, 0)`, `(1.6, 0, 0)`, `(0, 0, 1.6)`, and `(0, -1.6, 0)`, giving the mesh an asymmetric tetrahedral/perpendicular layout. The 389-node version has 2,621 triangular facets, while the 561-node version has 4,138 triangular facets, providing a finer surface discretization.

These solid-cluster meshes can be used when the desired object is a single deformable cluster-shaped body rather than a cluster assembled from several separate `OifCell` objects. In contrast, the `OifCluster` classes build clusters as groups of individual cells coupled by non-bonded cell-cell interactions, adhesion, collision, deformation, and contact-area analysis. :contentReference[oaicite:0]{index=0}


## Cell-cell and boundary interactions

`OifCluster` supports several interaction types between cells:

- Lennard-Jones adhesion via `set_lennard_jones_interactions()`.
- Morse adhesion via `set_morse_interactions()`.
- Soft-sphere repulsion via `set_soft_sphere_interactions()`.
- Self-cell soft-sphere interaction via `set_self_cell_soft_sphere_interactions()`.
- Membrane collision interaction via `set_membrane_collision_interactions()`.
- Boundary repulsion via `set_cell_boundary_interactions()`.

Interaction parameters are stored on the cluster object and passed to ESPResSo through `system.non_bonded_inter[...]`. Adhesive interactions are also used by the contact-analysis and bond-visualization routines to determine which mesh nodes are close enough to be considered in contact.

## Deformation and contact formation

The generic `deform()` method drives cells toward the cluster centroid by applying external forces to all mesh nodes of each cell. The simulation is then advanced for a number of deformation cycles, followed by relaxation cycles. During deformation and relaxation, optional VTK output can be written for each cell, together with line files showing current cell-cell bonds.

After deformation, the method compares the current number of detected contact areas with the expected `n_contact_areas`. If not enough contacts have formed, it raises an error, indicating that the deformation time was too short. `OifChainCluster` overrides this procedure and deforms the chain sequentially, bonding neighboring cells one pair at a time.

## Contact-area analysis and visualization

The cluster can estimate contact areas from mesh-node proximity. `count_current_contact_areas()` checks whether any mesh nodes of a cell pair lie within the active adhesive cutoff. `color_contact_areas_vtk()` marks contacting mesh nodes in VTK output and estimates an equivalent contact radius from the number of contacting nodes and the cell surface area. `output_vtk_cell_bonds()` writes line segments between close node pairs, useful for visualizing adhesive contacts.

## Save/load support

Clusters can be saved with `save_cluster()`. The method writes a `data.json` file containing the cluster name, origin, expected contact count, cell definitions, mesh positions, cell-type parameters, and optional interaction parameters. It also copies the required node and triangle mesh files into `nodes_files/` and `triangles_files/`. Optional inner particles can be saved per cell.

Clusters are restored with `load_cluster()`, which reconstructs cell types, creates the cells, restores mesh-node positions, resets the cluster origin, reloads stored interactions, and optionally loads inner particles. :contentReference[oaicite:3]{index=3}

## Notes and assumptions

This implementation treats a cell cluster as a group of separate deformable cells coupled by non-bonded adhesion/collision interactions, not as one fused mesh. Built-in cluster constructors assume spherical cells; variable-radius chains are handled separately by `OifVarChainCluster`. The implementation supports permanent-like adhesive contact formation for simulation setup and analysis, but it does not explicitly model biological processes such as dynamic adhesion remodeling, cell division, or active rearrangement inside a cluster.
