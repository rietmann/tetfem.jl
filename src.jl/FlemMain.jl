module FlemMain

using RunSimulation2d
using RunSimulation3d

# runsimulation2d("/home/rietmann/Dropbox/PostDoc/TetFem/mesh/meshgrid2d_1.vtk")

(h_min,error) = runsimulation2d("/scratch/tetfem/2x2_mesh_1.vtk")

# runsimulation3d("/scratch/tetfem_3d/2x2x2_mesh_minimal.vtk")
# (h_min,error) = runsimulation3d("/scratch/tetfem_3d/2x2x2_mesh_0.vtk")
# runsimulation3d_alt("/scratch/tetfem_3d/2x2x2_mesh_0.vtk")

end
