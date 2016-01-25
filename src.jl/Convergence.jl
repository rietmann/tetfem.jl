module Convergence

using SymPy
using JLD
using Element
using ProgressMeter
using PyPlot
using RunSimulation2d
using RunSimulation3d

function convergence_test2d()
    meshes = ["/home/rietmann/Dropbox/PostDoc/TetFem/mesh/meshgrid2d_1.vtk",
              "/home/rietmann/Dropbox/PostDoc/TetFem/mesh/meshgrid2d_2.vtk",
              "/home/rietmann/Dropbox/PostDoc/TetFem/mesh/meshgrid2d_3.vtk",
              "/home/rietmann/Dropbox/PostDoc/TetFem/mesh/meshgrid2d_4.vtk"
              ]
        
    h_min = zeros(length(meshes))
    error_inf = zeros(length(meshes))

    for (i,mesh) in enumerate(meshes)

        (h,error) = runsimulation(mesh)
        error_inf[i] = error
        h_min[i] = h
        println("Finished $(i) of $(length(meshes))")
    end

    println("hmin = $(h_min)\nerror = $(error_inf)")
    loglog(h_min,error_inf,"g*-",h_min,h_min.^4,"k--")

end


function convergence_test3d()

    meshes = ["/scratch/tetfem_3d/2x2x2_mesh_00.vtk",
              "/scratch/tetfem_3d/2x2x2_mesh_0.vtk",
              "/scratch/tetfem_3d/2x2x2_mesh_1.vtk",
              "/scratch/tetfem_3d/2x2x2_mesh_2.vtk"]
    # "/scratch/tetfem_3d/2x2x2_mesh_2.vtk"] 
    # "/scratch/tetfem_3d/2x2x2_mesh_2.vtk"] # 46 minutes T=2.3
              # "/scratch/tetfem_3d/2x2x2_mesh_3.vtk"
              
    h_min = zeros(length(meshes))
    error_inf = zeros(length(meshes))

    for (i,mesh) in enumerate(meshes)

        (h,error) = runsimulation3d(mesh)
        error_inf[i] = error
        h_min[i] = h
        println("Finished $(i) of $(length(meshes))")
    end

    println("hmin = $(h_min)\nerror = $(error_inf)")
    loglog(h_min,error_inf,"g*-",h_min,h_min.^4,"k--",h_min,5e-2*h_min,"k--")
    
end


end
