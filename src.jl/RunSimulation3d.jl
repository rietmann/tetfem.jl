module RunSimulation3d

export runsimulation3d

using Mesh
using Nodes
using FEM
using Element
using TimeStepping
using PyPlot
using Match

function runsimulation3d(meshfile)

    (v_x,v_y,v_z,EToV) = loadmesh(meshfile)

    # set polynomial order
    p_N = 3
    println("Setting order $(p_N) polynomials")
    @match p_N begin
        1 => tet = p1tetrahedra()
        2 => tet = p2tetrahedra()
        3 => tet = p3tetrahedra()
        _ => error("Polynomial order not supported")
    end    
    
    (dofindex,x_n,y_n,z_n) = buildxyz(v_x,v_y,v_z,EToV,tet)
    
    bc_array = dirichlet_bc(0.0,2.0,0.0,2.0,0.0,2.0,x_n,y_n,z_n)
    ndof = length(x_n)

    (M,Minv) = buildM(tet,v_x,v_y,v_z,EToV,dofindex,ndof)
    Ke = buildK(tet,v_x,v_y,v_z,EToV,ndof)

    element_radius = elementradius3d(v_x,v_y,v_z,EToV)
    h_min = minimum(element_radius)
    
    finaltime = 2 * 2/sqrt(3)
    @time (un,error) = run_timestepping(x_n,y_n,z_n,Minv,M,Ke,bc_array,dofindex,h_min,tet,finaltime)
    
    return (h_min,error)
    
end

# runsimulation3d("/scratch/tetfem_3d/2x2x2_mesh_0.vtk")

end
