module RunSimulation2d

export runsimulation2d

using Mesh
using Nodes
using FEM
using Element
using TimeStepping
using PyPlot
using Match

function runsimulation2d(meshfile)

    (v_x,v_y,v_z,EToV) = loadmesh(meshfile)

    p_N = "3"
    
    @match p_N begin
        "2"  => tri = p2triangle()
        "3"  => tri = p3triangle()
        "3f" => tri = p3triangle_fekete()
        _ => error("Polynomial order not supported")
    end
    
    (dofindex,x_n,y_n) = buildxy(v_x,v_y,EToV,tri)
    
    # visualize_nodes2d(x_n,y_n)

    bc_array = dirichlet_bc(0.0,2.0,0.0,2.0,x_n,y_n)
    
    ndof = length(x_n)

    (M,Minv) = buildM(tri,v_x,v_y,EToV,dofindex,ndof)
    # set dirichlet bc
    Minv_bc = bc_array.*Minv
    
    Ke = buildK(tri,v_x,v_y,EToV,ndof)
    # println("Ke[1]=$(Ke[1])")
    K = length(Ke)

    element_radius = elementradius2d(v_x,v_y,EToV)
    h_min = minimum(element_radius)

    finaltime = 4 * 2/sqrt(2)
    (un,error) = run_timestepping_rk4(x_n,y_n,M,Minv,Ke,bc_array,dofindex,h_min,tri,finaltime)
    return (h_min,error)
end

# runsimulation2d("/scratch/tetfem/2x2_mesh_1.vtk")

end
