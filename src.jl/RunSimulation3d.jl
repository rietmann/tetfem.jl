module RunSimulation3d

export runsimulation3d, runsimulation3d_elastic

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
    println("Loaded mesh with $(length(EToV)) elements and $(length(x_n)) unique degrees of freedom.")
    
    bc_array = dirichlet_bc(0.0,2.0,0.0,2.0,0.0,2.0,x_n,y_n,z_n)
    ndof = length(x_n)

    (M,Minv) = buildM(tet,v_x,v_y,v_z,EToV,dofindex,ndof)
    Ke = buildK(tet,v_x,v_y,v_z,EToV,ndof)

    element_radius = elementradius3d(v_x,v_y,v_z,EToV)
    h_min = minimum(element_radius)
    h_max = maximum(element_radius)

    finaltime = 2 * 2/sqrt(3)

    @time (un,error) = run_timestepping(x_n,y_n,z_n,Minv,Ke,bc_array,dofindex,h_min,tet,finaltime)
    
    return (h_min,h_max,error)
    
end

# runsimulation3d("/home/rietmann/Dropbox/PostDoc/TetFemJulia/meshes/3D_trelis_2x2x2/mesh_2x2x2_0.vtk")

function runsimulation3d_elastic(meshfile)

    (v_x, v_y, v_z, EToV) = loadmesh(meshfile)

    # set polynomial order
    p_N = 3
    println("Setting order $(p_N) polynomials")
    @match p_N begin
        1 => tet = p1tetrahedra()
        2 => tet = p2tetrahedra()
        3 => tet = p3tetrahedra()
        _ => error("Polynomial order not supported")
    end    
    
    (dofindex, x_n, y_n, z_n) = buildxyz(v_x, v_y, v_z, EToV, tet)
    
    bc = zeros(length(x_n), 3)

    # for pure P-wave in x-direction
    #bc[:, 1] = dirichlet_bc(0.0, 2.0, -1.0, 3.0, -1.0, 3.0, x_n, y_n, z_n)
    #bc[:, 2] = dirichlet_bc(0.0, 2.0,  0.0, 2.0, -1.0, 3.0, x_n, y_n, z_n)
    #bc[:, 3] = dirichlet_bc(0.0, 2.0, -1.0, 3.0,  0.0, 2.0, x_n, y_n, z_n)

    # for pure S-wave in x-direction, polarized in y or z-direction
    bc[:, 1] = dirichlet_bc(0.0, 2.0,  0.0, 2.0,  0.0, 2.0, x_n, y_n, z_n)
    bc[:, 2] = dirichlet_bc(0.0, 2.0, -1.0, 3.0, -1.0, 3.0, x_n, y_n, z_n)
    bc[:, 3] = dirichlet_bc(0.0, 2.0, -1.0, 3.0, -1.0, 3.0, x_n, y_n, z_n)

    ndof = length(x_n)

    (M, Minv) = buildM(tet, v_x, v_y, v_z, EToV, dofindex, ndof)
    # correct for 'measure' vs 'detJ'
    M *= 6.
    Minv /= 6.

    c_prime = build_c_prime(tet, v_x, v_y, v_z, EToV, ndof)

    element_radius = elementradius3d(v_x, v_y, v_z, EToV)
    h_min = minimum(element_radius)
    
    finaltime = 4.
    @time (un,error) = run_timestepping_elastic(x_n, y_n, z_n, Minv, c_prime,
                                                bc, dofindex, h_min, tet,
                                                finaltime)
    
    return (h_min, error)
    
end

end
