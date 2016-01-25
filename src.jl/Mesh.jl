module Mesh

export MeshConstants, loadmesh, meshgen1d, visualizemesh2d, initialize_savepoints2d, savepoints2d, dirichlet_bc, ElementT, initialize_savepoints3d, savepoints3d, testffi_0_4

using Element

using PyPlot
using PyCall

typealias ElementT Triangle

include("TetFemConfig.jl")

immutable MeshConstants{ElementT}
    p_DIM::Int # Dimensions
    
    # x,y,z-coordinates of element vertices
    v_x::Matrix{Float64}
    v_y::Matrix{Float64} 
    v_z::Matrix{Float64} 

    element_radius_h::Vector

    dt::Float64

    # Mass matrix and its inverse;
    # stored as a vector as both are diagonal
    M::Vector{Float64}
    Minv::Vector{Float64}

    
    # Stiffness matrix;
    # stored as vector of (p_Np x p_Np) matrices,
    # where a gather and scatter are required for evaluation on the
    # global degrees of freedom
    Ke::Vector{Matrix{Float64}}

    reference_element::ElementT
    
end

function testffi(blah::AbstractString)

    println("Before ccall")

    vector = Ptr{Cdouble}[0]
    # vector = Array(Ptr{Cdouble}, 1)
    ccall((:test_libvtk,vtklib_loc),Void,(Int32,Ptr{Ptr{Cdouble}}),42,vector)    

    println("First element: $(unsafe_load(vector[],1))")
    println("Second element: $(unsafe_load(vector[],2))")
    
    return vector
    
end

function getenv(var::AbstractString)
    val = ccall((:getenv, "libc"),
                Ptr{UInt8}, (Ptr{UInt8},), var)
    if val == C_NULL
        error("getenv: undefined variable: ", var)
    end
    env = bytestring(val)
    println("$(var) = $(env)")
    return env
end

function meshgen1d(xA :: Float64,xB :: Float64, num_elements :: Int)

    VX = collect(linspace(xA,xB,num_elements+1))
    EToV = Vector{Int}[]
    for k = 1:num_elements
        push!(EToV,[k,k+1])
    end
    return VX,EToV
    
end

function check_element_ordering(v_x,v_y,v_z,EToV)

    for k=1:length(EToV)
        
        v1 = [v_x[EToV[k][1]],v_y[EToV[k][1]],v_z[EToV[k][1]]]
        v2 = [v_x[EToV[k][2]],v_y[EToV[k][2]],v_z[EToV[k][2]]]
        v3 = [v_x[EToV[k][3]],v_y[EToV[k][3]],v_z[EToV[k][3]]]

        v0c = v2-v1;
        v1c = v3-v1;

        
        if cross(v0c,v1c)[3] < 0
            error("element $(k) not in counter-clockwise ordering!")
        end
        
    end
    
end

# Loads a triangular or tetrahedral mesh.  Interfaces with `vtklib`,
# our small library for interfacing with VTK files.
function loadmesh(mesh_file::AbstractString)

    println("Loading $(mesh_file)")

    c_verts_per_element = Cint[0]
    c_num_elements = Cint[0]
    c_num_verts = Cint[0]
    c_v_x = Ptr{Cdouble}[0]
    c_v_y = Ptr{Cdouble}[0]
    c_v_z = Ptr{Cdouble}[0]
    c_EToV = Ptr{Cint}[0]
    println("loading lib $(vtklib_loc)")
    # loadmesh_vtklib() loads the mesh in `mesh_file`, allocates and
    # loads the vertices and element indexes.
    ccall((:loadmesh_vtklib,vtklib_loc),Void,
          (Ptr{UInt8},Ptr{Cint},
           Ptr{Cint},Ptr{Cint},
           Ptr{Ptr{Cdouble}},Ptr{Ptr{Cdouble}},Ptr{Ptr{Cdouble}},
           Ptr{Ptr{Cint}}),
          mesh_file,c_verts_per_element,
          c_num_elements,c_num_verts,
          c_v_x,c_v_y,c_v_z,
          c_EToV)

    # c_verts_per_element points to the value verts_per_element. Thus
    # we need to "dereference" the pointer via `[]` to get its value.
    verts_per_element = c_verts_per_element[]
    num_verts = c_num_verts[]
    num_elements = c_num_elements[]

    v_x = zeros(Float64,num_verts)
    v_y = zeros(Float64,num_verts)
    v_z = zeros(Float64,num_verts)

    # c_v_x points to the C-allocated array where the vertices are
    # stored. `[]` dereferences the pointer allowing us to call
    # `unsafe_load` which accesses each vertex value.
    for i=1:num_verts
        v_x[i] = unsafe_load(c_v_x[],i)
        v_y[i] = unsafe_load(c_v_y[],i)
        if verts_per_element == 4
            v_z[i] = unsafe_load(c_v_z[],i)
        end
    end
    
    EToV = Vector{Int}[]
    # note [v1,v2,v3,v4] + 1 as Julia indexes by 1
    for i=1:num_elements        
        base_index = (i-1)*verts_per_element
        if verts_per_element == 3            
            this_element = [unsafe_load(c_EToV[],base_index+1),
                            unsafe_load(c_EToV[],base_index+2),
                            unsafe_load(c_EToV[],base_index+3)]+1
            push!(EToV,this_element)
        elseif verts_per_element == 4
            this_element = [unsafe_load(c_EToV[],base_index+1),
                            unsafe_load(c_EToV[],base_index+2),
                            unsafe_load(c_EToV[],base_index+3),
                            unsafe_load(c_EToV[],base_index+4)]+1
            push!(EToV,this_element)
        end
    end

    # the variables were allocated by the C-library, and must be
    # deallocated using the same library.
    ccall((:freedbl_vtklib,vtklib_loc),Void,(Ptr{Cdouble},),c_v_x[])
    ccall((:freedbl_vtklib,vtklib_loc),Void,(Ptr{Cdouble},),c_v_y[])
    if verts_per_element == 4
        ccall((:freedbl_vtklib,vtklib_loc),Void,(Ptr{Cdouble},),c_v_z[])
    end
    ccall((:freeint_vtklib,vtklib_loc),Void,(Ptr{Cint},),c_EToV[])
    
    return (v_x,v_y,v_z,EToV)
end

function visualizemesh2d(v_x,v_y,EToV)

    figure(1)
    for i=1:length(EToV)
        e_i = EToV[i]
        x = [v_x[e_i[1]],
             v_x[e_i[2]],
             v_x[e_i[3]],
             v_x[e_i[1]]]
        y = [v_y[e_i[1]],
             v_y[e_i[2]],
             v_y[e_i[3]],
             v_y[e_i[1]]]
        plot(x,y,"k-")
    end
    
end

function dirichlet_bc(x0,x1,y0,y1,x,y)

    tol = 1e-6;
    bc = ones(length(x))
    for i=1:length(bc)
        if abs(x[i]-x0) < tol || abs(x[i] - x1) < tol || abs(y[i] - y0) < tol || abs(y[i] - y1) < tol
            bc[i] = 0.0
        end
    end
    return bc
end

function dirichlet_bc(x0,x1,y0,y1,z0,z1,x,y,z)

    tol = 1e-8;
    bc = ones(length(x))
    for i=1:length(bc)
        if (abs(x[i] - x0) < tol
            || abs(x[i] - x1) < tol
            || abs(y[i] - y0) < tol
            || abs(y[i] - y1) < tol
            || abs(z[i] - z0) < tol
            || abs(z[i] - z1) < tol)
            bc[i] = 0.0
        end
    end
    return bc
end


function test_ref(a)
    a += 1
end

# In order to save the mesh, we need to pass in EToV, which is a
# Vector of Vectors. Index2 allows the C-function to access these
# values, which aren't directly accessible via C.
function index2(vector_vector_ptr::Ptr{Void},i::Cint,j::Cint)
    vector_vector = unsafe_pointer_to_objref(vector_vector_ptr)::Vector{Vector{Int}}
    # print("Hello from index2: vec[$(i)][$(j)]")
    result = vector_vector[i][j]
    # println("=$(result)")
    return convert(Cint,result)
end

function dofindex_func(dofindex_ptr::Ptr{Void},etov_local_ptr::Ptr{Void},k,ki,idxyz)

    println("Hello from dofindex_func")
    dofindex = unsafe_pointer_to_objref(dofindex_ptr)::Vector{Vector{Int}}
    etov_local = unsafe_pointer_to_objref(etov_local_ptr)::Vector{Vector{Int}}    
    myid = 1::Cint
    
    return myid::Cint
end

function initialize_savepoints2d(dofindex::Vector{Vector{Int}},
                                 K::Int,
                                 tri::Triangle,
                                 x_n::Vector{Float64},
                                 y_n::Vector{Float64},
                                 data::Vector{Float64},
                                 output_dir::AbstractString)
    const c_index2 = cfunction(index2,Cint,(Ptr{Void},Cint,Cint))    
    ccall((:initializeSavePoints2d,vtklib_loc),
          Void,
          (Ref{Cdouble},Cstring,Cstring,
           Cint,Cint,
           Ref{Cdouble},Ref{Cdouble},
           Ptr{Void},
           Cint,Ptr{Void},
           Ptr{Void}),
          data,"u(t)",output_dir,length(x_n),K,
          x_n,y_n,
          pointer_from_objref(tri.EToV_local), length(tri.EToV_local),
          pointer_from_objref(dofindex),
          c_index2)

end

function testffi_0_4()

    data = [42.0,2.0,3.0]
    tet = p3tetrahedra()
    const c_index2 = cfunction(index2,Cint,(Ptr{Void},Cint,Cint))    
    ccall((:test,vtklib_loc),
          Void,
          (Ref{Cdouble},Cstring,Cint,Ptr{Void},Ptr{Void}),
          data,"ourdata",length(data),pointer_from_objref(tet.EToV_local),c_index2)
    
end

function initialize_savepoints3d(dofindex::Vector{Vector{Int}},
                                 K::Int,
                                 tet::Tetrahedra,
                                 x_n::Vector{Float64},
                                 y_n::Vector{Float64},
                                 z_n::Vector{Float64},
                                 data::Vector{Float64},
                                 output_dir::AbstractString)    
    
    # c_index2 is a function accessible from C that allows it to
    # access the values in dofindex; a vector of vectors
    const c_index2 = cfunction(index2,Cint,(Ptr{Void},Cint,Cint))    
        
    ccall((:initializeSavePoints3d,vtklib_loc),
          Void,
          (Ref{Cdouble},Cstring,Cstring,
           Cint,Cint,
           Ref{Cdouble},Ref{Cdouble},Ref{Cdouble},
           Ptr{Void},Cint,
           Ptr{Void},
           Ptr{Void}),
          data,"u(t)",output_dir,length(x_n),K,
          x_n,y_n,z_n,
          pointer_from_objref(tet.EToV_local), length(tet.EToV_local),
          pointer_from_objref(dofindex),
          c_index2)
    
end

function savepoints2d(data,step,output_dir)

    ccall((:savePoints2d,vtklib_loc),
          Void,
          (Ptr{Cdouble},Cint,Cstring,Cint),
          data,length(data),output_dir,step)    
    
end

function savepoints3d(data,step,output_dir)

    ccall((:savePoints3d,vtklib_loc),
          Void,
          (Ptr{Cdouble},Cint,Cstring,Cint),
          data,length(data),output_dir,step)
    
end




end


