module Nodes

export buildx, buildxy, buildxyz, visualize_nodes2d, plotElements

using Element
using PyPlot
using BuildDOFIndex


using PyCall
unshift!(PyVector(pyimport("sys")["path"]), "")
@pyimport global_numbering

function rstoxy(v1,v2,v3,r,s)

    # Hesthaven & Warburton pg. 172
    xy = -(r+s)/2.0 * v1 +
    (r+1)/2 * v2 +
    (s+1)/2 * v3 

    x = xy[1]
    y = xy[2]

    return (x,y)
    
end

# Hesthaven & Warburton Version
function rst_toxyz_hest(v1,v2,v3,v4,r,s,t)

	xyz = (-(1+r+s+t)*v1 + 
	           (1+r)*v2+
	           (1+s)*v3+
	 	   (1+t)*v4)
    x = xyz[1]
    y = xyz[2]
    z = xyz[3]

    return (x,y,z)
 
	
end


function rst_toxyz(v1,v2,v3,v4,r,s,t)

    # see "rstoxyz.py"
    xyz = ((1-r-s-t)*v1 +
           r*v2 +
           s*v3 +
           t*v4
           )
    
    x = xyz[1]
    y = xyz[2]
    z = xyz[3]

    return (x,y,z)
    
end


function build_dofindex_python(xy,p_Np,K)
    # python version from martin
    (dofindex_linear,ndof) = global_numbering.get_global_tree(xy)
    dofindex = Vector{Int}[]
    for k=1:K
        push!(dofindex,Int[])
        for i=1:p_Np
            # +1 for julia's 1-indexing
            push!(dofindex[k], dofindex_linear[(k-1)*p_Np+i]+1)
        end
    end
    return (dofindex,ndof)
end

function rstoxy(v1,v2,v3,r,s)

    # Hesthaven & Warburton pg. 172
    xy = -(r+s)/2.0 * v1 +
    (r+1)/2 * v2 +
    (s+1)/2 * v3 

    x = xy[1]
    y = xy[2]

    return (x,y)
    
end


function rtox(v1,v2,ri)

    x = (ri+1)/2*(v2-v1)+v1
    
end

function buildx(v_x,EToV,elem)

    p_Np = length(elem.r)
    K = size(EToV)[1]
    x_all = zeros(p_Np,K)
    for k=1:K
        v1 = v_x[EToV[k][1]]
        v2 = v_x[EToV[k][2]]
        for i=1:p_Np
            x = rtox(v1,v2,elem.r[i])
            x_all[i,k] = x
        end
    end

    x_n = zeros(K*(p_Np-1)+1)
    counter = 1
    dofindex = Vector{Int}[]
    for k=1:K
        push!(dofindex,zeros(p_Np))
        for i=1:(p_Np-1)            
            x_n[counter] = x_all[i,k]
            dofindex[k][i] = counter
            counter += 1            
        end
        dofindex[k][p_Np] = counter
    end
    x_n[counter] = x_all[end,end]
    return (dofindex,x_n)
    
end
function buildxy(v_x,v_y,EToV,tri)    
    
    p_Np = length(tri.r)
    K = length(EToV)
    x_all = zeros(p_Np,K)
    y_all = zeros(p_Np,K)
    xy = zeros(2,p_Np*K)
    for k=1:K
        v1 = [v_x[EToV[k][1]],v_y[EToV[k][1]]]
        v2 = [v_x[EToV[k][2]],v_y[EToV[k][2]]]
        v3 = [v_x[EToV[k][3]],v_y[EToV[k][3]]]
        for i=1:p_Np
            (x,y) = rstoxy(v1,v2,v3,tri.r[i],tri.s[i])
            x_all[i,k] = x
            y_all[i,k] = y
            xy[1,(k-1)*p_Np+i] = x
            xy[2,(k-1)*p_Np+i] = y
        end        
    end

    # julia version
    (dofindex,ndof) = build_dofindex(x_all,y_all)
    
    # julia version is ~70x faster!
    # @time build_dofindex(x_all,y_all)
    # # python version
    # (dofindex2,ndof2) = build_dofindex_python(xy,p_Np,K)
    # @time build_dofindex_python(xy,p_Np,K)
    
    # if dofindex == dofindex2 && ndof == ndof2
    #     println("DOF methods are the same!")
    # end
    
    x_n = zeros(ndof)
    y_n = zeros(ndof)
    for k=1:K
        for i=1:p_Np
            x_n[dofindex[k][i]] = x_all[i,k]
            y_n[dofindex[k][i]] = y_all[i,k]
        end
    end
    return (dofindex,x_n,y_n)
    
end

function plotElement(tet)
    
    # visualize element
    figure(1)
    plot3D(tet.r,tet.s,tet.t,"ko")
    for k=1:length(tet.EToV_local)        
        p1 = tet.EToV_local[k][1]
        p2 = tet.EToV_local[k][2]
        p3 = tet.EToV_local[k][3]
        p4 = tet.EToV_local[k][4]        
        inner_element_r = [tet.r[p1],tet.r[p2],tet.r[p3],tet.r[p4],tet.r[p1],tet.r[p3],tet.r[p2],tet.r[p4]]
        inner_element_s = [tet.s[p1],tet.s[p2],tet.s[p3],tet.s[p4],tet.s[p1],tet.s[p3],tet.s[p2],tet.s[p4]]
        inner_element_t = [tet.t[p1],tet.t[p2],tet.t[p3],tet.t[p4],tet.t[p1],tet.t[p3],tet.t[p2],tet.t[p4]]        
        plot3D(inner_element_r,inner_element_s,inner_element_t,"k-")
    end    
    
    # f = open("reference_element_tetrahedralization.vtk","w")
    # write(f,"# vtk DataFile Version 2.0\nMesh\nASCII\nDATASET UNSTRUCTURED_GRID\n")
    # write(f,"POINTS $(length(tet.r)) double\n")
    # for n=1:length(tet.r)
    #     write(f,"$(tet.r[n]) $(tet.s[n]) $(tet.t[n])\n")
    # end
    # write(f,"CELLS $(length(tet.EToV_local)) $(length(tet.EToV_local)*5)\n")
    # for k=1:length(tet.EToV_local)
    #     write(f,"4 $(tet.EToV_local[k][1]-1) $(tet.EToV_local[k][2]-1) $(tet.EToV_local[k][3]-1) $(tet.EToV_local[k][4]-1)\n")
    # end
    # write(f,"CELL_TYPES $(length(tet.EToV_local))\n")
    # for k=1:length(tet.EToV_local)
    #     write(f,"10 ")
    # end
    # write(f,"\n")
    # close(f)

    figure(2)
    clf()
    pts_r = Float64[]
    pts_s = Float64[]
    pts_t = Float64[]
    tol = 0.01
    for n=1:length(tet.r)
        if 1 - (tet.r[n]+tet.s[n]+tet.t[n]) < tol
            push!(pts_r,tet.r[n])
            push!(pts_s,tet.s[n])
            push!(pts_t,tet.t[n])
        end
    end
    plot3D(pts_r,pts_s,pts_t,"ko-")
    plot3D([1.0, 0.0, 0.0,1.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0])
end

function plotElements(v_x,v_y,v_z,EToV,x_n,y_n,z_n)

    K = length(EToV)
    for k=1:K
        v1 = [v_x[EToV[k][1]],v_y[EToV[k][1]],v_z[EToV[k][1]]]
        v2 = [v_x[EToV[k][2]],v_y[EToV[k][2]],v_z[EToV[k][2]]]
        v3 = [v_x[EToV[k][3]],v_y[EToV[k][3]],v_z[EToV[k][3]]]
        v4 = [v_x[EToV[k][4]],v_y[EToV[k][4]],v_z[EToV[k][4]]]
        
        element_r = [v1[1],v2[1],v3[1],v4[1],v1[1],v3[1],v2[1],v4[1]]
        element_s = [v1[2],v2[2],v3[2],v4[2],v1[2],v3[2],v2[2],v4[2]]
        element_t = [v1[3],v2[3],v3[3],v4[3],v1[3],v3[3],v2[3],v4[3]]
        plot3D(element_r,element_s,element_t,"k-")
    end
    plot3D(x_n,y_n,z_n,"ro")
    
end

function buildxyz(v_x,v_y,v_z,EToV,tet)
    

    # plotElement(tet)
    # error("stop after plot")
    p_Np = length(tet.r)
    K = length(EToV)
    x_all = zeros(Float64,p_Np,K)
    y_all = zeros(Float64,p_Np,K)
    z_all = zeros(Float64,p_Np,K)
    xyz = zeros(3,p_Np*K)
    for k=1:K
        v1 = [v_x[EToV[k][1]],v_y[EToV[k][1]],v_z[EToV[k][1]]]
        v2 = [v_x[EToV[k][2]],v_y[EToV[k][2]],v_z[EToV[k][2]]]
        v3 = [v_x[EToV[k][3]],v_y[EToV[k][3]],v_z[EToV[k][3]]]
        v4 = [v_x[EToV[k][4]],v_y[EToV[k][4]],v_z[EToV[k][4]]]
        for i=1:p_Np
            (x,y,z) = rst_toxyz(v1,v2,v3,v4,tet.r[i],tet.s[i],tet.t[i])
            x_all[i,k] = x
            y_all[i,k] = y
            z_all[i,k] = z
            # xyz[1,(k-1)*p_Np+i] = x
            # xyz[2,(k-1)*p_Np+i] = y
            # xyz[3,(k-1)*p_Np+i] = z
        end        
    end
    
    @time (dofindex,ndof) = build_dofindex(x_all,y_all,z_all)

    # # python version
    # @time (dofindex2,ndof2) = build_dofindex_python(xyz,p_Np,K)
    
    # if ndof != ndof2
    # error("Julia ndof != Python ndof")
    # end
    
    x_n = zeros(Float64,ndof)
    y_n = zeros(Float64,ndof)
    z_n = zeros(Float64,ndof)
    for k=1:K
        for i=1:p_Np
            # test python vs. julia
            # if dofindex[k][i] != dofindex2[k][i]
            # error("dofindex_julia != dofindex_python")
            # end
            
            x_n[dofindex[k][i]] = x_all[i,k]
            y_n[dofindex[k][i]] = y_all[i,k]
            z_n[dofindex[k][i]] = z_all[i,k]
        end
    end
    return (dofindex,x_n,y_n,z_n)
    
end


function visualize_nodes2d(x::Vector{Float64},y::Vector{Float64})
    plot(x,y,"ro")
end

function visualize_nodes2d(x::Matrix{Float64},y::Matrix{Float64})
    for k=1:size(x)[2]
        plot(x[:,k],y[:,k],"ko")
    end
end

end
