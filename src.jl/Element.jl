module Element

using PyCall
using TetrahedraConstants
using JLD
using ComputeP3_alternate
using SEM
using SymPy

import Fekete

export Element1D, Triangle, Tetrahedra, p2triangle, p3triangle, p3triangle_fekete, elementradius1d, elementradius2d, elementradius3d, elementradius3d_viajacobian, p3tetrahedra, p1tetrahedra, p2tetrahedra, p3tetrahedra_noquad, p1element1d, p2element1d, p3element1d, p4element1d

@pyimport numpy as np

immutable Element1D
    p_N::Int64 # Polynomial order
    p_Np::Int64 # Number of points per element

    r::Vector{Float64}

    quadrature_weights::Vector{Float64}

    cfl_factor::Float64

    # Stiffness matrix constant derivative terms for (r) reference element ([-1,1])
    d_Phi_rr::Matrix{Float64}

    d_Phi_n::Matrix{Float64}

    diffX::Matrix{Float64}

end

immutable Triangle
    p_N::Int64 # Polynomial order
    p_Np::Int64 # Number of points per element

    r::Vector{Float64}
    s::Vector{Float64}
    
    quadrature_weights::Vector{Float64}

    cfl_factor::Float64

    # Stiffness matrix constant derivative terms for (r,s) reference triangle ([-1,1],[-1,1])
    d_Phi_rr::Matrix{Float64}
    d_Phi_rs::Matrix{Float64}
    d_Phi_sr::Matrix{Float64}
    d_Phi_ss::Matrix{Float64}

    # a triangularization of the local nodes within the element
    EToV_local :: Vector{Vector{Int}}
end

immutable Tetrahedra
    p_N::Int # Polynomial order
    p_Np::Int # Number of points per element

    r::Vector{Float64}
    s::Vector{Float64}
    t::Vector{Float64}
    
    quadrature_weights::Vector{Float64}

    cfl_factor::Float64
    
    # Stiffness matrix constant derivative terms for (r,s,t) reference tetrahedra ([0,1],[0,1],[0,1])
    d_Phi_rr::Matrix{Float64}
    d_Phi_rs::Matrix{Float64}
    d_Phi_sr::Matrix{Float64}
    d_Phi_rt::Matrix{Float64}
    d_Phi_tr::Matrix{Float64}
    d_Phi_ss::Matrix{Float64}
    d_Phi_st::Matrix{Float64}
    d_Phi_ts::Matrix{Float64}
    d_Phi_tt::Matrix{Float64}

    # a tetrahedralization of the local nodes within the element
    EToV_local::Vector{Vector{Int}}    
    
end

function p1element1d()

    r = [-1.0,1.0]
    quadrature_weights = deriveGaussQuadrature1d(r)

    d_Phi_rr = stiffness1d(r)

    cfl_factor = 0.5 # 1.0 should be limit
    
    return Element1D(1,length(r),r,quadrature_weights,
                     cfl_factor,d_Phi_rr,precompute_dphi_matrix(r),buildDerivativeMatrix(r))
end

function p2element1d()

    r = [-1.0,0.0,1.0]
    quadrature_weights = deriveGaussQuadrature1d(r)

    d_Phi_rr = stiffness1d(r)

    cfl_factor = 0.55

    return Element1D(1,length(r),r,quadrature_weights,
                     cfl_factor,d_Phi_rr,precompute_dphi_matrix(r),buildDerivativeMatrix(r))
end

function p3element1d()

    r = [-1.0000,-0.447213595499958,0.447213595499958,1.0000]
    quadrature_weights = deriveGaussQuadrature1d(r)

    d_Phi_rr = stiffness1d(r)

    cfl_factor = 0.33

    return Element1D(1,length(r),r,quadrature_weights,
                     cfl_factor,d_Phi_rr,precompute_dphi_matrix(r),buildDerivativeMatrix(r))
end

function p4element1d()

    r = [-1.000000000000000,  -0.654653670707977,  -0.000000000000000,   0.654653670707977, 1.0]
    quadrature_weights = deriveGaussQuadrature1d(r)

    d_Phi_rr = stiffness1d(r)

    cfl_factor = 0.2

    return Element1D(1,length(r),r,quadrature_weights,
                     cfl_factor,d_Phi_rr,precompute_dphi_matrix(r),buildDerivativeMatrix(r))
end

# 2=>[-1.,0.,1.], 
#                   3=>[-1.0000;-0.447213595499958;0.447213595499958;1.0000], 
#                   4=>[-1.000000000000000;  -0.654653670707977;  -0.000000000000000;   0.654653670707977; 1.0]}

function p2triangle()
    
    r = [-1.0,-1.0,1.0,-1.0,0.0,0.0,-0.3333333333333333]
    s = [-1.0,1.0,-1.0,0.0,0.0,-1.0,-0.3333333333333333]
    cfl_factor = 0.6
    ws = 1.0/20.0
    we = 2.0/15.0
    wg = 9.0/20.0    
    quadrature_weights = [ws,ws,ws,we,we,we,wg]

    K_p2 = np.load("K_p2triangle.npz")
    d_Phi_rr = get(K_p2,"Krr")
    d_Phi_rs = get(K_p2,"Krs")
    d_Phi_sr = get(K_p2,"Ksr")
    d_Phi_ss = get(K_p2,"Kss")
    
    EToV_local = Vector{Int}[]
    # +1 for julia indexing
    push!(EToV_local,[0,5,6]+1)
    push!(EToV_local,[5,2,4]+1)
    push!(EToV_local,[5,4,6]+1)
    push!(EToV_local,[0,6,3]+1)
    push!(EToV_local,[3,6,4]+1)
    push!(EToV_local,[1,3,4]+1)

    return Triangle(2,length(r),r,s,quadrature_weights,cfl_factor,
                    d_Phi_rr,d_Phi_rs,d_Phi_sr,d_Phi_ss,
                    EToV_local)
    
end

function p3triangle()
    
    ws = 2*( 919*sqrt(7) + 2471 )/(124080*sqrt(7) + 330960)
    wa = 2*( sqrt(7)*(2+sqrt(7))^4 )/( 25280 + 9520*sqrt(7) )
    wb = 2*( 147 + 42*sqrt(7) ) / ( 400*sqrt(7) + 1280 )

    quadrature_weights = [ws,ws,ws,
                          wa,wa,wa,wa,wa,wa,
                          wb,wb,wb]    
        
    K_p3 = np.load("K_p3triangle.npz")
    # K_p3 = np.load("Kquad_p3triangle.npz")
    d_Phi_rr = get(K_p3,"Krr")
    d_Phi_rs = get(K_p3,"Krs")
    d_Phi_sr = get(K_p3,"Ksr")
    d_Phi_ss = get(K_p3,"Kss")
    rs = [(-1.0,-1.0),(-1.0,1.0),(1.0,-1.0), # S_{1,2,3}
          (-1.0,0.413),(-1.0,-0.413),(-0.413,0.413),(0.413,-0.413),(-0.413,-1.0),(0.413,-1.0), # M_{(123,123)}
          (-0.5853,0.1706),(-0.5853,-0.5853),(0.1706,-0.5853)]
    r = zeros(length(rs))
    s = zeros(length(rs))    
    for i=1:length(r)
        r[i] = rs[i][1]
        s[i] = rs[i][2]
    end
    cfl_factor = 0.322
    
    EToV_local = Vector{Int}[]
    # +1 for julia indexing
    push!(EToV_local,[0,7,10 ]+1)
    push!(EToV_local,[7,11,10]+1)
    push!(EToV_local,[7,8,11 ]+1)
    push!(EToV_local,[8,2,11 ]+1)
    push!(EToV_local,[0,10,4 ]+1)
    push!(EToV_local,[4,10,9 ]+1)
    push!(EToV_local,[10,11,9]+1)
    push!(EToV_local,[11,6,9 ]+1)
    push!(EToV_local,[11,2,6 ]+1)
    push!(EToV_local,[9,6,5  ]+1)
    push!(EToV_local,[4,9,3  ]+1)
    push!(EToV_local,[3,9,1  ]+1)
    push!(EToV_local,[9,5,1  ]+1)        
    
    return Triangle(3,length(r),r,s,quadrature_weights,cfl_factor,
                    d_Phi_rr,d_Phi_rs,d_Phi_sr,d_Phi_ss,
                    EToV_local)
    
end

function p3triangle_fekete()

    
    cfl_factor = 0.322
    EToV_local = Vector{Int}[]
    push!(EToV_local,[1,4,10])
    push!(EToV_local,[1,10,9])
    push!(EToV_local,[4,5,6])
    push!(EToV_local,[4,6,10])
    push!(EToV_local,[5,2,6])
    push!(EToV_local,[10,6,7])
    push!(EToV_local,[9,10,7])
    push!(EToV_local,[9,7,8])
    push!(EToV_local,[8,7,3])
    
    if !isfile("p3fekete_constants.jld")
        
        rs = Fekete.buildFeketePtsP3()
        (rn,sn) = rs
        wi = Fekete.buildFeketeWeights(rs)
        (d_Phi_rr,d_Phi_rs,d_Phi_ss,d_Phi_sr) = Fekete.buildReferenceMatrices(rs)
        jldopen("p3fekete_constants.jld","w") do file
            write(file,"rn",rn)
            write(file,"sn",sn)
            write(file,"wi",wi)
            write(file,"d_Phi_rr",d_Phi_rr)
            write(file,"d_Phi_rs",d_Phi_rs)
            write(file,"d_Phi_ss",d_Phi_ss)
            write(file,"d_Phi_sr",d_Phi_sr)        
        end
        return Triangle(3,length(rn),rn,sn,wi,cfl_factor,
                        d_Phi_rr,d_Phi_rs,d_Phi_sr,d_Phi_ss,
                        EToV_local)

    else
        jldopen("p3fekete_constants.jld") do file
            rn = read(file,"rn")
            sn = read(file,"sn")
            wi = read(file,"wi")
            d_Phi_rr = read(file,"d_Phi_rr")
            d_Phi_rs = read(file,"d_Phi_rs")
            d_Phi_ss = read(file,"d_Phi_ss")
            d_Phi_sr = read(file,"d_Phi_sr")
            
            return Triangle(3,length(rn),rn,sn,wi,cfl_factor,
                            d_Phi_rr,d_Phi_rs,d_Phi_sr,d_Phi_ss,
                            EToV_local)
        end
    end
end

function p1tetrahedra()
    if !isfile("p1constants.jld")
        buildP1constants()
    end

    jldopen("p1constants.jld") do file
        quadrature_weights = read(file,"quadrature_weights")
        r = read(file,"r")
        s = read(file,"s")
        t = read(file,"t")
        EToV_local = read(file,"EToV_local")
        d_Phi_rr = read(file,"d_Phi_rr")
        d_Phi_rs = read(file,"d_Phi_rs")
        d_Phi_rt = read(file,"d_Phi_rt")
        d_Phi_ss = read(file,"d_Phi_ss")
        d_Phi_st = read(file,"d_Phi_st")
        d_Phi_tt = read(file,"d_Phi_tt")

        cfl_factor = 0.1
    
        # note that d_Phi_sr = d_Phi_rs' (transpose)
        return Tetrahedra(1,length(r),r,s,t,quadrature_weights,cfl_factor,
                          d_Phi_rr,d_Phi_rs,d_Phi_rs',d_Phi_rt,d_Phi_rt',d_Phi_ss,d_Phi_st,d_Phi_st',d_Phi_tt,
                          EToV_local)
        
    end
    
end

function p2tetrahedra()

    if !isfile("p2constants.jld")
        buildP2constants()
    end

    jldopen("p2constants.jld") do file
        quadrature_weights = read(file,"quadrature_weights")
        r = read(file,"r")
        s = read(file,"s")
        t = read(file,"t")
        EToV_local = read(file,"EToV_local")
        d_Phi_rr = read(file,"d_Phi_rr")
        d_Phi_rs = read(file,"d_Phi_rs")
        d_Phi_rt = read(file,"d_Phi_rt")
        d_Phi_ss = read(file,"d_Phi_ss")
        d_Phi_st = read(file,"d_Phi_st")
        d_Phi_tt = read(file,"d_Phi_tt")

        # read symmetric terms
        d_Phi_sr = read(file,"d_Phi_sr")
        d_Phi_tr = read(file,"d_Phi_tr")
        d_Phi_ts = read(file,"d_Phi_ts")
        

        cfl_factor = 0.04

        # note that d_Phi_sr = d_Phi_rs' (transpose)
        return Tetrahedra(2,length(r),r,s,t,quadrature_weights,cfl_factor,
                          d_Phi_rr,d_Phi_rs,d_Phi_sr,d_Phi_rt,d_Phi_tr,d_Phi_ss,d_Phi_st,d_Phi_ts,d_Phi_tt,
                          EToV_local)
        
    end

    
end


function p3tetrahedra_noquad()

    jldopen("K_p3true.jld") do file
        d_Phi_rr = read(file,"K_rr")
        d_Phi_rs = read(file,"K_rs")
        d_Phi_rt = read(file,"K_rt")
        d_Phi_ss = read(file,"K_ss")
        d_Phi_st = read(file,"K_st")
        d_Phi_tt = read(file,"K_tt")

        # take advantage of symmetry
        d_Phi_sr = d_Phi_rs'
        d_Phi_tr = d_Phi_rt'
        d_Phi_ts = d_Phi_st'
        
        (r,s,t) = nodes_tetP3_hesthaven()
        num_nodes = length(r)
        # dummy variable
        quadrature_weights = 0*r
        cfl_factor = 0.01
        EToV_local = Vector{Int}[]
        # dummy entries for now...
        push!(EToV_local,[22,   20,   21,   19]+1)
        push!(EToV_local,[20,    8,   21,    7]+1)
        push!(EToV_local,[ 9,   21,    7,   19]+1)
        
        return Tetrahedra(3,length(r),r,s,t,quadrature_weights,cfl_factor,
                          d_Phi_rr,d_Phi_rs,d_Phi_sr,d_Phi_rt,d_Phi_tr,d_Phi_ss,d_Phi_st,d_Phi_ts,d_Phi_tt,
                          EToV_local)
    end
end

function p3tetrahedra_alternate()

    (r,s,t) = ComputeP3_alternate.nodes_tetP3_hesthaven()
    cfl_factor = 0.01
    EToV_local = Vector{Int}[]
    quadrature_weights = ones(length(r))/length(r)
    jldopen("K_p3true.jld") do file
        d_Phi_rr = read(file,"K_rr")
        d_Phi_rs = read(file,"K_rs")
        d_Phi_rt = read(file,"K_rt")
        d_Phi_ss = read(file,"K_ss")
        d_Phi_st = read(file,"K_st")
        d_Phi_tt = read(file,"K_tt")

        # take advantage of symmetry
        d_Phi_sr = d_Phi_rs'
        d_Phi_tr = d_Phi_rt'
        d_Phi_ts = d_Phi_st'

        return Tetrahedra(3,length(r),r,s,t,quadrature_weights,cfl_factor,
                          d_Phi_rr,d_Phi_rs,d_Phi_sr,d_Phi_rt,d_Phi_tr,
                          d_Phi_ss,d_Phi_st,d_Phi_ts,d_Phi_tt,
                          EToV_local)
        
    end
    
    
end

function p4tetrahedra_alternate()

    (r,s,t) = ComputeP3_alternate.nodes_tetP4_hesthaven()
    cfl_factor = 0.01
    EToV_local = Vector{Int}[]
    quadrature_weights = ones(length(r))/length(r)
    jldopen("K_p3true.jld") do file
        d_Phi_rr = read(file,"K_rr")
        d_Phi_rs = read(file,"K_rs")
        d_Phi_rt = read(file,"K_rt")
        d_Phi_ss = read(file,"K_ss")
        d_Phi_st = read(file,"K_st")
        d_Phi_tt = read(file,"K_tt")

        # take advantage of symmetry
        d_Phi_sr = d_Phi_rs'
        d_Phi_tr = d_Phi_rt'
        d_Phi_ts = d_Phi_st'

        return Tetrahedra(3,length(r),r,s,t,quadrature_weights,cfl_factor,
                          d_Phi_rr,d_Phi_rs,d_Phi_sr,d_Phi_rt,d_Phi_tr,
                          d_Phi_ss,d_Phi_st,d_Phi_ts,d_Phi_tt,
                          EToV_local)
        
    end
    
    
end

function p5tetrahedra_alternate()

    (r,s,t) = ComputeP3_alternate.nodes_tetP5_hesthaven()
    cfl_factor = 0.01
    EToV_local = Vector{Int}[]
    quadrature_weights = ones(length(r))/length(r)
    jldopen("K_p3true.jld") do file
        d_Phi_rr = read(file,"K_rr")
        d_Phi_rs = read(file,"K_rs")
        d_Phi_rt = read(file,"K_rt")
        d_Phi_ss = read(file,"K_ss")
        d_Phi_st = read(file,"K_st")
        d_Phi_tt = read(file,"K_tt")

        # take advantage of symmetry
        d_Phi_sr = d_Phi_rs'
        d_Phi_tr = d_Phi_rt'
        d_Phi_ts = d_Phi_st'

        return Tetrahedra(3,length(r),r,s,t,quadrature_weights,cfl_factor,
                          d_Phi_rr,d_Phi_rs,d_Phi_sr,d_Phi_rt,d_Phi_tr,
                          d_Phi_ss,d_Phi_st,d_Phi_ts,d_Phi_tt,
                          EToV_local)
        
    end
    
    
end


function p3tetrahedra()
    
    # buildP3constants()
    if !isfile("p3constants.jld")
        buildP3constants()
        # buildP3constants_mat()
    end

    jldopen("p3constants.jld") do file
        quadrature_weights = read(file,"quadrature_weights")
        r = read(file,"r")
        s = read(file,"s")
        t = read(file,"t")
        EToV_local = read(file,"EToV_local")
        d_Phi_rr = read(file,"d_Phi_rr")
        d_Phi_rs = read(file,"d_Phi_rs")
        d_Phi_rt = read(file,"d_Phi_rt")
        d_Phi_ss = read(file,"d_Phi_ss")
        d_Phi_st = read(file,"d_Phi_st")
        d_Phi_tt = read(file,"d_Phi_tt")
        # symmetric terms 
        d_Phi_sr = read(file,"d_Phi_sr")
        d_Phi_tr = read(file,"d_Phi_tr")
        d_Phi_ts = read(file,"d_Phi_ts")
        
        # take advantage of symmetry
        # d_Phi_sr = d_Phi_rs'
        # d_Phi_tr = d_Phi_rt'
        # d_Phi_ts = d_Phi_st'
        
        cfl_factor = 0.063

        # note that d_Phi_sr = d_Phi_rs' (transpose)
        return Tetrahedra(3,length(r),r,s,t,quadrature_weights,cfl_factor,
                          d_Phi_rr,d_Phi_rs,d_Phi_sr,d_Phi_rt,d_Phi_tr,d_Phi_ss,d_Phi_st,d_Phi_ts,d_Phi_tt,
                          EToV_local)
        
    end

    
end

# volume of tetrahedra
function measureK(v1,v2,v3,v4)
    x0 = v1[1];  y0 = v1[2];  z0 = v1[3];
    x1 = v2[1];  y1 = v2[2];  z1 = v2[3];
    x2 = v3[1];  y2 = v3[2];  z2 = v3[3];
    x3 = v4[1];  y3 = v4[2];  z3 = v4[3];
    
    # V = |a \cdot (b \cross c)|/6
    
    a = [x1-x0, y1-y0, z1-z0]
    b = [x2-x0, y2-y0, z2-z0]
    c = [x3-x0, y3-y0, z3-z0]
    
    m_k = abs(dot(cross(b,c),a))/6
    
    return m_k
    
end

function elementradius3d(v_x,v_y,v_z,EToV)

    # choose jacobian element radius
    return elementradius3d_viajacobian(v_x,v_y,v_z,EToV)
    
end

function elementradius3d_viajacobian(v_x,v_y,v_z,EToV)
    K = length(EToV)
    element_radius = zeros(K)
    for k=1:K
        v1 = [v_x[EToV[k][1]],v_y[EToV[k][1]],v_z[EToV[k][1]]]
        v2 = [v_x[EToV[k][2]],v_y[EToV[k][2]],v_z[EToV[k][2]]]
        v3 = [v_x[EToV[k][3]],v_y[EToV[k][3]],v_z[EToV[k][3]]]
        v4 = [v_x[EToV[k][4]],v_y[EToV[k][4]],v_z[EToV[k][4]]]

        vol = measureK(v1,v2,v3,v4)
        element_radius[k] = (vol)^(1/3)
    end
    
    return element_radius
    
end

function elementradius3d_shortestedge(v_x,v_y,v_z,EToV)

    K = length(EToV)
    element_radius = zeros(K)
    for k=1:K
        v1 = [v_x[EToV[k][1]],v_y[EToV[k][1]],v_z[EToV[k][1]]]
        v2 = [v_x[EToV[k][2]],v_y[EToV[k][2]],v_z[EToV[k][2]]]
        v3 = [v_x[EToV[k][3]],v_y[EToV[k][3]],v_z[EToV[k][3]]]
        v4 = [v_x[EToV[k][4]],v_y[EToV[k][4]],v_z[EToV[k][4]]]
        x1 = v1[1];  y1 = v1[2];  z1 = v1[3];
        x2 = v2[1];  y2 = v2[2];  z2 = v2[3];
        x3 = v3[1];  y3 = v3[2];  z3 = v3[3];
        x4 = v4[1];  y4 = v4[2];  z4 = v4[3];

        len1 = sqrt( (x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2) # 1-2
        len2 = sqrt( (x3-x2)^2 + (y3-y2)^2 + (z3-z2)^2) # 3-2
        len3 = sqrt( (x1-x3)^2 + (y1-y3)^2 + (z1-z3)^2) # 1-3
        len4 = sqrt( (x2-x4)^2 + (y2-y4)^2 + (z2-z4)^2) # 2-4
        len5 = sqrt( (x1-x4)^2 + (y1-y4)^2 + (z1-z4)^2) # 1-4
        len6 = sqrt( (x3-x4)^2 + (y3-y4)^2 + (z3-z4)^2) # 3-4

        h = minimum([len1,len2,len3,len4,len5,len6])
        if h==0.0
            println("EToV[$(k)]=$(EToV[k])")
            println("x1:$((x1,y1,z1)),x2$((x2,y2,z2))")
            println("v1:$(v1),v2:$(v2)")
            println("h=minimum([$([len1,len2,len3,len4,len5,len6])])")
            error("h == 0.0")
        end
        # use shortest side as element-size metric
        # TODO: Use better metric
        element_radius[k] = h
    end

    return element_radius
    
end


function elementradius2d(v_x,v_y,EToV)

    K = length(EToV)
    element_radius = zeros(K)
    for k=1:K
        (x1,y1) = (v_x[EToV[k][1]],v_y[EToV[k][1]])
        (x2,y2) = (v_x[EToV[k][2]],v_y[EToV[k][2]])
        (x3,y3) = (v_x[EToV[k][3]],v_y[EToV[k][3]])
        len1 = sqrt( (x1-x2)^2 + (y1-y2)^2 )
        len2 = sqrt( (x1-x3)^2 + (y1-y3)^2 )
        len3 = sqrt( (x3-x2)^2 + (y3-y2)^2 )
        sper = (len1+len2+len3)/2.0
        area = sqrt( sper * (sper-len1) * (sper-len2) * (sper-len3) )
        element_radius[k] = area / sper
    end

    return element_radius
    
end

function elementradius1d(v_x,EToV)

    K = length(EToV)
    element_radius = zeros(K)
    for k=1:K
        element_radius[k] = abs(v_x[EToV[k][2]] - v_x[EToV[k][1]])
    end
    return element_radius

end

end
