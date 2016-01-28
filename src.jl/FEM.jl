module FEM

export buildM, buildM_full, buildM_full_faster, buildK, buildK_quad,
       buildM_alt, measureK, geometric_components, build_c_prime

using Element
using PyCall
using ProgressMeter
using JLD
using SEM

@pyimport numpy as np

# area of triangle is norm of cross product divided by 2
function measureK(v1,v2,v3)
    # get relative triangle vectors (add 0.0 to allow for cross product)
    v0c = v2-v1; push!(v0c,0.0)
    v1c = v3-v1; push!(v1c,0.0)
    return norm(cross(v0c,v1c)) / 2.0
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

import SymPy

function buildM_full(element::Element1D,v_x,EToV::Vector{Vector{Int64}},dofindex::Vector{Vector{Int64}},ndof)

    K = length(EToV)
    p_Np = length(element.r)
    M = spzeros(ndof,ndof)
    x = SymPy.symbols("x")
    Me = zeros(p_Np,p_Np)
    lg = zeros(SymPy.Sym,p_Np)
    for i=1:p_Np
        lg[i] = lagrange1d(element.r,i,x)
    end
    for i=1:p_Np
        for j=1:p_Np
            Me[i,j] = 1/2*SymPy.integrate(lg[i]*lg[j],x,-1,1)
        end
    end
    
    
    for k = 1:K
        
        m_k = v_x[EToV[k][2]] - v_x[EToV[k][1]]
        indexE = dofindex[k]
        for i = 1:p_Np
            ie = indexE[i]
            for j=1:p_Np
                je = indexE[j]
                M[ie,je] += m_k*Me[i,j]
            end
        end        
    end
    return M
end


function buildM(element::Element1D,
                v_x::Vector{Float64},
                EToV::Vector{Vector{Int64}},
                dofindex::Vector{Vector{Int64}},
                ndof::Int64)

    K = length(EToV)
    p_Np = length(element.r)
    mass_matrix = zeros(Float64,ndof)
    for k=1:K
        v1 = v_x[EToV[k][1]]
        v2 = v_x[EToV[k][2]]

        m_k = v2-v1
        if m_k < 0
            error("Vertices must be ordered smallest to largest")
        end
        indexE = dofindex[k]
        for i=1:p_Np
            ie = indexE[i]
            wi = element.quadrature_weights[i]
            # assemble diagonal mass matrix
            mass_matrix[ie] += m_k*wi
        end
    end

    mass_matrix_inv = 1./mass_matrix
    return(mass_matrix,mass_matrix_inv)
    
end

function buildM_full(element::Triangle,v_x,v_y,EToV::Vector{Vector{Int64}},dofindex::Vector{Vector{Int64}},ndof)

    K = length(EToV)
    p_Np = length(element.r)
    M = spzeros(ndof,ndof)
    Mpy = np.load("M_p3triangle.npz")
    Me = get(Mpy,"M")    
    
    for k = 1:K
        v1 = [v_x[EToV[k][1]],v_y[EToV[k][1]]]
        v2 = [v_x[EToV[k][2]],v_y[EToV[k][2]]]
        v3 = [v_x[EToV[k][3]],v_y[EToV[k][3]]]
        
        m_k = measureK(v1,v2,v3)
        indexE = dofindex[k]
        for i = 1:p_Np
            ie = indexE[i]
            for j=1:p_Np
                je = indexE[j]
                M[ie,je] += m_k*Me[i,j]
            end
        end        
    end
    return M
end

function buildM_alt(element::Tetrahedra,v_x,v_y,v_z,EToV::Vector{Vector{Int64}},dofindex::Vector{Vector{Int64}},ndof)

    K = length(EToV)
    p_Np = length(element.r)
    println("Number of GB for full M = $(ndof^2 * 8 / 1e9) GB")
    M = spzeros(ndof,ndof)
    Md = Dict([((1,1),0.0)])
    # not feasible, even for small mesh of 2K elements (23 GB of memory)
    # M = zeros(ndof,ndof)    
    file = jldopen("M_p3true.jld")
    Me = read(file,"M")

    p = Progress(K, 1)
    
    for k=1:K
        v1 = [v_x[EToV[k][1]],v_y[EToV[k][1]],v_z[EToV[k][1]]]
        v2 = [v_x[EToV[k][2]],v_y[EToV[k][2]],v_z[EToV[k][2]]]
        v3 = [v_x[EToV[k][3]],v_y[EToV[k][3]],v_z[EToV[k][3]]]
        v4 = [v_x[EToV[k][4]],v_y[EToV[k][4]],v_z[EToV[k][4]]]

        m_k = measureK(v1,v2,v3,v4)
        indexE = dofindex[k]
        for i=1:p_Np
            ie = indexE[i]
            for j=1:p_Np
                je = indexE[j]
                if haskey(Md,(ie,je))
                    Md[(ie,je)] += m_k*Me[i,j]
                else               
                    Md[(ie,je)] = m_k*Me[i,j]
                end
                # too slow!
                # M[ie,je] += m_k*Me[i,j]
            end            
        end
        next!(p)
    end
    num_entries = length(Md)
    Is = zeros(Int64,num_entries)
    Js = zeros(Int64,num_entries)
    Vs = zeros(num_entries)
    println("length Md=$(length(Md))")
    for (idof,((i,j),Mij)) in enumerate(Md)
        Is[idof] = i
        Js[idof] = j
        Vs[idof] = Mij
    end
    
    return sparse(Is,Js,Vs)
end

    
function buildM_full_faster(element::Tetrahedra,v_x,v_y,v_z,EToV::Vector{Vector{Int64}},dofindex::Vector{Vector{Int64}},ndof)

    K = length(EToV)
    p_Np = length(element.r)
    println("Number of GB for full M = $(ndof^2 * 8 / 1e9) GB")
    M = spzeros(ndof,ndof)
    Md = Dict([((1,1),0.0)])
    # not feasible, even for small mesh of 2K elements (23 GB of memory)
    # M = zeros(ndof,ndof)    
    Mpy = np.load("M_p3tetrahedra.npz")
    Me = get(Mpy,"M")

    p = Progress(K, 1)
    
    for k=1:K
        v1 = [v_x[EToV[k][1]],v_y[EToV[k][1]],v_z[EToV[k][1]]]
        v2 = [v_x[EToV[k][2]],v_y[EToV[k][2]],v_z[EToV[k][2]]]
        v3 = [v_x[EToV[k][3]],v_y[EToV[k][3]],v_z[EToV[k][3]]]
        v4 = [v_x[EToV[k][4]],v_y[EToV[k][4]],v_z[EToV[k][4]]]

        m_k = measureK(v1,v2,v3,v4)
        indexE = dofindex[k]
        for i=1:p_Np
            ie = indexE[i]
            for j=1:p_Np
                je = indexE[j]
                if haskey(Md,(ie,je))
                    Md[(ie,je)] += m_k*Me[i,j]
                else               
                    Md[(ie,je)] = m_k*Me[i,j]
                end
                # too slow!
                # M[ie,je] += m_k*Me[i,j]
            end            
        end
        next!(p)
    end
    num_entries = length(Md)
    Is = zeros(Int64,num_entries)
    Js = zeros(Int64,num_entries)
    Vs = zeros(num_entries)
    println("length Md=$(length(Md))")
    for (idof,((i,j),Mij)) in enumerate(Md)
        Is[idof] = i
        Js[idof] = j
        Vs[idof] = Mij
    end
    
    return sparse(Is,Js,Vs)
end

function buildM_full(element::Tetrahedra,v_x,v_y,v_z,EToV::Vector{Vector{Int64}},dofindex::Vector{Vector{Int64}},ndof)

    K = length(EToV)
    p_Np = length(element.r)
    println("Number of GB for full M = $(ndof^2 * 8 / 1e9) GB")
    M = spzeros(ndof,ndof)
    # not feasible, even for small mesh of 2K elements (23 GB of memory)
    # M = zeros(ndof,ndof)    
    Mpy = np.load("M_p3tetrahedra.npz")
    Me = get(Mpy,"M")

    for k=1:K
        v1 = [v_x[EToV[k][1]],v_y[EToV[k][1]],v_z[EToV[k][1]]]
        v2 = [v_x[EToV[k][2]],v_y[EToV[k][2]],v_z[EToV[k][2]]]
        v3 = [v_x[EToV[k][3]],v_y[EToV[k][3]],v_z[EToV[k][3]]]
        v4 = [v_x[EToV[k][4]],v_y[EToV[k][4]],v_z[EToV[k][4]]]

        m_k = measureK(v1,v2,v3,v4)
        indexE = dofindex[k]
        for i=1:p_Np
            ie = indexE[i]
            for j=1:p_Np
                je = indexE[j]
                M[ie,je] += m_k*Me[i,j]
            end            
        end
    end

    return sparse(M)
end


# 3D
function buildM(element::Tetrahedra,
                v_x::Vector{Float64}, v_y::Vector{Float64}, v_z::Vector{Float64},
                EToV::Vector{Vector{Int64}},
                dofindex::Vector{Vector{Int64}}, ndof)

    K = length(EToV)
    p_Np = length(element.r)
    mass_matrix = zeros(Float64,ndof)
    for k=1:K
        v1 = [v_x[EToV[k][1]],v_y[EToV[k][1]],v_z[EToV[k][1]]]
        v2 = [v_x[EToV[k][2]],v_y[EToV[k][2]],v_z[EToV[k][2]]]
        v3 = [v_x[EToV[k][3]],v_y[EToV[k][3]],v_z[EToV[k][3]]]
        v4 = [v_x[EToV[k][4]],v_y[EToV[k][4]],v_z[EToV[k][4]]]

        m_k = measureK(v1,v2,v3,v4)

        indexE = dofindex[k]
        for i=1:p_Np
            ie = indexE[i]
            wi = element.quadrature_weights[i]
            # assemble diagonal mass matrix
            mass_matrix[ie] += m_k * wi
        end
    end

    mass_matrix_inv = 1 ./ mass_matrix
    if length(find(isnan(mass_matrix_inv))) > 0
        error("Minv contains NaN")
    end
    return(mass_matrix, mass_matrix_inv)
    
end

# 2D
function buildM(element::Triangle,v_x::Vector{Float64},v_y::Vector{Float64},EToV::Vector{Vector{Int64}},dofindex::Vector{Vector{Int64}},ndof)

    K = length(EToV)
    p_Np = length(element.r)
    mass_matrix = zeros(Float64,ndof)
    for k=1:K
        v1 = [v_x[EToV[k][1]],v_y[EToV[k][1]]]
        v2 = [v_x[EToV[k][2]],v_y[EToV[k][2]]]
        v3 = [v_x[EToV[k][3]],v_y[EToV[k][3]]]

        m_k = measureK(v1,v2,v3)
        indexE = dofindex[k]
        for i=1:p_Np
            ie = indexE[i]
            wi = element.quadrature_weights[i]
            # assemble diagonal mass matrix
            mass_matrix[ie] += m_k*wi
        end
    end

    mass_matrix_inv = 1./mass_matrix
    return(mass_matrix,mass_matrix_inv)
    
end

function jacobian_components(v1,v2,v3)    
    v2mv1_2 = (v2-v1)./2
    dxdr = v2mv1_2[1]
    dydr = v2mv1_2[2]
    v3mv1_2 = (v3-v1)./2
    dxds = v3mv1_2[1]
    dyds = v3mv1_2[2]
    return (dxdr,dydr,dxds,dyds)
end

function geometric_components2(v1,v2,v3,v4)

    xr = (v2-v1)[1];  xs = (v3-v1)[1];  xt = (v4-v1)[1];
    yr = (v2-v1)[2];  ys = (v3-v1)[2];  yt = (v4-v1)[2];
    zr = (v2-v1)[3];  zs = (v3-v1)[3];  zt = (v4-v1)[3];

    # from sympy
    # J = Matrix([[xr, xs, xt],
    # [yr, ys, yt],
    # [zr, zs, zt]])
    # det(J)*Jinv = 
    # Matrix([
    # [ ys*zt - yt*zs, -xs*zt + xt*zs,  xs*yt - xt*ys],
    # [-yr*zt + yt*zr,  xr*zt - xt*zr, -xr*yt + xt*yr],
    # [ yr*zs - ys*zr, -xr*zs + xs*zr,  xr*ys - xs*yr]])
    # det(J) = xr*ys*zt - xr*yt*zs - xs*yr*zt + xs*yt*zr + xt*yr*zs - xt*ys*zr

    
    detJ = xr*ys*zt - xr*yt*zs - xs*yr*zt + xs*yt*zr + xt*yr*zs - xt*ys*zr
    rx = (ys*zt - yt*zs)/detJ
    ry = (-xs*zt + xt*zs)/detJ
    rz = (xs*yt - xt*ys)/detJ
    sx = (-yr*zt + yt*zr)/detJ
    sy = (xr*zt - xt*zr)/detJ
    sz = (-xr*yt + xt*yr)/detJ
    tx = (yr*zs - ys*zr)/detJ
    ty = (-xr*zs + xs*zr)/detJ
    tz = (xr*ys - xs*yr)/detJ

    return (rx,sx,tx,ry,sy,ty,rz,sz,tz,detJ)
    
end

function geometric_components(v1,v2,v3)

    (dxdr,dydr,dxds,dyds) = jacobian_components(v1,v2,v3)
    J = [dxdr dydr;
         dxds dyds]
    Jinv = 1/det(J)
    detJ = det(J)
    rx = dyds./detJ
    ry = -dxds./detJ
    sx = -dydr./detJ
    sy = dxdr./detJ

    return (rx,sx,ry,sy,detJ)
    
end

function geometric_components(v1,v2,v3,v4)

    x1 = v1[1];  y1 = v1[2];  z1 = v1[3];
    x2 = v2[1];  y2 = v2[2];  z2 = v2[3];
    x3 = v3[1];  y3 = v3[2];  z3 = v3[3];
    x4 = v4[1];  y4 = v4[2];  z4 = v4[3];

    dxdr = (x2-x1);  dxds = (x3-x1); dxdt = (x4-x1);
    dydr = (y2-y1);  dyds = (y3-y1); dydt = (y4-y1);
    dzdr = (z2-z1);  dzds = (z3-z1); dzdt = (z4-z1);

    detJ = (dxdr*(dyds*dzdt-dzds*dydt)
            -dydr*(dxds*dzdt-dzds*dxdt)
            +dzdr*(dxds*dydt-dyds*dxdt))

    if detJ < 1e-8
        error("|Jacobian| too small")
    end
    
    drdx  =  (dyds*dzdt - dzds*dydt)/(detJ)
    drdy  = -(dxds*dzdt - dzds*dxdt)/(detJ)
    drdz  =  (dxds*dydt - dyds*dxdt)/(detJ)

    dsdx  = -(dydr*dzdt - dzdr*dydt)/(detJ)
    dsdy  =  (dxdr*dzdt - dzdr*dxdt)/(detJ)
    dsdz  = -(dxdr*dydt - dydr*dxdt)/(detJ)

    dtdx  =  (dydr*dzds - dzdr*dyds)/(detJ)
    dtdy  = -(dxdr*dzds - dzdr*dxds)/(detJ)
    dtdz  =  (dxdr*dyds - dydr*dxds)/(detJ)    
    
    return (drdx,dsdx,dtdx,
            drdy,dsdy,dtdy,
            drdz,dsdz,dtdz,
            detJ)
    
end

function geometric_components3(v1, v2, v3, v4)

    x1 = v1[1];  y1 = v1[2];  z1 = v1[3];
    x2 = v2[1];  y2 = v2[2];  z2 = v2[3];
    x3 = v3[1];  y3 = v3[2];  z3 = v3[3];
    x4 = v4[1];  y4 = v4[2];  z4 = v4[3];

    J = zeros(3,3)
    J[1,1] = (x2-x1);  J[1,2] = (x3-x1); J[1,3] = (x4-x1);
    J[2,1] = (y2-y1);  J[2,2] = (y3-y1); J[2,3] = (y4-y1);
    J[3,1] = (z2-z1);  J[3,2] = (z3-z1); J[3,3] = (z4-z1);
    detJ = det(J)

    # should be require the Jacobian to be positive? I think this is connected
    # to the node ordering.
    if abs(detJ) < 1e-8
        error("|Jacobian| too small")
    end

    invJ = inv(J)
    
    return J, invJ, detJ
    
end

function buildK_quad(element::Element1D,v_x,c_m,EToV,dofindex,ndof)

    if length(c_m) != ndof
        error("material c_m needs to be defined on all dofs")
    end

    K = length(EToV)
    p_Np = length(element.r)
    Ke = Matrix{Float64}[]
    
    dphi_i_n = precompute_dphi(element)

    for k=1:K
        v1 = v_x[EToV[k][1]]
        v2 = v_x[EToV[k][2]]
        rx = 1/(v2-v1)
        m_k = v2-v1
        c_m_k = zeros(element.p_Np)
        for n=1:element.p_Np
            c_m_k[n] = c_m[dofindex[k][n]]
        end
        d_Phi_rr = stiffness1d_with_cm(element.r,dphi_i_n,c_m_k,element.quadrature_weights)
        Kx = m_k .* (rx^2).*d_Phi_rr
        push!(Ke,Kx)
    end
    return Ke
    
end
    
function buildK(element::Element1D,v_x,c_m,EToV,ndof)
    K = length(EToV)
    p_Np = length(element.r)
    Ke = Matrix{Float64}[]
    for k=1:K
        v1 = v_x[EToV[k][1]]
        v2 = v_x[EToV[k][2]]
        rx = 2/(v2-v1)
        m_k = v2-v1
        Kx = (c_m[k]*m_k) .* (rx^2).*element.d_Phi_rr
        push!(Ke,Kx)
    end
    return Ke
end

function buildK(element::Element1D,v_x,EToV,ndof)
    K = length(EToV)
    p_Np = length(element.r)
    Ke = Matrix{Float64}[]
    for k=1:K
        v1 = v_x[EToV[k][1]]
        v2 = v_x[EToV[k][2]]
        rx = 2/(v2-v1)
        m_k = v2-v1
        Kx = (m_k) .* (rx^2).*element.d_Phi_rr
        push!(Ke,Kx)
        
    end
    return Ke
end
    
function buildK(element::Triangle,v_x,v_y,EToV,ndof)
    K = length(EToV)
    p_Np = length(element.r)
    Ke = Matrix{Float64}[]
    for k=1:K
        v1 = [v_x[EToV[k][1]],v_y[EToV[k][1]]]
        v2 = [v_x[EToV[k][2]],v_y[EToV[k][2]]]
        v3 = [v_x[EToV[k][3]],v_y[EToV[k][3]]]
        (dxdr,dydr,dxds,dyds) = jacobian_components(v1,v2,v3)
        J = [dxdr dydr;
             dxds dyds]
        Jinv = 1/det(J)
        detJ = det(J)
        rx = dyds./detJ
        ry = -dxds./detJ
        sx = -dydr./detJ
        sy = dxdr./detJ
        m_k = measureK(v1,v2,v3)

        Kx = (m_k) .* ((rx^2).*element.d_Phi_rr
                      + (rx*sx).*element.d_Phi_rs
                      + (rx*sx).*element.d_Phi_sr
                      + (sx^2).*element.d_Phi_ss)
        Ky = (m_k) .* ((ry^2).*element.d_Phi_rr
                      + (ry*sy).*element.d_Phi_rs
                      + (ry*sy).*element.d_Phi_sr
                      + (sy^2).*element.d_Phi_ss)        
        
        push!(Ke,Kx+Ky)

    end

    return Ke
end

function fcmp(a,b)
    return abs(a-b) < 1e-8
end

# 3d version
function buildK(element::Tetrahedra,v_x,v_y,v_z,EToV,ndof)
    
    K = length(EToV)
    p_Np = length(element.r)
    Ke = Matrix{Float64}[]
    for k=1:K
        v1 = [v_x[EToV[k][1]],v_y[EToV[k][1]],v_z[EToV[k][1]]]
        v2 = [v_x[EToV[k][2]],v_y[EToV[k][2]],v_z[EToV[k][2]]]
        v3 = [v_x[EToV[k][3]],v_y[EToV[k][3]],v_z[EToV[k][3]]]
        v4 = [v_x[EToV[k][4]],v_y[EToV[k][4]],v_z[EToV[k][4]]]
        
        m_k = measureK(v1,v2,v3,v4) # = abs(det(J) * 6)
        
        (rx,sx,tx,
         ry,sy,ty,
         rz,sz,tz,
         detJ) = geometric_components(v1,v2,v3,v4)

        
        Kx = (m_k) * ((rx*rx).*element.d_Phi_rr  
                    + (rx*sx).*element.d_Phi_rs
                    + (rx*sx).*element.d_Phi_sr
                    + (sx*sx).*element.d_Phi_ss
                    + (rx*tx).*element.d_Phi_rt
                    + (rx*tx).*element.d_Phi_tr
                    + (sx*tx).*element.d_Phi_st
                    + (sx*tx).*element.d_Phi_ts
                    + (tx*tx).*element.d_Phi_tt)
        
        Ky = (m_k) * ((ry*ry).*element.d_Phi_rr  
                    + (ry*sy).*element.d_Phi_rs
                    + (ry*sy).*element.d_Phi_sr
                    + (sy*sy).*element.d_Phi_ss
                    + (ry*ty).*element.d_Phi_rt
                    + (ry*ty).*element.d_Phi_tr
                    + (sy*ty).*element.d_Phi_st
                    + (sy*ty).*element.d_Phi_ts
                    + (ty*ty).*element.d_Phi_tt)

        Kz = (m_k) * ((rz*rz).*element.d_Phi_rr  
                    + (rz*sz).*element.d_Phi_rs
                    + (rz*sz).*element.d_Phi_sr
                    + (sz*sz).*element.d_Phi_ss
                    + (rz*tz).*element.d_Phi_rt
                    + (rz*tz).*element.d_Phi_tr
                    + (sz*tz).*element.d_Phi_st
                    + (sz*tz).*element.d_Phi_ts
                    + (tz*tz).*element.d_Phi_tt)

        
        push!(Ke,Kx+Ky+Kz)

    end

    return Ke
end



# 3d elastic version
function build_c_prime(element::Tetrahedra, v_x, v_y, v_z, EToV, ndof)
    
    K = length(EToV)
    p_Np = length(element.r)
    c_prime = Array{Float64,4}[]

    c_iso = c_ijkl_iso(1., 1.)

    for k=1:K
        v1 = [v_x[EToV[k][1]], v_y[EToV[k][1]], v_z[EToV[k][1]]]
        v2 = [v_x[EToV[k][2]], v_y[EToV[k][2]], v_z[EToV[k][2]]]
        v3 = [v_x[EToV[k][3]], v_y[EToV[k][3]], v_z[EToV[k][3]]]
        v4 = [v_x[EToV[k][4]], v_y[EToV[k][4]], v_z[EToV[k][4]]]

        m_k = measureK(v1,v2,v3,v4) # = abs(det(J) / 6)
        
        (J, invJ, detJ) = geometric_components3(v1, v2, v3, v4)

        #println("$(m_k), $(detJ / 6)")
        
        ce = c_prime_ijkl(c_iso, invJ, detJ)
        
        push!(c_prime, ce)

    end

    return c_prime
end


function c_ijkl_iso_element(lambda::Float64, mu::Float64, i::Int, j::Int, k::Int, l::Int)
    # compute the elasticity tensor element wise for an isotropic medium
    return (lambda * (i == j) * (k == l) 
            + mu * ((i == k) * (j == l) + (i == l) * (j == k)))
end


function c_ijkl_iso(lambda::Float64, mu::Float64)
    # compute the elasticity tensor for an isotropic medium

    c_ijkl = zeros(3, 3, 3, 3)
    for i = 1:3
        for j = 1:3
            for k = 1:3
                for l = 1:3
                    c_ijkl[i, j, k, l] = c_ijkl_iso_element(
                        lambda, mu, i, j, k, l)
                end
            end
        end
    end
    return c_ijkl
end


function c_prime_ijkl(c_ijkl::Array{Float64,4}, invJ::Array{Float64,2}, detJ::Float64)
    # ignores the thermodynamic symmetry c_prime[ijkl] = c_prime[klij], hence
    # uses 81 floats in memory instead of 45

    c_prime_ijkl = zeros(3, 3, 3, 3)
    for i = 1:3
        for j = 1:3
            for k = 1:3
                for l = 1:3
                    for m = 1:3
                        for n = 1:3
                            c_prime_ijkl[i, j, k, l] += 
                                detJ * invJ[j, m] * invJ[l, n] * c_ijkl[i, m, k, n]
                        end
                    end
                end
            end
        end
    end
    return c_prime_ijkl

end


end
