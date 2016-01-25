module PolySpace

using SymPy
using JLD
import MAT
using Element
using ProgressMeter
using PyPlot

export buildPhi,buildPhiXYZ, polyspace, buildCoefficients

function P2Tilde_bt(r,s,t)
    btilde = r*s*t*(1-r-s-t)
    P2t = [1,r,s,t,r*s,r*t,t*s,r^2,s^2,t^2]
    P2t_bt = btilde .* P2t
    return P2t_bt
end

function P2_b(r,s,t)
    b_1 = r*s*(1-r-s)
    b_2 = r*t*(1-r-t)
    b_3 = t*s*(1-t-s)
    b_4 = (1-r-s)*(1-r-t)*(1-t-s)
    P2_b_1 = b_1 .* [r,s,r*s,r^2,s^2]
    P2_b_2 = b_2 .* [r,t,r*t,r^2,t^2]
    P2_b_3 = b_3 .* [t,s,t*s,t^2,s^2]
    rr = (1-r-t)
    ss = (1-s-t)
    P2_b_4 = b_4 .* [rr,ss,rr^2,ss^2,rr*ss]
    return [P2_b_1; P2_b_2; P2_b_3; P2_b_4]
end

# attempt to fix current approach
function P2_b2(r,s,t)
    b_1 = r*s*(1-r-s-t)
    b_2 = r*t*(1-r-s-t)
    b_3 = t*s*(1-r-s-t)
    b_4 = r*s*t
    P2_b_1 = b_1 .* [r,s,r*s,r^2,s^2]
    P2_b_2 = b_2 .* [r,t,r*t,r^2,t^2]
    P2_b_3 = b_3 .* [t,s,t*s,t^2,s^2]
    rr = (1-r-t)
    ss = (1-s-t)
    P2_b_4 = b_4 .* [rr,ss,rr^2,ss^2,rr*ss]
    return [P2_b_1; P2_b_2; P2_b_3; P2_b_4]
end

function buildCoefficients(rs,Pall,r,s)
    (rn,sn) = rs
    N = length(rn)
    A = zeros(N,N)
    for i=1:N
        for j=1:N
            # println("Pall[$(j)] = $(Pall[j])")
            A[i,j] = subs(Pall[j],(r,rn[i]),(s,sn[i]))
        end
    end

    condA = cond(A)
    println("Condition number of A = $(condA)")

    if condA > (1e14)
        error("Legendre Matrix 'A' likely singular -- check polynomial terms")
    end
    
    P_a = Vector{Float64}[]
    for n=1:N
        an = A\(eye(N)[:,n])
        push!(P_a,an)
    end
    return (P_a,A)
end

function buildCoefficients(rst,Pall,r,s,t)
    (rn,sn,tn) = rst
    N = length(rn)
    A = zeros(N,N)
    for i=1:N
        for j=1:N
            # println("Pall[$(j)] = $(Pall[j])")
            A[i,j] = subs(Pall[j],(r,rn[i]),(s,sn[i]),(t,tn[i]))
        end
    end

    condA = cond(A)
    println("Condition number of A = $(condA)")

    if condA > (1e14)
        error("Legendre Matrix 'A' likely singular -- check polynomial terms")
    end
    
    P_a = Vector{Float64}[]
    for n=1:N
        an = A\(eye(N)[:,n])
        push!(P_a,an)
    end
    return (P_a,A)
end

function polyspace(r,s,t)

    P2t_bt = P2Tilde_bt(r,s,t)

    P2_b1234 = P2_b(r,s,t)

    P3 = [1,
          r,r^2,r^3,s,s^2,s^3,t,t^2,t^3,
          r*s,r^2*s,r*s^2,
          r*t,r^2*t,r*t^2,
          s*t,s^2*t,s*t^2,
          r*s*t]

    Pall = [P3; P2_b1234; P2t_bt]
    return Pall
    
end

function buildPhiXY(rs_symbols,xy)

    (r,s) = rs_symbols
    
    Pall = [1,r,s,r*s,r^2,s^2,r^3,s^3,
            r^2*s,r*s^2,
            r^3*s + s^2*r^2,r^2 * s^2 + s^3 * r]
    
    # if !isfile("p3coefficients.jld")
    (pa,A) = buildCoefficients(xy,Pall,r,s)
    
    # computed via mathematica
    # file = MAT.matopen("/home/rietmann/Dropbox/PostDoc/TetFemJulia/util/p_coefficients_new.mat")    
    # pa = MAT.read(file,"Pa")
    # MAT.close(file)
    println("pa=$(typeof(pa))")
    println("len pa=$(length(pa))")
    # else
    # file = jldopen("p3coefficients.jld")
    # pa = read(file,"pa")
    # pa_v2 = read(file,"pa_v2")
    # close(file)
    # end    
    
    phi = Sym[]

    for i=1:length(Pall)
        # Pa_i = vec(pa[i,:])
        phi_i = dot(pa[i],Pall)
        # println("phi_i=$(phi_i)")
        push!(phi,phi_i)
    end
    
    return phi
end


function buildPhiXYZ(rst_symbols,xyz)

    (r,s,t) = rst_symbols

    P2t_bt = P2Tilde_bt(r,s,t)

    P2_b1234 = P2_b2(r,s,t)

    P3 = [1,
          r,r^2,r^3,s,s^2,s^3,t,t^2,t^3,
          r*s,r^2*s,r*s^2,
          r*t,r^2*t,r*t^2,
          s*t,s^2*t,s*t^2,
          r*s*t]

    Pall = [P3; P2_b1234; P2t_bt]

    # if !isfile("p3coefficients.jld")
    (pa,A) = buildCoefficients(xyz,Pall,r,s,t)
    
    # computed via mathematica
    # file = MAT.matopen("/home/rietmann/Dropbox/PostDoc/TetFemJulia/util/p_coefficients_new.mat")    
    # pa = MAT.read(file,"Pa")
    # MAT.close(file)
    println("pa=$(typeof(pa))")
    println("len pa=$(length(pa))")
    # else
    # file = jldopen("p3coefficients.jld")
    # pa = read(file,"pa")
    # pa_v2 = read(file,"pa_v2")
    # close(file)
    # end    
    
    phi = Sym[]

    for i=1:length(Pall)
        # Pa_i = vec(pa[i,:])
        phi_i = dot(pa[i],Pall)
        # println("phi_i=$(phi_i)")
        push!(phi,phi_i)
    end
    
    return phi
end


function buildPhiXYZ_mathematica(rst_symbols,xyz)

    (r,s,t) = rst_symbols

    P2t_bt = P2Tilde_bt(r,s,t)

    P2_b1234 = P2_b(r,s,t)

    P3 = [1,
          r,r^2,r^3,s,s^2,s^3,t,t^2,t^3,
          r*s,r^2*s,r*s^2,
          r*t,r^2*t,r*t^2,
          s*t,s^2*t,s*t^2,
          r*s*t]

    Pall = [P3; P2_b1234; P2t_bt]

    # if !isfile("p3coefficients.jld")
    # (pa_old,A) = buildCoefficients(xyz,Pall,r,s,t)
    
    # computed via mathematica
    file = MAT.matopen("/home/rietmann/Dropbox/PostDoc/TetFemJulia/util/p_coefficients_new_0_5.mat")
    pa = MAT.read(file,"Pa")
    MAT.close(file)
    println("pa=$(typeof(pa))")
    println("len pa=$(length(pa))")
    # else
    # file = jldopen("p3coefficients.jld")
    # pa = read(file,"pa")
    # pa_v2 = read(file,"pa_v2")
    # close(file)
    # end    
    
    phi = Sym[]

    for i=1:length(Pall)
        Pa_i = vec(pa[i,:])
        phi_i = dot(Pa_i,Pall)
        # println("phi_i=$(phi_i)")
        push!(phi,phi_i)
    end
    
    return phi
end

function testPhi()

    
    tet = p3tetrahedra()
    (r,s,t) = symbols("r,s,t")

    rst = (tet.r,tet.s,tet.t)

    P2t_bt = P2Tilde_bt(r,s,t)

    P2_b1234 = P2_b(r,s,t)

    P3 = [1,
          r,r^2,r^3,s,s^2,s^3,t,t^2,t^3,
          r*s,r^2*s,r*s^2,
          r*t,r^2*t,r*t^2,
          s*t,s^2*t,s*t^2,
          r*s*t]

    Pall = [P3; P2_b1234; P2t_bt]

    # if !isfile("p3coefficients.jld")
    (pa,A) = buildCoefficients(rst,Pall,r,s,t)
    # else
    # file = jldopen("p3coefficients.jld")
    # pa = read(file,"pa")
    # pa_v2 = read(file,"pa_v2")
    # close(file)
    # end    

    phi = Sym[]

    N = length(tet.r)
    for pai in pa
        push!(phi,dot(pai,Pall))    
    end
    npts = length(tet.r)
    int_phi_r = zeros(npts)
    println("phi[1]=$(phi[1])")
    for i=1:npts
        int_phi_r[i] = integrate(diff(phi[i],r),(r,0,1-s-t),(s,0,1-t),(t,0,1))
        println("i=$(i)")
    end
    figure(2)
    PyPlot.plot(1:npts,int_phi_r,"k*-")
    
end


function buildPhi(rst_sym,tet)

    (r,s,t) = rst_sym    

    rst = (tet.r,tet.s,tet.t)

    P2t_bt = P2Tilde_bt(r,s,t)

    P2_b1234 = P2_b2(r,s,t)

    P3 = [1,
          r,r^2,r^3,s,s^2,s^3,t,t^2,t^3,
          r*s,r^2*s,r*s^2,
          r*t,r^2*t,r*t^2,
          s*t,s^2*t,s*t^2,
          r*s*t]

    Pall = [P3; P2_b1234; P2t_bt]

    # if !isfile("p3coefficients.jld")
    (pa,A) = buildCoefficients(rst,Pall,r,s,t)
    # else
    # file = jldopen("p3coefficients.jld")
    # pa = read(file,"pa")
    # pa_v2 = read(file,"pa_v2")
    # close(file)
    # end    

    phi = Sym[]

    N = length(tet.r)
    for pai in pa
        push!(phi,dot(pai,Pall))    
    end

    return phi
end

# include("/home/rietmann/Dropbox/PostDoc/TetFemJulia/util/PolyLibHs/A.jl")

function testCoefficients()

    (r,s,t) = symbols("r,s,t")
    tet = p3tetrahedra()

    rst = (tet.r,tet.s,tet.t)

    P2t_bt = P2Tilde_bt(r,s,t)

    P2_b1234 = P2_b(r,s,t)

    P3 = [1,
          r,r^2,r^3,s,s^2,s^3,t,t^2,t^3,
          r*s,r^2*s,r*s^2,
          r*t,r^2*t,r*t^2,
          s*t,s^2*t,s*t^2,
          r*s*t]

    Pall = [P3; P2_b1234; P2t_bt]

    # if !isfile("p3coefficients.jld")
    (pa,A) = buildCoefficients(rst,Pall,r,s,t)
    Ahs = hsA()
    return (pa,A,Ahs)
    
end

function testQuadratureRule()

    r,s,t = symbols("r s t")
    tetP3 = p3tetrahedra()
    phi = buildPhi((r,s,t),tetP3)

    tetP3 = p3tetrahedra()
    
    wi_test = zeros(length(phi))
    for n=1:length(phi)
        wi_test[n] = 6*integrate(phi[n],(r,0,1-s-t),(s,0,1-t),(t,0,1))
    end
    println("wi_test vs. paper quadrature")
    if maximum(abs(wi_test - tetP3.quadrature_weights)) > 1e-8
        error("A quadrature weight was incorrect!")
    else
        println("All quadrature rules correct")
    end
    
    test_polys = Sym[]
    for i=0:7
        for j=0:7
            for k=0:7
                if i+j+k == 0
                    # push!(test_polys,1+0*r)                                        
                elseif i+j+k <= 7
                    if i==0
                        push!(test_polys,s^j*t^k)
                    elseif j==0
                        push!(test_polys,r^i*t^k)
                    elseif k==0
                        push!(test_polys,r^i*s^j)
                    end
                end
            end
        end
    end
        
    # test_polys1 = [r^7,                   
    #                r^6*s,
    #                r^5*s^2,
    #                r^4*s^3,
    #                r^3*s^4,
    #                r^2*s^5,
    #                r^1*s^6,
    #                r^6*t,
    #                r^5*t^2,
    #                r^4*t^3,
    #                r^3*t^4,
    #                r^2*t^5,
    #                r^1*t^6,
    #                r^5*s*t,
    #                r^4*s^2*t,
    #                r^4*s*t^2,
    #                r^3*s^3*t,
    #                r^3*s^2*t^2,
    #                r^3*s^1*t^3,
    #                r^2*s^4*t]
    # test_polys2 = [
    #                r^2*s^4*t,
    #                r*s^4*t^2,
    #                r*s^3*t^3,
    #                r*s^2*t^4,
    #                r*s^1*t^5,
    #                r^2*s^5,                   
    #                r^1*s^6,
    #                r^1*s^5*t,
    #                r^1*s^4*t^2,
    #                r^1*s^3*t^3,
    #                r^1*s^2*t^4,
    #                r^1*s^1*t^5,
    #                s^7]
    # test_polys3 = [s^6*t^1,
    #                s^5*t^2,
    #                s^4*t^3,
    #                s^3*t^4,
    #                s^2*t^5,
    #                s^2*t^5,
    #                s^1*t^6,                   
    #                t^7
    #                ]
    
    # test_polys = vcat(test_polys1,test_polys2,test_polys3)
    println("Testing $(length(test_polys))")
    
    # should be 1/6
    mKref = float(integrate(1+0*r,(r,0,1-s-t),(s,0,1-t),(t,0,1)))
    
    for (i,pt) in enumerate(test_polys)
        println("on $(i)th poly: $(pt)")
        exact = float(integrate(pt,(r,0,1-s-t),(s,0,1-t),(t,0,1)))
        via_quadrature = 0.0        
        for i=1:length(tetP3.r)
            ri = tetP3.r[i]
            si = tetP3.s[i]
            ti = tetP3.t[i]
            qi = tetP3.quadrature_weights[i]
            via_quadrature += qi*float(pt(ri,si,ti))
            # println("pt(r,s,t) = $(pt(ri,si,ti))")
        end
        # weights are given for normalized tet with volume 1. We are
        # computing on the "reference" tet with volume 1/6, hence the
        # rescaling
        via_quadrature *= mKref

        if (abs(exact - via_quadrature) < 1e-10)
            print("Integrating: OK!; ")
        else
            print("Integrating ($(pt)): FAILED!; $(abs(exact - via_quadrature)) < 1e-10 == $(abs(exact - via_quadrature) < 1e-10); ")            
        end
        println("exact=$(exact) vs. $(via_quadrature), diff=$(abs(exact-via_quadrature))")        
    end

end

# testQuadratureRule()

end
