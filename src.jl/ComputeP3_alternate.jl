module ComputeP3_alternate

using PyPlot
using SymPy
using JLD
using ProgressMeter

export nodes_tetP3_hesthaven

function nodes_tetP3_hesthaven()

    # given from Hesthaven & Warburton
    r = [ -1 , -0.447213595499958 ,
          0.447213595499958 , 1 , -1 , -0.333333333333333 ,
          0.447213595499958 , -1 , -0.447213595499958 , -1 , -1 ,
          -0.333333333333333 , 0.447213595499958 , -1 , -0.333333333333333 ,
          -1 , -1 , -0.447213595499958 , -1 , -1 ]

    s = [ -1 , -1 , -1 , -1 , -0.447213595499958 , -0.333333333333333,
          -0.447213595499958 , 0.447213595499958 , 0.447213595499958 , 1 ,
          -1 , -1 , -1 , -0.333333333333333 , -0.333333333333333 ,
          0.447213595499958 , -1 , -1 , -0.447213595499958 , -1 ]
    
    t = [ -1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 ,
          -0.447213595499958 , -0.333333333333333 , -0.447213595499958 ,
          -0.333333333333333 , -0.333333333333333 , -0.447213595499958 ,
          0.447213595499958 , 0.447213595499958 , 0.447213595499958 , 1 ]

    # shift and rescale to [0,1]x3 reference tetrahedra
    r = (r + 1)/2
    s = (s + 1)/2
    t = (t + 1)/2
    
    return (r,s,t)
    
end

function nodes_tetP4_hesthaven()
    
    r = [                -1 , -0.654653670707977 ,                 0 , 0.654653670707977 ,                 1 ,                -1 , -0.551583572090994 , 0.103167144181987 , 0.654653670707977 ,                -1 , -0.551583572090994 , -2.34617731248651e-17 ,                -1 , -0.654653670707977 ,                -1 ,                -1 , -0.551583572090994 , 0.103167144181987 , 0.654653670707977 ,                -1 ,              -0.5 , 0.103167144181987 ,                -1 , -0.551583572090994 ,                -1 ,                -1 , -0.551583572090994 ,                 0 ,                -1 , -0.551583572090994 ,                -1 ,                -1 , -0.654653670707977 ,                -1 ,                -1 ]
    s = [               -1 ,                -1 ,                -1 ,                -1 ,                -1 , -0.654653670707977 , -0.551583572090994 , -0.551583572090994 , -0.654653670707977 , -6.40987562127854e-17 , 0.103167144181987 , -6.40987562127854e-17 , 0.654653670707977 , 0.654653670707977 ,                 1 ,                -1 ,                -1 ,                -1 ,                -1 , -0.551583572090994 ,              -0.5 , -0.551583572090994 , 0.103167144181987 , 0.103167144181987 , 0.654653670707977 ,                -1 ,                -1 ,                -1 , -0.551583572090993 , -0.551583572090993 ,                 0 ,                -1 ,                -1 , -0.654653670707977 ,                -1 ]
    t = [               -1 ,                -1 ,                -1 ,                -1 ,                -1 ,                -1 ,                -1 ,                -1 ,                -1 ,                -1 ,                -1 ,                -1 ,                -1 ,                -1 ,                -1 , -0.654653670707977 , -0.551583572090993 , -0.551583572090994 , -0.654653670707977 , -0.551583572090993 ,              -0.5 , -0.551583572090993 , -0.551583572090994 , -0.551583572090994 , -0.654653670707977 ,                 0 , 0.103167144181987 ,                 0 , 0.103167144181987 , 0.103167144181987 ,                 0 , 0.654653670707977 , 0.654653670707977 , 0.654653670707977 ,                 1 ]

    # shift and rescale to [0,1]x3 reference tetrahedra
    r = (r + 1)/2
    s = (s + 1)/2
    t = (t + 1)/2
    
    return (r,s,t)
    
end
function nodes_tetP5_hesthaven()
r = [                -1 , -0.765055323929465 , -0.285231516480645 , 0.285231516480645 , 0.765055323929465 ,                 1 ,                -1 , -0.68854346464994 , -0.165752193680209 , 0.377086929299879 , 0.765055323929465 ,                -1 , -0.668495612639582 , -0.165752193680209 , 0.285231516480645 ,                -1 , -0.68854346464994 , -0.285231516480645 ,                -1 , -0.765055323929465 ,                -1 ,                -1 , -0.68854346464994 , -0.165752193680209 , 0.377086929299879 , 0.765055323929465 ,                -1 , -0.622331815219382 , -0.133004554341855 , 0.377086929299879 ,                -1 , -0.622331815219382 , -0.165752193680209 ,                -1 , -0.68854346464994 ,                -1 ,                -1 , -0.668495612639582 , -0.165752193680209 , 0.285231516480645 ,                -1 , -0.622331815219382 , -0.165752193680209 ,                -1 , -0.668495612639582 ,                -1 ,                -1 , -0.68854346464994 , -0.285231516480645 ,                -1 , -0.68854346464994 ,                -1 ,                -1 , -0.765055323929465 ,                -1 ,                -1 ]
s = [               -1 ,                -1 ,                -1 ,                -1 ,                -1 ,                -1 , -0.765055323929465 , -0.68854346464994 , -0.668495612639582 , -0.68854346464994 , -0.765055323929465 , -0.285231516480645 , -0.165752193680209 , -0.165752193680209 , -0.285231516480645 , 0.285231516480645 , 0.377086929299879 , 0.285231516480645 , 0.765055323929465 , 0.765055323929465 ,                 1 ,                -1 ,                -1 ,                -1 ,                -1 ,                -1 , -0.68854346464994 , -0.622331815219382 , -0.622331815219382 , -0.68854346464994 , -0.165752193680209 , -0.133004554341855 , -0.165752193680209 , 0.377086929299879 , 0.377086929299879 , 0.765055323929465 ,                -1 ,                -1 ,                -1 ,                -1 , -0.668495612639582 , -0.622331815219382 , -0.668495612639582 , -0.165752193680209 , -0.165752193680209 , 0.285231516480645 ,                -1 ,                -1 ,                -1 , -0.688543464649939 , -0.688543464649939 , -0.285231516480645 ,                -1 ,                -1 , -0.765055323929465 ,                -1 ]
t=[               -1 ,                -1 ,                -1 ,                -1 ,                -1 ,                -1 ,                -1 ,                -1 ,                -1 ,                -1 ,                -1 ,                -1 ,                -1 ,                -1 ,                -1 ,                -1 ,                -1 ,                -1 ,                -1 ,                -1 ,                -1 , -0.765055323929465 , -0.688543464649939 , -0.668495612639582 , -0.688543464649939 , -0.765055323929465 , -0.688543464649939 , -0.622331815219382 , -0.622331815219382 , -0.688543464649939 , -0.668495612639582 , -0.622331815219382 , -0.668495612639582 , -0.688543464649939 , -0.688543464649939 , -0.765055323929465 , -0.285231516480645 , -0.165752193680209 , -0.165752193680209 , -0.285231516480645 , -0.165752193680209 , -0.133004554341855 , -0.165752193680209 , -0.165752193680209 , -0.165752193680209 , -0.285231516480645 , 0.285231516480645 , 0.377086929299879 , 0.285231516480645 , 0.377086929299879 , 0.377086929299879 , 0.285231516480645 , 0.765055323929465 , 0.765055323929465 , 0.765055323929465 ,                 1 ]

# shift and rescale to [0,1]x3 reference tetrahedra
    r = (r + 1)/2
    s = (s + 1)/2
    t = (t + 1)/2
    
    return (r,s,t)
    
end
    
function buildCoefficients(rst,rst_n,Pall)
    (r,s,t) = rst
    (rn,sn,tn) = rst_n
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

    if condA > (1/1e-10)
        error("Legendre Matrix 'A' likely singular -- check polynomial terms")
    end
    
    P_a = Vector{Float64}[]
    for n=1:N
        an = A\(eye(N)[:,n])
        push!(P_a,an)
    end
    return P_a
end

function buildKij(ij_phi_rst)

    (ij,phi,rst) = ij_phi_rst
    (i,j) = ij
    (r,s,t) = rst
    mKref = 1/6
    phiI_x_phiJ_rr = diff(phi[i],r)*diff(phi[j],r)
    Kij_rr = (1/mKref)*integrate(phiI_x_phiJ_rr,(r,0,1-s-t),(s,0,1-t),(t,0,1))
    
    phiI_x_phiJ_rs = diff(phi[i],r)*diff(phi[j],s)
    Kij_rs = (1/mKref)*integrate(phiI_x_phiJ_rs,(r,0,1-s-t),(s,0,1-t),(t,0,1))
    
    phiI_x_phiJ_rt = diff(phi[i],r)*diff(phi[j],t)
    Kij_rt = (1/mKref)*integrate(phiI_x_phiJ_rt,(r,0,1-s-t),(s,0,1-t),(t,0,1))
    
    phiI_x_phiJ_ss = diff(phi[i],s)*diff(phi[j],s)
    Kij_ss = (1/mKref)*integrate(phiI_x_phiJ_ss,(r,0,1-s-t),(s,0,1-t),(t,0,1))
    
    phiI_x_phiJ_st = diff(phi[i],s)*diff(phi[j],t)
    Kij_st = (1/mKref)*integrate(phiI_x_phiJ_st,(r,0,1-s-t),(s,0,1-t),(t,0,1))
    
    phiI_x_phiJ_tt = diff(phi[i],t)*diff(phi[j],t)
    Kij_tt = (1/mKref)*integrate(phiI_x_phiJ_tt,(r,0,1-s-t),(s,0,1-t),(t,0,1))
    
    return (Kij_rr,Kij_rs,Kij_rt,Kij_ss,Kij_st,Kij_tt)
    
end

function buildMij(ij_phi_rst)

    (ij,phi,rst) = ij_phi_rst
    (i,j) = ij
    (r,s,t) = rst
    mKref = 1/6
    phiI_x_phiJ = phi[i]*phi[j]
    Mij = (1/mKref)*integrate(phiI_x_phiJ,(r,0,1-s-t),(s,0,1-t),(t,0,1))    
    return Mij
    
end

function buildPhiP3Alt()

    (rn,sn,tn) = nodes_tetP3_hesthaven()

    r,s,t = symbols("r,s,t")

    Pall = [1+0*r,
            r,r^2,r^3,
            s,s^2,s^3,
            t,t^2,t^3,
            r*s,r^2*s,r*s^2,
            r*t,r^2*t,r*t^2,
            s*t,s^2*t,s*t^2,
            r*s*t]

    pa = buildCoefficients((r,s,t),(rn,sn,tn),Pall)

    phi = Sym[]

    for pai in pa
        push!(phi,dot(pai,Pall))
    end

end

function buildPhiP3AltXYZ(rst_symbols,x,y,z)

    (r,s,t) = rst_symbols

    Pall = [1+0*r,
            r,r^2,r^3,
            s,s^2,s^3,
            t,t^2,t^3,
            r*s,r^2*s,r*s^2,
            r*t,r^2*t,r*t^2,
            s*t,s^2*t,s*t^2,
            r*s*t]

    pa = buildCoefficients((r,s,t),(x,y,z),Pall)

    phi = Sym[]

    for pai in pa
        push!(phi,dot(pai,Pall))
    end
    return phi
end

function buildPhiP4AltXYZ(rst_symbols,x,y,z)

    (r,s,t) = rst_symbols

    # 35 members
    Pall = [1+0*r,
            r,r^2,r^3,r^4,
            s,s^2,s^3,s^4,
            t,t^2,t^3,t^4,
            r*s,r^2*s,r*s^2,r^3*s,r*s^3,r^2*s^2,
            r*t,r^2*t,r*t^2,r^3*t,r*t^3,r^2*t^2,
            s*t,s^2*t,s*t^2,s^3*t,s*t^3,s^2*t^2,
            r*s*t,r^2*s*t,r*s^2*t,r*s*t^2
            ]

    if length(Pall) != length(x)
        error("number of points ($(length(x))) and elements in span ($(length(Pall))) must be identical")
    end
    
    pa = buildCoefficients((r,s,t),(x,y,z),Pall)

    phi = Sym[]

    for pai in pa
        push!(phi,dot(pai,Pall))
    end
    return phi
end

function buildPhiP5AltXYZ(rst_symbols,x,y,z)

    (r,s,t) = rst_symbols

    # 56 members
    Pall = [1+0*r,
            r,r^2,r^3,r^4,r^5,
            s,s^2,s^3,s^4,s^5,
            t,t^2,t^3,t^4,t^5,
            r*s,r^2*s,r*s^2,r^3*s,r*s^3,r^2*s^2, r^4*s,r^3*s^2,r^2*s^3,r*s^4,
            r*t,r^2*t,r*t^2,r^3*t,r*t^3,r^2*t^2, r^4*t,r^3*t^2,r^2*t^3,r*t^4,
            s*t,s^2*t,s*t^2,s^3*t,s*t^3,s^2*t^2, s^4*t,s^3*t^2,s^2*t^3,s*t^4,
            r*s*t,r^2*s*t,r*s^2*t,r*s*t^2,       r^3*s*t,r^2*s^2*t,r^2*s*t^2,r*s^2*t^2,r*s^3*t,r*s*t^3
            ]

    if length(Pall) != length(x)
        error("number of points ($(length(x))) and elements in span ($(length(Pall))) must be identical")
    end
    
    pa = buildCoefficients((r,s,t),(x,y,z),Pall)

    phi = Sym[]

    for pai in pa
        push!(phi,dot(pai,Pall))
    end
    return phi
end


function computeK()

    (rn,sn,tn) = nodes_tetP3_hesthaven()

    r,s,t = symbols("r,s,t")

    Pall = [1+0*r,
            r,r^2,r^3,
            s,s^2,s^3,
            t,t^2,t^3,
            r*s,r^2*s,r*s^2,
            r*t,r^2*t,r*t^2,
            s*t,s^2*t,s*t^2,
            r*s*t]

    pa = buildCoefficients((r,s,t),(rn,sn,tn),Pall)

    phi = Sym[]

    for pai in pa
        push!(phi,dot(pai,Pall))
    end

    figure(1)
    clf()
    plot3D(rn,sn,tn,"ko")

    println("Number of Nodes: $(length(rn)), typeof(phi)=$(typeof(phi))")

    K_rr = zeros(length(pa),length(pa))
    K_rs = zeros(length(pa),length(pa))
    K_rt = zeros(length(pa),length(pa))
    K_ss = zeros(length(pa),length(pa))
    K_st = zeros(length(pa),length(pa))
    K_tt = zeros(length(pa),length(pa))
    
    # ij = ((Int64,Int64),Array{Sym,1},(Sym,Sym,Sym))[]
    p = Progress(length(pa), 1)
    for i=1:length(pa)
        for j=1:length(pa)
            # push!(ij,((i,j),phi,(r,s,t)))
            (Kij_rr,Kij_rs,Kij_rt,Kij_ss,Kij_st,Kij_tt) = buildKij(((i,j),phi,(r,s,t)))
            K_rr[i,j] = Kij_rr
            K_rs[i,j] = Kij_rs
            K_rt[i,j] = Kij_rt
            K_ss[i,j] = Kij_ss
            K_st[i,j] = Kij_st
            K_tt[i,j] = Kij_tt
        end
        next!(p)
    end
    
    println("Finished buildK!")
    file = jldopen("K_p3true.jld","w")
    write(file,"K_rr",K_rr)
    write(file,"K_rs",K_rs)
    write(file,"K_rt",K_rt)
    write(file,"K_ss",K_ss)
    write(file,"K_st",K_st)
    write(file,"K_tt",K_tt)
    close(file)
end

function computeM()

(rn,sn,tn) = nodes_tetP3_hesthaven()

    r,s,t = symbols("r,s,t")

    Pall = [1+0*r,
            r,r^2,r^3,
            s,s^2,s^3,
            t,t^2,t^3,
            r*s,r^2*s,r*s^2,
            r*t,r^2*t,r*t^2,
            s*t,s^2*t,s*t^2,
            r*s*t]

    pa = buildCoefficients((r,s,t),(rn,sn,tn),Pall)

    phi = Sym[]

    for pai in pa
        push!(phi,dot(pai,Pall))
    end

    figure(1)
    clf()
    plot3D(rn,sn,tn,"ko")
    
    println("Number of Nodes: $(length(rn)), typeof(phi)=$(typeof(phi))")

    ij = Tuple{Tuple{Int64,Int64},Array{Sym,1},Tuple{Sym, Sym, Sym}}[]
    
    nterms = length(pa)
    M = zeros(nterms,nterms)
    p = Progress(length(pa), 1)
    for i=1:length(pa)        
        for j=1:length(pa)            
            mKref = 1/6            
            M[i,j] = (1/mKref)*integrate(phi[i]*phi[j],(r,0,1-s-t),(s,0,1-t),(t,0,1))
        end
        next!(p)
    end
    
    println("Finished buildM!")
    file = jldopen("M_p3true.jld","w")
    write(file,"M",M)
    close(file)
end

end
