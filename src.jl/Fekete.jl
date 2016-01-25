module Fekete

using SymPy
using PyPlot

using Optim
using JLD

function buildFeketePtsP4()

    # initial alpha guess    
    alpha_min = 0.55
    alpha_max = 0.9
    beta_min = 0.05
    beta_max = 0.9
    
    alpha_opt = 0.8273268324217696
    beta_opt = 0.3503732658239263
    alpha0 = alpha_opt
    beta0 = beta_opt

    
    v1 = [-1,-1]
    v2 = [1,-1]
    v3 = [-1,1]
    

    rn(alpha,beta) = [-1,1,-1,
                      0,0,-1,
                      (alpha*v1+(1-alpha)*v2)[1],((1-alpha)*v1+alpha*v2)[1],
                      (alpha*v2+(1-alpha)*v3)[1],((1-alpha)*v2+alpha*v3)[1],
                      (alpha*v3+(1-alpha)*v1)[1],((1-alpha)*v3+alpha*v1)[1],
                      (beta*v1+(1-beta)*(v1+v2+v3)/3)[1],
                      (beta*v2+(1-beta)*(v1+v2+v3)/3)[1],
                      (beta*v3+(1-beta)*(v1+v2+v3)/3)[1]]
    sn(alpha,beta) = [-1,-1,1,
                      -1,0,0,
                      (alpha*v1+(1-alpha)*v2)[2],((1-alpha)*v1+alpha*v2)[2],
                      (alpha*v2+(1-alpha)*v3)[2],((1-alpha)*v2+alpha*v3)[2],
                      (alpha*v3+(1-alpha)*v1)[2],((1-alpha)*v3+alpha*v1)[2],
                      (beta*v1+(1-beta)*(v1+v2+v3)/3)[2],
                      (beta*v2+(1-beta)*(v1+v2+v3)/3)[2],
                      (beta*v3+(1-beta)*(v1+v2+v3)/3)[2]]
    

    if !isfile("p4_alpha_beta_opt.jld")
        
        plotElement2D(v1,v2,v3,1)
        figure(1)
        PyPlot.plot(rn(alpha0,beta0),sn(alpha0,beta0),"ko")

        r,s = symbols("r,s")
        P4 = [1,r,s,r^2,s^2,r*s,r^2*s,r*s^2,r^3,s^3,r^2*s^2,r^3*s,r*s^3,r^4,s^4]
        println("P4=$(P4)")

        npts = 30
        alpha = linspace(0.51,0.99,npts)
        beta = linspace(0.1,0.9,npts)
        detV = zeros(npts)
        detV2 = zeros(npts)
        for (n,(alpha_n,beta_n)) in enumerate(zip(alpha,beta))
            rna = rn(alpha_n,beta_opt)
            sna = sn(alpha_n,beta_opt)
            rna2 = rn(alpha_opt,beta_n)
            sna2 = sn(alpha_opt,beta_n)
            detV[n] = det(Vandermonde2D(P4,r,s,rna,sna))
            detV2[n] = det(Vandermonde2D(P4,r,s,rna2,sna2))
        end
        figure(2)
        clf()
        subplot(2,1,1)
        PyPlot.plot(alpha,detV,alpha_opt,det(Vandermonde2D(P4,r,s,rn(alpha_opt,beta_opt),sn(alpha_opt,beta_opt))),"ro")
        subplot(2,1,2)
        PyPlot.plot(beta,detV2,beta_opt,det(Vandermonde2D(P4,r,s,rn(alpha_opt,beta_opt),sn(alpha_opt,beta_opt))),"ro")
        
        # Find biggest Vandermonde determinant as a function of alpha.
        # Have to make negative because `optimize` finds minimums not maximums        
        f_opt(alpha_beta) = -det(Vandermonde2D(P4,r,s,rn(alpha_beta[1],alpha_beta[2]),sn(alpha_beta[1],alpha_beta[2])))
        d1 = DifferentiableFunction(f_opt)
        # result = optimize(f_opt,[alpha0,beta0])
        l = [alpha_min,beta_min]
        u = [alpha_max,beta_max]
        result = fminbox(d1,[alpha0,beta0],l,u)
        alphabeta_opt = result.minimum
        alpha_opt=alphabeta_opt[1]
        beta_opt=alphabeta_opt[2]
        println("alpha_opt=$(alpha_opt)")
        println("beta_opt=$(beta_opt)")
        println("Saving alpha_opt")
        jldopen("p4_alpha_beta_opt.jld","w") do file
            write(file,"alpha_opt",alphabeta_opt[1])
            write(file,"beta_opt",alphabeta_opt[2])
        end
        # println("result=$(result)")
        # figure(2)
        # clf()
        # PyPlot.plot(alpha,detV,"k*-")
        # PyPlot.plot([alpha_opt],[-f_opt(alpha_opt)],"ro")
        
    else
        println("Loading precomputed alphabeta_opt")
        jldopen("p4_alpha_beta_opt.jld") do file
            alpha_opt = read(file,"alpha_opt")
            beta_opt = read(file,"beta_opt")
            println("alpha_opt=$(alpha_opt)")
            println("beta_opt=$(beta_opt)")
        end
    end
    
    return (rn(alpha_opt,beta_opt),sn(alpha_opt,beta_opt))
        
end



function buildFeketePtsP3()

    # initial alpha guess
    alpha0 = 0.72
    alpha_opt = 0.7236067977499789
    
    v1ref = [-1,-1]
    v2ref = [1,-1]
    v3ref = [-1,1]
    
    rn(alpha) = [-1,1,-1,(1-2*alpha),2*alpha-1,
                 (alpha*[1,-1]+(1-alpha)*[-1,1])[1],
                 ((1-alpha)*[1,-1]+alpha*[-1,1])[1],
                 -1,-1,((v1ref+v2ref+v3ref)/3)[1]]
    sn(alpha) = [-1,-1,1,-1,-1,
                 (alpha*[1,-1]+(1-alpha)*[-1,1])[2],
                 ((1-alpha)*[1,-1]+alpha*[-1,1])[2],
                 2*alpha-1,1-2*alpha,((v1ref+v2ref+v3ref)/3)[2]]    

    if !isfile("p3_alpha_opt.jld")
        
        plotElement2D(v1ref,v2ref,v3ref,1)
        figure(1)
        PyPlot.plot(rn(alpha_opt),sn(alpha_opt),"ko")

        r,s = symbols("r,s")
        P3 = [1,r,s,r^2,s^2,r*s,r^2*s,r*s^2,r^3,s^3]
        println("P3=$(P3)")

        npts = 50
        alpha = linspace(0.51,0.99,npts)
        detV = zeros(npts)
        for (n,alpha_n) in enumerate(alpha)
            rna = rn(alpha_n)
            sna = sn(alpha_n)
            detV[n] = det(Vandermonde2D(P3,r,s,rna,sna))
        end

        # Find biggest Vandermonde determinant as a function of alpha.
        # Have to make negative because `optimize` finds minimums not maximums
        f_opt(alpha) = -det(Vandermonde2D(P3,r,s,rn(alpha),sn(alpha)))
        result = optimize(f_opt,0.6,0.8)
        alpha_opt = result.minimum
        println("Saving alpha_opt")
        jldopen("p3_alpha_opt.jld","w") do file
            write(file,"alpha_opt",alpha_opt)
        end
        println("result=$(result)")
        figure(2)
        clf()
        PyPlot.plot(alpha,detV,"k*-")
        PyPlot.plot([alpha_opt],[-f_opt(alpha_opt)],"ro")
            
    else
        println("Loading precomputed alpha_opt")
        jldopen("p3_alpha_opt.jld") do file
            alpha_opt = read(file,"alpha_opt")
        end
    end
    
    return (rn(alpha_opt),sn(alpha_opt))
        
end

function buildPhi(rs_n,r,s)

    (rn,sn) = rs_n
    P3 = [1,r,s,r^2,s^2,r*s,r^2*s,r*s^2,r^3,s^3]
    V = Vandermonde2D(P3,r,s,rn,sn)

    # coefficients for Lagrange polys
    P_a = Vector{Float64}[]
    N = length(P3)
    for n=1:N
        an = V\(eye(N)[:,n])
        push!(P_a,an)
    end    

    # build Lagrange poly
    phi = Sym[]    
    for i=1:length(P3)
        phi_i = dot(P_a[i],P3)
        push!(phi,phi_i)
    end

    return phi
end

function buildFeketeWeights(rs)
    (rn,sn) = rs
    r,s = symbols("r,s")

    phi=buildPhi(rs,r,s)
    
    m_ref = integrate(1.0+0*r,(r,-1,-s),(s,-1,1))
    # build Lagrange poly
    N = length(phi)
    wi = zeros(N)
    for i=1:N        
        wi[i] = 1/m_ref*integrate(phi[i],(r,-1,-s),(s,-1,1))
    end

    # test quadrature rule
    P3 = [1,r,s,r^2,s^2,r*s,r^2*s,r*s^2,r^3,s^3]
    for p in vcat(P3,[r^2*s^2,r^4,s^4])
        int_p_exact = integrate(p,(r,-1,-s),(s,-1,1))
        int_p_quadr = 0.0
        for i=1:N
            int_p_quadr += wi[i]*subs(p,(r,rn[i]),(s,sn[i]))
        end
        int_p_quadr *= m_ref
        println("$(p):int_p_exact - int_p_quadr=$(int_p_exact-int_p_quadr)")
    end    
    
    return wi
end

function buildReferenceMatrices(rs_n)

    (rn,sn) = rs_n
    npts = length(rn)
    Krr = zeros(npts,npts)
    Krs = zeros(npts,npts)
    Ksr = zeros(npts,npts)
    Kss = zeros(npts,npts)
        
    if !isfile("p3_fekete_matrix.jld")
        println("Computing Krr,etc, matrices for P3 fekete points")
        (rn,sn) = rs_n
        r,s = symbols("r,s")
        phi = buildPhi(rs_n,r,s)

        m_ref = integrate(1.0+0*r,(r,-1,-s),(s,-1,1))
        npts = length(phi)
        
        phi_r = zeros(Sym,npts)
        phi_s = zeros(Sym,npts)
        for i=1:npts
            phi_r[i] = diff(phi[i],r)
            phi_s[i] = diff(phi[i],s)
        end
        for i=1:npts
            for j=1:npts
                Krr[i,j] = 1/m_ref*integrate(phi_r[i]*phi_r[j],(r,-1,-s),(s,-1,1))
                Krs[i,j] = 1/m_ref*integrate(phi_r[i]*phi_s[j],(r,-1,-s),(s,-1,1))
                Kss[i,j] = 1/m_ref*integrate(phi_s[i]*phi_s[j],(r,-1,-s),(s,-1,1))            
            end
        end

        # don't recompute, just use tranpose
        Ksr = Krs'
        
        jldopen("p3_fekete_matrix.jld","w") do file
            write(file,"Krr",Krr)
            write(file,"Krs",Krs)
            write(file,"Kss",Kss)
            write(file,"Ksr",Ksr)
        end
    else
        println("Loading precomputed Krr,etc, matrices for P3 fekete points")
        jldopen("p3_fekete_matrix.jld") do file
            Krr = read(file,"Krr")
            Krs = read(file,"Krs")
            Kss = read(file,"Kss")
            Ksr = read(file,"Ksr")            
        end        
    end
    return (Krr,Krs,Kss,Ksr)
end

function Vandermonde2D(P3,r,s,rn,sn)

    Nt = length(P3)
    V = zeros(Nt,Nt)
    for i=1:Nt
        for j=1:Nt
            V[i,j] = subs(P3[j],(r,rn[i]),(s,sn[i]))
        end
    end
    condV = cond(V)
    println("Condition number of V = $(condV), |V|=$(det(V))")    

    if condV > (1e14)
        # error("Legendre Matrix 'V' likely singular -- check polynomial terms")
        println("Legendre Matrix 'V' likely singular -- check polynomial terms")
    end
    return V
    
end


function plotElement2D(v1,v2,v3,fig; clearfig=true)
    x_n = [v1[1],v2[1],v3[1]]
    y_n = [v1[2],v2[2],v3[2]]

    element_r = [v1[1],v2[1],v3[1],v1[1],v3[1],v2[1]]
    element_s = [v1[2],v2[2],v3[2],v1[2],v3[2],v2[2]]
    figure(fig)
    if clearfig
        clf()
    end
    PyPlot.plot(element_r,element_s,"k-")
    PyPlot.plot([x_n[1]],[y_n[1]],"ro")
    PyPlot.plot([x_n[2]],[y_n[2]],"go")
    PyPlot.plot([x_n[3]],[y_n[3]],"bo")
    axis(:equal)
    PyPlot.xlabel("x")
    PyPlot.ylabel("y")
end


end
