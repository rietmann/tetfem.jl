module SEM

using SymPy

export lagrange1d, deriveGaussQuadrature1d, stiffness1d, stiffness1d_with_cm, precompute_dphi, precompute_dphi_matrix, buildDerivativeMatrix

function lagrange1d(r :: Vector{Float64},n :: Int,x :: Sym)
    
    if n > length(r)
        error("Lagrange polynomial n cannot be higher than number of collocation points")
    end
    poly = 1.0
    
    for j=1:length(r)
        if j != n
            r_n=r[n]
            r_j=r[j]
            poly = poly*(x-r[j])/(r[n]-r[j]);
        end
    end
    return poly :: Sym
end

function deriveGaussQuadrature1d(r :: Vector{Float64})

    Nn = length(r);

    wi = zeros(length(r))
    x = symbols("x")
    for n=1:Nn
        phi_n = lagrange1d(r,n,x)
        wi[n] = 1/2*integrate(phi_n,(x,-1.0,1.0));        
    end
    
    return wi
end

function buildDerivativeMatrix(r :: Vector{Float64})

    Nn = length(r);
    x = symbols("x")
    diffX = zeros(Float64,Nn,Nn)
    for i=1:Nn
        for j=1:Nn
            d_phi_i = diff(lagrange1d(r,j,x),x)
            if Nn > 2
                # reference element has length 2
                diffX[i,j] = 2*d_phi_i(r[i])
            else
                diffX[i,j] = 2*d_phi_i
            end
        end
    end
    
    return diffX
    
end

function precompute_dphi(element)

    N = length(element.r)
    x = symbols("x")
    d_phi = Vector{Float64}[]
    for i=1:N
        push!(d_phi, zeros(Float64,N))
        d_phi_i = 2*diff(lagrange1d(element.r,i,x),x)
        for n=1:N
            if N > 2
                d_phi[i][n] = d_phi_i(element.r[n])
            else
                d_phi[i][n] = d_phi_i
            end
        end
    end
    return d_phi
end

function precompute_dphi_matrix(r)
   
    N = length(r)
    x = symbols("x")
    d_phi = zeros(N,N)
    for i=1:N
        d_phi_i = 2*diff(lagrange1d(r,i,x),x)
        for n=1:N
            if N>2
                d_phi[n,i] = d_phi_i(r[n])
            else
                d_phi[n,i] = d_phi_i
            end
        end
    end
    return d_phi
end

function stiffness1d_with_cm(r,d_phi_i_n,cm,quad)

    N = length(r)
            
    d_Phi_rr = zeros(N,N)
    for i=1:N
        for j=1:N            
            # normalized for reference element has measure = 2.0
            d_Phi_rr_sum = (2.0)*sum([quad[n]*cm[n]*d_phi_i_n[i][n]*d_phi_i_n[j][n] for (n,rn) in enumerate(r)])

            # using gaussian quadrature is exact for this stiffness matrix!
            # d_Phi_rr_exact = (1.0/2.0)*integrate(diff(phi[i],x)*diff(phi[j],x),(x,-1,1))
            # println("diff = $(d_Phi_rr_exact) vs. $(d_Phi_rr_sum)")
            d_Phi_rr[i,j] = d_Phi_rr_sum
        end
    end

    return d_Phi_rr
    
end


function stiffness1d(r)

    N = length(r)
    x = symbols("x")
    phi = zeros(Sym,N)
    for n=1:N
        phi[n] = lagrange1d(r,n,x)
    end

    d_Phi_rr = zeros(N,N)
    for i=1:N
        for j=1:N            
            # normalized for reference element has measure = 2.0 (two
            # derivatives cancel one integral scaling)
            d_Phi_rr[i,j] = 2*integrate(diff(phi[i],x)*diff(phi[j],x),(x,-1,1))
        end
    end

    return d_Phi_rr
    
end

end
