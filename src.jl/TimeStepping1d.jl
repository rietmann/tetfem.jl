module TimeStepping1d

export run_timestepping, run_timestepping_adjoint, run_timestepping_dsu, run_timestepping_dsp

using Mesh
using Element
using Source
using Base.Test
using PyPlot
using Devectorize
using ProgressMeter

function applyKun(Ke::Vector{Matrix{Float64}},un::Vector{Float64},bc::Vector{Float64},dofindex::Vector{Vector{Int}})
    K = length(dofindex)
    p_Np = length(dofindex[1])
    un_gather = zeros(p_Np)
    ndof = length(un)
    an = zeros(ndof)

    # faster version calls BLAS gemv directly and avoids GC with a temporary variable
    an_k = zeros(p_Np)
    
    for k=1:K
        # gather
        for i=1:p_Np
            idof = dofindex[k][i]
            un_gather[i] = bc[idof]*un[idof]
        end

        an_k[:] = 0.0        
        Base.LinAlg.BLAS.gemv!('N',1.0,Ke[k],un_gather,0.0,an_k)

        # a bit slower than calling gemv directly (~25% slower)
        # an_k = Ke[k]*un_gather
        
        # scatter
        for i=1:p_Np
            idof = dofindex[k][i]
            # note assembly '+='
            an[idof] += bc[idof]*an_k[i]            
        end
    end
    return an
end

function applyKun_mfree(elem::Element1D,m_k_rx2::Vector{Float64},
                        c_m::Vector{Float64},un::Vector{Float64},bc::Vector{Float64},
                        dofindex::Vector{Vector{Int}})

    K=length(dofindex)
    p_Np = length(elem.r)
    un_gather = zeros(elem.p_Np)
    c_m_k = zeros(elem.p_Np)
    an_k = zeros(elem.p_Np)
    ndof = length(un)
    an = zeros(ndof)
    for k=1:K
        un_gather[:] = 0
        an_k[:] = 0
        c_m_k[:] = 0
        # gather
        for i=1:p_Np
            idof = dofindex[k][i]
            un_gather[i] = bc[idof]*un[idof]
            c_m_k[i] = c_m[dofindex[k][i]]
        end        
        for i=1:p_Np
            for j=1:p_Np            
                # apply Kun
                for n=1:elem.p_Np
                    an_k[i] += un_gather[j]*elem.quadrature_weights[n]*c_m_k[n]*elem.d_Phi_n[n,i]*elem.d_Phi_n[n,j]
                end
            end
            an_k[i] *= (m_k_rx2[k])
            # scatter to global dof
            an[dofindex[k][i]] += an_k[i]
        end
    end
    return an
end

# 1D: 2nd-order time-stepping for adjoint wavefield
function run_timestepping_adjoint(x_n::Vector{Float64},
                                  Minv::Vector{Float64},
                                  c_m::Vector{Float64},
                                  m_k_rx2::Vector{Float64}, # scaling for on-the-fly stiffness-matrix calc,
                                  bc::Vector{Float64},
                                  adjoint_source_traces::Matrix{Float64},
                                  adjoint_source_trace_locations::Vector{Int},
                                  dofindex::Vector{Vector{Int}},
                                  h_min::Float64, elem::Element1D,finaltime::Float64)

    dt = elem.cfl_factor*h_min # 1.6 for 4th order leapfrog
    Nsteps = round(Int64,ceil(finaltime/dt))
    
    dt = finaltime/Nsteps    
    ndof = length(x_n)
    un = zeros(ndof,Nsteps+1)
    time_n = zeros(Nsteps+1)
    vn = zeros(ndof)
    an = zeros(ndof)
    an0 = zeros(ndof)

    plot_enabled = false
    plot_every = 10
    
    adjoint_source_traces_reversed = flipdim(adjoint_source_traces,1)
    
    for it=1:Nsteps
        
        # 2nd order newmark        
        b = applyKun_mfree(elem,m_k_rx2,c_m,un[:,it],bc,dofindex)
        t_i = (it)*dt
        time_n[it+1] = t_i
        Nsources = size(adjoint_source_traces)[2]
        for s=1:Nsources
            b[adjoint_source_trace_locations[s]] -= adjoint_source_traces_reversed[it,s]
        end
        
        for n=1:ndof
            vn[n] = vn[n] - bc[n]*dt*Minv[n]*b[n]
            un[n,it+1] = un[n,it] + dt*vn[n]
        end
        
        
        if plot_enabled && it%plot_every == 0
            if plot_enabled
                figure(1)
                clf()
                plot(x_n,un[:,it+1],"k-*")
                title("it=$(it)")
                ylim(-2,2)
                sleep(0.01)
            end
        end
        
    end
    return (un,time_n)
    
end

function applyLdL(elem::Element1D,m_k_rx2::Vector{Float64},
                  c_m::Vector{Float64},un::Vector{Float64},
                  s::Vector{Float64},LdL::Vector{Matrix{Float64}},
                  dofindex::Vector{Vector{Int}})

    K=length(dofindex)
    p_Np = length(elem.r)
    un_gather = zeros(elem.p_Np)
    s_gather = zeros(elem.p_Np)
    rhs_k = zeros(elem.p_Np)
    ndof = length(un)
    rhs = zeros(ndof)
    for k=1:K
        un_gather[:] = 0
        rhs_k[:] = 0
        # gather
        for i=1:p_Np
            idof = dofindex[k][i]
            un_gather[i] = un[idof]
            s_gather[i] = s[idof]
        end

        rhs_k[:] = 0.0
        
        for i=1:p_Np            
            # probably a bit slower than calling gemv directly
            rhs_k[i] = (s_gather.'*LdL[i]*un_gather)[1]
        end
        
        # scatter
        for i=1:p_Np
            idof = dofindex[k][i]
            # note assembly '+='
            # and scaling to current element
            # (rx^2 * m_k) (rx^2 for 2 derivatives, m_k for integral)
            rhs[idof] += (m_k_rx2[k]/2.0)*rhs_k[i]
        end
    end
    return rhs
        
end

# 1D: computation of \delta_{s_p} for T1+T2 with dist. RHS and dsu source
# an "adjoint" type simulation, so run with zero end conditions in reverse time
function run_timestepping_dsp(x_n::Vector{Float64},
                              Minv::Vector{Float64},
                              c_m::Vector{Float64},
                              m_k_rx2::Vector{Float64},
                              # special LdL matrices for RHS
                              LdL::Vector{Matrix{Float64}},
                              bc::Vector{Float64}, dofindex::Vector{Vector{Int}},
                              h_min::Float64, elem::Element1D,finaltime::Float64,
    pn_rhs_reversed::Matrix{Float64},s::Vector{Float64},
    dsu_source_reversed::Matrix{Float64}, station_indexes::Vector{Int}
    )

    dt = elem.cfl_factor*h_min # 1.6 for 4th order leapfrog
    Nsteps = round(Int64,ceil(finaltime/dt))
    
    dt = finaltime/Nsteps    
    ndof = length(x_n)
    un = zeros(ndof,Nsteps+1)
    time_n = zeros(Nsteps+1)
    vn = zeros(ndof)
    an = zeros(ndof)
    an0 = zeros(ndof)

    plot_enabled = false
    plot_every = 10

    for it=1:Nsteps
        
        # 2nd order newmark        
        b = applyKun_mfree(elem,m_k_rx2,c_m,un[:,it],bc,dofindex)

        # build RHS
        # rhs = applyLdL(elem,m_k_rx2,c_m,pn_rhs_reversed[:,it],s,LdL,dofindex)
        # faster, approximate way using quadrature and original stiffness formulation
        rhs = applyKun_mfree(elem,m_k_rx2,s,pn_rhs_reversed[:,it],bc,dofindex)
        
        # -(Ku + F)
        b += rhs
        
        # dsu source
        for (r,idx) in enumerate(station_indexes)
            b[idx] += dsu_source_reversed[r,it]
        end
        
        t_i = (it-1)*dt
        time_n[it] = t_i        
        
        for n=1:ndof
            vn[n] = vn[n] - bc[n]*dt*Minv[n]*b[n]
            un[n,it+1] = un[n,it] + dt*vn[n]
        end
        
        
        if plot_enabled && it%plot_every == 0
            if plot_enabled
                figure(1)
                clf()
                plot(x_n,un[:,it+1],"k-*")
                ylim(-6,6)
                sleep(0.5)
            end
        end
        
    end
    time_n[end] = Nsteps*dt
    return (un,time_n)
    
end


# 1D: computation of \delta_{s_u} for T3 with dist. RHS
function run_timestepping_dsu(x_n::Vector{Float64},
                              Minv::Vector{Float64},
                              c_m::Vector{Float64},
                              m_k_rx2::Vector{Float64},
                              # special LdL matrices for RHS
                              # LdL::Vector{Matrix{Float64}},
                              bc::Vector{Float64}, dofindex::Vector{Vector{Int}},
                              h_min::Float64, elem::Element1D,finaltime::Float64,
                              un_rhs::Matrix{Float64},s::Vector{Float64})

    dt = elem.cfl_factor*h_min # 1.6 for 4th order leapfrog
    Nsteps = round(Int64,ceil(finaltime/dt))
    dt = finaltime/Nsteps    
    ndof = length(x_n)
    un = zeros(ndof,Nsteps+1)
    time_n = zeros(Nsteps+1)
    vn = zeros(ndof)
    an = zeros(ndof)
    an0 = zeros(ndof)

    plot_enabled = false
    plot_every = 10

    for it=1:Nsteps
        
        # 2nd order newmark        
        b = applyKun_mfree(elem,m_k_rx2,c_m,un[:,it],bc,dofindex)

        # build RHS
        # rhs = applyLdL(elem,m_k_rx2,c_m,un_rhs[:,it],s,LdL,dofindex)
        
        # using original stiffness uses quadrature approximation, but
        # is faster and reduces code footprint. It also proves the
        # concept to take advantage of it in SPECFEM3D.
        rhs = applyKun_mfree(elem,m_k_rx2,s,un_rhs[:,it],bc,dofindex)

        # -(K_c u + K_s u)
        b -= rhs
        
        t_i = (it-1)*dt
        time_n[it] = t_i
        
        for n=1:ndof
            vn[n] = vn[n] - bc[n]*dt*Minv[n]*b[n]
            un[n,it+1] = un[n,it] + dt*vn[n]
        end
        
        if any(isnan(un[:,it+1]))
            error("dsu exploded on it=$(it)")
        end
        
        if plot_enabled && it%plot_every == 0
            if plot_enabled
                figure(1)
                clf()
                plot(x_n,un[:,it+1],"k-*")
                ylim(-6,6)
                sleep(0.5)
            end
        end
        
    end
    time_n[end] = Nsteps*dt
    return (un,time_n)
    
end

# 1D: non-exact version with dirichlet bcs
function run_timestepping(x_n::Vector{Float64},
                          Minv::Vector{Float64},
                          c_m::Vector{Float64},
                          m_k_rx2::Vector{Float64}, # scaling for on-the-fly stiffness-matrix calc
                          bc::Vector{Float64},
                          forcing,
                          dofindex::Vector{Vector{Int}},
                          h_min::Float64, elem::Element1D,finaltime::Float64)

    dt = elem.cfl_factor*h_min # 1.6 for 4th order leapfrog
    Nsteps = round(Int64,ceil(finaltime/dt))    
    dt = finaltime/Nsteps    
    ndof = length(x_n)
    un = zeros(ndof,Nsteps+1)
    time_n = zeros(Nsteps+1)
    vn = zeros(ndof)
    an = zeros(ndof)
    an0 = zeros(ndof)

    plot_enabled = false
    plot_every = 10

    check_enabled = true
    check_every = 100
    
    for it=1:Nsteps
        
        # 4th order leapfrog
        # b = applyKun(Ke,un,bc,dofindex)

        # # apply source
        # t_i = it*dt
        # b[forcing.source_idx] -= forcing.source_func(t_i)
        
        # for n=1:ndof
        #     an0[n] = bc[n]*Minv[n]*b[n]
        # end
        
        # b2 = applyKun(Ke,an0,bc,dofindex)
        # # b2[forcing.source_idx] -= forcing.source_func(t_i)        
        # for n=1:ndof
        #     an[n] = bc[n]*Minv[n]*b2[n]
        #     unp1[n] = 2*un[n] - unm1[n] - dt^2*an0[n] + dt^4/12.0*an[n]
        # end

        # unm1 = copy(un)
        # un = copy(unp1)
        
        # 2nd order newmark        
        b = applyKun_mfree(elem,m_k_rx2,c_m,un[:,it],bc,dofindex)

        # test = (1+0.025.*sin(2*pi*x_n)).*(1 + 0.05*exp(-(x_n-6.0).^2/2))
        # Ktest = (Minv).*applyKun_mfree2(elem,m_k_rx2,ones(length(c_m)),test,bc,dofindex)
        # plot(x_n,test,"k-",x_n,Ktest,"g-")
        # ylim(0,1.1)
        # error("stopping")
        # figure(1)
        # plot(x_n,abs(b-b2),"k-")
        # sleep(1)
        t_i = (it-1)*dt
        time_n[it] = t_i
        b[forcing.source_idx] -= forcing.source_func(t_i)
        for n=1:ndof
            vn[n] = vn[n] - bc[n]*dt*Minv[n]*b[n]
            un[n,it+1] = un[n,it] + dt*vn[n]
        end
        # println("un[121]=$(un[121,it+1]), forcing.source_func(t_i)=$(forcing.source_func(t_i))")
        if check_enabled && it%check_every == 0 && maximum(abs(un[:,it+1])) > 1000
            error("Solution blowing up")
        end
        
        if plot_enabled && it%plot_every == 0
            if plot_enabled
                figure(1)
                clf()
                plot(x_n,un[:,it+1],"k-*")
                ylim(-6,6)
                sleep(0.5)
            end
        end
        
    end
    time_n[end] = Nsteps*dt
    return (un,time_n)
    
end

# 1D with exact solution using lowest mode sin wave
function run_timestepping_exact(x_n::Vector{Float64},
                          Minv::Vector{Float64},
                          Ke::Vector{Matrix{Float64}},
                          bc::Vector{Float64},
                          dofindex::Vector{Vector{Int}},
                          h_min::Float64, elem::Element1D,finaltime::Float64)

    dt = elem.cfl_factor*h_min*1.6 # 1.6 for 4th order leapfrog
    Nsteps = int(ceil(finaltime/dt))
    dt = finaltime/Nsteps
    println("$(Nsteps) steps at dt=$(dt)")
    un = cos(pi/4*sqrt(2)*0*dt)*sin(pi/2*x_n)
    unm1 = cos(pi/4*sqrt(2)*(-1)*dt)*sin(pi/2*x_n)
    unp1 = zeros(length(un))
    vn = zeros(length(un))
    an = zeros(length(un))
    an0 = zeros(length(un))

    un = sin(pi/2*x_n)
    vn = zeros(length(un))

    ndof = length(x_n)
    
    error_check_exact = true
    for it=1:Nsteps

        # 4th order leapfrog
        b = applyKun(Ke,un,bc,dofindex)
        for n=1:ndof
            an0[n] = Minv[n]*b[n]
        end
        
        b2 = applyKun(Ke,an0,bc,dofindex)
        for n=1:ndof
            an[n] = Minv[n]*b2[n]
            unp1[n] = 2*un[n] - unm1[n] - dt^2*an0[n] + dt^4/12.0*an[n]
        end

        unm1 = copy(un)
        un = copy(unp1)
        
        # 2nd order Newmark
        # an = applyKun(Ke,un,bc,dofindex)
        # an = -Minv.*an

        # # half step first step
        # if it==1
        #     vn = vn + dt/2*an;
        # else
        #     vn = vn + dt*an;
        # end
        
        # un = un + dt*vn
        
        if it%1 == 0
            if error_check_exact
                un_exact = cos(pi/4*sqrt(2)*it*dt)*sin(pi/2*x_n)
                error_exact = norm(un-un_exact,Inf)

                clf()
                plot(x_n,un,"k-*",x_n,un_exact,"r-o")
                ylim(-1,1)
                sleep(0.1)
                
                println("|Error|_inf exact @ T=$(it*dt): $(error_exact)")
                if error_exact > 1
                    error("Check \"norm(un-un_exact,Inf)=$(norm(un-un_exact,Inf))\" failed on step $(it)")
                end
            end
            
        end
    end
    
    if error_check_exact
        un_exact_end = cos(pi/4*sqrt(2)*Nsteps*dt)*sin(pi/2*x_n)
        final_error = norm(un-un_exact_end,Inf)
        # final_error = maximum(abs(un-un_exact_end))  # norm(un-un_exact_end,Inf)
        println("|Error|_inf exact @ Tend=$(finaltime): $(final_error)")
        return (un,final_error)
    else
        return (un,0.0)
    end
        
end


end
