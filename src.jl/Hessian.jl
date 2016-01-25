module Hessian

using FEM
using SEM
using Element
using Adjoint
using SymPy
using TimeStepping
using PyPlot
import PyPlot

export hessianvector

function hessianvector(elem,v_x,EToV, # general simulation context variables
                       x_n,dofindex,cm,
                       alpha,
                       bc_array,forcing,T,
                       station_indexes, # stations at x_n[station_indexes]
                       data_time, # truth recorded at time
                       data_at_stations, # true data recordings
                       s::Vector{Float64}) # test vector

    (un,time_un) = fwd_simulation(elem,v_x,EToV,x_n,dofindex,cm,bc_array,forcing,T)

    if any(isnan(un))
        error("un exploded")
    end

    println("maximum un = $(maximum(abs(un)))")
    
    (adjoint_source_traces,
     adjoint_source_time_n) = buildadjoint_source((un,time_un),
                                                  station_indexes,
                                                  data_time,
                                                  data_at_stations)
    
    (pn,time_un) = adjoint_simulation(elem,v_x,EToV,x_n,dofindex,cm,bc_array,
                                      adjoint_source_traces, adjoint_source_time_n,
                                      station_indexes, T)


    if any(isnan(pn))
        error("pn exploded")
    end
    
    # T3 computation
    (dsu,time_dsu) = fwd_simulation_dsu(elem,v_x,EToV,
                                        x_n,dofindex,cm,bc_array,T,
                                        s,un)

    if any(isnan(dsu))
        error("dsu exploded")
    end
    
    dt = time_dsu[2] - time_dsu[1]
    T3 = zeros(length(cm))
    
    println("length(time_dsu)=$(length(time_dsu))")
    for n=1:length(time_dsu)            
        dx_dsu = fieldtograd_x(elem,elem.diffX,v_x,EToV,dofindex,dsu[:,n])
        dx_pn = fieldtograd_x(elem,elem.diffX,v_x,EToV,dofindex,pn[:,n])
        T3 += dt*(dx_dsu.*dx_pn)
    end
    
    # T1+T2 computation

    # prepare dsu source
    num_stations = length(station_indexes)
    dsu_source = zeros(num_stations,length(time_dsu))
    for r=1:num_stations
        dsu_source[r,:] = dsu[station_indexes[r],:]
    end

    (dsp,time_dsp) = adj_simulation_dsp(elem,v_x,EToV,x_n,
                                        dofindex,cm,bc_array,T,
                                        s,pn,dsu_source,station_indexes)

    if any(isnan(dsp))
        error("dsp exploded")
    end
    
    T1pT2 = zeros(length(cm))
    for n=1:length(time_dsp)
        dx_dsp = fieldtograd_x(elem,elem.diffX,v_x,EToV,dofindex,dsp[:,n])
        dx_un = fieldtograd_x(elem,elem.diffX,v_x,EToV,dofindex,un[:,n])
        T1pT2 += dt*(dx_dsp.*dx_un)
    end

    Kx = buildK_forgrad(elem,v_x,EToV,dofindex,length(x_n))
    T4 = alpha*Kx*s
    
    Hs = -T1pT2 + T3 + T4
    return Hs

    # (M,Minv) = buildM(elem,v_x,EToV,dofindex,length(x_n))
    
    # figure(2)
    # clf()
    # subplot(4,1,1)
    # PyPlot.plot(x_n,-T1pT2,"k-")
    # subplot(4,1,2)    
    # PyPlot.plot(x_n,T3,"k-")
    # subplot(4,1,3)
    # PyPlot.plot(x_n,T4,"k-")
    # subplot(4,1,4)
    # PyPlot.plot(x_n,-T1pT2+T3+T4,"k-")
    
end

# build the sequence of LdL matrices for RHS of T3
function buildLdL(elem::Element1D)

    npts = length(elem.r)
    x = symbols("x")
    LdL = Matrix{Float64}[]
    for i=1:npts

        dli = diff(lagrange1d(elem.r,i,x),x)

        LdLi = zeros(npts,npts)
        for n=1:npts
            ln = lagrange1d(elem.r,n,x)
            for j=1:npts
                dlj = diff(lagrange1d(elem.r,j,x),x)
                LdLi[n,j] = integrate(ln*dlj*dli,(x,-1,1))
            end
        end

        push!(LdL,LdLi)
        
    end
    return LdL
end

# forward simulation for hessian-vector product term T3 (\delta_{s_u})
function fwd_simulation_dsu(elem, v_x, EToV, x_n, dofindex,
                            velocity_model :: Vector{Float64}, bc_array,
                            T,s,un)

    c_m = velocity_model
    ndof = length(x_n)
    
    (M,Minv) = buildM(elem,v_x,EToV,dofindex,ndof)
    
    element_radius = elementradius1d(v_x,EToV)
    h_min = minimum(element_radius./maximum(c_m))

    K=length(EToV)
    m_k_rx2 = zeros(K)
    for k=1:K
        v1 = v_x[EToV[k][1]]
        v2 = v_x[EToV[k][2]]
        rx = 1/(v2-v1)
        m_k = v2-v1
        m_k_rx2[k] = m_k*rx^2
    end

    # LdL = buildLdL(elem)
    
    (dsu,dsu_time_n) = run_timestepping_dsu(x_n,Minv,c_m,m_k_rx2,
                                            bc_array,dofindex,h_min,elem,T,un,s)
    return (dsu,dsu_time_n)

end

# forward simulation for hessian-vector product term T3 (\delta_{s_u})
function adj_simulation_dsp(elem, v_x, EToV, x_n, dofindex,
                            velocity_model :: Vector{Float64}, bc_array,
                            T,s,pn,dsu_source,station_indexes)

    c_m = velocity_model
    ndof = length(x_n)
    
    (M,Minv) = buildM(elem,v_x,EToV,dofindex,ndof)
    
    element_radius = elementradius1d(v_x,EToV)
    h_min = minimum(element_radius./maximum(c_m))

    K=length(EToV)
    m_k_rx2 = zeros(K)
    for k=1:K
        v1 = v_x[EToV[k][1]]
        v2 = v_x[EToV[k][2]]
        rx = 1/(v2-v1)
        m_k = v2-v1
        m_k_rx2[k] = m_k*rx^2
    end

    LdL = buildLdL(elem)

    pn_reversed = flipdim(pn,2)
    dsu_source_reversed = flipdim(dsu_source,2)
    
    (dsp_reversed,
     dsp_time_n) = run_timestepping_dsp(x_n,Minv,c_m,m_k_rx2,LdL,
                                        bc_array,dofindex,h_min,elem,T,
                                        pn_reversed,s,
                                        dsu_source_reversed,station_indexes)

    dsp = flipdim(dsp_reversed,2)
    
    return (dsp,dsp_time_n)

end



end # module
