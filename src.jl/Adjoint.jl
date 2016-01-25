module Adjoint

using Match

using Element
using Mesh
using Nodes
using PyPlot
using PyCall
using FEM
using TimeStepping
using SEM

import SymPy

import Source

import Ipopt
# import NLopt

using Grid

using MathProgBase

export buildadjoint_source, mesh_setup, ground_truth, fwd_simulation, adjoint_simulation, fieldtograd_x,fieldtograd_x2, misfit, applyK_forgrad, buildK_forgrad, regularization, prepare_gradient, find_optimal_model

@pyimport scipy.integrate as scipy_integrate
@pyimport scipy.interpolate as scipy_interpolate

function buildadjoint_source(un_t :: Tuple{Matrix{Float64},Vector{Float64}}, # (un,time)
                             station_idxs :: Vector{Int},
                             data_time :: Vector{Float64},
                             data_at_stations :: Matrix{Float64})                            

    (un, un_time) = un_t
    dt = un_time[2]-un_time[1]
    
    num_stations = length(station_idxs)
    
    num_steps = size(un)[2] # un = zeros(ndof,Nsteps+1)
    adjoint_source = zeros(num_steps,num_stations)

    num_stations = length(station_idxs)
    plot_stations = 2
    
    for (i,station_idx) in enumerate(station_idxs)        
        interp_data = zeros(length(un_time))
        data_data = data_at_stations[:,i]
        # "ground truth data" may not have same time points as
        # "synthetic" un_t, so we interpolate.
        data_interp = InterpIrregular(data_time,data_data,BCnearest, InterpLinear)
        # data_interp_f = scipy_interpolate.interp1d(data_time,data_data,"cubic")
        for (n,time_n) in enumerate(un_time)
            # fix for broken "BCnearest". We are getting NaN when running of edges of data_time
            data_at_time_n = data_interp[time_n]
            if isnan(data_at_time_n)
                println("fixing NaN in adjoint source")                
                data_at_time_n = data_data[end]
            end
            # data_at_time_n = data_interp_f(time_n)[1]
            interp_data[n] = data_at_time_n
            adjoint_source[n,i] = un[station_idx,n] - data_at_time_n
            
        end
        if any(isnan(adjoint_source[:,i]))
            error("adjoint_source has NaN @ i=$(i)\n adjoint_source=$(adjoint_source[:,i])")
        end
        # if i <= plot_stations
        #     subplot(plot_stations,1,i)
        #     # plot(un_time,adjoint_source[:,i])
        #     plot(un_time,abs(interp_data-data_data),"k-*")
        # end
    end
    
    
    return (adjoint_source,un_time)
    
end

function materialVelocity_perturbation2(x_n::Vector{Float64},c0::Float64,x0::Float64,amt::Float64)
    
    ndof = length(x_n)
    c_m = zeros(ndof)
    for n=1:ndof
        c_m[n] = c0 + amt*exp(-(x_n[n]-x0)^2/2)
    end    
    figure(2)
    clf()
    plot(x_n,c_m,"k-*")
    ylim(0,2)
    title("velocity model")
    return c_m
    
end

function materialVelocity_perturbation(x_n::Vector{Float64},c0::Float64,x0::Float64,amt::Float64)
    
    ndof = length(x_n)
    c_m = zeros(ndof)
    for n=1:ndof
        c_m[n] = c0 + amt*exp(-(x_n[n]-x0)^2/2)
    end    
    figure(2)
    clf()
    plot(x_n,c_m,"k-*")
    ylim(0,2)
    title("velocity model")
    return c_m
    
end

function mesh_setup(x0 :: Float64,x1 :: Float64, K :: Int)

    v_x,EToV = meshgen1d(x0,x1,K)
    p_N = 1
    @match p_N begin
        1 => elem = p1element1d()
        2 => elem = p2element1d()
        3 => elem = p3element1d()
        4 => elem = p4element1d()
        _ => error("Polynomial order not supported")
    end    

    (dofindex,x_n) = buildx(v_x,EToV,elem)

    return (v_x,EToV,elem,dofindex,x_n)
    
end

function regularization(elem,v_x,EToV,dofindex,c)
    
    K=length(dofindex)
    Np = length(elem.r)
    reg_final = 0.0    
    for k=1:K
        v1 = v_x[EToV[k][1]]
        v2 = v_x[EToV[k][2]]
        rx = 1/(v2-v1)
        m_k = v2-v1
        c_k = zeros(Np)
        # gather to element
        for n=1:Np
            c_k[n] = c[dofindex[k][n]]
        end
        reg_final += m_k * (rx^2) * (c_k.')*elem.d_Phi_rr*c_k
    end
    return reg_final     
    
end

function prepare_gradient(v_x, EToV, elem, dofindex, x_n, forcing,
                          M, Minv, T,
                          desired_station_locations,
                          un_truth_stations,
                          station_indexes,
                          c_m_gt, time_n_gt, c_m_current, alpha)
    
    
    ndof = length(x_n)
    bc_array = ones(ndof)
    # dirichlet boundary conditions
    bc_array[1] = 0.0; bc_array[end] = 0.0
    
    (un,time_n_flat) = fwd_simulation(elem,v_x,EToV,x_n,
                                      dofindex,c_m_current,bc_array,
                                      forcing,T)
    

    # setup recorded field to build Adjoint source
    num_stations = length(station_indexes)
    un_stations = zeros(length(time_n_flat),num_stations)
    for (i,idx) in enumerate(station_indexes)
        un_stations[:,i] = un[idx,:]'
    end
    
    (adjoint_source_traces,time_adjoint_source) = buildadjoint_source((un,time_n_flat),station_indexes,time_n_gt,un_truth_stations)

        
    # run adjoint - returns adjoint in "forward" time, so that it can be directly compared with forward simulation
    (un_adj, time_n) = adjoint_simulation(elem,v_x,EToV,x_n,
                                          dofindex,c_m_current,bc_array,
                                          adjoint_source_traces, time_adjoint_source, station_indexes,T)

    
    # build gradient (sensitivity)
    grad_c = zeros(length(c_m_current))
    # grad_c2 = zeros(length(c_m_current))
    dt = time_n_flat[2]-time_n_flat[1]

    diffX = elem.diffX
    for it=1:(length(time_n)-1)
        un_grad_x = fieldtograd_x(elem,diffX,v_x,EToV,dofindex,un[:,it])
        un_adj_grad_x = fieldtograd_x(elem,diffX,v_x,EToV,dofindex,un_adj[:,it])
        # the riemann_sum compares *very* favorably against the
        # simpson numerical integrator
        grad_c += dt*un_grad_x.*un_adj_grad_x
    end
       
    return grad_c
    
end


function misfit(elem,v_x,EToV,dofindex,un_stations,time_un,data_stations,time_data,c_m)

    num_stations = size(un_stations)[2]
    misfit = zeros(num_stations)
    dt_un = time_un[2]-time_un[1]
    for i=1:num_stations
        data_station = InterpIrregular(time_data,data_stations[:,i],BCnearest, InterpLinear)
        # scipy interpolate is tooooooo slow
        # data_station_f = scipy_interpolate.interp1d(time_data,data_stations[:,i],"linear")
        misfit_t = zeros(length(time_un))    
        for (it,ti) in enumerate(time_un)
            data_station_time = data_station[ti]

            # catch bug: when time is just left or right of
            # "time_data", it yields NaN instead of the nearest point
            # ("BCnearest"). Just set to zero instead. Should only
            # happen at endpoints of interval.
            if isnan(data_station_time)
                println("fixing NaN")
                data_station_time = data_stations[end,i]
            end

            # apply misfit using interpolation.
            misfit_t[it] = (un_stations[it,i] - data_station_time)^2
            # only works if exact same dt
            # misfit_t[it] = (un_stations[it,i] - data_stations[it,i])^2          
        end
        # integrate station misfit in time (using Simpson integration)
        # misfit[i] = scipy_integrate.simps(misfit_t,dx=dt_un)
        misfit[i] = dt_un*sum(misfit_t)
    end

    # regularization
    misfit_all_stations = 0.5*sum(misfit)
    return misfit_all_stations
    # regularization_is = regularization(elem,v_x,EToV,dofindex,c_m)
    # misfit_with_regularization = misfit_all_stations
    # println("misfit=$(misfit_with_regularization)=$(misfit_all_stations) + $(alpha)/2*$(regularization_is)")
    # # println("misfit_reg=$(regularization_is)")
    # return misfit_with_regularization
    # # return regularization_is
end


function fwd_simulation(elem, v_x, EToV, x_n, dofindex, velocity_model :: Vector{Float64}, bc_array,
                      forcing,T)

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
    
    (un,time_n) = run_timestepping(x_n,Minv,c_m,m_k_rx2,bc_array,forcing,dofindex,h_min,elem,T)
    return (un,time_n)

end


function adjoint_simulation(elem, v_x, EToV, x_n, dofindex, velocity_model :: Vector{Float64}, bc_array,
                            adjoint_source_traces,adjoint_source_time_n,station_indexes,T)

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

    (un_adj,time_n) = run_timestepping_adjoint(x_n,Minv,c_m,m_k_rx2,bc_array,
                                               adjoint_source_traces,station_indexes,
                                               dofindex,h_min,elem,T)
    if any(isnan(un_adj))
        for (it,t) in enumerate(time_n)
            if any(isnan(un_adj[:,it]))
                error("un_adj[:,$(it)] = $(un_adj[:,it])")
            end
        end
    end
    # adjoint was simulated "backwards" (with adjoint source
    # reversed), so we flip the time
    
    return (flipdim(un_adj,2),time_n)

end

function ground_truth(elem, v_x, EToV, x_n,
                      dofindex, velocity_model :: Vector{Float64}, bc_array,
                      forcing, station_locations :: Vector{Float64},T)
    
    
    (un,time_n) = fwd_simulation(elem,v_x,EToV,x_n,
                                 dofindex,velocity_model,
                                 bc_array,forcing,T)
    
    station_indexes = zeros(Int,length(station_locations))
    for (i,loc_x) in enumerate(station_locations)

        station_indexes[i] = Source.getClosest_x_idx(x_n,loc_x)
        if loc_x > maximum(x_n) && loc_x < minimum(x_n)
            error("station located outside mesh")
        else
            println("Station $(i) location |error|: $(abs(x_n[station_indexes[i]]-loc_x))")
        end
        
    end
    
    println("size(un) = $(size(un)), length(time_n)=$(length(time_n)), station_indexes=$(station_indexes)")
    Nsteps = size(un)[2]
    un_truth_at_stations = zeros(Nsteps,length(station_indexes))
    for (i,idx) in enumerate(station_indexes)
        un_truth_at_stations[:,i] = un[idx,:]
    end

    return (un,un_truth_at_stations,station_indexes,time_n)
    
end

function fieldtograd_x2(elem::Element1D,diffX::Matrix{Float64},v_x,EToV,dofindex,un::Vector{Float64})

    K = length(dofindex)
    diff_x = zeros(length(un))
    h = v_x[2]-v_x[1]
    for n=2:(length(un)-1)
        diff_x[n] = (un[n+1] - un[n-1])/(2*h)
    end
    diff_x[1] = (-1/2*un[3] + 2*un[2] - 3/2*un[1])/h
    diff_x[end] = -(-1/2*un[end-2] + 2*un[end-1] - 3/2*un[end])/h
    return diff_x
end

function fieldtograd_x(elem::Element1D,diffX::Matrix{Float64},v_x,EToV,dofindex,un::Vector{Float64})

    K = length(dofindex)
    diff_x = zeros(length(un))
    
    for k=1:K
        un_elem = zeros(length(elem.r))
        m_k = v_x[EToV[k][2]] - v_x[EToV[k][1]]        
        for n=1:length(elem.r)
            un_elem[n] = un[dofindex[k][n]]
        end
        diff_x_tmp = (1/m_k)*diffX * un_elem
        for n=1:length(elem.r)            
            # if abs(diff_x[dofindex[k][n]]) == 0
            diff_x[dofindex[k][n]] = diff_x_tmp[n]
            # else
            # diff_x[dofindex[k][n]] = diff_x_tmp[n]
            # diff_x[dofindex[k][n]] *= 0.5
            # end
        end
    end
    return diff_x
end

function buildK_forgrad(elem::Element1D,v_x::Vector{Float64},
                        EToV::Vector{Vector{Int}},dofindex::Vector{Vector{Int}},ndof::Int)

    K = length(EToV)
    p_Np = length(elem.r)
    println("Number of GB for full M = $(ndof^2 * 8 / 1e9) GB")
    Kx = spzeros(ndof,ndof)
    Kd = Dict([((1,1),0.0)])

    for k=1:K
        v1 = v_x[EToV[k][1]]
        v2 = v_x[EToV[k][2]]

        m_k = v2-v1
        rx = 1/(v2-v1)
        indexE = dofindex[k]
        if k<10
            println("indexE=$(indexE)")
        end
        for i=1:p_Np
            ie = indexE[i]
            for j=1:p_Np
                je = indexE[j]
                if haskey(Kd,(ie,je))
                    Kd[(ie,je)] += (m_k*rx^2)*elem.d_Phi_rr[i,j]
                else               
                    Kd[(ie,je)] = (m_k*rx^2)*elem.d_Phi_rr[i,j]
                end
                # too slow!
                # M[ie,je] += m_k*Me[i,j]
            end            
        end
        
    end
    num_entries = length(Kd)
    Is = zeros(Int64,num_entries)
    Js = zeros(Int64,num_entries)
    Vs = zeros(num_entries)
    println("length Kd=$(length(Kd))")
    for (idof,((i,j),Kij)) in enumerate(Kd)
        Is[idof] = i
        Js[idof] = j
        Vs[idof] = Kij
    end
    
    return sparse(Is,Js,Vs)
    
end

function applyK_forgrad(elem::Element1D,v_x::Vector{Float64},
                        EToV::Vector{Vector{Int}},dofindex::Vector{Vector{Int}},
                        un::Vector{Float64})

    K=length(dofindex)
    p_Np = length(elem.r)
    un_gather = zeros(elem.p_Np)
    an_k = zeros(elem.p_Np)
    ndof = length(un)
    an = zeros(ndof)
    for k=1:K
        un_gather[:] = 0
        an_k[:] = 0

        # m_k_rx2
        v1 = v_x[EToV[k][1]]
        v2 = v_x[EToV[k][2]]
        rx = 1/(v2-v1)
        m_k = v2-v1
        m_k_rx2 = m_k*rx^2
        # gather
        for i=1:p_Np
            idof = dofindex[k][i]
            un_gather[i] = un[idof]
        end        
        for i=1:p_Np
            for j=1:p_Np            
                # apply Kun
                for n=1:elem.p_Np
                    an_k[i] += un_gather[j]*elem.quadrature_weights[n]*elem.d_Phi_n[n,i]*elem.d_Phi_n[n,j]
                end
            end
            # reference element has length 2
            an_k[i] *= (m_k_rx2)
            # scatter to global dof
            an[dofindex[k][i]] += an_k[i]
        end
    end
    return an
end

type FWI <: MathProgBase.AbstractNLPEvaluator
    v_x                       :: Vector{Float64}
    EToV                      :: Vector{Vector{Int}}
    elem                      :: Element1D
    dofindex                  :: Vector{Vector{Int}}
    x_n                       :: Vector{Float64}
    forcing                   :: Source.WaveSourceParameters
    M                         :: Vector{Float64}                        
    Minv                      :: Vector{Float64}                     
    T                         :: Float64                        
    desired_station_locations :: Vector{Float64}
    un_truth_stations         :: Matrix{Float64}
    station_indexes           :: Vector{Int}          
    c_m_gt                    :: Vector{Float64}                   
    time_n_gt                 :: Vector{Float64}                
    alpha                     :: Float64
end

function MathProgBase.initialize(d::FWI, requested_features::Vector{Symbol})   
    for feat in requested_features
        # have to have :Jac here, despite that we won't actually support it...
        if !(feat in [:Grad,:Jac])            
            error("Unsupported feature $feat")
            # TODO: implement Jac-vec and Hess-vec products
            # for solvers that need them
        end
    end
end

MathProgBase.features_available(d::FWI) = [:Grad]

function MathProgBase.eval_f(d::FWI, cm)

    ndof = length(d.x_n)
    bc_array = ones(ndof)
    # dirichlet boundary conditions
    bc_array[1] = 0.0; bc_array[end] = 0.0
    
    (un,time_n_flat) = fwd_simulation(d.elem,
                                      d.v_x,
                                      d.EToV,
                                      d.x_n,
                                      d.dofindex,cm,bc_array,
                                      d.forcing,
                                      d.T)
    
    # check misfit
    num_stations = length(d.station_indexes)
    un_stations = zeros(length(time_n_flat),num_stations)
    for (i,idx) in enumerate(d.station_indexes)
        un_stations[:,i] = un[idx,:]'
    end

    if any(isnan(cm))
        error("Velocity model cm has NaN")
    end
    reg = regularization(d.elem,d.v_x,d.EToV,d.dofindex,cm)[1]
    reg_gt = regularization(d.elem,d.v_x,d.EToV,d.dofindex,d.c_m_gt)[1]
    # println("reg true vs. current = $(reg_gt) vs. $(reg)")
    misfit_data = misfit(d.elem,d.v_x,d.EToV,d.dofindex,un_stations,time_n_flat,d.un_truth_stations,d.time_n_gt,cm)[1]
    return (misfit_data + d.alpha/2*reg)

end

function MathProgBase.eval_grad_f(d::FWI, grad_f, cm)
    grad_fwdadj = prepare_gradient(d.v_x,
                                   d.EToV,
                                   d.elem,
                                   d.dofindex,
                                   d.x_n,
                                   d.forcing,
                                   d.M,
                                   d.Minv,
                                   d.T,
                                   d.desired_station_locations,
                                   d.un_truth_stations, d.station_indexes,
                                   d.c_m_gt, d.time_n_gt, cm, d.alpha)

    # gradient of the normalization
    diff_c_x = applyK_forgrad(d.elem,d.v_x,d.EToV,d.dofindex,cm)
    grad_f[:] = (-1*grad_fwdadj[:] + d.alpha*diff_c_x[:])*1e-3
end

# seemingly necessary, but we don't implement Jacobian
MathProgBase.jac_structure(d::FWI) = Int[],Int[]


function find_optimal_model(v_x,EToV,elem,dofindex,x_n,forcing,M,Minv,T,desired_station_locations, un_truth_stations, station_indexes, c_m_gt, time_n_gt, cm0,alpha)
    # using Ipopt via MathProgBase
    
    m = MathProgBase.model(Ipopt.IpoptSolver(max_iter=25,limited_memory_max_history=5)); println("starting Ipopt")
    # m = MathProgBase.model(NLopt.NLoptSolver(algorithm=:LD_LBFGS, maxtime=120)); println("starting NLopt")
    l = zeros(length(cm0))
    u = 10*ones(length(cm0))
    lb = Float64[]
    ub = Float64[]
    
    MathProgBase.loadnonlinearproblem!(m,length(cm0),0,l,u,lb,ub,:Min,FWI(v_x,
                                                                          EToV,
                                                                          elem,
                                                                          dofindex,
                                                                          x_n,
                                                                          forcing,
                                                                          M,
                                                                          Minv,
                                                                          T,
                                                                          desired_station_locations,
                                                                          un_truth_stations,
                                                                          station_indexes,
                                                                          c_m_gt,
                                                                          time_n_gt,
                                                                          alpha                    
                                                                          ))

    MathProgBase.setwarmstart!(m,cm0)
    
    @time MathProgBase.optimize!(m)
    println("Finished NLopt")
    stat = MathProgBase.status(m)
    println("status: $(stat)")
    cm_opt = MathProgBase.getsolution(m)
    figure(1)
    clf()
    plot(x_n,cm_opt,"k-",x_n,c_m_gt,"r--")    

    # file = JLD.jldopen("cm_opt.jld","w")
    # JLD.write(file,"cm_opt",cm_opt)
    # JLD.close(file)
    
    return cm_opt
    
    
    result = Optim.optimize(f_eval_misfit, f_eval_grad!,cm0,
                            method = :cg,
                            grtol = 1e-12,
                            ftol = 1e-10,
                            iterations = 100,
                            store_trace = true,
                            show_trace = true)

    return result
    
end



end
