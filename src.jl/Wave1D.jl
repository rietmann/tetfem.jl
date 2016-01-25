module Wave1D

using Match

using Element
using Mesh
using Nodes
using PyPlot
using PyCall
using FEM
using SEM
using TimeStepping
using Source
using Adjoint
using Hessian

using ProgressMeter

import JLD

import Optim
import Optimizer

import SymPy

using MathProgBase

function materialVelocity_perturbation(x_n::Vector{Float64},c0::Float64,x0::Float64,amt::Float64)
    
    ndof = length(x_n)
    c_m = zeros(ndof)
    for n=1:ndof
        c_m[n] = c0 + amt*exp(-(x_n[n]-x0)^2/2)
    end    
    # figure(2)
    # clf()
    # plot(x_n,c_m,"k-*")
    # ylim(0,2)
    # title("perturbed velocity model")
    return c_m
    
end


function materialVelocity_perturbation(v_x,EToV,c0,x0,amt)
    # one c_m per element
    K = size(EToV)[1]
    c_m = zeros(K)
    for k=1:K
        x1 = v_x[EToV[k][1]]
        c_m[k] = c0 + amt*exp(-(x1-x0)^2/2)
    end    
    # figure(2)
    # clf()
    # plot(v_x[1:end-1],c_m,"k-*")
    # ylim(0,2)
    # title("velocity model")
    return c_m
    
end

function materialVelocity_step(v_x,EToV,xab,a,b)
    
    # one c_m per element
    K = size(EToV)[1]
    c_m = zeros(K)
    for k=1:K
        x1 = v_x[EToV[k][1]]
        if x1 < xab
            c_m[k] = a
        else
            c_m[k] = b
        end
    end
    return c_m
end

function simple_wave1d_fwd(T :: Float64,K::Int)

    v_x,EToV = meshgen1d(0.0,10.0,K)
    
    println("v_x=$(v_x)\nEToV=$(EToV)")

    p_N = 3
    @match p_N begin
        1 => elem = p1element1d()
        2 => elem = p2element1d()
        3 => elem = p3element1d()
        4 => elem = p4element1d()
        _ => error("Polynomial order not supported")
    end    
    
    (dofindex,x_n) = buildx(v_x,EToV,elem)

    forcing = buildSource(3.0,x_n)
    
    ndof = length(x_n)
    (M,Minv) = buildM(elem,v_x,EToV,dofindex,ndof)

    bc_array = ones(ndof)
    # dirichlet boundary conditions
    bc_array[1] = 0.0; bc_array[end] = 0.0

    # c_m = materialVelocity_step(v_x,EToV,5.0,1.0,5.0)
    # c_m = materialVelocity_perturbation(v_x,EToV,1,6.0,0.1)
    c_m = ones(ndof)
    @time Ke = buildK_quad(elem,v_x,c_m,EToV,dofindex,ndof)

    element_radius = elementradius1d(v_x,EToV)
    h_min = minimum(element_radius./maximum(c_m))

    @time un = run_timestepping(x_n,Minv,Ke,bc_array,forcing,dofindex,h_min,elem,T)
    # @time (un,error) = run_timestepping_exact(x_n,Minv,Ke,bc_array,dofindex,h_min,elem,T)
    
    return (h_min,un)
    
end

function prepare_groundtruth(v_x,EToV,elem,dofindex,x_n,forcing,ndof,M,Minv,T,desired_station_locations)

    bc_array = ones(ndof)
    # dirichlet boundary conditions
    bc_array[1] = 0.0; bc_array[end] = 0.0

    # c_m = materialVelocity_step(v_x,EToV,5.0,1.0,5.0)
    c_m = materialVelocity_perturbation(x_n,1.0,6.0,0.05)    
    
    
    
    # ground truth simulation
    (un_gt,un_truth_stations,station_indexes,time_n) = ground_truth(elem,v_x,EToV,x_n,
                                                                    dofindex,c_m,bc_array,
                                                                    forcing, 
                                                                    desired_station_locations,
                                                                    T)
    


    file = JLD.jldopen("gt_data.jld","w")
    JLD.write(file,"un_gt",un_gt)
    JLD.write(file,"time_n",time_n)
    JLD.write(file,"un_truth_stations",un_truth_stations)
    JLD.write(file,"station_indexes",station_indexes)
    JLD.write(file,"c_m_gt",c_m)
    JLD.close(file)

    return (un_truth_stations, station_indexes, c_m, time_n)
    # (un,time_n_flat) = fwd_simulation(elem,v_x,EToV,x_n,
    #                                   dofindex,c_m_flat,bc_array,
    #                                   forcing,T)
    
    # un_stations = zeros(size(un)[2],length(station_indexes))
    # for (i,idx) in enumerate(station_indexes)
    #     un_stations[:,i] = un[idx,:]'
    # end

    # (adjoint_source,time_adjoint_source) = buildadjoint_source((un,time_n_flat),station_indexes,time_n,un_truth_stations)
    # println("lengh time_n=$(length(time_n)), size adjoint_source = $(size(adjoint_source))")
    # figure(1)
    # clf()
    # for (i,loc_x) in enumerate(desired_station_locations)
    #     subplot(length(desired_station_locations),1,i)
    #     title("Recording at at x=$(loc_x)")
    #     plot(time_n,un_truth_stations[:,i],"r--")
    #     plot(time_adjoint_source,adjoint_source[:,i],"g--")
    #     plot(time_n_flat,un[station_indexes[i],:]',"k-")
    #     plot(time_n_flat,un_stations[:,i],"k-")
    # end
end

@pyimport scipy.integrate as scipy_integrate


function prepare_gradient_threesteps(v_x, EToV, elem, dofindex, x_n, forcing,
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

    un_time = time_n_flat
    
    println("fwd simulation took $(length(time_n_flat)) steps")
    
    # setup recorded field to build Adjoint source
    num_stations = length(station_indexes)
    un_stations = zeros(length(time_n_flat),num_stations)
    for (i,idx) in enumerate(station_indexes)
        un_stations[:,i] = un[idx,:]'
    end
    
    (adjoint_source_traces,time_adjoint_source) = buildadjoint_source((un,time_n_flat),station_indexes,time_n_gt,un_truth_stations)

        
    # run adjoint - returns adjoint in "forward" time, so that it can be directly compared with forward simulation
    (un_adj, time_n_adj) = adjoint_simulation(elem,v_x,EToV,x_n,
                                              dofindex,c_m_current,bc_array,
                                              adjoint_source_traces, time_adjoint_source, station_indexes,T)


    figure(10)
    clf()
    interp_data = zeros(length(un_time))
    subplot(3,1,1)
    plot(un_time,un[station_indexes[1],:].',"k-*",un_time,un_truth_stations[:,1],"g-o")
    subplot(3,1,2)
    plot(un_time,abs(un[station_indexes[1],:].'-un_truth_stations[:,1]),"k-*")    
    subplot(3,1,3)
    plot(time_n_adj,un_adj[station_indexes[1],:].',"k-*")
    
    
    println("adj simulation took $(length(time_n_adj)) steps")
    
    # build gradient (sensitivity)
    grad_c_t = zeros(length(c_m_current),length(time_n_flat))
    grad_un_t = zeros(length(c_m_current),length(time_n_flat))
    grad_pn_t = zeros(length(c_m_current),length(time_n_flat))
    XX = zeros(length(c_m_current),length(time_n_flat))
    TT = zeros(length(c_m_current),length(time_n_flat))
    grad_c = zeros(length(c_m_current))
    # grad_c2 = zeros(length(c_m_current))
    dt = time_n_flat[2]-time_n_flat[1]

    diffX = elem.diffX

    for it=1:(length(time_n_flat))
        un_grad_x = fieldtograd_x(elem,diffX,v_x,EToV,dofindex,un[:,it])
        un_adj_grad_x = fieldtograd_x(elem,diffX,v_x,EToV,dofindex,un_adj[:,it])
        # the riemann_sum compares *very* favorably against the
        # simpson numerical integrator
        grad_c += dt*un_grad_x.*un_adj_grad_x
        grad_c_t[:,it] = un_grad_x.*un_adj_grad_x
        grad_un_t[:,it] = un_grad_x
        grad_pn_t[:,it] = un_adj_grad_x
        
        # XX[:,it] = x_n        
    end

    figure(3)
    clf()    
    subplot(2,1,1)
    plot(x_n,un[:,2]/maximum(un[:,2]),"b-o",x_n,un_adj[:,2]/maximum(un_adj[:,2]),"r--*")
    ylim(-0.1,1.1)
    xlim(5.87,6.13)
    subplot(2,1,2)
    plot(x_n,grad_un_t[:,2]/maximum(grad_un_t[:,2]),"b-o",x_n,grad_pn_t[:,2]/maximum(grad_pn_t[:,2]),"r--*")
    ylim(-1.1,1.1)
    xlim(5.87,6.13)
    # plot(x_n,grad_c,"k*-")
    
    # subplot(6,1,1)
    # plot(x_n,grad_c_t[:,1])
    # subplot(6,1,2)
    # plot(x_n,grad_c_t[:,2])
    # subplot(6,1,3)
    # plot(x_n,grad_c_t[:,3])
    
    # subplot(6,1,1)
    # plot(x_n,grad_un_t[:,1])
    # subplot(6,1,2)
    # plot(x_n,grad_un_t[:,2])
    # subplot(6,1,3)
    # plot(x_n,grad_un_t[:,3])
    # subplot(6,1,4)
    # plot(x_n,grad_pn_t[:,1])
    # subplot(6,1,5)
    # plot(x_n,grad_pn_t[:,2])
    # subplot(6,1,6)
    # plot(x_n,grad_pn_t[:,3])
    
    # for n=1:length(x_n)
    # TT[n,:] = time_n
    # end
    
    # figure(3)
    # clf()
    # surf(XX,TT,grad_un_t)
    # xlabel("x")
    # ylabel("t")
    
    # figure(4)
    # clf()
    # surf(XX,TT,grad_pn_t)
    # xlabel("x")
    # ylabel("t")
    
    # figure(5)
    # clf()
    # surf(XX,TT,grad_c_t)
    # xlabel("x")
    # ylabel("t")
    
    
    # grad_ct = zeros(length(time_n))
    # for n=1:length(grad_c)
    #     grad_ct[:] = grad_c_t[n,:]
    #     grad_c2[n] = scipy_integrate.simps(grad_ct,dx=dt)
    # end
    
    # figure(3)
    # clf()
    # plot(x_n,grad_c,"r--",x_n,grad_c2,"k-")
    
    return grad_c
    
end


function error_test()

    T = 10*2*4/sqrt(2)
    K = [10,20,40,80,160]
    error_K = zeros(length(K))
    hK = zeros(length(K))
    for (i,Kn) in enumerate(K)
        (h,error_T) = wave1d_fwd(T,Kn)
        error_K[i] = error_T
        hK[i] = h
    end
    println("hK=$(hK),error_K=$(error_K)")
    # figure(1);
    # clf()
    # loglog(hK,error_K,"k*-",hK,hK.^5,"k--")
    
end

function three_step_setup(T)

    (v_x,EToV,elem,dofindex,x_n) = mesh_setup(0.0,10.0,200)
    # (v_x,EToV,elem,dofindex,x_n) = mesh_setup(0.0,10.0,800)
    forcing = buildSourceImmediate(6.0,x_n)
    println("force at $(forcing.source_idx)")
    ndof = length(x_n)
    (M,Minv) = buildM(elem,v_x,EToV,dofindex,ndof)
    # T = 15.0
    # T = 30.0
    # T = 80*0.02
    num_stations = 1
    desired_station_locations = [6.0]
    if isfile("gt_data_dontuse.jld")
        file = JLD.jldopen("gt_data.jld","r")
        un_truth_stations = JLD.read(file,"un_truth_stations")
        time_n_gt = JLD.read(file,"time_n")
        station_indexes = JLD.read(file,"station_indexes")
        c_m_gt = JLD.read(file,"c_m_gt")
        JLD.close(file)
    else
        (un_truth_stations, station_indexes, c_m_gt, time_n_gt) = prepare_groundtruth(v_x,EToV,elem,dofindex,x_n,forcing,ndof,M,Minv,T,desired_station_locations)
    end
    println("station at idx=$(station_indexes)")
    c_m_flat = ones(length(c_m_gt))
    alpha=0.1

    return (v_x,EToV,elem,dofindex,x_n,forcing,M,Minv,T,desired_station_locations, un_truth_stations, station_indexes, c_m_gt, time_n_gt, c_m_flat,alpha)
    
end


function example_setup()

    (v_x,EToV,elem,dofindex,x_n) = mesh_setup(0.0,10.0,200)
    # (v_x,EToV,elem,dofindex,x_n) = mesh_setup(0.0,10.0,800)
    forcing = buildSource(3.0,x_n)
    ndof = length(x_n)
    (M,Minv) = buildM(elem,v_x,EToV,dofindex,ndof)
    # T = 15.0
    T = 30.0
    # T = 22.0
    num_stations = 20
    desired_station_locations = collect(linspace(0.1,9.9,num_stations))
    if isfile("gt_data_dontuse.jld")
        file = JLD.jldopen("gt_data.jld","r")
        un_truth_stations = JLD.read(file,"un_truth_stations")
        time_n_gt = JLD.read(file,"time_n")
        station_indexes = JLD.read(file,"station_indexes")
        c_m_gt = JLD.read(file,"c_m_gt")
        JLD.close(file)
    else
        (un_truth_stations, station_indexes, c_m_gt, time_n_gt) = prepare_groundtruth(v_x,EToV,elem,dofindex,x_n,forcing,ndof,M,Minv,T,desired_station_locations)
    end
    c_m_flat = ones(length(c_m_gt))
    # alpha=0.00001
    alpha=100
    
    return (v_x,EToV,elem,dofindex,x_n,forcing,M,Minv,T,desired_station_locations, un_truth_stations, station_indexes, c_m_gt, time_n_gt, c_m_flat,alpha)
    
end

function go_hessian()

    (v_x,EToV,elem,dofindex,x_n,forcing,M,Minv,T,desired_station_locations, un_truth_stations, station_indexes, c_m_gt, time_n_gt, c_m_flat,alpha) = example_setup()

    s = rand(length(c_m_flat))

    ndof = length(x_n)
    bc_array = ones(ndof)
    # dirichlet boundary conditions
    bc_array[1] = 0.0; bc_array[end] = 0.0
    
    # should use "final" model for a better hessian for hessian-vector applications
    if isfile("cm_opt.jld")
        file = JLD.jldopen("cm_opt.jld","r")
        cm = JLD.read(file,"cm_opt")
        JLD.close(file)
        println("using OPTIMIZED starting model for hessian")
    else
        cm = c_m_flat
        println("using FLAT starting model for hessian")
    end
    hessianvector(elem,v_x,EToV,x_n,
                  dofindex,cm,alpha,
                  bc_array,
                  forcing,T,station_indexes,time_n_gt,un_truth_stations,s)
    
end

# test only 3 steps of gradient
function test_gradient_threestep()

    (v_x,EToV,elem,dofindex,x_n,forcing,M,Minv,T,desired_station_locations, un_truth_stations, station_indexes, c_m_gt, time_n_gt, c_m_flat,alpha) = three_step_setup(0.06)

    println("forcing.source_idx = $(forcing.source_idx)")
    println("station idx = $(station_indexes)")
    
    println("truth took $(length(time_n_gt)) steps")
    
    s = rand(length(c_m_flat))

    ndof = length(x_n)
    bc_array = ones(ndof)
    # dirichlet boundary conditions
    bc_array[1] = 0.0; bc_array[end] = 0.0
    
    # should use "final" model for a better hessian for hessian-vector applications
    if isfile("cm_opt.jld")
        file = JLD.jldopen("cm_opt.jld","r")
        cm0 = JLD.read(file,"cm_opt")
        JLD.close(file)
        println("using OPTIMIZED starting model for hessian")
    else
        cm0 = c_m_flat
        println("using FLAT starting model for hessian")
    end

    eval_misfit = (cm::Vector) -> begin

        ndof = length(x_n)
        bc_array = ones(ndof)
        # dirichlet boundary conditions
        bc_array[1] = 0.0; bc_array[end] = 0.0
        
        (un,time_n_flat) = fwd_simulation(elem,v_x,EToV,x_n,
                                          dofindex,cm,bc_array,
                                          forcing,T)
        
        # check misfit
        num_stations = length(station_indexes)
        un_stations = zeros(length(time_n_flat),num_stations)
        for (i,idx) in enumerate(station_indexes)
            un_stations[:,i] = un[idx,:]'
        end

        # figure(2)
        # clf()
        # plot(x_n,c_m_gt,"r--",x_n,cm,"k-")
        alpha=0.0
        station_misfit = misfit(elem,v_x,EToV,dofindex,un_stations,time_n_flat,un_truth_stations,time_n_gt,cm)[1]
        regularization_contribution = alpha/2*regularization(elem,v_x,EToV,dofindex,cm)

        return (station_misfit + regularization_contribution)
        
    end
    
    eval_grad = (cm::Vector) -> begin
        grad_fwdadj = prepare_gradient_threesteps(v_x,EToV,elem,dofindex,x_n,forcing,
                                                  M,Minv,T,desired_station_locations,
                                                  un_truth_stations, station_indexes,
                                                  c_m_gt, time_n_gt,cm,alpha)

        # gradient of the normalization
        diff_c_x = applyK_forgrad(elem,v_x,EToV,dofindex,cm)

        alpha=0.0
        return (-1*grad_fwdadj[:] + alpha*diff_c_x[:])
    end

    Mfull = buildM_full(elem,v_x,EToV,dofindex,ndof)
    
    grad_cm0 = eval_grad(cm0)
    grad_fd = zeros(length(cm0))
    grad_exact = zeros(length(cm0))
    h = 0.000000000001
    p = Progress(length(cm0),1)
    for n=1:length(cm0)
        pert_c = zeros(ndof)
        pert_c[n] = 1.0
        grad_fd[n] = (eval_misfit(cm0+h*pert_c)[1] - eval_misfit(cm0)[1])/h
        grad_exact[n] = (pert_c.'*(Mfull*grad_cm0))[1]
        next!(p)
    end
    figure(1)
    clf()
    plot(x_n,grad_fd,"r--*",x_n,grad_exact,"k-o")
    xlim(5.87,6.13)
    # xlim(0,10)
    # ylim(-140,0)
    
    h0 = 0.0001
    N=15
    d_cm = 0.05*rand(length(cm0))+1.0
    grad_at_d_cm = (d_cm.'*Mfull*grad_cm0)[1]
    diff_misfit = zeros(N)
    diff_grad = zeros(N)
    diff_grad_rel = zeros(N)
    h_n = zeros(N)
    for n=1:N
        h = h0/2^n
        h_n[n] = h
        diff_misfit[n] = (eval_misfit(cm0+h*d_cm)[1] - eval_misfit(cm0)[1])/h
        diff_grad[n] = diff_misfit[n]-grad_at_d_cm
        diff_grad_rel[n] = abs((diff_misfit[n]-grad_at_d_cm))/abs(grad_at_d_cm)
    end
    
    figure(9)
    clf()
    subplot(2,1,1)
    semilogx(h_n,diff_grad,"k-*")
    subplot(2,1,2)
    semilogx(h_n,diff_grad_rel,"k-*")

    println("diff_grad=$(diff_grad)")
    println("diff_grad_rel=$(diff_grad_rel)")
    
    
end


function test_gradient()

    (v_x,EToV,elem,dofindex,x_n,forcing,M,Minv,T,desired_station_locations, un_truth_stations, station_indexes, c_m_gt, time_n_gt, c_m_flat,alpha) = example_setup()

    s = rand(length(c_m_flat))

    ndof = length(x_n)
    bc_array = ones(ndof)
    # dirichlet boundary conditions
    bc_array[1] = 0.0; bc_array[end] = 0.0
    
    # should use "final" model for a better hessian for hessian-vector applications
    if isfile("cm_opt.jld")
        file = JLD.jldopen("cm_opt.jld","r")
        cm0 = JLD.read(file,"cm_opt")
        JLD.close(file)
        println("using OPTIMIZED starting model for hessian")
    else
        cm0 = c_m_flat
        println("using FLAT starting model for hessian")
    end

    eval_misfit = (cm::Vector) -> begin

        ndof = length(x_n)
        bc_array = ones(ndof)
        # dirichlet boundary conditions
        bc_array[1] = 0.0; bc_array[end] = 0.0
        
        (un,time_n_flat) = fwd_simulation(elem,v_x,EToV,x_n,
                                          dofindex,cm,bc_array,
                                          forcing,T)
        
        # check misfit
        num_stations = length(station_indexes)
        un_stations = zeros(length(time_n_flat),num_stations)
        for (i,idx) in enumerate(station_indexes)
            un_stations[:,i] = un[idx,:]'
        end

        # figure(2)
        # clf()
        # plot(x_n,c_m_gt,"r--",x_n,cm,"k-")
        alpha=0.0
        station_misfit = misfit(elem,v_x,EToV,dofindex,un_stations,time_n_flat,un_truth_stations,time_n_gt,cm)[1]
        regularization_contribution = alpha/2*regularization(elem,v_x,EToV,dofindex,cm)

        return (station_misfit + regularization_contribution)
        
    end
    
    eval_grad = (cm::Vector) -> begin
        grad_fwdadj = prepare_gradient(v_x,EToV,elem,dofindex,x_n,forcing,
                                       M,Minv,T,desired_station_locations,
                                       un_truth_stations, station_indexes,
                                       c_m_gt, time_n_gt,cm,alpha)

        # gradient of the normalization
        diff_c_x = applyK_forgrad(elem,v_x,EToV,dofindex,cm)

        alpha=0.0
        return (-1*grad_fwdadj[:] + alpha*diff_c_x[:])
    end

    Mfull = buildM_full(elem,v_x,EToV,dofindex,ndof)
    
    grad_cm0 = eval_grad(cm0)
    grad_fd = zeros(length(cm0))
    grad_exact = zeros(length(cm0))
    h = 0.000000001
    p = Progress(length(cm0),1)
    for n=1:length(cm0)
        pert_c = zeros(ndof)
        pert_c[n] = 1.0
        grad_fd[n] = (eval_misfit(cm0+h*pert_c)[1] - eval_misfit(cm0)[1])/h
        grad_exact[n] = (pert_c.'*(Mfull*grad_cm0))[1]
        next!(p)
    end
    figure(1)
    clf()
    plot(x_n,grad_fd,"r--",x_n,grad_exact)
    xlim(0,10)
    # ylim(-140,0)
    
    h0 = 0.0001
    N=15
    d_cm = 0.05*rand(length(cm0))+1.0
    grad_at_d_cm = (d_cm.'*Mfull*grad_cm0)[1]
    diff_misfit = zeros(N)
    diff_grad = zeros(N)
    diff_grad_rel = zeros(N)
    h_n = zeros(N)
    for n=1:N
        h = h0/2^n
        h_n[n] = h
        diff_misfit[n] = (eval_misfit(cm0+h*d_cm)[1] - eval_misfit(cm0)[1])/h
        diff_grad[n] = diff_misfit[n]-grad_at_d_cm
        diff_grad_rel[n] = abs((diff_misfit[n]-grad_at_d_cm))/abs(grad_at_d_cm)
    end
    
    figure(9)
    clf()
    subplot(2,1,1)
    semilogx(h_n,diff_grad,"k-*")
    subplot(2,1,2)
    semilogx(h_n,diff_grad_rel,"k-*")
    
end

function test_hessian(T)

    (v_x,EToV,elem,dofindex,x_n,forcing,M,Minv,T,desired_station_locations, un_truth_stations, station_indexes, c_m_gt, time_n_gt, c_m_flat,alpha) = example_setup()

    s = rand(length(c_m_flat))

    ndof = length(x_n)
    bc_array = ones(ndof)
    # dirichlet boundary conditions
    bc_array[1] = 0.0; bc_array[end] = 0.0
    
    # should use "final" model for a better hessian for hessian-vector applications
    if isfile("cm_opt.jld")
        file = JLD.jldopen("cm_opt.jld","r")
        cm0 = JLD.read(file,"cm_opt")
        JLD.close(file)
        println("using OPTIMIZED starting model for hessian")
    else
        cm0 = c_m_flat
        println("using FLAT starting model for hessian")
    end

    eval_misfit = (cm::Vector) -> begin

        ndof = length(x_n)
        bc_array = ones(ndof)
        # dirichlet boundary conditions
        bc_array[1] = 0.0; bc_array[end] = 0.0
        
        (un,time_n_flat) = fwd_simulation(elem,v_x,EToV,x_n,
                                          dofindex,cm,bc_array,
                                          forcing,T)
        
        # check misfit
        num_stations = length(station_indexes)
        un_stations = zeros(length(time_n_flat),num_stations)
        for (i,idx) in enumerate(station_indexes)
            un_stations[:,i] = un[idx,:]'
        end

        station_misfit = misfit(elem,v_x,EToV,dofindex,un_stations,time_n_flat,un_truth_stations,time_n_gt,cm)[1]
        regularization_contribution = alpha/2*regularization(elem,v_x,EToV,dofindex,cm)

        return (station_misfit + regularization_contribution)
        
    end
    
    eval_grad = (cm::Vector) -> begin
        grad_fwdadj = prepare_gradient(v_x,EToV,elem,dofindex,x_n,forcing,
                                       M,Minv,T,desired_station_locations,
                                       un_truth_stations, station_indexes,
                                       c_m_gt, time_n_gt,cm,alpha)

        # gradient of the normalization
        diff_c_x = applyK_forgrad(elem,v_x,EToV,dofindex,cm)

        return (-1*grad_fwdadj[:] + alpha*diff_c_x[:])
    end

    grad_cm0 = eval_grad(cm0)
    misfit_cm0 = eval_misfit(cm0)[1]
    hessian_fd = zeros(ndof)
    hessian_fdg = zeros(ndof)
    hessian_ex = zeros(ndof)
    h = 0.00000001
    p = Progress(length(cm0),1)
    Mfull = buildM_full(elem,v_x,EToV,dofindex,ndof)
    # for n=1:length(cm0)
    for n=150
        pert_c = zeros(ndof)
        pert_c[n] = 1.0
        grad_pertc = eval_grad(cm0 + h*pert_c)        
        Lcphdc = eval_misfit(cm0+h*pert_c)[1]
        Lcmhdc = eval_misfit(cm0-h*pert_c)[1]
        println("typof=$(typeof(Lcphdc)) and $(typeof(misfit_cm0)) and $(typeof(Lcmhdc))")
        hessian_fd[n] = (Lcphdc - 2.0*misfit_cm0 + Lcmhdc)/h^2
        Hs = hessianvector(elem,v_x,EToV,x_n,
                           dofindex,cm0,alpha,
                           bc_array,
                           forcing,T,station_indexes,
                           time_n_gt,un_truth_stations,pert_c)
        # Hs = -(T1pT2-T3-T4)
        hessian_ex[n]  = (pert_c.'*Mfull*Hs)[1]
        hessian_fdg[n] = (pert_c.'*Mfull*(grad_pertc-grad_cm0)/h)[1]
        
        # find best combination
        # T123inv = pinv([T1pT2 T3])
        # beta = T123inv*diff_grad_h
        # println("beta=$(beta)")
        
        figure(1)
        clf()
        # # subplot(2,1,1)
        # # plot(x_n,diff_grad_h)
        # subplot(2,1,1)
        # plot(x_n,-T1pT2)
        # println("vx[2]-vx[1]=$(v_x[2]-v_x[1])")
        # Hs = -0.25*T1pT2+0.75*T3+T4        
        # plot(x_n,-T1pT2,"b-",x_n,T3,"g-",x_n,T4,"r-")
        # subplot(2,1,2)
        plot(x_n,(grad_pertc-grad_cm0)/h,"r--",x_n,Hs,"k-")
        # plot(x_n,Hs-(grad_pertc-grad_cm0)/h)
        error("stopping")
        # abs_error = maximum(abs((grad_pertc-grad_cm0)/h - Hs))
        # l2_error = sum(((grad_pertc-grad_cm0)/h - Hs).^2)
        # return l2_error
        next!(p)
    end
    
    figure(1)
    clf()
    plot(x_n,hessian_fd,"r-",x_n,hessian_fdg,"g--",x_n,hessian_ex,"k-")
    
    
end

function test_hessian_time()

    dt = 0.02
    errors = zeros(10)
    for n=1:10
        errors[n] = test_hessian(dt*2^n)
    end
    figure(1)
    clf()
    n=collect(1:10)
    loglog(dt*2.^n,errors,"k*-")
end

function go_simple()
    (v_x,EToV,elem,dofindex,x_n,forcing,M,Minv,T,desired_station_locations, un_truth_stations, station_indexes, c_m_gt, time_n_gt, c_m_flat,alpha) = example_setup()

    model = (1+0.025*sin(2*pi*x_n)).*c_m_gt
    # for xi=1:length(x_n)
    #     if(x_n[xi]<5)
    #         model[xi] = 1+0.05/5*x_n[xi]
    #     else
    #         model[xi] = (1+0.1)-0.05/5*x_n[xi]
    #     end
    # end
    # figure(1)
        # clf()
        # plot(x_n,model,"k-",x_n,c_m_gt,"r--")
        
    return go(v_x,EToV,elem,dofindex,x_n,forcing,M,Minv,T,desired_station_locations, un_truth_stations, station_indexes, c_m_gt, time_n_gt, model,alpha)
end

function go(filename::String)
    (v_x,EToV,elem,dofindex,x_n,forcing,M,Minv,T,desired_station_locations, un_truth_stations, station_indexes, c_m_gt, time_n_gt, c_m_flat,alpha) = example_setup()
    file = JLD.jldopen(filename,"r")
    model = read(file,"model")
    return go_global(v_x,EToV,elem,dofindex,x_n,forcing,M,Minv,T,desired_station_locations, un_truth_stations, station_indexes, c_m_gt, time_n_gt, model,alpha)
end

function go(model::Vector{Float64})
    (v_x,EToV,elem,dofindex,x_n,forcing,M,Minv,T,desired_station_locations, un_truth_stations, station_indexes, c_m_gt, time_n_gt, c_m_flat,alpha) = example_setup()
    return go(v_x,EToV,elem,dofindex,x_n,forcing,M,Minv,T,desired_station_locations, un_truth_stations, station_indexes, c_m_gt, time_n_gt, model,alpha)
end

function go()
    (v_x,EToV,elem,dofindex,x_n,forcing,M,Minv,T,desired_station_locations, un_truth_stations, station_indexes, c_m_gt, time_n_gt, c_m_flat,alpha) = example_setup()
    return go_global(v_x,EToV,elem,dofindex,x_n,forcing,M,Minv,T,desired_station_locations, un_truth_stations, station_indexes, c_m_gt, time_n_gt, c_m_flat,alpha)
end

function go()
    (v_x,EToV,elem,dofindex,x_n,forcing,M,Minv,T,desired_station_locations, un_truth_stations, station_indexes, c_m_gt, time_n_gt, c_m_flat,alpha) = example_setup()
    return go_global(v_x,EToV,elem,dofindex,x_n,forcing,M,Minv,T,desired_station_locations, un_truth_stations, station_indexes, c_m_gt, time_n_gt, c_m_flat,alpha)
end


function f_eval_grad_reg(cm)

    x = SymPy.symbols("x")

    N = length(gbl_elem.r)
    
    # precompute dd_phi
    dd_phi = zeros(N,N)
    for i=1:N
        dd_phi_i = diff(lagrange1d(gbl_elem.r,i,x),x,x)
        for n=1:N
            dd_phi[n,i] = dd_phi_i(gbl_elem.r[n])
        end
    end
    
    K = length(gbl_EToV)
    d2c2 = zeros(length(cm))
    d2_lg_n = [diff(lagrange1d(gbl_elem.r,n,x),x,x) for n=1:length(gbl_elem.r)]
    for k=1:K

        v1 = gbl_v_x[gbl_EToV[k][1]]
        v2 = gbl_v_x[gbl_EToV[k][2]]
        
        m_k = v2-v1
        rx = 2/(v2-v1)
        cm_k = zeros(length(gbl_elem.r))
        for m=1:length(gbl_elem.r)
            i = gbl_dofindex[k][m]
            cm_k[m] = cm[i]
        end
        dd_cm_k = rx*dd_phi*cm_k
        for m=1:length(gbl_elem.r)
            i = gbl_dofindex[k][m]
            d2c2[i] = dd_cm_k[m]
        end
        # d2c2[i] += rx^2*d2_lg_n[m](elem.r[m])*cm[i]
    end        
    return -d2c2

end


function go(v_x,EToV,elem,dofindex,x_n,forcing,M,Minv,T,desired_station_locations, un_truth_stations, station_indexes, c_m_gt, time_n_gt, cm0,alpha)

    eval_misfit = (cm::Vector) -> begin
        
        ndof = length(x_n)
        bc_array = ones(ndof)
        # dirichlet boundary conditions
        bc_array[1] = 0.0; bc_array[end] = 0.0
        
        (un,time_n_flat) = fwd_simulation(elem,v_x,EToV,x_n,
                                          dofindex,cm,bc_array,
                                          forcing,T)
        
        # check misfit
        num_stations = length(station_indexes)
        un_stations = zeros(length(time_n_flat),num_stations)
        for (i,idx) in enumerate(station_indexes)
            un_stations[:,i] = un[idx,:]'
        end

        if any(isnan(cm))
            error("Velocity model cm has NaN")
        end
        reg = regularization(elem,v_x,EToV,dofindex,cm)[1]
        misfit_data = misfit(elem,v_x,EToV,dofindex,un_stations,time_n_flat,un_truth_stations,time_n_gt,cm)[1]
        return (misfit_data + alpha/2*reg)        
    end

    eval_reg = (cm::Vector) -> begin
        reg = regularization(elem,v_x,EToV,dofindex,cm)        
        return reg[1]
    end
    
    eval_grad! = (cm::Vector,grad_storage::Vector) -> begin
        grad_fwdadj = prepare_gradient(v_x,EToV,elem,dofindex,x_n,forcing,
                                       M,Minv,T,desired_station_locations,
                                       un_truth_stations, station_indexes,
                                       c_m_gt, time_n_gt, cm,alpha)

        # gradient of the normalization
        diff_c_x = applyK_forgrad(elem,v_x,EToV,dofindex,cm)

        # visualize grad and model
        figure(1)
        clf()
        plot(x_n,1+0.05*grad_fwdadj/maximum(grad_fwdadj),"k-",x_n,c_m_gt,"r--",x_n,cm,"g-")
        ylim(0.94,1.08)
    
        grad_storage[:] = (-1*0.5*grad_fwdadj[:] + alpha*diff_c_x[:])*1e-3
    end
    
    eval_grad_reg! = (cm::Vector,grad_storage::Vector) -> begin
        
        # gradient of the normalization
        diff_c_x = applyK_forgrad(elem,v_x,EToV,dofindex,cm)

        # visualize grad and model
        figure(1)
        clf()
        plot(x_n,1+0.05*diff_c_x/maximum(diff_c_x),"k-",x_n,c_m_gt,"r--",x_n,cm,"g-")
        ylim(0.94,1.08)
    
        grad_storage[:] = diff_c_x[:]
    end

    eval_grad = (cm::Vector) -> begin
        grad_storage = zeros(length(cm))
        eval_grad!(cm,grad_storage)
        return grad_storage
    end

    # my optimization routine
    # my_result = Optimizer.optimize(eval_misfit,eval_grad,cm0,
    #                                iterations=15,verbosity=0,method=:bfgs,alpha0=1e-4)

    # using Ipopt via MathProgBase
    m = MathProgBase.model(MathProgBase.defaultNLPsolver)
    l = zeros(length(cm0))
    u = 10*ones(length(cm0))
    lb = Float64[]
    ub = Float64[]
    MathProgBase.loadnonlinearproblem!(m,length(cm0),0,l,u,lb,ub,:Min,FWI())

    MathProgBase.setwarmstart!(m,cm0)
    MathProgBase.optimize!(m)
    stat = MathProgBase.status(m)
    return stat

    
    # using Optim.jl
    # result = Optim.optimize(eval_misfit, eval_grad!,cm0,
    #                         method = :bfgs,
    #                         grtol = 1e-12,
    #                         ftol = 1e-10,
    #                         iterations = 100,
    #                         store_trace = true,
    #                         show_trace = true)

    # return result
end

function testgradient()
    (v_x,EToV,elem,dofindex,x_n,forcing,M,Minv,T,desired_station_locations, un_truth_stations, station_indexes, c_m_gt, time_n_gt, c_m_flat,alpha) = example_setup()
    
    eval_misfit = (cm::Vector) -> begin

        ndof = length(x_n)
        bc_array = ones(ndof)
        # dirichlet boundary conditions
        bc_array[1] = 0.0; bc_array[end] = 0.0
        
        (un,time_n_flat) = fwd_simulation(elem,v_x,EToV,x_n,
                                          dofindex,cm,bc_array,
                                          forcing,T)
        
        # check misfit
        num_stations = length(station_indexes)
        un_stations = zeros(length(time_n_flat),num_stations)
        for (i,idx) in enumerate(station_indexes)
            un_stations[:,i] = un[idx,:]'
        end

        # figure(2)
        # clf()
        # plot(x_n,c_m_gt,"r--",x_n,cm,"k-")
        
        return misfit(elem,v_x,EToV,dofindex,un_stations,time_n_flat,un_truth_stations,time_n_gt,alpha,cm)[1]
        
    end

    eval_grad = (cm::Vector) -> begin
        grad = prepare_gradient(v_x,EToV,elem,dofindex,x_n,forcing,M,Minv,T,desired_station_locations, un_truth_stations, station_indexes, c_m_gt, time_n_gt, cm,alpha)
        # grad_storage[1] = 0.0
        # grad_storage[end] = 0.0
        return -0.5*grad[:]
    end

    N=10
    h0 = 1e-4
    cm0 = c_m_flat
    # d_cm = 0.05*rand(length(cm0))+1.0
    d_cm = zeros(length(c_m_flat))
    d_cm[45] = 1.0
    grad_at_d_cm = (d_cm.'*M.*eval_grad(cm0))[1]
    diff_misfit = zeros(N)
    diff_grad = zeros(N)
    diff_grad_rel = zeros(N)
    h_n = zeros(N)
    for n=1:N
        h = h0/2^n
        h_n[n] = h
        diff_misfit[n] = (eval_misfit(cm0+h*d_cm)[1] - eval_misfit(cm0)[1])/h
        diff_grad[n] = abs(diff_misfit[n]-grad_at_d_cm)
        diff_grad_rel[n] = abs(diff_misfit[n]-grad_at_d_cm)/abs(grad_at_d_cm)
    end
    println("diff_grad=$(diff_grad), grad_at_d_cm=$(grad_at_d_cm)")
    figure(2)
    clf()
    subplot(2,1,1)
    loglog(h_n,diff_grad,"k-*")
    subplot(2,1,2)
    loglog(h_n,diff_grad_rel,"k-*")
    
    ndof = length(c_m_flat)    
    grad_c = eval_grad(c_m_flat)
    println("compared d_cm: $(cm0.'*(M.*grad_c))")
    h = 0.001 # compared d_cm: fd vs exact [-67.21808473722923] vs [-453.2381036592239]
    h = 0.001/2 # compared d_cm: fd vs exact [-425.4921374599911] vs [-453.2381036592239]
    h = 0.001/4 # compared d_cm: fd vs exact [-438.2640075573665] vs [-453.2381036592239]
    h = 0.001/8 # compared d_cm: fd vs exact [-444.6499740337622] vs [-453.2381036592239]
    # h = 0.0001 # compared d_cm: fd vs exact [-445.9271698675859] vs [-453.2381036592239]
    # h = 0.00001 # compared d_cm: fd vs exact [-450.52508362215525] vs [-453.2381036592239]
    # h = 0.000001 # compared d_cm: fd vs exact [-450.98489224670857] vs [-453.2381036592239]
    # h = 0.0000001 # compared d_cm: fd vs exact [-451.03105430044366] vs [-453.2381036592239]
    pts_check = collect(1:1:ndof)
    p = Progress(length(pts_check),1)
    grad_fd = zeros(length(pts_check))
    x_n_check = x_n[pts_check]
    for (ni,n) in enumerate(pts_check)
        pert_c = zeros(ndof)
        pert_c[n] = 1.0
        grad_fd[ni] = (eval_misfit(c_m_flat+h*pert_c)[1] - eval_misfit(c_m_flat)[1])/h
        next!(p)
    end
    figure(3)
    clf()
    # subplot(2,1,1)
    plot(x_n_check,grad_fd,"k-o",x_n,M.*grad_c,"r--")
    println("compared d_cm: fd vs exact $(cm0.'*grad_fd) vs $(cm0.'*(M.*grad_c))")
    # subplot(2,1,2)
    # plot()
    
    # ylim(-0.05,0.05)
    
end

function plot_convergence()

    figure(1)
    clf()
    h = [0.001, 0.001/2, 0.001/4, 0.001/8, 0.0001, 0.00001, 0.000001, 0.0000001]
    fd = [-67.21808473722923,
          -425.4921374599911,
          -438.2640075573665,
          -444.6499740337622,
          -445.9271698675859,
          -450.52508362215525,
          -450.98489224670857,
          -451.03105430044366]

    ex = -453.2381036592239

    fd_ex = fd-ex
    rel_ex = 
    subplot(2,1,1)
    loglog(h,fd_ex)
    subplot(2,1,2)
    loglog(h,fd_ex/abs(ex))
    
    
    h = 0.001 # compared d_cm: fd vs exact      [-67.21808473722923] vs  [-453.2381036592239]
    h = 0.001/2 # compared d_cm: fd vs exact    [-425.4921374599911] vs  [-453.2381036592239]
    h = 0.001/4 # compared d_cm: fd vs exact    [-438.2640075573665] vs  [-453.2381036592239]
    h = 0.001/8 # compared d_cm: fd vs exact    [-444.6499740337622] vs  [-453.2381036592239]
    # h = 0.0001 # compared d_cm: fd vs exact   [-445.9271698675859] vs  [-453.2381036592239]
    # h = 0.00001 # compared d_cm: fd vs exact  [-450.52508362215525] vs [-453.2381036592239]
    # h = 0.000001 # compared d_cm: fd vs exact [-450.98489224670857] vs [-453.2381036592239]
    # h = 0.0000001 # compared d_cm: fd vsexact [-451.03105430044366] vs [-453.2381036592239]
    
end

type Regularization <: MathProgBase.AbstractNLPEvaluator
end
    
function MathProgBase.initialize(d::Regularization, requested_features::Vector{Symbol})
    for feat in requested_features
        # have to have :Jac here, despite that we won't actually support it...
        if !(feat in [:Grad,:Jac])            
            error("Unsupported feature $feat")
            # TODO: implement Jac-vec and Hess-vec products
            # for solvers that need them
        end
    end
end
MathProgBase.features_available(d::Regularization) = [:Grad]

MathProgBase.eval_f(d::Regularization, cm) = (0.5*cm.'*gbl_Kx*cm)[1]

MathProgBase.eval_grad_f(d::Regularization, grad_f, cm) = f_eval_grad_reg!(cm,grad_f)

# seemingly necessary, but we don't implement Jacobian
MathProgBase.jac_structure(d::Regularization) = Int[],Int[]

function f_eval_grad_reg2!(cm,grad_f)

    Kc = gbl_Kx*cm
    grad_f[:] = Kc[:]/1000
    
end

function f_eval_grad_reg!(cm,grad_f)

    x = SymPy.symbols("x")

    N = length(gbl_elem.r)
    
    # precompute dd_phi
    dd_phi = zeros(N,N)
    for i=1:N
        dd_phi_i = diff(lagrange1d(gbl_elem.r,i,x),x,x)
        for n=1:N
            dd_phi[n,i] = dd_phi_i(gbl_elem.r[n])
        end
    end
    
    K = length(gbl_EToV)
    d2c2 = zeros(length(cm))
    d2_lg_n = [diff(lagrange1d(gbl_elem.r,n,x),x,x) for n=1:length(gbl_elem.r)]
    for k=1:K

        v1 = gbl_v_x[gbl_EToV[k][1]]
        v2 = gbl_v_x[gbl_EToV[k][2]]
        
        m_k = v2-v1
        rx = 2/(v2-v1)
        cm_k = zeros(length(gbl_elem.r))
        for m=1:length(gbl_elem.r)
            i = gbl_dofindex[k][m]
            cm_k[m] = cm[i]
        end
        # dd_cm_k = rx^2*dd_phi*cm_k
        dd_cm_k = rx*dd_phi*cm_k # note rx instead of rx^2 (tests show this is correct...)
        for m=1:length(gbl_elem.r)
            i = gbl_dofindex[k][m]
            d2c2[i] = dd_cm_k[m]
        end
        # d2c2[i] += rx^2*d2_lg_n[m](elem.r[m])*cm[i]
    end        
    # grad_storage[:] = d2c2[:]
    grad_f[:] = -d2c2[:]

    # figure(1)
    # clf()
    # plot(gbl_x_n,1+0.05*grad_f/maximum(grad_f),"k-",gbl_x_n,cm,"g-",gbl_x_n,gbl_cm0,"y-")
    # ylim(0.94,1.08)
    # sleep(0.1)
    
end

function testregularization_optimize2()
    (v_x,EToV,elem,dofindex,x_n,forcing,M,Minv,T,desired_station_locations, un_truth_stations, station_indexes, c_m_gt, time_n_gt, c_m_flat,alpha) = example_setup()
    global gbl_v_x                       = v_x
    global gbl_EToV                      = EToV
    global gbl_elem                      = elem
    global gbl_dofindex                  = dofindex
    global gbl_x_n                       = x_n
    global gbl_forcing                   = forcing
    global gbl_M                         = M                        
    global gbl_Minv                      = Minv                     

    cm1 = (1+0.0125*sin(2*pi*x_n)).*c_m_gt
    cm0 = (1+0.025*sin(2*pi*x_n)).*c_m_gt
    global gbl_cm0 = cm0
    Kx = buildK_forgrad(elem,v_x,EToV,dofindex,length(x_n))
    global gbl_Kx = Kx

    m = MathProgBase.model(Ipopt.IpoptSolver(max_iter=100)); println("starting Ipopt")
    # m = MathProgBase.model(NLopt.NLoptSolver(algorithm=:LD_LBFGS, maxtime=10)); println("starting NLopt")
    l = zeros(length(cm0))
    u = 1.2*ones(length(cm0))
    lb = Float64[]
    ub = Float64[]
    MathProgBase.loadnonlinearproblem!(m,length(cm0),0,l,u,lb,ub,:Min,Regularization())

    MathProgBase.setwarmstart!(m,cm0)
    
    @time MathProgBase.optimize!(m)
    println("Finished NLopt")
    stat = MathProgBase.status(m)
    println("status: $(stat)")
    cm_opt = MathProgBase.getsolution(m)
    figure(2)
    clf()
    plot(x_n,cm_opt,"k-",x_n,c_m_gt,"r--",x_n,cm0,"g--")
    
end

function testregularization_optimize()

    (v_x,EToV,elem,dofindex,x_n,forcing,M,Minv,T,desired_station_locations, un_truth_stations, station_indexes, c_m_gt, time_n_gt, c_m_flat,alpha) = example_setup()

    cm1 = (1+0.0125*sin(2*pi*x_n)).*c_m_gt
    cm0 = (1+0.025*sin(2*pi*x_n)).*c_m_gt

    Kx = buildK_forgrad(elem,v_x,EToV,dofindex,length(x_n))
    println("Kx is sym? $(issym(Kx))")
    

    eval_misfit_reg = (cm::Vector) -> begin

        regularization_is = 0.5*cm.'*Kx*cm # regularization(elem,v_x,EToV,dofindex,cm)
        println("misfit=$(regularization_is)")
        return regularization_is[1]
        
    end

    x = SymPy.symbols("x")
    d_lg_n = [diff(lagrange1d(elem.r,n,x),x) for n=1:length(elem.r)]

    eval_grad_g2! = (cm::Vector,grad_storage::Vector) -> begin
        
        N = length(elem.r)
    
        dd_phi = zeros(N,N)
        for i=1:N
            dd_phi_i = diff(lagrange1d(elem.r,i,x),x,x)
            for n=1:N
                dd_phi[n,i] = dd_phi_i(elem.r[n])
            end
        end
        
        K = length(EToV)
        d2c2 = zeros(length(cm))
        d2_lg_n = [diff(lagrange1d(elem.r,n,x),x,x) for n=1:length(elem.r)]
        for k=1:K

            v1 = v_x[EToV[k][1]]
            v2 = v_x[EToV[k][2]]
            
            m_k = v2-v1
            rx = 2/(v2-v1)
            cm_k = zeros(length(elem.r))
            for m=1:length(elem.r)
                i = dofindex[k][m]
                cm_k[m] = cm[i]
            end
            dd_cm_k = rx^2*dd_phi*cm_k
            for m=1:length(elem.r)
                i = dofindex[k][m]
                d2c2[i] = dd_cm_k[m]
            end
            # d2c2[i] += rx^2*d2_lg_n[m](elem.r[m])*cm[i]
        end        
        # grad_storage[:] = d2c2[:]
        grad_storage[:] = -d2c2[:]/100
        # visualize grad and model

        println("grad eval")
        figure(1)
        clf()
        plot(x_n,1+0.05*grad_storage/maximum(grad_storage),"k-",x_n,c_m_gt,"r--",x_n,cm,"g-",x_n,cm0,"y-")
        ylim(0.94,1.08)
        sleep(0.01)
    end

    Kx = buildK_forgrad(elem,v_x,EToV,dofindex,length(x_n))

    eval_grad_K! = (cm::Vector,grad_storage::Vector) -> begin

        Kc = Kx*cm
        grad_storage[:] = Kc[:]

        println("grad eval")
        figure(1)
        clf()
        plot(x_n,1+0.05*grad_storage/maximum(grad_storage),"k-",x_n,c_m_gt,"r--",x_n,cm,"g-",x_n,cm0,"y-")
        ylim(0.94,1.08)
        sleep(0.01)
        
    end
    
    result = Optim.optimize(eval_misfit_reg, eval_grad_g2!,cm0,
                            method = :bfgs,
                            grtol = 1e-12,
                            ftol = 1e-10,
                            iterations = 30,
                            store_trace = true,
                            show_trace = true)
    
    return result
    
end

function testregularization()

    h0 = 0.001
    N = 20

    lagrangian_diff = zeros(N)
    
    (v_x,EToV,elem,dofindex,x_n,forcing,M,Minv,T,desired_station_locations, un_truth_stations, station_indexes, c_m_gt, time_n_gt, c_m_flat,alpha) = example_setup()

    cm1 = (1+0.0125*sin(2*pi*x_n)).*c_m_gt
    cm0 = (1+0.025*sin(2*pi*x_n)).*c_m_gt

    Kx = buildK_forgrad(elem,v_x,EToV,dofindex,length(x_n))
    println("Kx is sym? $(issym(Kx))")


    eval_misfit = (cm::Vector) -> begin

        regularization_is = 0.5*cm.'*Kx*cm # regularization(elem,v_x,EToV,dofindex,cm)
        return regularization_is
        
    end

    x = SymPy.symbols("x")
    d_lg_n = [diff(lagrange1d(elem.r,n,x),x) for n=1:length(elem.r)]
    
    eval_grad_g = (cm::Vector) -> begin

        bi = zeros(length(cm))
        K = length(EToV)
        for k=1:K
            v1 = v_x[EToV[k][1]]
            v2 = v_x[EToV[k][2]]
            
            m_k = v2-v1
            rx = 2/(v2-v1)
            for m=1:length(elem.r)
                i = dofindex[k][m]
                cndl = 0*x
                for n=1:length(elem.r)
                    ii = dofindex[k][n]
                    cndl += cm[ii]*d_lg_n[n]
                end
                bi[i] += (m_k*rx)*SymPy.integrate(cndl*d_lg_n[m],(x,-1,1))
            end
        end
        gi = Minv.*bi

        # figure(1)
        # clf()
        # plot(x_n,grad_c,"k-",x_n,gi,"r-",x_n,cm,"g-")
        
        # fix ends
        # gi[1] = 0.0
        # gi[end] = 0.0
        return gi
    end
    
    eval_grad_g2 = (cm::Vector) -> begin
        
        N = length(elem.r)
    
        dd_phi = zeros(N,N)
        for i=1:N
            dd_phi_i = diff(lagrange1d(elem.r,i,x),x,x)
            for n=1:N
                dd_phi[n,i] = dd_phi_i(elem.r[n])
            end
        end
        
        K = length(EToV)
        d2c2 = zeros(length(cm))
        d2_lg_n = [diff(lagrange1d(elem.r,n,x),x,x) for n=1:length(elem.r)]
        for k=1:K

            v1 = v_x[EToV[k][1]]
            v2 = v_x[EToV[k][2]]
            
            m_k = v2-v1
            rx = 2/(v2-v1)
            cm_k = zeros(length(elem.r))
            for m=1:length(elem.r)
                i = dofindex[k][m]
                cm_k[m] = cm[i]
            end
            dd_cm_k = rx*dd_phi*cm_k
            for m=1:length(elem.r)
                i = dofindex[k][m]
                d2c2[i] = dd_cm_k[m]
            end
            # d2c2[i] += rx^2*d2_lg_n[m](elem.r[m])*cm[i]
        end
        return -d2c2
    end
    
    eval_grad = (cm::Vector) -> begin
        grad_c = Kx*cm # applyK_forgrad(elem,v_x,EToV,dofindex,cm)

        bi = zeros(length(cm))
        K = length(EToV)
        for k=1:K
            v1 = v_x[EToV[k][1]]
            v2 = v_x[EToV[k][2]]
            
            m_k = v2-v1
            rx = 2/(v2-v1)
            for m=1:length(elem.r)
                i = dofindex[k][m]
                cndl = 0*x
                for n=1:length(elem.r)
                    ii = dofindex[k][n]
                    cndl += cm[ii]*d_lg_n[n]
                end
                bi[i] += (m_k*rx^2)*SymPy.integrate(cndl*d_lg_n[m],(x,-1,1))
            end
        end
        gi = Minv.*bi

        # figure(1)
        # clf()
        # plot(x_n,grad_c,"k-",x_n,gi,"r-",x_n,cm,"g-")
        
        # grad_c[1] = 0.0
        # grad_c[end] = 0.0
        return grad_c
    end
    
    @time grad_g_cm0 = eval_grad_g(cm0)
    @time grad_g2_cm0 = eval_grad_g2(cm0)
    figure(1)
    clf()
    plot(x_n,grad_g_cm0,"k-",x_n,grad_g2_cm0,"r-")
    # error("stopping")
    
    # println("L(cm0)=$(eval_misfit(cm0)), L(cm1)=$(eval_misfit(cm1))")
    
    # d_cm = 0.5*rand(length(cm0))+0.5
    # grad_at_d_cm = (eval_grad(cm0).'*d_cm)[1]
    # println("grad_at_d_cm=$(grad_at_d_cm)")
    # diff_misfit = zeros(N)
    # diff_grad = zeros(N)
    # diff_grad_rel = zeros(N)
    # h_n = zeros(N)
    # for n=1:N
    #     h = h0/2^n
    #     h_n[n] = h
    #     diff_misfit[n] = (eval_misfit(cm0+h*d_cm)[1] - eval_misfit(cm0)[1])/h
    #     diff_grad[n] = diff_misfit[n]-grad_at_d_cm
    #     diff_grad_rel[n] = (diff_misfit[n]-grad_at_d_cm)/grad_at_d_cm
    # end
    # println("")
    # figure(2)
    # clf()
    # subplot(2,1,1)
    # loglog(h_n,diff_grad,"k-*")
    # subplot(2,1,2)
    # loglog(h_n,diff_grad_rel,"k-*")

    # figure(1)
    # clf()
    # subplot(2,1,1)
    # plot(x_n,cm0,x_n,c_m_gt,"r--")
    # subplot(2,1,2)
    # plot(x_n,eval_grad(cm0),"k-*")
    
    # println("diff_grad=$(diff_grad)\ndiff_grad_rel=$(diff_grad_rel)")
    v1 = v_x[EToV[1][1]]
    v2 = v_x[EToV[1][2]]
        
    ndof = length(cm0)
    grad_fd = zeros(ndof)
    grad_eval_at_h = zeros(ndof)
    grad_c = eval_grad(cm0)
    grad_c[1] = 0.0
    grad_c[end] = 0.0
    h = 0.00000001
    grad_g_check = zeros(ndof)
    grad_g2_check = zeros(ndof)
    for n=1:ndof
        pert_c = zeros(ndof)
        pert_c[n] = 1.0        
        grad_fd[n] = (eval_misfit(cm0+h*pert_c)[1] - eval_misfit(cm0)[1])/h
        grad_g_check[n] = (grad_g_cm0.'*Kx*pert_c)[1]
        grad_g2_check[n] = (grad_g2_cm0.'*Kx*pert_c)[1]
    end
figure(3)
clf()
# subplot(2,1,1)
plot(x_n,grad_fd,"k-",x_n,grad_c,"r--",x_n,grad_g_check,"g--",x_n,grad_g2_check,"y--")    
ylim(-0.05,0.05)
# subplot(2,1,2)
# plot(x_n,grad_g_cm0,"b-*",x_n,grad_g2_cm0,"g-*",)
end

end
