module ResolutionAnalysis

using PyPlot
using Adjoint
using Hessian
using Source
using FEM
import JLD
using ProgressMeter

using ResolutionTools

function materialVelocity_perturbation_with_resolution(x_n::Vector{Float64},c0::Float64,
                                                       x0::Float64,amt::Float64,
                                                       wavelength::Float64,A::Float64)
    
    ndof = length(x_n)
    c_m = zeros(ndof)
    for n=1:ndof
        c_m[n] = c0 + amt*exp(-(x_n[n]-x0)^2.0/1.0)*(1+A*sin(x_n[n]*(2*pi*1.0/wavelength)))
    end    
    return c_m
    
end


function prepare_groundtruth_with_resolution(v_x,EToV,elem,dofindex,x_n,forcing,ndof,M,Minv,T,desired_station_locations,wavelength,A)

    bc_array = ones(ndof)
    # dirichlet boundary conditions
    bc_array[1] = 0.0; bc_array[end] = 0.0

    c_m = materialVelocity_perturbation_with_resolution(x_n,1.0,6.0,0.05,wavelength,A)    
    
    # ground truth simulation
    (un_gt,un_truth_stations,station_indexes,time_n) = ground_truth(elem,v_x,EToV,x_n,
                                                                    dofindex,c_m,bc_array,
                                                                    forcing, 
                                                                    desired_station_locations,
                                                                    T)
    
    return (un_truth_stations, station_indexes, c_m, time_n)
    
end

# test resolution analysis with different station densities
function resolution_analysis_example()
    
    (v_x,EToV,elem,dofindex,x_n) = mesh_setup(0.0,10.0,200)
    # (v_x,EToV,elem,dofindex,x_n) = mesh_setup(0.0,10.0,800)
    forcing = buildSource(3.0,x_n)
    ndof = length(x_n)
    (M,Minv) = buildM(elem,v_x,EToV,dofindex,ndof)
    T = 10.0
    # T = 30.0
    # T = 22.0    
    # desired_station_locations = [linspace(0.1,4.9,25); linspace(5.0,9.9,5)]
    # desired_station_locations = [linspace(0.1,4.9,5); linspace(5.0,9.9,25)]
    desired_station_locations = collect(linspace(0.1,9.9,60))
    num_stations = length(desired_station_locations)
    if isfile("gt_data_dontuse.jld")
        file = JLD.jldopen("gt_data.jld","r")
        un_truth_stations = JLD.read(file,"un_truth_stations")
        time_n_gt = JLD.read(file,"time_n")
        station_indexes = JLD.read(file,"station_indexes")
        c_m_gt = JLD.read(file,"c_m_gt")
        JLD.close(file)
    else
        (un_truth_stations, station_indexes, c_m_gt, time_n_gt) = prepare_groundtruth_with_resolution(v_x,EToV,elem,dofindex,x_n,forcing,ndof,M,Minv,T,desired_station_locations,0.5,0.2)
    end
    c_m_flat = ones(length(c_m_gt))
    # alpha=0.00001
    alpha=30
    
    return (v_x,EToV,elem,dofindex,x_n,forcing,M,Minv,T,desired_station_locations, un_truth_stations, station_indexes, c_m_gt, time_n_gt, c_m_flat,alpha)
    
end

function resolution_analysis_example(stations)
    
    (v_x,EToV,elem,dofindex,x_n) = mesh_setup(0.0,10.0,200)
    forcing = buildSource(3.0,x_n)
    ndof = length(x_n)
    (M,Minv) = buildM(elem,v_x,EToV,dofindex,ndof)
    T = 10.0
    desired_station_locations = stations
    num_stations = length(desired_station_locations)
    if isfile("gt_data_dontuse.jld")
        file = JLD.jldopen("gt_data.jld","r")
        un_truth_stations = JLD.read(file,"un_truth_stations")
        time_n_gt = JLD.read(file,"time_n")
        station_indexes = JLD.read(file,"station_indexes")
        c_m_gt = JLD.read(file,"c_m_gt")
        JLD.close(file)
    else
        (un_truth_stations, station_indexes, c_m_gt, time_n_gt) = prepare_groundtruth_with_resolution(v_x,EToV,elem,dofindex,x_n,forcing,ndof,M,Minv,T,desired_station_locations,0.5,0.2)
    end
    c_m_flat = ones(length(c_m_gt))
    # alpha=0.00001
    alpha=30
    
    return (v_x,EToV,elem,dofindex,x_n,forcing,M,Minv,T,desired_station_locations, un_truth_stations, station_indexes, c_m_gt, time_n_gt, c_m_flat,alpha)
    
end


function test_resolution_station_scaling()

    num_stations_tests = [5,10,20,40,80]
    num_tests = length(num_stations_tests)
    sigma = zeros(num_tests)
    A = zeros(num_tests)
    avg_hv_cc = Vector{Float64}[]
    for (i,num_stations) in enumerate(num_stations_tests)        
        desired_stations = collect(linspace(0.1,9.9,num_stations))
        (avg_hv_cc_i, (A_i,sigma_i)) = get_resolution(desired_stations)
        push!(avg_hv_cc,avg_hv_cc_i)
        A[i] = A_i
        sigma[i] = sigma_i
    end
    println("sigma=$(sigma)")
    spltidx = 1
    figure(2)
    clf()
    for i=1:num_tests
        x = linspace(0,10,length(avg_hv_cc[i]))
        subplot(num_tests+1,1,i)
        plot(x,avg_hv_cc[i],x,A[i]*exp(-((x-5.0).^2)/sigma[i]^2))
        title("avg_hv_cc[$(i)]")
    end
    subplot(num_tests+1,1,num_tests+1)
    semilogx(num_stations_tests,sigma)
    title("sigma")
    
end

function get_resolution(stations)

    (v_x,EToV,elem,dofindex,x_n,forcing,M,Minv,T,desired_station_locations, un_truth_stations, station_indexes, c_m_gt, time_n_gt, c_m_flat,alpha) = resolution_analysis_example(stations)
    ndof = length(x_n)
    bc_array = ones(ndof)
    # dirichlet boundary conditions
    bc_array[1] = 0.0; bc_array[end] = 0.0
    
    
    # if isfile("cm_opt_resolution2.jld")
    #     println("Using precomputed cm")
    #     file = JLD.jldopen("cm_opt_resolution2.jld","r")
    #     cm0 = JLD.read(file,"cm_opt")
    #     JLD.close(file)
    # else
    #     # run optimization to get to near optimum
    #     println("Computing and saving optimal cm")
        
    #     file = JLD.jldopen("cm_opt_resolution2.jld","w")
    #     JLD.write(file,"cm_opt",cm0)
    #     JLD.close(file)        
    # end

    cm0 = find_optimal_model(v_x,EToV,elem,dofindex,x_n,forcing,M,Minv,T,desired_station_locations, un_truth_stations, station_indexes, c_m_gt, time_n_gt, c_m_flat,alpha)    
    # remove alpha from hessian
    alpha=0.1
    num_hessian_vectors = 10
    
    cc_hessian_vector_avg = zeros(2*length(cm0)-1)
    p = Progress(num_hessian_vectors,1)
    for n=1:num_hessian_vectors
        s = randn(length(c_m_flat))/100
        hv = hessianvector(elem,v_x,EToV,x_n,
                           dofindex,cm0,alpha,
                           bc_array,
                           forcing,T,station_indexes,
                           time_n_gt,un_truth_stations,s)
        
        figure(2)
        subplot(2,1,1)
        plot(x_n,s)
        subplot(2,1,2)
        plot(x_n,hv)
        cc_hv = xcorr(hv,hv)        
        cc_hessian_vector_avg += cc_hv
        next!(p)
    end
    cc_hessian_vector_avg = cc_hessian_vector_avg / num_hessian_vectors
    xcc = linspace(0,10,length(cc_hessian_vector_avg))
    
    (Ares,x0res,sigma_res) = ResolutionTools.get_gaussian_fit(cc_hessian_vector_avg,
                                                              (maximum(cc_hessian_vector_avg),5.0,1.0))
    return (cc_hessian_vector_avg,(Ares,sigma_res))
    
end

function test_resolution_analysis()
    
    (v_x,EToV,elem,dofindex,x_n,forcing,M,Minv,T,desired_station_locations, un_truth_stations, station_indexes, c_m_gt, time_n_gt, c_m_flat,alpha) = resolution_analysis_example()
    ndof = length(x_n)
    bc_array = ones(ndof)
    # dirichlet boundary conditions
    bc_array[1] = 0.0; bc_array[end] = 0.0
    
    
    if isfile("cm_opt_resolution.jld")
        println("Using precomputed cm")
        file = JLD.jldopen("cm_opt_resolution.jld","r")
        cm0 = JLD.read(file,"cm_opt")
        JLD.close(file)
    else
        # run optimization to get to near optimum
        println("Computing and saving optimal cm")
    cm0 = find_optimal_model(v_x,EToV,elem,dofindex,x_n,forcing,M,Minv,T,desired_station_locations, un_truth_stations, station_indexes, c_m_gt, time_n_gt, c_m_flat,alpha)    
        file = JLD.jldopen("cm_opt_resolution.jld","w")
        JLD.write(file,"cm_opt",cm0)
        JLD.close(file)        
    end
    figure(1)
    clf()
    plot(x_n,cm0,x_n,c_m_gt,"r--")
    title("final model")

    idx4 = getClosest_x_idx(x_n,4.0)
    idx6 = getClosest_x_idx(x_n,6.0)

    # check single column hessian for x=4.0 and x=6.0
    s = zeros(length(cm0))
    s[idx4] = 1.0
    hv = hessianvector(elem,v_x,EToV,x_n,
                           dofindex,cm0,alpha,
                           bc_array,
                           forcing,T,station_indexes,
                           time_n_gt,un_truth_stations,s)

    figure(4)
    clf()
    subplot(2,1,1)
    plot(x_n,hv)
    title("dc at 4.0")

    s = zeros(length(cm0))
    s[idx6] = 1.0
    hv = hessianvector(elem,v_x,EToV,x_n,
                           dofindex,cm0,alpha,
                           bc_array,
                           forcing,T,station_indexes,
                           time_n_gt,un_truth_stations,s)
    subplot(2,1,2)
    plot(x_n,hv)
    title("dc at 6.0")
    
    # error("stopping")
    
    num_hessian_vectors = 5
    
    cc_hessian_vector_avg = zeros(2*length(cm0)-1)
    p = Progress(num_hessian_vectors,1)
    for n=1:num_hessian_vectors
        s = randn(length(c_m_flat))/100
        hv = hessianvector(elem,v_x,EToV,x_n,
                           dofindex,cm0,alpha,
                           bc_array,
                           forcing,T,station_indexes,
                           time_n_gt,un_truth_stations,s)
        
        figure(2)
        subplot(2,1,1)
        plot(x_n,s)
        subplot(2,1,2)
        plot(x_n,hv)
        cc_hv = xcorr(hv,hv)        
        cc_hessian_vector_avg += cc_hv
        next!(p)
    end
    cc_hessian_vector_avg = cc_hessian_vector_avg / num_hessian_vectors
    xcc = linspace(0,10,length(cc_hessian_vector_avg))
    
    (Ares,x0res,sigma_res) = ResolutionTools.get_gaussian_fit(cc_hessian_vector_avg,
                                                              (maximum(cc_hessian_vector_avg),5.0,1.0))

    println("Initial objective' = ",sum((maximum(cc_hessian_vector_avg)*
                                         exp(-((xcc-5.0).^2)/1.0^2) - cc_hessian_vector_avg).^2))
    println("final fit=",sum((Ares*exp(-((xcc-x0res).^2)/sigma_res^2) - cc_hessian_vector_avg).^2))
    
    
    
    figure(3)
    clf()
    plot(xcc,cc_hessian_vector_avg,xcc,Ares*exp(-((xcc-x0res).^2)/sigma_res^2))

    file = JLD.jldopen("resolution_gaussian.jld","w")
    JLD.write(file,"resolution_gaussian",cc_hessian_vector_avg)
    JLD.close(file)        
    
end



end
