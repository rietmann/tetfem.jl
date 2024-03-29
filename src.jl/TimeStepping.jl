module TimeStepping

export run_timestepping, run_timestepping_adjoint, run_timestepping_exact,
    run_timestepping_rk4, run_timestepping_solver, run_timestepping_dsu,
    run_timestepping_dsp, run_timestepping_elastic

using Mesh
using Element
using Source
using Base.Test
using PyPlot
using Devectorize
using ProgressMeter

include("TetFemConfig.jl")

function applyKun(Ke::Vector{Matrix{Float64}}, un::Vector{Float64}, 
                  bc::Vector{Float64}, dofindex::Vector{Vector{Int}})

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

# 3D elastic
function applyKun_elastic(c_prime::Vector{Array{Float64, 4}}, un::Array{Float64, 2}, 
                          bc::Array{Float64, 2}, 
                          dofindex::Vector{Vector{Int}}, tet::Tetrahedra)

    K = length(dofindex)
    p_Np = length(dofindex[1])
    un_gather = zeros(p_Np, 3)
    ndof = length(un[:, 1])
    an = zeros(ndof, 3)

    # faster version calls BLAS gemv directly and avoids GC with a temporary variable
    an_k = zeros(p_Np, 3)
    buff_j = zeros(p_Np)
    
    for k=1:K
        # gather
        for i=1:p_Np
            idof = dofindex[k][i]
            un_gather[i, :] = bc[idof, :] .* un[idof, :]
        end

        #an_k[:] = 0.0        
        #Base.LinAlg.BLAS.gemv!('N',1.0,Ke[k],un_gather,0.0,an_k)

        # a bit slower than calling gemv directly (~25% slower)
        an_k[:, :] = 0.0        
        #an_k[:, 1] += c_prime[k][1, 1, 1, 1] * (tet.d_Phi_rr * un_gather[:, 1])
        #an_k[:, 1] += c_prime[k][1, 1, 1, 2] * (tet.d_Phi_rs * un_gather[:, 1])
        #an_k[:, 1] += c_prime[k][1, 1, 1, 3] * (tet.d_Phi_rt * un_gather[:, 1])
        #an_k[:, 1] += c_prime[k][1, 2, 1, 1] * (tet.d_Phi_sr * un_gather[:, 1])
        #an_k[:, 1] += c_prime[k][1, 2, 1, 2] * (tet.d_Phi_ss * un_gather[:, 1])
        #an_k[:, 1] += c_prime[k][1, 2, 1, 3] * (tet.d_Phi_st * un_gather[:, 1])
        #an_k[:, 1] += c_prime[k][1, 3, 1, 1] * (tet.d_Phi_tr * un_gather[:, 1])
        #an_k[:, 1] += c_prime[k][1, 3, 1, 2] * (tet.d_Phi_ts * un_gather[:, 1])
        #an_k[:, 1] += c_prime[k][1, 3, 1, 3] * (tet.d_Phi_tt * un_gather[:, 1])

        for j=1:3
            #buff_j = tet.d_Phi_rr * un_gather[:, j]
            buff_j[:] = 0.0        
            Base.LinAlg.BLAS.gemv!('N', 1.0, tet.d_Phi_rr, un_gather[:, j], 0.0, buff_j)
            for i=1:3
                an_k[:, i] += c_prime[k][i, 1, j, 1] * buff_j
            end
        end
        for j=1:3
            #buff_j = tet.d_Phi_ss * un_gather[:, j]
            buff_j[:] = 0.0        
            Base.LinAlg.BLAS.gemv!('N', 1.0, tet.d_Phi_ss, un_gather[:, j], 0.0, buff_j)
            for i=1:3
                an_k[:, i] += c_prime[k][i, 2, j, 2] * buff_j
            end
        end
        for j=1:3
            #buff_j = tet.d_Phi_tt * un_gather[:, j]
            buff_j[:] = 0.0        
            Base.LinAlg.BLAS.gemv!('N', 1.0, tet.d_Phi_tt, un_gather[:, j], 0.0, buff_j)
            for i=1:3
                an_k[:, i] += c_prime[k][i, 3, j, 3] * buff_j
            end
        end

        for j=1:3
            #buff_j = tet.d_Phi_rs * un_gather[:, j]
            buff_j[:] = 0.0        
            Base.LinAlg.BLAS.gemv!('N', 1.0, tet.d_Phi_rs, un_gather[:, j], 0.0, buff_j)
            for i=1:3
                an_k[:, i] += c_prime[k][i, 1, j, 2] * buff_j
            end
        end
        for j=1:3
            #buff_j = tet.d_Phi_sr * un_gather[:, j]
            buff_j[:] = 0.0        
            #Base.LinAlg.BLAS.gemv!('N', 1.0, tet.d_Phi_sr, un_gather[:, j], 0.0, buff_j)
            Base.LinAlg.BLAS.gemv!('T', 1.0, tet.d_Phi_rs, un_gather[:, j], 0.0, buff_j)
            for i=1:3
                an_k[:, i] += c_prime[k][i, 2, j, 1] * buff_j
            end
        end

        for j=1:3
            #buff_j = tet.d_Phi_rt * un_gather[:, j]
            buff_j[:] = 0.0        
            Base.LinAlg.BLAS.gemv!('N', 1.0, tet.d_Phi_rt, un_gather[:, j], 0.0, buff_j)
            for i=1:3
                an_k[:, i] += c_prime[k][i, 1, j, 3] * buff_j
            end
        end
        for j=1:3
            #buff_j = tet.d_Phi_tr * un_gather[:, j]
            buff_j[:] = 0.0        
            #Base.LinAlg.BLAS.gemv!('N', 1.0, tet.d_Phi_tr, un_gather[:, j], 0.0, buff_j)
            Base.LinAlg.BLAS.gemv!('T', 1.0, tet.d_Phi_rt, un_gather[:, j], 0.0, buff_j)
            for i=1:3
                an_k[:, i] += c_prime[k][i, 3, j, 1] * buff_j
            end
        end

        for j=1:3
            #buff_j = tet.d_Phi_st * un_gather[:, j]
            buff_j[:] = 0.0        
            Base.LinAlg.BLAS.gemv!('N', 1.0, tet.d_Phi_st, un_gather[:, j], 0.0, buff_j)
            for i=1:3
                an_k[:, i] += c_prime[k][i, 2, j, 3] * buff_j
            end
        end
        for j=1:3
            #buff_j = tet.d_Phi_ts * un_gather[:, j]
            buff_j[:] = 0.0        
            #Base.LinAlg.BLAS.gemv!('N', 1.0, tet.d_Phi_ts, un_gather[:, j], 0.0, buff_j)
            Base.LinAlg.BLAS.gemv!('T', 1.0, tet.d_Phi_st, un_gather[:, j], 0.0, buff_j)
            for i=1:3
                an_k[:, i] += c_prime[k][i, 3, j, 2] * buff_j
            end
        end
        
        # scatter
        for i=1:p_Np
            idof = dofindex[k][i]
            # note assembly '+='
            an[idof, :] += bc[idof, :] .* an_k[i, :]
        end
    end
    return an
end

# 2D RK4 timestepping with exact solution
function run_timestepping_rk4(x_n::Vector{Float64},y_n::Vector{Float64},
                             M::Vector{Float64},Minv::Vector{Float64},
                             Ke::Vector{Matrix{Float64}},
                             bc::Vector{Float64},
                             dofindex::Vector{Vector{Int}},
                             h_min::Float64, tri::Triangle,finaltime)

    dt = tri.cfl_factor*h_min*1.5
    Nsteps = int(ceil(finaltime/dt))
    dt = finaltime/Nsteps
    
    # initial condition
    un = sin(pi/2*x_n).*sin(pi/2*y_n)
    vn = zeros(length(un))

    if visualization_to_disk
        initialize_savepoints2d(dofindex,length(Ke),tri,x_n,y_n,un,visualization_dir)
    end
        
    println("Simulating for T=$(finaltime) in $(Nsteps) steps @ dt=$(dt)")
    for it=1:Nsteps
        # Explicit Newmark 2nd-order time stepping
        an = applyKun(Ke,un,bc,dofindex)
        k1v = -Minv.*an
        k1u = vn

        an = applyKun(Ke,un+dt/2*k1u,bc,dofindex)
        k2v = -Minv.*an        
        k2u = vn + dt/2*k1v

        an = applyKun(Ke,un+dt/2*k2u,bc,dofindex)
        k3v = -Minv.*an
        k3u = vn + dt/2*k2v

        an = applyKun(Ke,un+dt*k3u,bc,dofindex)
        k4v = -Minv.*an
        k4u = vn + dt*k3v

        un = un + dt/6*(k1u + 2*k2u + 2*k3u + k4u)
        vn = vn + dt/6*(k1v + 2*k2v + 2*k3v + k4v)
        
        if it%error_check_every == 0
            if visualization_to_disk
                savepoints2d(un,it,visualization_dir)
            end
            
            if error_check_exact
                un_exact = cos(pi/2*sqrt(2)*it*dt)*sin(pi/2*x_n).*sin(pi/2*y_n)
                error_exact = norm(un-un_exact,Inf)
                println("|Error|_inf exact: $(error_exact)")
                if error_exact > 1
                    error("Check \"norm(un-un_exact,Inf)=$(norm(un-un_exact,Inf))\" failed on step $(it)")
                end
            end
            
        end
            
    end

    if error_check_exact
        un_exact_end = cos(pi/2*sqrt(2)*Nsteps*dt)*sin(pi/2*x_n).*sin(pi/2*y_n)
        final_error = norm(un-un_exact_end,Inf)
        println("|Error|_inf exact: $(final_error)")
        if final_error > 1
            error("Check \"norm(un-un_exact,Inf)=final_error\" incorrect at end!")
        end
        return (un,final_error)
    else
        return (un,0.0)
    end
    
end

# 3D version of RK4 with exact solution
function run_timestepping_rk4(x_n::Vector{Float64},y_n::Vector{Float64},z_n::Vector{Float64},
                             Minv::Vector{Float64},
                             Ke::Vector{Matrix{Float64}},
                             bc::Vector{Float64},
                             dofindex::Vector{Vector{Int}},
                             h_min::Float64, tet::Tetrahedra,finaltime)

    dt = tet.cfl_factor*h_min*1.5
    Nsteps = int(ceil(finaltime/dt))
    dt = finaltime/Nsteps
    
    # initial condition
    un = sin(pi/2*x_n).*sin(pi/2*y_n).*sin(pi/2*z_n)
    vn = zeros(length(un))

    if visualization_to_disk
        initialize_savepoints2d(dofindex,length(Ke),tet,x_n,y_n,un)
    end
    
    println("Simulating for T=$(finaltime) in $(Nsteps) steps @ dt=$(dt)")
    p = Progress(Nsteps, 10)
    for it=1:Nsteps
        # Explicit Newmark 2nd-order time stepping
        an = applyKun(Ke,un,bc,dofindex)
        k1v = -Minv.*an
        k1u = vn

        an = applyKun(Ke,un+dt/2*k1u,bc,dofindex)
        k2v = -Minv.*an        
        k2u = vn + dt/2*k1v

        an = applyKun(Ke,un+dt/2*k2u,bc,dofindex)
        k3v = -Minv.*an
        k3u = vn + dt/2*k2v

        an = applyKun(Ke,un+dt*k3u,bc,dofindex)
        k4v = -Minv.*an
        k4u = vn + dt*k3v

        # @devec begin
        un = un + dt./6.*(k1u + 2.*k2u + 2.*k3u + k4u)
        vn = vn + dt./6.*(k1v + 2.*k2v + 2.*k3v + k4v)
        # end
        
        if it%error_check_every == 0
            if visualization_to_disk
                savepoints2d(un,it)
            end
            
            if error_check_exact
                un_exact = cos(pi/2*sqrt(3)*it*dt)*sin(pi/2*x_n).*sin(pi/2*y_n).*sin(pi/2*z_n)
                error_exact = norm(un-un_exact,Inf)
                println("|Error|_inf exact @ T=$(it*dt): $(error_exact)")
                if error_exact > 1
                    error("Check \"norm(un-un_exact,Inf)=$(norm(un-un_exact,Inf))\" failed on step $(it)")
                end
            end            
        end
        next!(p)
    end

    if error_check_exact
        un_exact_end = cos(pi/2*sqrt(3)*Nsteps*dt)*sin(pi/2*x_n).*sin(pi/2*y_n).*sin(pi/2*z_n)
        final_error = norm(un-un_exact_end,Inf)
        println("|Error|_inf exact @ Tend=$(finaltime): $(final_error)")
        if final_error > 1
            error("Check \"norm(un-un_exact,Inf)=final_error\" incorrect at end!")
        end
        return (un,final_error)
    else
        return (un,0.0)
    end
    
end

# 2D Newmark 2nd-order timestepping
function run_timestepping(x_n::Vector{Float64},y_n::Vector{Float64},
                          M::Vector{Float64},Minv::Vector{Float64},
                          Ke::Vector{Matrix{Float64}},
                          bc::Vector{Float64},
                          dofindex::Vector{Vector{Int}},
                          h_min::Float64, tri::Triangle,finaltime::Float64)

    
    dt = tri.cfl_factor*h_min/1.3
    Nsteps = int(ceil(finaltime/dt))
    dt = finaltime/Nsteps
    
    # initial condition
    un = sin(pi/2*x_n).*sin(pi/2*y_n)
    # println("un0 = $(un)")    
    # println("x_n=$(x_n)")
    # println("y_n=$(y_n)")
    vn = zeros(length(un))

    if visualization_to_disk
        initialize_savepoints2d(dofindex,length(Ke),tri,x_n,y_n,un,visualization_dir)
    end
    println("Simulating for T=$(finaltime) in $(Nsteps) steps @ dt=$(dt)")
    
    for it=1:Nsteps
        # Explicit Newmark 2nd-order time stepping
        an = applyKun(Ke,un,bc,dofindex)
        an = -Minv.*an
        # half step first step
        if it==1
            vn = vn + dt/2*an;
        else
            vn = vn + dt*an;
        end
        
        un = un + dt*vn    

        if it == 1
            # println("an = $(an)")
        end

        if it%error_check_every == 0
            if visualization_to_disk
                savepoints2d(un,it,visualization_dir)
            end
            
            if error_check_exact
                un_exact = cos(pi/2*sqrt(2)*it*dt)*sin(pi/2*x_n).*sin(pi/2*y_n)
                error_exact = norm(un-un_exact,Inf)
                println("|Error|_inf exact @ T=$(it*dt): $(error_exact)")
                if error_exact > 1
                    error("Check \"norm(un-un_exact,Inf)=$(norm(un-un_exact,Inf))\" failed on step $(it)")
                end
            end
            
        end
            
    end

    if error_check_exact
        un_exact_end = cos(pi/2*sqrt(2)*Nsteps*dt)*sin(pi/2*x_n).*sin(pi/2*y_n)
        final_error = norm(un-un_exact_end,Inf)
        # final_error = maximum(abs(un-un_exact_end))  # norm(un-un_exact_end,Inf)
        println("|Error|_inf exact @ Tend=$(finaltime): $(final_error)")
        return (un,final_error)
    else
        return (un,0.0)
    end
    
end
    

# 3D 4th-order Leap-frog timestepping
function run_timestepping(x_n::Vector{Float64},y_n::Vector{Float64},z_n::Vector{Float64},
                          Minv::Vector{Float64},
                          Ke::Vector{Matrix{Float64}},
                          bc::Vector{Float64},
                          dofindex::Vector{Vector{Int}},
                          h_min::Float64, tet::Tetrahedra,finaltime::Float64)

        
    dt = tet.cfl_factor * h_min
    Nsteps = round(Int,ceil(finaltime/dt))
    dt = finaltime/Nsteps
    
    println("$(Nsteps) steps at dt=$(dt) for T=$(finaltime)")

    # initial condition Leap-Frog
    un = cos(pi/2*sqrt(3)*0*dt)*sin(pi/2*x_n).*sin(pi/2*y_n).*sin(pi/2*z_n)
    unm1 = cos(pi/2*sqrt(3)*(-1)*dt)*sin(pi/2*x_n).*sin(pi/2*y_n).*sin(pi/2*z_n)
    
    #un = sin(pi / 2 * x_n)
    #unm1 = cos(pi / 2 * dt) * sin(pi / 2 * x_n)
    
    unp1 = zeros(length(un))
    vn = zeros(length(un))
    an = zeros(length(un))
    an0 = zeros(length(un))
    
    if visualization_to_disk
        initialize_savepoints3d(dofindex,length(Ke),tet,x_n,y_n,z_n,un,visualization_dir)
    end
    
    ndof = length(un)

    p = Progress(Nsteps, 10)

    println("Simulating for T=$(finaltime) in $(Nsteps) steps @ dt=$(dt)")
    
    for it=1:Nsteps
                
        # Explicit 4th-order time stepping
        b = applyKun(Ke,un,bc,dofindex)
        
        # # # 4th order leapfrog
        an0 = Minv.*b
        
        b2 = applyKun(Ke,an0,bc,dofindex)
        for n=1:ndof
            # an = Minv.*b2
            unp1[n] = 2*un[n] - unm1[n] - dt^2*an0[n] + dt^4/12.0*Minv[n]*b2[n]
        end
        
        unm1 = copy(un)
        un = copy(unp1)
                        
        if it%error_check_every == 0                        
            if error_check_exact
                un_exact = cos(pi/2*sqrt(3)*(it)*dt)*sin(pi/2*x_n).*sin(pi/2*y_n).*sin(pi/2*z_n)                
                # error_exact = norm(un-un_exact,Inf)
                abs_error = abs(un-un_exact)
                
                (error_exact,idx) = findmax(abs_error)
                
                if visualization_to_disk
                    savepoints3d(un,it,visualization_dir)
                end
                
                println("|Error|_inf(t[$(it)]=$(it*dt)) exact: = $(error_exact) @ ($(x_n[idx]),$(y_n[idx]),$(z_n[idx]))")
                if error_exact > 2
                    error("Check \"norm(un-un_exact,Inf)=$(norm(un-un_exact,Inf))\" failed on step $(it)")
                end
            end
            
        end

        next!(p)
        
    end

    if error_check_exact
        un_exact_end = cos(pi/2*sqrt(3)*(Nsteps)*dt)*sin(pi/2*x_n).*sin(pi/2*y_n).*sin(pi/2*z_n)
        final_error = norm(un-un_exact_end,Inf)
        # final_error = maximum(abs(un-un_exact_end))  # norm(un-un_exact_end,Inf)
        println("|Error|_inf exact @ Tend=$(Nsteps*dt): $(final_error)")
        return (un,final_error)
    else
        return (un,0.0)
    end
    
end


# ELASTIC
# 3D 4th-order Leap-frog timestepping
function run_timestepping_elastic(x_n::Vector{Float64}, y_n::Vector{Float64}, z_n::Vector{Float64},
                                  Minv::Vector{Float64},
                                  c_prime::Vector{Array{Float64,4}},
                                  bc::Array{Float64, 2},
                                  dofindex::Vector{Vector{Int}},
                                  h_min::Float64, tet::Tetrahedra, finaltime::Float64)

        
    dt = tet.cfl_factor * h_min / sqrt(3)
    Nsteps = round(Int, ceil(finaltime / dt))
    dt = finaltime / Nsteps
    
    println("$(Nsteps) steps at dt=$(dt) for T=$(finaltime)")

    # initial condition Leap-Frog
    un = zeros(length(x_n), 3)
    unm1 = zeros(length(x_n), 3)

    # P wave
    #un[:,1] = sin(pi / 2 * x_n)
    #unm1[:,1] = cos(pi / 2 * sqrt(3) * dt) * sin(pi / 2 * x_n)

    # S wave in y direction
    un[:,2] = sin(pi / 2 * x_n)
    unm1[:,2] = cos(pi / 2 * dt) * sin(pi / 2 * x_n)

    unp1 = zeros(length(x_n), 3)
    vn = zeros(length(x_n), 3)
    an = zeros(length(x_n), 3)
    an0 = zeros(length(x_n), 3)
    
    if visualization_to_disk
        initialize_savepoints3d(dofindex, length(c_prime), tet, x_n, y_n, z_n, un[:, 2], visualization_dir)
    end
    
    ndof = length(un[:, 1])

    p = Progress(Nsteps, 10)

    println("Simulating for T=$(finaltime) in $(Nsteps) steps @ dt=$(dt)")
    
    for it=1:Nsteps
                
        # Explicit Newmark 2nd/4th-order time stepping
        b = applyKun_elastic(c_prime, un, bc, dofindex, tet)
        
        # # # 4th order leapfrog
        for l=1:3
            an0[:, l] = Minv .* b[:, l]
        end
        
        b2 = applyKun_elastic(c_prime, an0, bc, dofindex, tet)
        for l=1:3
            for n=1:ndof
                # an = Minv.*b2
                unp1[n, l] = (2 * un[n, l] - unm1[n, l] - dt^2 * an0[n, l] +
                              dt^4 / 12.0 * Minv[n] * b2[n, l])
            end
        end
        
        unm1 = copy(un)
        un = copy(unp1)
                        
        if it%error_check_every == 0                        
            if error_check_exact
                # P-wave
                # un_exact = cos(pi / 2 * sqrt(3) * it * dt) * sin(pi / 2 * x_n)
                # abs_error = abs(un[:, 1] - un_exact)

                # S wave in y direction
                un_exact = cos(pi / 2 * it * dt) * sin(pi / 2 * x_n)
                abs_error = abs(un[:, 2] - un_exact)
                
                (error_exact, idx) = findmax(abs_error)
                
                if visualization_to_disk
                    savepoints3d(un[:, 2], it, visualization_dir)
                end
                
                println("|Error|_inf(t[$(it)]=$(it*dt)) exact: = $(error_exact) @ ($(x_n[idx]),$(y_n[idx]),$(z_n[idx]))")
                if error_exact > 2
                    error("Check \"norm(un-un_exact,Inf)=$(norm(un[:,2]-un_exact,Inf))\" failed on step $(it)")
                end
            end
            
        end

        next!(p)
        
    end

    if error_check_exact
        un_exact_end = cos(pi / 2 * Nsteps * dt) * sin(pi / 2 * x_n)
        final_error = norm(un[:, 2] - un_exact_end, Inf)
        
        println("|Error|_inf exact @ Tend=$(Nsteps*dt): $(final_error)")
        return (un,final_error)
    else
        return (un,0.0)
    end
    
end

end

