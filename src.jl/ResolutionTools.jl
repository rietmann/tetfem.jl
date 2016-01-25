module ResolutionTools

using JuMP
using PyPlot
import JLD
import MathProgBase
export get_gaussian_fit


function test_get_gaussian_fit()
    
    file = JLD.jldopen("resolution_gaussian.jld","r")
    resolution_gaussian = JLD.read(file,"resolution_gaussian")
    JLD.close(file)        

    npts = length(resolution_gaussian)
    x = linspace(0,10,npts)    
    data = resolution_gaussian
    A0 = maximum(data)/2.35
    sigma0 = 1.0
    x0_0 = 5.0
    figure(1)
    clf()
    plot(x,data,"k-",x,A0*exp(-((x-x0_0).^2)/sigma0^2),"r--")
    # println("Initial objective' = ",sum((A0*exp(-((x-5.0).^2)/1.0^2) - data).^2))
    println("Initial objective' = ",sum((A0*exp(-((x-x0_0).^2)/sigma0^2)-data).^2))
    println("intiial values=",(A0,5.0,1.0))
    (Aopt,x0opt,sigmaopt) = get_gaussian_fit(resolution_gaussian,(A0,5.0,1.0))    
    println("optimized values=",(Aopt,x0opt,sigmaopt))
    println("final objective' = ",sum((Aopt*exp(-((x-x0opt).^2)/sigmaopt^2)-data).^2))
    plot(x,Aopt*exp(-((x-x0opt).^2)/sigmaopt^2),"g")
    # 
end

function get_gaussian_fit(resolution_gaussian::Vector{Float64},initial_guess)

    (A0,x0_0,sigma0) = initial_guess
    
    npts = length(resolution_gaussian)
    x = linspace(0,10,npts)
    
    data = resolution_gaussian
    x_data = zip(x,data)
    
    m = Model()
    
    @defVar(m, A >= 0, start=A0)
    @defVar(m, 0 <= x0 <= 10, start=x0_0)
    @defVar(m, sigma >= 0, start=sigma0)
    
    @setNLObjective(m, :Min, sum{(A*exp(-((xi-x0)^2)/sigma^2)-di)^2,(xi,di) = x_data})

    solve(m)

    Aopt = getValue(A)
    x0opt = getValue(x0)
    sigma_opt = getValue(sigma)
           
    return (Aopt,x0opt,sigma_opt)
    
end

function testJuMP()

    m = Model()
    @defVar(m, x, start = 0.0)
    @defVar(m, y, start = 0.0)

    @setNLObjective(m, Min, (1-x)^2 + 100(y-x^2)^2)

    solve(m)
    println("x = ", getValue(x), " y = ", getValue(y))

    # adding a (linear) constraint
    @addConstraint(m, x + y == 10)
    solve(m)
    println("x = ", getValue(x), " y = ", getValue(y))
    
end



end
