module Source

# kdtree
# @pyimport scipy.spatial as spatial

export WaveSourceParameters, buildSource, buildSourceImmediate, getClosest_x_idx

function getClosest_x_idx(x_target::Vector{Float64},x_desired::Float64)
    # tree = spatial.KDTree(map(z -> (z,0.0),x_target))
    # (dist_source,source_index) = tree[:query]([x_desired,0.0])
    x_diff = abs(x_target .- x_desired)
    return indmin(x_diff)
    # +1 for python->julia
    # return (source_index+1)
end 

immutable WaveSourceParameters
    desired_source_loc :: Float64 # originally desired source location
    source_loc :: Float64 # actual source location (snapped to x_n grid)
    source_idx :: Int # source index (indexed in global degrees of freedom)
    source_func :: Function
end

function buildSource(x_desired :: Float64, x_n :: Vector{Float64})
    
    source_t = (t::Float64) -> begin
        t0 = 0.2;
        sigma = 2.0;
        M = 50.0;
        return M*exp(-(t-t0)^2/sigma^2)*sin(2*pi/sigma*(t-t0))
    end

    idx = getClosest_x_idx(x_n,x_desired)
    println("$(x_desired),$(x_n[idx]),$(idx),$(source_t(1.2))")
    return WaveSourceParameters(x_desired,x_n[idx],idx,source_t)
    
end

function buildSourceImmediate(x_desired :: Float64, x_n :: Vector{Float64})
    
    source_t = (t::Float64) -> begin
        if t==0.0
            return 10
        else
            return 0
        end
        # t0 = 0.0;
        # sigma = 0.05;
        # M = 50.0;
            # return M*exp(-(t-t0)^2/sigma^2)*sin(2*pi/sigma*(t-t0))
        end

    idx = getClosest_x_idx(x_n,x_desired)
    println("$(x_desired),$(x_n[idx]),$(idx),$(source_t(1.2))")
    return WaveSourceParameters(x_desired,x_n[idx],idx,source_t)
    
end


end
