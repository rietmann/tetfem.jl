module BuildDOFIndex

export LookupTreeNode, build_dofindex, test_build_dofindex

type LookupTreeNode

    child1::Union{Void,LookupTreeNode}
    child2::Union{Void,LookupTreeNode}
    x::Float64    
    y::Float64    
    z::Float64
    id::Int64

    tol::Float64

    function LookupTreeNode(x,y,id,tol)
        new(nothing,nothing,x,y,0.0,id,tol)
    end

    function LookupTreeNode(x,y,z,id,tol)
        new(nothing,nothing,x,y,z,id,tol)
    end       
    
end

type IdRef

    id::Int64
    
end

function node_equal(node,x,y)
    return ( sqrt( (node.x-x)^2 + (node.y-y)^2 ) < node.tol )
end
function node_equal(node,x,y,z)
    return ( sqrt( (node.x-x)^2 + (node.y-y)^2 + (node.z-z)^2 ) < node.tol )
end

function dist(node,x,y)
    return sqrt( (node.x-x)^2 + (node.y-y)^2 )
end

function dist(node,x,y,z)
    return sqrt( (node.x-x)^2 + (node.y-y)^2 + (node.z-z)^2 )
end

function find(this_node,x,y)

    if node_equal(this_node,x,y)
        return this_node.id
    else
        if this_node.child1 == nothing
            return nothing            
        elseif this_node.child2 == nothing
            # search child1
            if node_equal(this_node.child1,x,y)
                # child 1 matched, return its id
                return this_node.child1.id            
            else
                # child 1 didn't match, so we didn't find the node on this branch
                return nothing
            end
        else
            # search children
            dist1 = dist(this_node.child1,x,y)
            dist2 = dist(this_node.child2,x,y)
            # if within tolerance of same distance to both points (i.e. on the
            # cutting plane), search on both sides:
            if abs(dist1 - dist2) < this_node.tol
                child1_search = find(this_node.child1,x,y)
                child2_search = find(this_node.child2,x,y)
                if child1_search != nothing
                    return child1_search
                elseif child2_search != nothing
                    return child2_search
                else
                    return nothing
                end                                    
            elseif (dist1 < dist2)
                return find(this_node.child1,x,y)
            else
                return find(this_node.child2,x,y)
            end
        end
    end
end

function find(this_node,x,y,z)

    if node_equal(this_node,x,y,z)
        return this_node.id
    else
        if this_node.child1 == nothing
            return nothing            
        elseif this_node.child2 == nothing
            # search child1
            if node_equal(this_node.child1,x,y,z)
                # child 1 matched, return its id
                return this_node.child1.id            
            else
                # child 1 didn't match, so we didn't find the node on this branch
                return nothing
            end
        else
            # search children
            dist1 = dist(this_node.child1,x,y,z)
            dist2 = dist(this_node.child2,x,y,z)
            # if within tolerance of same distance to both points (i.e. on the
            # cutting plane), search on both sides:
            if abs(dist1 - dist2) < this_node.tol
                child1_search = find(this_node.child1,x,y,z)
                child2_search = find(this_node.child2,x,y,z)
                if child1_search != nothing
                    return child1_search
                elseif child2_search != nothing
                    return child2_search
                else
                    return nothing
                end                                    
            elseif (dist1 < dist2)
                return find(this_node.child1,x,y,z)
            else
                return find(this_node.child2,x,y,z)
            end
        end
    end
end

function findorinsert(this_node,x,y,id_next)
    if node_equal(this_node,x,y)
        return this_node.id
    else

        if this_node.child1 == nothing
            # no match possible, add it as child 1
            this_node.child1 = LookupTreeNode(x,y,id_next.id,this_node.tol)
            id_next.id += 1
            return this_node.child1.id
        elseif this_node.child2 == nothing
            # check child 1
            if node_equal(this_node.child1,x,y)
                # child 1 matched, return its id
                return this_node.child1.id            
            else                
                # child 1 didn't match, add it as child2
                this_node.child2 = LookupTreeNode(x,y,id_next.id,this_node.tol)
                id_next.id += 1
                return this_node.child2.id
            end
        else
            # search children
            dist1 = dist(this_node.child1,x,y)
            dist2 = dist(this_node.child2,x,y)

            # if within tolerance of same distance to both points (i.e. on the
            # cutting plane), search on both sides:
            if abs(dist1-dist2) < this_node.tol
                child1_search = find(this_node.child1,x,y)
                child2_search = find(this_node.child2,x,y)
                if child1_search != nothing
                    return child1_search
                elseif child2_search != nothing
                    return child2_search
                end
                # if neither child contains node, we add it below, and
                # it will be found when exploring both branches.
            end
                            
            if dist1 < dist2
                # closer to child 1
                return findorinsert(this_node.child1,x,y,id_next)
            else
                # closer to child 2
                return findorinsert(this_node.child2,x,y,id_next)
            end
            
        end        
    end
end

function findorinsert_cpp(this_node,x,y,z,id_next)
    if node_equal(this_node,x,y,z)
        return this_node.id
    else

        if this_node.child1 == nothing
            # no match possible, add it as child 1
            this_node.child1 = LookupTreeNode(x,y,z,id_next.id,this_node.tol)
            id_next.id += 1
            return this_node.child1.id
        elseif this_node.child2 == nothing
            # check child 1
            if node_equal(this_node.child1,x,y,z)
                # child 1 matched, return its id
                return this_node.child1.id            
            else                
                # child 1 didn't match, add it as child2
                this_node.child2 = LookupTreeNode(x,y,z,id_next.id,this_node.tol)
                id_next.id += 1
                return this_node.child2.id
            end
        else
            # search children
            dist1 = dist(this_node.child1,x,y,z)
            dist2 = dist(this_node.child2,x,y,z)

            # if within tolerance of same distance to both points (i.e. on the
            # cutting plane), search on both sides:
                                        
            if dist1 < (dist2 + this_node.tol)
                # closer to child 1
                return findorinsert(this_node.child1,x,y,z,id_next)
            else
                # closer to child 2
                return findorinsert(this_node.child2,x,y,z,id_next)
            end
            
        end        
    end
end

function findorinsert(this_node,x,y,z,id_next)
    if node_equal(this_node,x,y,z)
        return this_node.id
    else

        if this_node.child1 == nothing
            # no match possible, add it as child 1
            this_node.child1 = LookupTreeNode(x,y,z,id_next.id,this_node.tol)
            id_next.id += 1
            return this_node.child1.id
        elseif this_node.child2 == nothing
            # check child 1
            if node_equal(this_node.child1,x,y,z)
                # child 1 matched, return its id
                return this_node.child1.id            
            else                
                # child 1 didn't match, add it as child2
                this_node.child2 = LookupTreeNode(x,y,z,id_next.id,this_node.tol)
                id_next.id += 1
                return this_node.child2.id
            end
        else
            # search children
            dist1 = dist(this_node.child1,x,y,z)
            dist2 = dist(this_node.child2,x,y,z)

            # if within tolerance of same distance to both points (i.e. on the
            # cutting plane), search on both sides:
            if abs(dist1-dist2) < this_node.tol
                child1_search = find(this_node.child1,x,y,z)
                child2_search = find(this_node.child2,x,y,z)
                if child1_search != nothing
                    return child1_search
                elseif child2_search != nothing
                    return child2_search
                end
                # if neither child contains node, we add it below, and
                # it will be found when exploring both branches.
            end
                            
            if dist1 < dist2
                # closer to child 1
                return findorinsert(this_node.child1,x,y,z,id_next)
            else
                # closer to child 2
                return findorinsert(this_node.child2,x,y,z,id_next)
            end
            
        end        
    end
end


function build_dofindex(x_all,y_all)
    tree_tol = sqrt((x_all[1,1]-x_all[2,1])^2 + (y_all[1,1]-y_all[2,1])^2) / 1e4
    root = LookupTreeNode(x_all[1,1],y_all[1,1],1,tree_tol)
    idnext = IdRef(2) # pass by reference to maintain global count
    dofindex = Vector{Int}[]
    for k=1:size(x_all)[2]
        push!(dofindex,Array(Int64,size(x_all)[1]))
        for i=1:size(x_all)[1]
            found_id = findorinsert(root,x_all[i,k],y_all[i,k],idnext)
            dofindex[k][i] = found_id
        end
    end
    
    # total number of unique nodes (i.e., degrees of freedom) is one less than final idnext
    ndof = idnext.id - 1
    
    return (dofindex,ndof)
end

function build_dofindex(x_all,y_all,z_all)
    tree_tol = sqrt((x_all[1,1]-x_all[2,1])^2 + (y_all[1,1]-y_all[2,1])^2 + (z_all[1,1]-z_all[2,1])^2) / 1e5
    root = LookupTreeNode(x_all[1,1],y_all[1,1],z_all[1,1],1,tree_tol)
    idnext = IdRef(2) # pass by reference to maintain global count
    dofindex = Vector{Int}[]
    for k=1:size(x_all)[2]
        push!(dofindex,Array(Int64,size(x_all)[1]))
        for i=1:size(x_all)[1]
            found_id = findorinsert(root,x_all[i,k],y_all[i,k],z_all[i,k],idnext)
            dofindex[k][i] = found_id
        end
    end
    
    # total number of unique nodes (i.e., degrees of freedom) is one less than final idnext
    ndof = idnext.id - 1
    
    return (dofindex,ndof)
end

function test_build_dofindex(x_all,y_all,z_all)
    
    println("Testing dofindex!")
    (dofindex_orig,ndof) = build_dofindex(x_all,y_all,z_all)
    tol = 1e-8
    # horrible n^2 way for building dofindex
    (npts,K) = size(x_all)
    dofindex = Vector{Int}[]
    added_pts = Tuple{Float64,Float64,Float64}[]
    for k=1:K
        push!(dofindex,Array(Int64,size(x_all)[1]))
        for i=1:npts
            xi = x_all[i,k]
            yi = y_all[i,k]
            zi = z_all[i,k]
            already_added = false
            # look for pt in existing list of points
            for n=1:length(added_pts)
                (xn,yn,zn) = added_pts[n]
                # if found, set dofindex to current n
                if sqrt( (xi-xn)^2 + (yi-yn)^2 + (zi-zn)^2 ) < tol
                    already_added = true
                    dofindex[k][i] = n
                    break
                end                
            end

            # if not found, add to list of new points
            if !already_added
                push!(added_pts,(xi,yi,zi))
                dofindex[k][i] = length(added_pts)
            end
            
        end
    end
    ndof_test = length(added_pts)
    println("ndof vs. ndof_test = $(ndof) vs. $(ndof_test)")
    for k=1:K
        for i=1:npts
            if dofindex[k][i] != dofindex_orig[k][i]
                println("Dofindex not the same! [$(k)][$(i)]")
            end
        end
    end
    return (dofindex_orig,ndof)

end
end
