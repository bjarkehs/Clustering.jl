#
#  Re-picks centers that get no samples assigned to them.
#
function p_repick_unused_centers{T<:FloatingPoint}(
    x::DArray{T},           # in: the sample set (d x n)
    costs::Vector{T},       # in: the current assignment costs (n)
    centers::Matrix{T},     # to be updated: the centers (d x k)
    unused::Vector{Int})    # in: the set of indices of centers to be updated

    # pick new centers using a scheme like kmeans++
    ds = similar(costs)
    tcosts = copy(costs)
    n = size(x, 2)

    for i in unused
        j = wsample(1:n, tcosts)
        tcosts[j] = 0
        v = Float64[ elem for elem in x[:,j] ]
        centers[:,i] = v

        ds = dcosts(x,v)
        tcosts = min(tcosts, ds)
    end
end

function local_p_update_centers(x, centers, cweights, to_update, assignments)
        quote
            l_x = localpart($x)
            l_centers = $centers
            l_cweights = $cweights
            accumulate_cols_u!(l_centers, l_cweights, l_x, $assignments, $to_update)
            (l_centers, l_cweights)
        end
    end

function p_update_centers!{T<:FloatingPoint}(
    x::DArray{T},                   # in: sample matrix (d x n)
    w::Nothing,                     # in: sample weights
    assignments::Vector{Int},       # in: assignments (n)
    to_update::Vector{Bool},        # in: whether a center needs update (k)
    centers::Matrix{T},             # out: updated centers (d x k)
    cweights::Vector{T})            # out: updated cluster weights (k)
    Tl = eltype(x)
    n::Int = size(x, 2)
    k::Int = size(centers, 2)

    # initialize center weights
    for i = 1 : k
        if to_update[i]
            cweights[i] = 0.
        end
    end

    # accumulate columns
    part_ids = procs(x)
    amount_of_procs = length(part_ids)
    tmp_cweights = zeros(Int64, (amount_of_procs, k))
    tmp_centers = Array(Array{Tl,2}, amount_of_procs)



    results = @sync [ @spawnat part_ids[i] eval(local_p_update_centers(x, centers, cweights, to_update, assignments[x.indexes[i][2]])) for i=1:amount_of_procs ]

    for i=1:length(results)
        tmp_centers[i], tmp_cweights[i,:] = take(results[i])
    end

    cweights = sum(tmp_cweights,1)
    update_range = find(to_update)
    # Sets the vector of the center to 0 if it has been updated in a process, thus having a weight larger than 0
    centers[:,update_range] = centers[:,update_range] .* (cweights[update_range]' == 0)
    # Adds the center vectors from every process if that process has a weight of the center, which is larger than 0
    centers[:,update_range] = centers[:,update_range] .+ sum([ tmp_centers[i][:,update_range] .* (tmp_cweights[i,update_range] .> 0) for i=1:amount_of_procs ])

    # sum ==> mean
    for i = 1 : k
        if to_update[i]
            @inbounds ci::Tl = 1 / cweights[i]
            multiply!(view(centers,:,i), ci)
        end
    end
end

function local_update_assignments(x, centers, is_init, assignments, costs, counts, to_update, old_to_update, k, dmat_all)        # out: the list of centers get no samples assigned to it
    quote
        l=localpart($x)
        l_is_init = $is_init
        l_to_update = $to_update
        l_centers = $centers
        # dmat = pairwise(SqEuclidean(), l_centers, l)
        if l_is_init || $dmat_all
            dmat = pairwise(SqEuclidean(), l_centers, l)
        else
            # if only a small subset is affected, only compute for that subset
            affected_inds = find($old_to_update)
            dmat_p = pairwise(SqEuclidean(), l_centers[:, affected_inds], l)
            dmat[affected_inds, :] = dmat_p
        end
        l_assignments = $assignments
        l_costs = $costs
        l_counts = $counts
        # process each sample
        for j = 1 : size(l, 2)

            # find the closest cluster to the i-th sample
            a::Int = 1
            c = dmat[1, j]
            for i = 2 : $k
                ci = dmat[i, j]
                if ci < c
                    a = i
                    c = ci
                end
            end

            # set/update the assignment

            if l_is_init
                l_assignments[j] = a
            else  # update
                pa = l_assignments[j]
                if pa != a
                    # if assignment changes,
                    # both old and new centers need to be updated
                    l_assignments[j] = a
                    l_to_update[a] = true
                    l_to_update[pa] = true
                end
            end

            # set costs and counts accordingly
            l_costs[j] = c
            l_counts[a] += 1
        end
        (l_assignments, l_costs, l_counts, l_to_update)
    end
end

function p_update_assignments!{T<:FloatingPoint}(
    x::DArray{T},               # in: sample matrix (d x n)
    centers::Matrix{T},         # in: matrix of centers (d x k)
    is_init::Bool,              # in:  whether it is the initial run
    assignments::Vector{Int},   # out: assignment vector (n)
    costs::Vector{T},           # out: costs of the resultant assignment (n)
    counts::Vector{Int},        # out: number of samples assigned to each cluster (k)
    to_update::Vector{Bool},    # out: whether a center needs update (k)
    unused::Vector{Int};
    dmat_all=false)        # out: the list of centers get no samples assigned to it

    k::Int = size(centers, 2)
    n::Int = size(x, 2)

    # re-initialize the counting vector
    fill!(counts, 0)

    old_to_update = copy(to_update)
    if is_init
        fill!(to_update, true)
    else
        fill!(to_update, false)
        if !isempty(unused)
            empty!(unused)
        end
    end

    part_ids = procs(x)
    amount_of_procs = length(part_ids)
    tmp_counts = Array(Int64, (amount_of_procs, k))
    tmp_to_update = Array(Bool, (amount_of_procs, k))

    # process each sample
    # @sync @parallel for i=1:amount_of_procs
    #     part = procs(x)[i]
    #     range = x.indexes[i][2]
    #     op = local_update_assignments(x, centers, is_init, assignments[range], costs[range], counts, to_update, k)
    #     assignments[range], costs[range], tmp_counts[i,:], tmp_to_update[i,:] = @fetchfrom part eval(op)
    # end

    refs = @sync [ @spawnat part_ids[i] eval(local_update_assignments(x, centers, is_init, assignments[x.indexes[i][2]], costs[x.indexes[i][2]], counts, to_update, old_to_update, k, dmat_all)) for i=1:amount_of_procs ]
    for i=1:length(refs)
        range = x.indexes[i][2]
        assignments[range], costs[range], tmp_counts[i,:], tmp_to_update[i,:] = take(refs[i])
    end

    counts[:] = sum(tmp_counts, 1)[:]

    to_update[:] = (sum(tmp_to_update, 1) .> 0)[:]

    # look for centers that have no associated samples

    for i = 1 : k
        if counts[i] == 0
            push!(unused, i)
            to_update[i] = false # this is handled using different mechanism
        end
    end
end

function _pkmeans!{T<:FloatingPoint}(
    x::DArray{T},                  # in: sample matrix (d x n)
    w::Union(Nothing, Vector{T}),  # in: sample weights (n)
    centers::Matrix{T},            # in/out: matrix of centers (d x k)
    assignments::Vector{Int},      # out: vector of assignments (n)
    costs::Vector{T},              # out: costs of the resultant assignments (n)
    counts::Vector{Int},           # out: the number of samples assigned to each cluster (k)
    cweights::Vector{T},           # out: the weights of each cluster
    opts::KmeansOpts)              # in: options

    # process options

    tol::Float64 = opts.tol
    max_iters::Int = opts.max_iters
    display::Symbol = opts.display

    disp_level =
        display == :none ? 0 :
        display == :final ? 1 :
        display == :iter ? 2 :
        throw(ArgumentError("Invalid value for the option 'display'."))

    # initialize

    k = size(centers, 2)
    to_update = Array(Bool, k) # indicators of whether a center needs to be updated
    unused = Int[]
    num_affected::Int = k # number of centers, to which the distances need to be recomputed

    p_update_assignments!(x, centers, true, assignments, costs, counts, to_update, unused)
    objv = w == nothing ? sum(costs) : dot(w, costs)

    # main loop

    if disp_level >= 2
        @printf "%7s %18s %18s | %8s \n" "Iters" "objv" "objv-change" "affected"
        println("-------------------------------------------------------------")
    end

    t = 0

    converged = false
    while !converged && t < opts.max_iters
        t = t + 1

        # update (affected) centers
        p_update_centers!(x, w, assignments, to_update, centers, cweights)

        if !isempty(unused)
            p_repick_unused_centers(x, costs, centers, unused)
        end

        # # update pairwise distance matrix

        if !isempty(unused)
            to_update[unused] = true
        end

        # println("Beyond this point doesn't work!")

        dmat_all = false
        if t == 1 || num_affected > 0.75 * k
            dmat_all = true
        end

        # # update assignments

        p_update_assignments!(x, centers, false, assignments, costs, counts, to_update, unused; dmat_all=dmat_all)
        num_affected = sum(to_update) + length(unused)

        # compute change of objective and determine convergence

        prev_objv = objv
        objv = w == nothing ? sum(costs) : dot(w, costs)
        objv_change = objv - prev_objv

        if objv_change > tol
            warn("The objective value changes towards an opposite direction")
        end

        if abs(objv_change) < tol
            converged = true
        end

        # display iteration information (if asked)

        if disp_level >= 2
            @printf "%7d %18.6e %18.6e | %8d\n" t objv objv_change num_affected
        end
    end

    if disp_level >= 1
        if converged
            println("K-means converged with $t iterations (objv = $objv)")
        else
            println("K-means terminated without convergence after $t iterations (objv = $objv)")
        end
    end

    return KmeansResult(centers, assignments, costs, counts, cweights, float64(objv), t, converged)
end

##### Distributed means calculation #####

function dmean(disArr::DArray)
    op = quote
            l=localpart($disArr)
            (sum(l), length(l))
        end
    localvalues=[ @fetchfrom part eval(op) for part in disArr.pmap ]
    sum([ localvalue[1] for localvalue in localvalues ] )/sum([ localvalue[2] for localvalue in localvalues ] )
end

function unite(arrOfArrs::Array)
    totalLength=sum(map(length, arrOfArrs))
    givenType=eltype(arrOfArrs[1])
    pointer = 1
    result=Array(givenType,totalLength)
    for arr in arrOfArrs
        result[pointer:pointer+length(arr)-1]=arr
        pointer += length(arr)
    end
    result
end

function dcosts(disArr::DArray, v)
    op = quote
            l=localpart($disArr)
            colwise(SqEuclidean(), $v, l)
        end
    localvalues=[ @fetchfrom part eval(op) for part in disArr.pmap ]
    unite(localvalues)
end

function p_dmat(disArr::DArray, v)
    op = quote
            l=localpart($disArr)
            pairwise(SqEuclidean(), $v, l)
        end
    localvalues=[ @fetchfrom part eval(op) for part in disArr.pmap ]
    unite(localvalues)
end

function pkmeans_initialize!{T<:FloatingPoint}(x::DArray{T}, centers::Matrix{T})
    n = size(x, 2)
    k = size(centers, 2)

    # randomly pick the first center
    si = rand(1:n)
    v = Float64[ elem for elem in x[:,si] ]
    centers[:,1] = v

    # initialize the cost vector
    costs = dcosts(x, v)

    # pick remaining (with a chance proportional to cost)
    for i = 2 : k
        si = wsample(1:n, costs)
        v = Float64[ elem for elem in x[:,si] ]
        centers[:,i] = v

        # update costs

        if i < k
            new_costs = dcosts(x, v)
            costs = min(costs, new_costs)
        end
    end
end

function pkmeans!{T<:FloatingPoint}(
    x::DArray{T},
    centers::Matrix{T},
    opts::KmeansOpts)

    m::Int, n::Int = size(x)
    m2::Int, k::Int = size(centers)
    if m != m2
        throw(ArgumentError("Mismatched dimensions in x and init_centers."))
    end
    check_k(n, k)

    w = opts.weights
    if w != nothing
        if length(w) != size(x, 2)
            throw(ArgumentError("The length of w must match the number of columns in x."))
        end
    end

    assignments = zeros(Int, n)
    costs = zeros(n)
    counts = Array(Int, k)
    weights = opts.weights
    cweights = Array(T, k)

    if isa(weights, Vector)
        if !(eltype(weights) == T)
            throw(ArgumentError("The element type of weights must be the same as that of samples."))
        end
    end

    _pkmeans!(x, weights, centers, assignments, costs, counts, cweights, opts)
end

function pkmeans{T<:FloatingPoint}(
    x::DArray{T},
    init_centers::Matrix{T},
    opts::KmeansOpts)

    pkmeans!(x, copy(init_centers), opts)
end

pkmeans(x::DArray, init_centers::Matrix; opts...) = pkmeans(x, init_centers, kmeans_opts(;opts...))

function pkmeans(x::DArray, k::Int, opts::KmeansOpts)
    m, n = size(x)
    check_k(n, k)
    init_centers = Array(eltype(x), (m, k))
    pkmeans_initialize!(x, init_centers)
    pkmeans!(x, init_centers, opts)
end

pkmeans(x::DArray, k::Int; opts...) = pkmeans(x, k, kmeans_opts(;opts...))