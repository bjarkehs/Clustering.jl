module Clustering
    using NumericExtensions
    using Distance
    using StatsBase

    import Base.show

    export kmeans, kmeans!, kmeans_opts, update!
    export pkmeans, pkmeans!

    export AffinityPropagationOpts
    export affinity_propagation

    include("utils.jl")
    include("seeding.jl")
    include("kmeans.jl")
    include("affprop.jl")
    include("dbscan.jl")
    include("pkmeans.jl")
end
