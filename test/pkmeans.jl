addprocs(7)

@everywhere using Clustering

a = drand(3,2000)
l_a = convert(Array{Float64,2}, a)

seeds = rand(3,5)

pk = @elapsed b = pkmeans(a, seeds)

k = @elapsed c = kmeans(l_a, seeds)

println("Pkmeans Assignments == Kmeans Assignments? $(b.assignments == c.assignments)")

println("Pkmeans duration: $pk")
println("Kmeans duration: $k")

a = drand(3,950000)
l_a = convert(Array{Float64,2}, a)

seeds = rand(3,10)

pk = @elapsed b = pkmeans(a, seeds)

k = @elapsed c = kmeans(l_a, seeds)

println("Pkmeans Assignments == Kmeans Assignments? $(b.assignments == c.assignments)")

println("Pkmeans duration: $pk")
println("Kmeans duration: $k")