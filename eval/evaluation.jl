using JuMP
using GLPK
using Gurobi

include("../data/14_burma_6.tsp")
include("../src/parser.jl")
include("../src/robust.jl")
include("../src/static.jl")
include("../src/amelioration_locale.jl")
include("../src/reparation_dual.jl")
(n, L, W, K, B, w_v, W_v, lh, coordinates, distances) = parse()

heuristique_reparation(n, L, W, K, B, w_v, W_v, lh, distances)

