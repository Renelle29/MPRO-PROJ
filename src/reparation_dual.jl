using JuMP
using GLPK
using Gurobi

include("parser.jl")
include("robust.jl")
include("static.jl")
include("amelioration_locale.jl")


function relax_robust_dual(n, L, W, K, B, w_v, W_v, lh, distances; TimeLimit=30)

    mod = Model(Gurobi.Optimizer)
    set_optimizer_attribute(mod, "OutputFlag", 0)
    set_optimizer_attribute(mod, "TimeLimit", TimeLimit)

    @variable(mod, x[1:n,1:n] >= 0)
    @variable(mod, y[1:n,1:K] >= 0)
    @variable(mod, Δ_1 >= 0)
    @variable(mod, Δ1[1:n,1:n] >= 0)
    @variable(mod, Δ_2[1:K] >= 0)
    @variable(mod, Δ2[1:n,1:K] >= 0)

    println("--------- Starting greedy heuristic ---------")
    y_start, x_start = regret_greedy_robust(n, W, K, B, w_v, W_v, lh, distances; maxIter=10000)

    if y_start !== nothing && is_feasable(n, B, W, w_v, W_v, y_start)
        println("--------- Greedy heuristic done - Found a solution of cost $(robust_value_feasable_solution(n, L, lh, distances, x_start)) ---------")

        for i in 1:n, k in 1:K
            set_start_value(y[i,k], y_start[i,k])
        end

        for i in 1:n, j in 1:n
            set_start_value(x[i,j], x_start[i,j])
        end

    else
        println("--------- Greedy heuristic done - No solution found ---------")
    end

    # Symmetry breaking
    #@constraint(mod, [i in 1:n, k in i+1:K], y[i,k] == 0)
    #@constraint(mod, y[1,1] == 0)
    #@constraint(mod, [k in 1:K-1], sum(y[i,k] for i in 1:n) >= sum(y[i,k+1] for i in 1:n))

    # Reformulation objectif dual - Incertitude sur les distances
    @constraint(mod, [i in 1:n, j in i+1:n], Δ_1 + Δ1[i,j] >= (lh[i] + lh[j]) * x[i,j])

    # Reformulation contraintes duales - Incertitude sur les poids
    @constraint(mod, [k in 1:K], sum(w_v[i] * y[i,k] + W_v[i] * Δ2[i,k] for i in 1:n) + W * Δ_2[k] <= B)
    @constraint(mod, [i in 1:n, k in 1:K], Δ_2[k] + Δ2[i,k] >= w_v[i] * y[i,k])

    # Contraintes statiques inchangées
    @constraint(mod, [i in 1:n], sum(y[i,k] for k in 1:K) == 1)
    @constraint(mod, [i in 1:n, j in 1:n, k in 1:K], x[i,j] >= y[i,k] + y[j,k] - 1)

    @objective(mod, Min, L * Δ_1 + sum(3 * Δ1[i,j] + distances[i,j] * x[i,j] for i in 1:n, j in i+1:n))

    optimize!(mod)

    optimum = JuMP.objective_value(mod)
    lb = MOI.get(mod, MOI.ObjectiveBound())
    solve_time = MOI.get(mod, MOI.SolveTimeSec())
    nodes = MOI.get(mod, MOI.NodeCount())
    y_opt = JuMP.value(y)

    println("L'optimum vaut $(optimum). Meilleure borne inf $(lb)\n$(nodes) noeuds ont été explorés en $(round(solve_time, digits=3)) seconds")

    return (optimum, lb, solve_time, nodes, y_opt)
end

function reparation_bete(n, K, B, w_v, W_v, y_frac)

    y = zeros(Int, n, K)
    affectations = zeros(Int, n)

    for i in 1:n
        k = argmax(y_frac[ i, :])
        y[i, k] = 1
        affectations[i] = k
    end

    poids_cliques = zeros(Float64, K)
    for i in 1:n
        poids_cliques[affectations[i]] += w_v[i]*(1+W_v[i])
    end

    for k in 1:K
        while  poids_cliques[k] > B

            Ik = [i for i in 1:n if affectations[i] == k]
            sort!(Ik, by = i -> y_frac[i, k])
            moved = false
            for i in Ik
                candidates = sortperm(y_frac[i, :], rev = true)
                for k2 in candidates
                    if k2 != k && poids_cliques[k2] + w_v[i]*(1+W_v[i]) <= B
                        y[i, k] = 0
                        y[i, k2] = 1
                        affectations[i] = k2

                        poids_cliques[k]  -= w_v[i]*(1+W_v[i])
                        poids_cliques[k2] += w_v[i]*(1+W_v[i])
                        moved = true
                        break
                    end
                end
                moved && break
            end
            if !moved
                error("Réparation impossible : capacité trop restrictive")
            end
        end
    end

    return y
end

function heuristique_reparation(n, L, W, K, B, w_v, W_v, lh, distances)
    (optimum, lb, solve_time, nodes, y_opt) = relax_robust_dual(n, L, W, K, B, w_v, W_v, lh, distances)
    println("Valeur de la relaxation duale obtenue : ", optimum)
    y = reparation_mieux(n, L, W, K, B , w_v, W_v, y_opt)
    if !isnothing(y)
        x = creer_x(y, n, K)
        (optimumf, solve_time1, δ1_opt)  = sub_solve_1(n, L, lh, distances, x)
        println("Valeur obtenue apres reparation: ", optimumf)
        return x, optimumf
    end
end 

function couts_transferts_continu(n, K, lh, distances, y)   
    couts_trans = zeros(n, K, K)
    for i in 1:n
        for k1 in 1:K
            for k2 in 1:K
            couts_trans[i, k1, k2] = sum((distances[i, j] + 3*(lh[i] + lh[j]))*y[j, k2]*y[i, k1]
                                        for j in 1:n if i != j)
            end 
        end
    end 
    return couts_trans
end

function maj_couts_trans(n , K, lh, distances, y, couts_trans, i, k1, k2)
    for j in 1:n 
        for q1 in 1:K
            for q2 in 1:K
                if j == i
                    if q1 == k1
                        couts_trans[j, q1, q2] = 0 
                    elseif q1 == k2
                        couts_trans[j, q1, q2] *= 1+y[i, k1]
                    end
                else 
                    d = (distances[i, j] + 3*(lh[i]+lh[j]))*y[i, k1]*y[j, q1]
                    if q2 == k1
                        couts_trans[j, q1, q2] -= d 
                    elseif q2 == k2
                        couts_trans[j, q1, q2] += d
                    end
                end
            end
        end
    end
    return couts_trans
end 


function reparation_mieux(n, L, W, K, B, w_v, W_v, y_frac)

    pire_poids = [sum(w_v[i]*(1+W_v[i])*y_frac[i, k] for i in 1:n) for k in 1:K]
    couts_trans = couts_transferts_continu(n, K, lh, distances, y_frac)
    y = copy(y_frac) 
    stop = maximum([y[i, k]*(1-y[i, k]) for i in 1:n for k in 1:K]) 
    vals = [(couts_trans[j, q1, q2] - couts_trans[j, q1, q1], j, q1, q2)
           for j in 1:n for q1 in 1:K for q2 in 1:K if (y[j, q1]*y[j, q2] != 0 && q1 != q2)]
    
    while !isempty(vals) #tant que des transferts sont possibles

        ok = false 
        (best_tuple, _) = findmin(vals)
        minval, i, k1, k2 = best_tuple
        while !ok #on impose de verifier les contraintes de poids
            if pire_poids[k2] + y[i, k1]*w_v[i]*(1+W_v[i]) <= B
                ok = true
            else 
                filter!(x -> x != (minval, i, k1, k2), vals)
                if isempty(vals)
                    println("Tableau vide prematurement")
                    return nothing
                end
                (best_tuple, _) = findmin(vals)
                minval, i, k1, k2 = best_tuple
            end
        end

        y[i, k2] += y[i, k1]
        pire_poids[k1] -= y[i, k1]*w_v[i]*(1+W_v[i])
        pire_poids[k2] += y[i, k1]*w_v[i]*(1+W_v[i])
        maj_couts_trans(n , K, lh, distances, y, couts_trans, i, k1, k2) 
        y[i, k1] = 0
        vals = [(couts_trans[j, q1, q2] - couts_trans[j, q1, q1], j, q1, q2)
           for j in 1:n for q1 in 1:K for q2 in 1:K if (y[j, q1]*y[j, q2] != 0 && q1 != q2)]
    end
    y = round.(Int, y)
       
    poids_max = [sub_solve_2(n, W, w_v, W_v, y, k)[1] for k in 1:K]
    while maximum(poids_max) > B
        println("Solution reparee non admissible")
        println(y)
        return nothing
    end
    x = creer_x(y, n, K)
    return x
end
