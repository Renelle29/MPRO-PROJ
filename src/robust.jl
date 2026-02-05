function solve_robust_dual(n, L, W, K, B, w_v, W_v, lh, distances; TimeLimit=20)

    count = min_nb_pairs(n,K)

    mod = Model(Gurobi.Optimizer)
    set_optimizer_attribute(mod, "OutputFlag", 0)
    set_optimizer_attribute(mod, "TimeLimit", TimeLimit)

    @variable(mod, 1 >= x[i=1:n,j=i+1:n] >= 0)
    @variable(mod, y[1:n,1:K], Bin)
    @variable(mod, Δ_1 >= 0)
    @variable(mod, Δ1[1:n,1:n] >= 0)
    @variable(mod, Δ_2[1:K] >= 0)
    @variable(mod, Δ2[1:n,1:K] >= 0)

    println("--------- Starting greedy heuristic ---------")
    y_start, x_start = regret_greedy_robust(n, W, K, B, w_v, W_v, lh, distances; maxIter=10000)
    y_start, x_start = canonicalize_solution(y_start)

    if y_start !== nothing && is_feasable(n, B, W, w_v, W_v, y_start)
        println("--------- Greedy heuristic done - Found a solution of cost $(robust_value_feasable_solution(n, L, lh, distances, x_start)) ---------")

        for i in 1:n, k in 1:K
            set_start_value(y[i,k], y_start[i,k])
        end

        for i in 1:n
            for j in i+1:n
                set_start_value(x[i,j], x_start[i,j])
            end
        end

    else
        println("--------- Greedy heuristic done - No solution found ---------")
    end

    # Symmetry breaking
    fix(y[1, 1], 1; force=true)
    for i in 1:n
        for k in i+1:K
            fix(y[i, k], 0; force=true)
        end
    end
    #@constraint(mod, [k in 1:K-1], sum(y[i,k] for i in 1:n) >= sum(y[i,k+1] for i in 1:n))

    # Reformulation objectif dual - Incertitude sur les distances
    @constraint(mod, [i in 1:n, j in i+1:n], Δ_1 + Δ1[i,j] >= (lh[i] + lh[j]) * x[i,j])

    # Reformulation contraintes duales - Incertitude sur les poids
    @constraint(mod, [k in 1:K], sum(w_v[i] * y[i,k] + W_v[i] * Δ2[i,k] for i in 1:n) + W * Δ_2[k] <= B)
    @constraint(mod, [i in 1:n, k in 1:K], Δ_2[k] + Δ2[i,k] >= w_v[i] * y[i,k])

    # Contraintes statiques inchangées
    @constraint(mod, [i in 1:n], sum(y[i,k] for k in 1:K) == 1)
    @constraint(mod, [i in 1:n, j in i+1:n, k in 1:K], x[i,j] >= y[i,k] + y[j,k] - 1) 

    # Strenghtening the relaxation
    @constraint(mod, sum(x[i,j] for i in 1:n, j in i+1:n) >= count)

    @objective(mod, Min, L * Δ_1 + sum(3 * Δ1[i,j] + distances[i,j] * x[i,j] for i in 1:n, j in i+1:n))

    optimize!(mod)

    optimum = JuMP.objective_value(mod)
    lb = MOI.get(mod, MOI.ObjectiveBound())
    gap = MOI.get(mod, MOI.RelativeGap())
    solve_time = MOI.get(mod, MOI.SolveTimeSec())
    nodes = MOI.get(mod, MOI.NodeCount())
    y_opt = JuMP.value(y)

    println("L'optimum vaut $(optimum). Meilleure borne inf $(lb)\n$(nodes) noeuds ont été explorés en $(round(solve_time, digits=3)) seconds")

    return (optimum, lb, gap, solve_time, nodes, y_opt)
end

function solve_cutting_planes_CB(n, L, W, K, B, w_v, W_v, lh, distances; TimeLimit=20)

    count = min_nb_pairs(n,K)

    mod = Model(Gurobi.Optimizer)
    set_optimizer_attribute(mod, "LazyConstraints", 1)
    set_optimizer_attribute(mod, "OutputFlag", 0)
    set_optimizer_attribute(mod, "TimeLimit", TimeLimit)

    @variable(mod, 1 >= x[i=1:n,j=i+1:n] >= 0)
    @variable(mod, y[1:n,1:K], Bin)
    @variable(mod, z >= 0)

    println("--------- Starting greedy heuristic ---------")
    y_start, x_start = regret_greedy_robust(n, W, K, B, w_v, W_v, lh, distances; maxIter=10000)
    y_start, x_start = canonicalize_solution(y_start)

    if y_start !== nothing && is_feasable(n, B, W, w_v, W_v, y_start)
        println("--------- Greedy heuristic done - Found a solution of cost $(robust_value_feasable_solution(n, L, lh, distances, x_start)) ---------")

        for i in 1:n, k in 1:K
            set_start_value(y[i,k], y_start[i,k])
        end

        for i in 1:n
            for j in i+1:n
                set_start_value(x[i,j], x_start[i,j])
            end
        end

    else
        println("--------- Greedy heuristic done - No solution found ---------")
    end

    # Symmetry breaking
    fix(y[1, 1], 1; force=true)
    for i in 1:n
        for k in i+1:K
            fix(y[i, k], 0; force=true)
        end
    end
    #@constraint(mod, [k in 1:K-1], sum(y[i,k] for i in 1:n) >= sum(y[i,k+1] for i in 1:n))

    @constraint(mod, [i in 1:n], sum(y[i,k] for k in 1:K) == 1)
    @constraint(mod, [i in 1:n, j in i+1:n, k in 1:K], x[i,j] >= y[i,k] + y[j,k] - 1) 

    # Strenghtening the relaxation
    @constraint(mod, sum(x[i,j] for i in 1:n, j in i+1:n) >= count)

    @objective(mod, Min, z)

    function cp_callback(cb_data)
        status = callback_node_status(cb_data, mod)

        if status != MOI.CALLBACK_NODE_STATUS_INTEGER
            return
        end

        x_val = callback_value.(cb_data, x)
        y_val = callback_value.(cb_data, y)
        z_val = callback_value(cb_data, z)

        (optimum1, solve_time1, δ1_opt) = sub_solve_1(n, L, lh, distances, x_val)

        if abs(optimum1 - z_val) > 0.001
            con = @build_constraint(sum((distances[i,j] + δ1_opt[i,j] * (lh[i] + lh[j])) * x[i, j] for i in 1:n, j in i+1:n) <= z)
            MOI.submit(mod, MOI.LazyConstraint(cb_data), con)
        end

        for k in 1:K

            (optimum2, solve_time2, δ2_opt) = sub_solve_2(n, W, w_v, W_v, y_val, k)

            if optimum2 - B > 0.001
                con = @build_constraint(sum(w_v[i] * (1 + δ2_opt[i]) * y[i,k] for i in 1:n) <= B)
                MOI.submit(mod, MOI.LazyConstraint(cb_data), con)           
            end
        end

    end

    MOI.set(mod, MOI.LazyConstraintCallback(), cp_callback)

    optimize!(mod)

    optimum = JuMP.objective_value(mod)
    lb = MOI.get(mod, MOI.ObjectiveBound())
    gap = MOI.get(mod, MOI.RelativeGap())
    solve_time = MOI.get(mod, MOI.SolveTimeSec())
    nodes = MOI.get(mod, MOI.NodeCount())
    y_opt = JuMP.value(y)
    x_opt = JuMP.value(x)

    println("L'optimum vaut $(optimum). Meilleure borne inf $(lb)\n$(nodes) noeuds ont été explorés en $(round(solve_time, digits=3)) seconds")

    return (optimum, lb, gap, solve_time, nodes, y_opt, x_opt)

end

function solve_cutting_planes_noCB(n, L, W, K, B, w_v, W_v, lh, distances)

    # Initialize values
    optimum = 0
    y_opt = zeros((n,K))
    (U1, U2) = init_incertitude_sets(n,w_v,distances)
    optimal = false
    t0 = time_ns()

    # Boucle d'optimisation via plans coupants
    while (!optimal)
        optimal = true
        
        # Résoudre problème principal
        (optimum, solve_time, nodes, y_opt, x_opt) = main_solve_cp(n, K, B, U1, U2)

        # Ajout d'une coupe si on a surévalué l'optimum robuste
        (optimum1, solve_time1, δ1_opt) = sub_solve_1_fast(n, L, lh, distances, x_opt)
        if abs(optimum1 - optimum) > 0.001
            optimal = false
            l1 = zeros(n,n)
            for i in 1:n, j in 1:n
                l1[i,j] = distances[i,j] + δ1_opt[i,j] * (lh[i] + lh[j])
            end
            push!(U1, copy(l1))
        end

        # Ajout de coupes si on a violé des contraintes robustes
        for k in 1:K
            (optimum2, solve_time2, δ2_opt) = sub_solve_2_fast(n, W, w_v, W_v, y_opt, k)
            if optimum2 - B > 0.001
                optimal = false
                w2 = zeros(n)
                for i in 1:n
                    w2[i] = w_v[i] * (1 + δ2_opt[i])
                end
                push!(U2, copy(w2))
            end
        end

    end
    elapsed = (time_ns() - t0) / 1e9

    println("L'optimum robuste vaut $(optimum)\nTrouvé en $(round(elapsed, digits=3)) seconds")

    return (optimum, elapsed, y_opt)
end

function main_solve_cp(n, K, B, U1, U2)

    mod = Model(Gurobi.Optimizer)
    set_optimizer_attribute(mod, "OutputFlag", 0)
    set_optimizer_attribute(mod, "TimeLimit", 60)

    @variable(mod, x[1:n,1:n] >= 0)
    @variable(mod, y[1:n,1:K], Bin)
    @variable(mod, z)

    # Coupes d'optimalité - Incertitude sur les distances
    @constraint(mod, [l in eachindex(U1)], sum(U1[l][i, j] * x[i, j] for i in 1:n, j in i+1:n) <= z)

    # Coupes de faisabilité - Incertitude sur les poids
    @constraint(mod, [k in 1:K, u in eachindex(U2)], sum(U2[u][i] * y[i,k] for i in 1:n) <= B)

    # Contraintes statiques inchangées
    @constraint(mod, [i in 1:n], sum(y[i,k] for k in 1:K) == 1)
    @constraint(mod, [i in 1:n, j in 1:n, k in 1:K], x[i,j] >= y[i,k] + y[j,k] - 1)

    @objective(mod, Min, z)

    optimize!(mod)

    optimum = JuMP.objective_value(mod)
    solve_time = MOI.get(mod, MOI.SolveTimeSec())
    nodes = MOI.get(mod, MOI.NodeCount())
    y_opt = JuMP.value(y)
    x_opt = JuMP.value(x)

    println("L'optimum vaut $(optimum)\n$(nodes) noeuds ont été explorés en $(round(solve_time, digits=3)) seconds")

    return (optimum, solve_time, nodes, y_opt, x_opt)
end

# Trouver la meilleure coupe d'optimalité
function sub_solve_1(n, L, lh, distances, x_opt)

    mod = Model(Gurobi.Optimizer)
    set_optimizer_attribute(mod, "OutputFlag", 0)
    set_optimizer_attribute(mod, "TimeLimit", 60)

    @variable(mod, 0 <= δ1[i=1:n,j=i+1:n] <= 3)

    @constraint(mod, sum(δ1[i,j] for i in 1:n, j in i+1:n) <= L)

    @objective(mod, Max, sum(x_opt[i,j] * (distances[i,j] + δ1[i,j] * (lh[i] + lh[j])) for i in 1:n, j in i+1:n))

    optimize!(mod)

    optimum = JuMP.objective_value(mod)
    solve_time = MOI.get(mod, MOI.SolveTimeSec())
    δ1_opt = JuMP.value(δ1)

    #println("L'optimum vaut $(optimum)\nTrouvé en $(round(solve_time, digits=3)) seconds")

    return (optimum, solve_time, δ1_opt)
end

function sub_solve_1_fast(n, L, lh, distances, x_opt)

    pairs = Tuple{Int,Int,Float64}[]

    for i in 1:n, j in i+1:n
        coeff = x_opt[i,j] * (lh[i] + lh[j])
        coeff > 0 && push!(pairs, (i, j, coeff))
    end

    sort!(pairs, by = x -> -x[3])

    δ1 = zeros(n, n)
    remaining = L

    for (i, j, c) in pairs
        remaining <= 0 && break
        δ = min(3.0, remaining)
        δ1[i,j] = δ
        remaining -= δ
    end

    # Compute objective value
    value = 0.0
    for i in 1:n, j in i+1:n
        value += x_opt[i,j] * (distances[i,j] + δ1[i,j] * (lh[i] + lh[j]))
    end

    return (value, nothing, δ1)
end


# Trouver la meilleure coupe de faisabilité pour une partie k donnée
function sub_solve_2(n, W, w_v, W_v, y_opt, k)

    mod = Model(Gurobi.Optimizer)
    set_optimizer_attribute(mod, "OutputFlag", 0)
    set_optimizer_attribute(mod, "TimeLimit", 60)
    
    @variable(mod, 0 <= δ2[i=1:n] <= W_v[i])
    
    @constraint(mod, sum(δ2[i] for i in 1:n) <= W)
    
    @objective(mod, Max, sum(y_opt[i,k] * w_v[i] * (1 + δ2[i]) for i in 1:n))
    
    optimize!(mod)
    
    optimum = JuMP.objective_value(mod)
    solve_time = MOI.get(mod, MOI.SolveTimeSec())
    δ2_opt = JuMP.value(δ2)
    
    #println("L'optimum vaut $(optimum)\nTrouvé en $(round(solve_time, digits=3)) seconds")
    
    return (optimum, solve_time, δ2_opt)
end

function sub_solve_2_fast(n, W, w_v, W_v, y_opt, k)

    items = Tuple{Int,Float64}[]

    for i in 1:n
        if y_opt[i,k] >= 0.5 && W_v[i] > 0
            push!(items, (i, w_v[i]))
        end
    end

    sort!(items, by = x -> -x[2])  # descending w_i

    δ2 = zeros(n)
    remaining = W

    for (i, _) in items
        remaining <= 0 && break
        δ = min(W_v[i], remaining)
        δ2[i] = δ
        remaining -= δ
    end

    value = 0.0
    for i in 1:n
        value += y_opt[i,k] * w_v[i] * (1 + δ2[i])
    end

    return (value, nothing, δ2)
end


function init_incertitude_sets(n,w_v,distances)
    l1 = zeros(n,n)
    for i in 1:n, j in 1:n
        l1[i,j] = distances[i,j]
    end
    U1 = [copy(l1)]

    w2 = zeros(n)
    for i in 1:n
        w2[i] = w_v[i]
    end
    U2 = [copy(w2)]

    return (U1, U2)
end

function is_feasable(n, B, W, w_v, W_v, y_opt)

    feasable = true

    for k in 1:K

        (optimum2, solve_time, δ2_opt) = sub_solve_2(n, W, w_v, W_v, y_opt, k)

        if optimum2 - B > 0.001
            feasable = false
        end

    end
    
    return feasable
end

function robust_value_feasable_solution(n, L, lh, distances, x_opt)

    (optimum1, solve_time, δ1_opt) = sub_solve_1(n, L, lh, distances, x_opt)

    return optimum1
end

function main_solve_cp_lp(n, K, B, U1_all, U2_all)
    mod = Model(Gurobi.Optimizer)
    set_optimizer_attribute(mod, "OutputFlag", 0)
    set_optimizer_attribute(mod, "TimeLimit", 20)

    @variable(mod, 1 >= x[1:n,1:n] >= 0)
    @variable(mod, y[1:n,1:K], Bin)
    @variable(mod, z)

    # Coupes d'optimalité - Incertitude sur les distances
    @constraint(mod, [l in 1:10], sum(U1_all[l][i, j] * x[i, j] for i in 1:n, j in i+1:n) <= z)

    # Coupes de faisabilité - Incertitude sur les poids
    @constraint(mod, [k in 1:K, u in 1:10], sum(U2_all[u][i] * y[i,k] for i in 1:n) <= B)

    # Contraintes statiques inchangées
    @constraint(mod, [i in 1:n], sum(y[i,k] for k in 1:K) == 1)
    @constraint(mod, [i in 1:n, j in 1:n, k in 1:K], x[i,j] >= y[i,k] + y[j,k] - 1)

    @objective(mod, Min, z)

    optimize!(mod)

    optimum = JuMP.objective_value(mod)
    solve_time = MOI.get(mod, MOI.SolveTimeSec())
    y_opt = JuMP.value(y)
    x_opt = JuMP.value(x)
    lb = MOI.get(mod, MOI.ObjectiveBound())

    println("Borne duale : $(lb)\nObtenue en $(round(solve_time, digits=3)) seconds")

    return (lb, solve_time, y_opt, x_opt)
end

function regret_greedy_robust(n, W, K, B, w_v, W_v, lh, distances; maxIter=10000)
    y = zeros(Int, n, K)
    cost = Inf

    for iter in 1:maxIter
        y_temp = zeros(Int, n, K)
        load = zeros(Float64, K)
        assigned = falses(n)

        cluster_cost = zeros(Float64, n, K)
        cluster_lh_max = zeros(Float64, K)

        for step in 1:n
            best_regret = -Inf
            best_i = 0
            best_k = 0

            for i in 1:n
                assigned[i] && continue

                best1 = Inf
                best2 = Inf
                bestk = 0

                for k in 1:K
                    if load[k] + w_v[i] * (1 + W_v[i]) <= B
                        c = cluster_cost[i,k] + 3 * (lh[i] + cluster_lh_max[k]) + 1e-6 * rand()

                        if c < best1
                            best2 = best1
                            best1 = c
                            bestk = k
                        elseif c < best2
                            best2 = c
                        end
                    end
                end

                regret = best2 - best1

                if regret > best_regret
                    best_regret = regret
                    best_i = i
                    best_k = bestk
                end
            end

            try
                # Assign
                y_temp[best_i,best_k] = 1
                assigned[best_i] = true
                load[best_k] += w_v[best_i] * (1 + W_v[best_i])
            catch
                break
            end

            @inbounds for i in 1:n
                cluster_cost[i,best_k] += distances[i,best_i]
            end
            cluster_lh_max[best_k] = max(cluster_lh_max[best_k], lh[best_i])
        end

        if sum(assigned[i] for i in 1:n) == n && get_value_static(n,K,distances,y_temp) < cost
            cost = get_value_static(n,K,distances,y_temp)
            y = copy(y_temp)
        end
    end

    x = zeros((n,n))
    for i in 1:n, j in 1:n
        x[i,j] = sum(y[i,k] * y[j,k] for k in 1:K)
    end

    if x[1,1] == 1
        return y, x
    else
        return nothing, nothing
    end
end

function min_nb_pairs(n,K)

    d = div(n,K)
    r = n - K * d

    count = 0

    for i in 1:r
        for j in 1:d
            count += j
        end
    end

    for i in 1:(K-r)
        for j in 1:(d-1)
            count += j
        end
    end
    println("Minimum number of pairs $count")

    return count
end

# From a solution, return a solution that respect Symmetry constraints
function canonicalize_solution(y)
    n, K = size(y)

    # Get smallest item of a cluster
    sig = fill(Inf, K)
    for k in 1:K
        for i in 1:n
            if y[i,k] == 1
                sig[k] = i
                break
            end
        end
    end

    perm = sortperm(sig) # Index of clusters by increasing smallest item

    # Reorder y
    y_new = y[:, perm]

    # Recompute x
    x_new = zeros(n, n)
    for i in 1:n, j in 1:n
        x_new[i,j] = sum(y_new[i,k] * y_new[j,k] for k in 1:K)
    end

    return y_new, x_new
end
