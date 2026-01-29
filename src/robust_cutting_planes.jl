function solve_cutting_planes_CB(n, L, W, K, B, w_v, W_v, lh, distances)

    mod = Model(Gurobi.Optimizer)
    set_optimizer_attribute(mod, "LazyConstraints", 1)
    set_optimizer_attribute(mod, "OutputFlag", 0)
    set_optimizer_attribute(mod, "TimeLimit", 60)

    @variable(mod, x[1:n,1:n] >= 0)
    @variable(mod, y[1:n,1:K], Bin)
    @variable(mod, z >= 0)

    @constraint(mod, [i in 1:n], sum(y[i,k] for k in 1:K) == 1)
    @constraint(mod, [i in 1:n, j in 1:n, k in 1:K], x[i,j] >= y[i,k] + y[j,k] - 1)

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
    solve_time = MOI.get(mod, MOI.SolveTimeSec())
    nodes = MOI.get(mod, MOI.NodeCount())
    y_opt = JuMP.value(y)
    x_opt = JuMP.value(x)

    println("L'optimum vaut $(optimum)\n$(nodes) noeuds ont été explorés en $(round(solve_time, digits=3)) seconds")

    return (optimum, solve_time, nodes, y_opt, x_opt)

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
        (optimum1, solve_time1, δ1_opt) = sub_solve_1(n, L, lh, distances, x_opt)
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
            (optimum2, solve_time2, δ2_opt) = sub_solve_2(n, W, w_v, W_v, y_opt, k)
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

    return (optimum, y_opt)
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

    @variable(mod, 0 <= δ1[1:n,1:n] <= 3)

    @constraint(mod, sum(δ1[i,j] for i in 1:n, j in i+1:n) <= L)

    @objective(mod, Max, sum(x_opt[i,j] * (distances[i,j] + δ1[i,j] * (lh[i] + lh[j])) for i in 1:n, j in i+1:n))

    optimize!(mod)

    optimum = JuMP.objective_value(mod)
    solve_time = MOI.get(mod, MOI.SolveTimeSec())
    δ1_opt = JuMP.value(δ1)

    #println("L'optimum vaut $(optimum)\nTrouvé en $(round(solve_time, digits=3)) seconds")

    return (optimum, solve_time, δ1_opt)
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

function is_feasable()

    feasable = true

    for k in 1:K

        (optimum, solve_time, δ2_opt) = sub_solve_2(n, W, w_v, W_v, y_opt, k)

        if optimum - B > 0.001
            feasable = false
        end

    end
    
    return feasable
end