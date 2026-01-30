function solve_robust_dual(n, L, W, K, B, w_v, W_v, lh, distances; TimeLimit=20)

    mod = Model(Gurobi.Optimizer)
    set_optimizer_attribute(mod, "OutputFlag", 0)
    set_optimizer_attribute(mod, "TimeLimit", TimeLimit)

    @variable(mod, x[1:n,1:n] >= 0)
    @variable(mod, y[1:n,1:K], Bin)
    @variable(mod, Δ_1 >= 0)
    @variable(mod, Δ1[1:n,1:n] >= 0)
    @variable(mod, Δ_2[1:K] >= 0)
    @variable(mod, Δ2[1:n,1:K] >= 0)

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