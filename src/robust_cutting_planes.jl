function main_solve_cp(n, K, B, U1, U2)

    mod = Model(Gurobi.Optimizer)

    @variable(mod, x[1:n,1:n] >= 0)
    @variable(mod, y[1:n,1:K], Bin)
    @variable(mod, z)

    # Coupes d'optimalité - Incertitude sur les distances
    @constraint(mod, [l1 in U1], sum(l1[i,j] * x[i,j] for i in 1:n, j in i+1:n) <= z)

    # Coupes de faisabilité - Incertitude sur les poids
    @constraint(mod, [k in 1:K, w2 in U2], sum(w2[i] * y[i,k] for i in 1:n) <= B)

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