function solve_static(n,K,B,w_v,distances)

    mod = Model(Gurobi.Optimizer)
    set_optimizer_attribute(mod, "OutputFlag", 0)
    set_optimizer_attribute(mod, "TimeLimit", 60)

    @variable(mod, x[1:n,1:n] >= 0)
    @variable(mod, y[1:n,1:K], Bin)

    @constraint(mod, [k in 1:K], sum(w_v[i] * y[i,k] for i in 1:n) <= B) # Max weight for a set K_i
    @constraint(mod, [i in 1:n], sum(y[i,k] for k in 1:K) == 1) # Each vertex belongs to a set
    @constraint(mod, [i in 1:n, j in 1:n, k in 1:K], x[i,j] >= y[i,k] + y[j,k] - 1) # i & j belong to k => i & j belong to same set

    @objective(mod, Min, 0.5 * sum(distances[i,j] * x[i,j] for i in 1:n, j in 1:n))

    optimize!(mod)

    optimum = JuMP.objective_value(mod)
    solve_time = MOI.get(mod, MOI.SolveTimeSec())
    nodes = MOI.get(mod, MOI.NodeCount())
    y_opt = JuMP.value(y)

    println("L'optimum vaut $(optimum)\n$(nodes) noeuds ont été explorés en $(round(solve_time, digits=3)) seconds")

    return (optimum, solve_time, nodes, y_opt)
end