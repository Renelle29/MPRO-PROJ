function solve_static(n,K,B,w_v,distances; TimeLimit=20)

    mod = Model(Gurobi.Optimizer)
    set_optimizer_attribute(mod, "OutputFlag", 0)
    set_optimizer_attribute(mod, "TimeLimit", TimeLimit)

    @variable(mod, x[1:n,1:n] >= 0)
    @variable(mod, y[1:n,1:K], Bin)

    @constraint(mod, [k in 1:K], sum(w_v[i] * y[i,k] for i in 1:n) <= B) # Max weight for a set K_i
    @constraint(mod, [i in 1:n], sum(y[i,k] for k in 1:K) == 1) # Each vertex belongs to a set
    @constraint(mod, [i in 1:n, j in 1:n, k in 1:K], x[i,j] >= y[i,k] + y[j,k] - 1) # i & j belong to k => i & j belong to same set

    @objective(mod, Min, 0.5 * sum(distances[i,j] * x[i,j] for i in 1:n, j in 1:n))

    optimize!(mod)

    optimum = JuMP.objective_value(mod)
    lb = MOI.get(mod, MOI.ObjectiveBound())
    solve_time = MOI.get(mod, MOI.SolveTimeSec())
    nodes = MOI.get(mod, MOI.NodeCount())
    y_opt = JuMP.value(y)

    println("L'optimum vaut $(optimum). Meilleure borne inf $(lb)\n$(nodes) noeuds ont été explorés en $(round(solve_time, digits=3)) seconds")

    return (optimum, lb, solve_time, nodes, y_opt)
end

function greedy_solution_static(n,K,B,w_v,distances)
    y = zeros((n,K))
    done = zeros(n)

    for _ in 1:n

        total_cost = -1
        item = -1
        insert = -1

        for i in 1:n
            
            if done[i] == 0

                cost = Inf
                set = -1

                for k in 1:K
                    c = sum(distances[i,k] * y[i,k] for i in 1:n)

                    if sum(w_v[j] * y[j,k] for j in 1:n) + w_v[i] > B
                        c = Inf
                    end

                    if c < cost
                        cost = c
                        set = k
                    end
                end

                if cost > total_cost
                    total_cost = cost
                    item = i
                    insert = set
                end
            end
        end

        if !(total_cost == -1)
            done[item] = 1
            y[item, insert] = 1
        
        else
            println("Couldn't find any feasable solution")
            break
        end
    end

    return y
end

function get_value_static(n,K,distances,y)
    total = 0.0
    for k in 1:K
        for i in 1:n
            for j in i+1:n
                total += distances[i,j] * y[i,k] * y[j,k]
            end
        end
    end
    return total
end