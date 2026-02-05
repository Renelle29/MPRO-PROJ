function solve_static(n,K,B,w_v,distances; TimeLimit=20)

    count = min_nb_pairs(n,K)

    mod = Model(Gurobi.Optimizer)
    set_optimizer_attribute(mod, "OutputFlag", 0)
    set_optimizer_attribute(mod, "TimeLimit", TimeLimit)

    @variable(mod, 1 >= x[i=1:n,j=i+1:n] >= 0)
    @variable(mod, y[1:n,1:K], Bin)

    println("--------- Starting greedy heuristic ---------")
    y_start, x_start = regret_greedy_static(n,K,B,w_v,distances)

    if y_start !== nothing
        println("--------- Greedy heuristic done - Found a solution of cost $(get_value_static(n,K,distances,y_start)) ---------")

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

    @constraint(mod, [k in 1:K], sum(w_v[i] * y[i,k] for i in 1:n) <= B) # Max weight for a set K_i
    @constraint(mod, [i in 1:n], sum(y[i,k] for k in 1:K) == 1) # Each vertex belongs to a set
    @constraint(mod, [i in 1:n, j in i+1:n, k in 1:K], x[i,j] >= y[i,k] + y[j,k] - 1) # i & j belong to k => i & j belong to same set
    #@constraint(mod, [i in 1:n, j in 1:n, k in 1:K], x[i,j] >= y[i,k] * y[j,k]) # i & j belong to k => i & j belong to same set NON LINEAR

    # Strenghtening the relaxation
    @constraint(mod, sum(x[i,j] for i in 1:n, j in i+1:n) >= count)
    #Triangle inequalities
    #@constraint(mod, [i in 1:n, j in i+1:n, l in i+1:j-1, k in 1:K], x[i,j] >= x[i,l] + x[l,j] - 1)

    @objective(mod, Min, sum(distances[i,j] * x[i,j] for i in 1:n, j in i+1:n))

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
                    c = sum(distances[i,j] * y[i,k] for j in 1:n)

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

function regret_greedy_static(n,K,B,w_v,distances; maxIter=10000)
    y = zeros(Int, n, K)
    cost = Inf

    for iter in 1:maxIter
        y_temp = zeros(Int, n, K)
        load = zeros(Float64, K)
        assigned = falses(n)

        for step in 1:n
            best_regret = -Inf
            best_i = 0
            best_k = 0

            for i in 1:n
                assigned[i] && continue

                costs = Float64[]
                ks = Int[]

                for k in 1:K
                    if load[k] + w_v[i] <= B
                        c = sum(distances[i,j] * y_temp[j,k] for j in 1:n) + 1e-6 * rand()
                        push!(costs, c)
                        push!(ks, k)
                    end
                end

                if length(costs) == 0
                    #println("No feasible cluster for item $i")
                    break
                end

                perm = sortperm(costs)
                c1 = costs[perm[1]]
                k1 = ks[perm[1]]
                c2 = length(costs) > 1 ? costs[perm[2]] : Inf

                regret = c2 - c1

                if regret > best_regret
                    best_regret = regret
                    best_i = i
                    best_k = k1
                end
            end

            # Assign the selected item
            try
                y_temp[best_i, best_k] = 1
                load[best_k] += w_v[best_i]
                assigned[best_i] = true 
            catch
                break
            end
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

function simple_lower_bound(n,K,distances)

    d_vector = []
    for i in 1:n
        for j in i+1:n
            push!(d_vector, distances[i,j])
        end
    end
    sort!(d_vector)

    count = min_nb_pairs(n,K)

    slb = 0
    for i in 1:count
        slb += d_vector[i]
    end
    println("Lowest bound found  $slb")

    return slb
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