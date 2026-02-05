function solve_sansPoids(n, K, distances)
    mod = Model(Gurobi.Optimizer)
    set_optimizer_attribute(mod, "OutputFlag", 0)

    @variable(mod, x[1:n,1:n] >= 0)
    @variable(mod, y[1:n,1:K], Bin)

    @constraint(mod, [i in 1:n], sum(y[i,k] for k in 1:K) == 1)
    @constraint(mod, [i in 1:n, j in 1:n, k in 1:K], x[i,j] >= y[i,k] + y[j,k] - 1)

    @objective(mod, Min, sum(x[i, j]*distances[i, j] for i in 1:n, j in i+1:n))

    optimize!(mod)

    optimum = JuMP.objective_value(mod)
    y_opt = JuMP.value(y)
    x_opt = JuMP.value(x)
    return optimum, y_opt
end 

#=function remplissage_homogene(n, L, W, K, B, w_v, W_v, lh, distances)
    cout_sur_poids = zeros(n)
    poids_moyen = sum(w_v[i]*(1+ W_v[i]) for i in 1:n)/ K
    affectation = zeros(n)
    nb_affectes = 0
    val = 0
    y = zeros(Int, n, K)
    for k in 1:K
        poids_clique = 0
        while poids_clique < poids_moyen && nb_affectes < n
            i = argmin(cout_sur_poids)
            println(cout_sur_poids[i])
            affectation[i] = k
            y[i, k] = 1
            nb_affectes += 1
            poids_clique += w_v[i]*(1+ W_v[i])
            val += w_v[i]*(1+ W_v[i])*cout_sur_poids[i]
            cout_sur_poids[i] = Inf
            for j in 1:n
                if affectation[j] == 0
                    cout_sur_poids[j] += distances[i, j] + 3*(lh[i] + lh[j])/ w_v[j]*(1+ W_v[j])
                end
            end
        end
        for j in 1:n 
            if affectation[j] == 0
                cout_sur_poids[j] = 0
            end
        end
    end 
    return val, y
end 
NULLE =#