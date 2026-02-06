#Fontions 

function creer_x(y, n, K)
    x = zeros(Int, n, n) 
    for i in 1:n-1
        for j in i+1:n
            x[i, j] = sum(y[i, k]*y[j, k] for k in 1:K)
        end
    end 
    return x
end 

function solution_poids_egaux(n, L, W, K, B, w_v, W_v, lh, distances)
    poids_moyen = sum(w_v[i]*(1+ W_v[i]) for i in 1:n)/ K
    affectations = zeros(Int, n)
    place_restante = zeros(K)
    poids_cliques = zeros(K)
    y = zeros(Int, n, K)
    for k in 1:K
        place_restante[k] = poids_moyen
    end

    for i in 1:n
        best_k_plus, best_k_moins = 0, 0
        best_r_plus, best_r_moins = Inf, -Inf
        for k in 1:K
            if place_restante[k] >= w_v[i]*(1 + W_v[i]) && best_r_plus > place_restante[k]
                best_k_plus = k 
                best_r_plus = place_restante[k]
            elseif place_restante[k] < w_v[i]*(1 + W_v[i]) && best_r_moins < place_restante[k]
                best_k_moins = k
                best_r_moins = place_restante[k]
            end 
        end 
        if best_k_plus != 0
            affectations[i] = best_k_plus
            place_restante[best_k_plus] -= w_v[i]*(1 + W_v[i])
            poids_cliques[best_k_plus] +=  w_v[i]*(1 + W_v[i])
            y[i, best_k_plus] = 1
        else 
            affectations[i] = best_k_moins
            place_restante[best_k_moins] = -Inf
            poids_cliques[best_k_moins] +=  w_v[i]*(1 + W_v[i])
            y[i, best_k_moins] = 1
        end 
    end
    valeur = sum( (distances[i, j] + 3*(lh[i]+lh[j]))*y[i, k]*y[j, k] for i in 1:n,  
                    j in 1:n, k in 1:K if i < j )
    return affectations, y, poids_cliques, valeur
end 

function couts_transferts(n, K, lh, distances, y)   
    couts_trans = zeros(n, K)
    for i in 1:n
        for k in 1:K
            couts_trans[i, k] = sum((distances[i, j] + 3*(lh[i] + lh[j]))*y[j, k] 
                                        for j in 1:n if i != j)
        end
    end 
    return couts_trans
end


#PARTIE 1 : TRANSFERT
function transfert_viable(B, w_v, W_v, poids_cliques, affectations, couts_trans, i, k)
    if k != affectations[i]
        accepte = B >= poids_cliques[k] + w_v[i]*(1+W_v[i])
        delta = couts_trans[i, k] - couts_trans[i, affectations[i]]
    end
    return delta, accepte
end

function transfert(n, B, w_v, W_v, lh, distances, poids_cliques, affectations, y, couts_trans, i, k)
    copie_poids = copy(poids_cliques)
    copie_affectations = copy(affectations)
    copie_y = copy(y)
    copie_couts = copy(couts_trans)
    ki = affectations[i]
 
    copie_poids[k] += w_v[i]*(1 + W_v[i])
    copie_poids[ki] -= w_v[i]*(1 + W_v[i])
    copie_y[i, ki], copie_y[i, k] = 0, 1
    copie_affectations[i] = k 
    for j in 1:n
        if j != i
            copie_couts[j, k] += distances[i, j] + 3*(lh[i]+ lh[j])
            copie_couts[j, ki] -= distances[i, j] + 3*(lh[i]+ lh[j])
        end

    end
    return copie_poids, copie_affectations, copie_y, copie_couts 
end 

function meilleur_transfert(n, B, w_v, W_v, lh, distances, poids_cliques, affectations, y, couts_trans)
    copie_poids = copy(poids_cliques)
    copie_affectations = copy(affectations)
    copie_y = copy(y)
    copie_couts = copy(couts_trans)
    dlta = 0
    for i in 1:n
        for k in 1:K
            if k != affectations[i]
                dl, acc = transfert_viable(B, w_v, W_v, poids_cliques, affectations, couts_trans, i, k)
                if dl < dlta && acc
                    copie_poids, copie_affectations, copie_y, copie_couts = transfert(n, B, w_v, W_v, lh, distances, 
                                                                 poids_cliques, affectations, y, couts_trans, i, k)
                    dlta = dl
                end
            end
        end
    end 
    return copie_poids, copie_affectations, copie_y, copie_couts, dlta
end

function heuristique_meilleur_transfert(n, L, W, K, B, w_v, W_v, lh, distances)
    #= affectations, y, poids_cliques, valeur = 
    solution_poids_egaux(n, L, W, K, B, w_v, W_v, lh, distances)
    couts_trans = couts_transferts(n, K, lh, distances, y)=#

    y, x = regret_greedy_robust(n, W, K, B, w_v, W_v, lh, distances)
    while isnothing(y)
        y, x = regret_greedy_robust(n, W, K, B, w_v, W_v, lh, distances)
    end
    (valeur,t, dlt) = sub_solve_1(n, L, lh, distances, x)
    println(valeur)
    
    affectations = zeros(Int, n)
    poids_cliques = zeros(K)
    for i in 1:n
        affectations[i] = argmax(y[i, :])
        poids_cliques[affectations[i]] += w_v[i]*(1 + W_v[i]) 
    end
    couts_trans = couts_transferts(n, K, lh, distances, y)
    cont = -1
    delta = -1
    while delta < 0
        poids_cliques, affectations, y, couts_trans, delta = 
        meilleur_transfert(n, B, w_v, W_v, lh, distances, poids_cliques, affectations, y, couts_trans)
        cont += 1
    end
    println("Nombre de transferts : ", cont)

    x = creer_x(y, n , K)
    (val,t, dlt) = sub_solve_1(n, L, lh, distances, x)
    println(val)
    return y, val, cont, delta
end



# PARTIE 2 : ECHANGE 
function echange_possible(B, w_v, W_v, affectations, poids_cliques, i, j)
    ki = affectations[i]
    kj = affectations[j]
    b1 = B >= poids_cliques[ki] - w_v[i]*(1 + W_v[i]) + w_v[j]*(1 + W_v[j])
    b2 = B >= poids_cliques[kj] - w_v[j]*(1 + W_v[j]) + w_v[i]*(1 + W_v[i])
    return b1 && b2
end 

function echange_viable(n, B, w_v, W_v, lh, distances, poids_cliques, affectations, y, couts_trans, i, j)
    ki = affectations[i]
    kj = affectations[j]
   delta = couts_trans[i, kj] + couts_trans[j, ki]- couts_trans[i, ki] - couts_trans[j, kj]
   accepte = echange_possible(B, w_v, W_v, affectations, poids_cliques, i, j)
   return delta, accepte
end 

function echange(n, B, w_v, W_v, lh, distances, poids_cliques, affectations, y, couts_trans, i, j)
    copie_poids = copy(poids_cliques)
    copie_affectations = copy(affectations)
    copie_y = copy(y)
    copie_couts = copy(couts_trans)
    ki = affectations[i]
    kj = affectations[j]

    copie_poids[ki] += w_v[j]*(1 + W_v[j]) - w_v[i]*(1 + W_v[i])
    copie_poids[kj] += w_v[i]*(1 + W_v[i]) - w_v[j]*(1 + W_v[j])
    copie_y[i, kj], copie_y[i, ki] = 1, 0
    copie_y[j, ki], copie_y[j, kj] = 1, 0
    copie_affectations[i], copie_affectations[j] = kj, ki

    for t in 1:n
        if t == i
            copie_couts[t, ki] += distances[t, j] + 3*lh[j]
            copie_couts[t, kj] -= distances[t, j] + 3*lh[j]
        elseif t == j 
            copie_couts[t, ki] -= distances[t, i] + 3*lh[i]
            copie_couts[t, kj] += distances[t, i] + 3*lh[i]
        else
            copie_couts[t, ki] += distances[t, j] + 3*lh[j]- distances[t, i] - 3*lh[i]
            copie_couts[t, kj] += distances[t, i] + 3*lh[i]- distances[t, j] - 3*lh[j]
        end       
    end

    return copie_poids, copie_affectations, copie_y, copie_couts
end

function meilleur_echange(n, B, w_v, W_v, lh, distances, poids_cliques, affectations, y, couts_trans)
    copie_poids = copy(poids_cliques)
    copie_affectations = copy(affectations)
    copie_y = copy(y)
    copie_couts = copy(couts_trans)
    dlta = 0
    for i in 1:n-1
        for j in i+1:n
            if affectations[i] != affectations[j]
                dl, acc = echange_viable(n, B, w_v, W_v, lh, distances, poids_cliques, affectations, y, couts_trans, i, j)
                if dl < dlta && acc
                    copie_poids, copie_affectations, copie_y, copie_couts = echange(n, B, w_v, W_v, lh, 
                                                distances, poids_cliques, affectations, y, couts_trans, i, j)
                    dlta = dl
                end
            end
        end 
    end 
    return copie_poids, copie_affectations, copie_y, copie_couts, dlta               
end 

function heuristique_meilleur_echange(n, L, W, K, B, w_v, W_v, lh, distances)
    #=affectations, y, poids_cliques, valeur = 
    solution_poids_egaux(n, L, W, K, B, w_v, W_v, lh, distances)=#
    y, x = regret_greedy_robust(n, W, K, B, w_v, W_v, lh, distances)
    while isnothing(y)
        y, x = regret_greedy_robust(n, W, K, B, w_v, W_v, lh, distances)
    end 
    (val,t, dlt) = sub_solve_1(n, L, lh, distances, x)
    println(val)

    affectations = zeros(Int, n)
    poids_cliques = zeros(K)
    for i in 1:n
        affectations[i] = argmax(y[i, :])
        poids_cliques[affectations[i]] += w_v[i]*(1 + W_v[i]) 
    end
    couts_trans = couts_transferts(n, K, lh, distances, y)
    cont = -1
    delta = -1
    while delta < 0
        poids_cliques, affectations, y, couts_trans, delta = 
        meilleur_echange(n, B, w_v, W_v, lh, distances, poids_cliques, affectations, y, couts_trans)
        cont += 1
    end
    println("Nombre d'échanges : ", cont)

    x = creer_x(y, n , K)
    (val,t, dlt) = sub_solve_1(n, L, lh, distances, x)
    println(val)
    return y, val, cont, delta
end 




