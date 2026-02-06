using CSV
using DataFrames
using JuMP
using Gurobi

include("../src/parser.jl")
include("../src/static.jl")
include("../src/robust.jl")

# --- Configuration des fonctions à évaluer ---
# Chaque entrée contient :
#   :args => les indices dans le tuple renvoyé par parse() à utiliser
#   :keep => les valeurs à garder dans le résultat
functions_config = Dict(
    :solve_static => Dict(
        :args => [:n, :K, :B, :w_v, :distances],
        :keep => [:optimum]
    ),
    :solve_robust_dual => Dict(
        :args => [:n, :L, :W, :K, :B, :w_v, :W_v, :lh, :distances],
        :keep => [:optimum, :time, :lb]
    ),
    :solve_cutting_planes_CB => Dict(
        :args => [:n, :L, :W, :K, :B, :w_v, :W_v, :lh, :distances],
        :keep => [:optimum, :time, :lb]
    ),
    :solve_cutting_planes_noCB => Dict(
        :args => [:n, :L, :W, :K, :B, :w_v, :W_v, :lh, :distances],
        :keep => [:optimum, :time]
    ),
    :regret_greedy_robust => Dict(
        :args => [:n, :W, :K, :B, :w_v, :W_v, :lh, :distances],
        :keep => [:sub_solve_1]  # spécial, appeler sub_solve_1 si la fonction ne renvoie pas nothing
    )
)

# --- Fonction pour récupérer les valeurs depuis le tuple de parse() ---
function get_args_from_tuple(data_tuple, names)
    # mapping des noms vers positions dans le tuple renvoyé par parse()
    names_map = Dict(
        :n => 1,
        :L => 2,
        :W => 3,
        :K => 4,
        :B => 5,
        :w_v => 6,
        :W_v => 7,
        :lh => 8,
        :coordinates => 9,
        :distances => 10
    )
    return [data_tuple[names_map[name]] for name in names]
end

# --- Fonction pour extraire les valeurs utiles après appel ---
function extract_values(fname::Symbol, result)
    if fname == :regret_greedy_robust
        y, x = result
        if y === nothing && x === nothing
            return Dict(:sub_solve_1 => nothing)
        else
            # Appel sub_solve_1(n, L, lh, distances, x)
            n, L, W, K, B, w_v, W_v, lh, coordinates, distances = parse()
            sub_result = sub_solve_1(n, L, lh, distances, x)
            return return Dict(
                :optimum => sub_result[1], 
                :time => sub_result[2]
            )
        end
    else
        # Noms des résultats renvoyés par chaque fonction
        result_names = Dict(
            :solve_static => [:optimum, :lb, :gap, :time, :nodes, :y_opt, :x_opt],
            :solve_robust_dual => [:optimum, :lb, :gap, :time, :nodes, :y_opt],
            :solve_cutting_planes_CB => [:optimum, :lb, :gap, :time, :nodes, :y_opt, :x_opt],
            :solve_cutting_planes_noCB => [:optimum, :elapsed,  :y_opt]
        )
        names = result_names[fname]
        mapping = Dict()
        for (i, name) in enumerate(names)
            mapping[name] = result[i]
        end
        keep_vals = functions_config[fname][:keep]
        return Dict(val => mapping[val] for val in keep_vals)
    end
end

function extract_values(fname::Symbol, result)
    if fname == :regret_greedy_robust
        y, x = result
        if y === nothing && x === nothing
            return Dict(:optimum => nothing, :time => nothing)
        else
            # On garde votre appel local à parse()
            n, L, W, K, B, w_v, W_v, lh, coordinates, distances = parse()
            sub_result = sub_solve_1(n, L, lh, distances, x)
            return Dict(
                :optimum => sub_result[1], 
                :time => sub_result[2]
            )
        end
    else
        result_names = Dict(
            :solve_static => [:optimum, :lb, :gap, :solve_time, :nodes, :y_opt, :x_opt],
            :solve_robust_dual => [:optimum, :lb, :gap, :solve_time, :nodes, :y_opt],
            :solve_cutting_planes_CB => [:optimum, :lb, :gap, :solve_time, :nodes, :y_opt, :x_opt],
            :solve_cutting_planes_noCB => [:optimum, :elapsed, :y_opt]
        )
        
        names = result_names[fname]
        mapping = Dict()
        for (i, name) in enumerate(names)
            # Unification du nom de la clé pour le temps
            key = (name == :solve_time || name == :elapsed) ? :time : name
            mapping[key] = result[i]
        end
        
        # On ne garde que ce qui est défini dans functions_config
        keep_vals = functions_config[fname][:keep]
        # On transforme :solve_time ou :elapsed en :time dans keep_vals également
        final_dict = Dict()
        for v in keep_vals
            k_mapped = (v == :solve_time || v == :elapsed) ? :time : v
            final_dict[k_mapped] = mapping[k_mapped]
        end
        return final_dict
    end
end

# --- Fonction principale ---
function evaluate_file(file_path::String)
    # [Inclusions des fichiers src et data]
    for file in readdir("src")
        if endswith(file, ".jl") include(joinpath("src", file)) end
    end
    include(file_path)
    data_tuple = parse()

    all_results = Dict()

    # 1. Exécution et collecte des résultats
    for fname in keys(functions_config)
        func = getfield(Main, fname)
        try
            args = get_args_from_tuple(data_tuple, functions_config[fname][:args])
            result = func(args...)
            all_results[fname] = extract_values(fname, result)
        catch e
            all_results[fname] = nothing
        end
    end

    # 2. Calcul des références globales
    # MODIFICATION : On exclut explicitement :solve_static du calcul du min_opt
    valid_opts_robust = [res[:optimum] for (name, res) in all_results 
                         if name != :solve_static && 
                         res !== nothing && 
                         haskey(res, :optimum) && 
                         res[:optimum] !== nothing]
    
    # Meilleure borne inférieure (LB) parmi toutes les méthodes qui en fournissent
    valid_lbs = [(name == :solve_static ? res[:optimum] : res[:lb]) 
    for (name, res) in all_results if res !== nothing && haskey(res, (name == :solve_static ? :optimum : :lb))
         && res[(name == :solve_static ? :optimum : :lb)] !== nothing]

    # Détermination des valeurs de référence
    min_opt_robust = isempty(valid_opts_robust) ? NaN : minimum(valid_opts_robust)
    best_lb = isempty(valid_lbs) ? -Inf : maximum(valid_lbs)

    # 3. Affichage final
    println("\n=== RÉSULTATS : $(basename(file_path)) ===")
    
    # A) Écart du modèle Statique par rapport au meilleur résultat Robuste
    if haskey(all_results, :solve_static) && all_results[:solve_static] !== nothing
        st_opt = all_results[:solve_static][:optimum]
        if !isnan(min_opt_robust)
            gap_st = (min_opt_robust - st_opt) / st_opt
            println("Static Opt: $st_opt | Meilleur Robuste: $min_opt_robust")
            println("GAP Static vs Best Robust Opt: $(round(gap_st*100, digits=2))%")
        else
            println("Static Opt: $st_opt | (Aucun résultat robuste valide pour comparer)")
        end
    end

    # B) Détails des méthodes Robustes (comparaison au meilleur LB)
    println("\nDétails des méthodes robustes :")
    robust_methods = [:solve_robust_dual, :solve_cutting_planes_CB, :solve_cutting_planes_noCB, :regret_greedy_robust]
    
    for m in robust_methods
        res = haskey(all_results, m) ? all_results[m] : nothing
        print("  - Méthode $m : ")
        
        if res === nothing || !haskey(res, :optimum) || res[:optimum] === nothing
            println("N/A")
            continue
        end
        
        opt = res[:optimum]
        t = haskey(res, :time) ? res[:time] : "N/A"
        
        # Gap vs Best LB
        if best_lb != -Inf
            gap_lb = (opt - best_lb) / opt
            msg_gap = "$(round(gap_lb*100, digits=2))%"
        else
            msg_gap = "N/A (pas de LB)"
        end
        
        println("Opt = $opt | Temps = $(t)s | Gap/Best LB = $msg_gap")
    end
    println("-"^40)
end

evaluate_file("data/10_ulysses_6.tsp")