# test_eval.jl
using CSV
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
        :keep => [:optimum, :solve_time, :lb]
    ),
    :solve_cutting_planes_CB => Dict(
        :args => [:n, :L, :W, :K, :B, :w_v, :W_v, :lh, :distances],
        :keep => [:optimum, :solve_time, :lb]
    ),
    :solve_cutting_planes_noCB => Dict(
        :args => [:n, :L, :W, :K, :B, :w_v, :W_v, :lh, :distances],
        :keep => [:optimum, :elapsed]
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
                :solve_time => sub_result[2]
            )
        end
    else
        # Noms des résultats renvoyés par chaque fonction
        result_names = Dict(
            :solve_static => [:optimum, :lb, :gap, :solve_time, :nodes, :y_opt, :x_opt],
            :solve_robust_dual => [:optimum, :lb, :gap, :solve_time, :nodes, :y_opt],
            :solve_cutting_planes_CB => [:optimum, :lb, :gap, :solve_time, :nodes, :y_opt, :x_opt],
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

# --- Fonction principale ---
function evaluate_file(file_path::String)
    # Inclure tous les fichiers source nécessaires
    for file in readdir("src")
        if endswith(file, ".jl")
            include(joinpath("src", file))
        end
    end

    # Inclure le fichier de données
    include(file_path)

    # Appeler parse() pour récupérer tous les paramètres
    data_tuple = parse()

    println("=== Évaluation pour le fichier : $(file_path) ===")
    all_results = Dict()
    # Parcourir les fonctions
    for fname in keys(functions_config)
        func = getfield(Main, fname)
        try
            # Extraire uniquement les arguments nécessaires pour cette fonction
            args = get_args_from_tuple(data_tuple, functions_config[fname][:args])
            # Appeler la fonction
            result = func(args...)
            # Extraire les valeurs à afficher
# Stocker les valeurs extraites dans notre dictionnaire global
            all_results[fname] = extract_values(fname, result)
            
        catch e
            all_results[fname] = "ERREUR : $e"
        end
    end

    # --- AFFICHAGE UNIQUE À LA FIN ---
    println("\n--- RÉSUMÉ DES RÉSULTATS ---")
    for (fname, values) in all_results
        println("Fonction $fname :")
        if values isa String  # Cas d'erreur
            println("  $values")
        else
            for (key, val) in values
                println("  - $key = $val")
            end
        end
    end
    println("-"^30)
end

# --- Exemple d'utilisation ---
evaluate_file("data/14_burma_9.tsp")