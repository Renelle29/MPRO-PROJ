using CSV
using DataFrames
using JuMP
using Gurobi

include("src/parser.jl")
include("src/static.jl")
include("src/robust.jl")

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


function evaluate_file_for_csv(file_path::String)
    # Inclusion des sources et de la donnée
    for file in readdir("src")
        if endswith(file, ".jl") include(joinpath("src", file)) end
    end
    include(file_path)
    data_tuple = parse()

    all_results = Dict()
    for fname in keys(functions_config)
        func = getfield(Main, fname)
        try
            args = get_args_from_tuple(data_tuple, functions_config[fname][:args])
            result = func(args...)
            all_results[fname] = extract_values(fname, result)
        catch
            all_results[fname] = nothing
        end
    end

    # --- Calculs des indicateurs ---
    valid_opts_robust = [res[:optimum] for (name, res) in all_results 
                         if name != :solve_static && res !== nothing && haskey(res, :optimum) && res[:optimum] !== nothing]
    
    valid_lbs = [res[:lb] for res in values(all_results) 
                 if res !== nothing && haskey(res, :lb) && res[:lb] !== nothing]

    min_opt_robust = isempty(valid_opts_robust) ? NaN : minimum(valid_opts_robust)
    best_lb = isempty(valid_lbs) ? -Inf : maximum(valid_lbs)

    # --- Construction de la ligne du CSV ---
    row = Dict{Symbol, Any}(:Instance => basename(file_path))

    # 1. Gap Static-Robuste
    if haskey(all_results, :solve_static) && all_results[:solve_static] !== nothing && !isnan(min_opt_robust)
        st_opt = all_results[:solve_static][:optimum]
        row[:Gap_Static_Robuste] = (st_opt - min_opt_robust) / st_opt
    else
        row[:Gap_Static_Robuste] = NaN
    end

    # 2. Pour chaque méthode robuste : Temps et Gap Opt-LB
    robust_methods = [:solve_robust_dual, :solve_cutting_planes_CB, :solve_cutting_planes_noCB, :regret_greedy_robust]
    
    for m in robust_methods
        res = haskey(all_results, m) ? all_results[m] : nothing
        
        prefix = string(m)
        if res !== nothing && haskey(res, :optimum) && res[:optimum] !== nothing
            opt = res[:optimum]
            # Temps de calcul
            row[Symbol(prefix, "_Time")] = haskey(res, :time) ? res[:time] : NaN
            # Gap Opt-LB
            row[Symbol(prefix, "_Gap_LB")] = (best_lb != -Inf) ? (opt - best_lb) / opt : NaN
        else
            row[Symbol(prefix, "_Time")] = NaN
            row[Symbol(prefix, "_Gap_LB")] = NaN
        end
    end

    return row
end

function process_all_data(data_folder::String, output_csv::String)
    # Lister tous les fichiers de données (ajustez l'extension si nécessaire)
    files = filter(f -> endswith(f, ".tsp"), readdir(data_folder))
    
    all_rows = []
    for f in files
        println("Traitement de : $f")
        push!(all_rows, evaluate_file_for_csv(joinpath(data_folder, f)))
    end

    # Création du DataFrame
    df = DataFrame(all_rows)

    # --- Réorganisation forcée des colonnes dans l'ordre demandé ---
    ordered_cols = [:Instance, :Gap_Static_Robuste]
    robust_methods = [:solve_robust_dual, :solve_cutting_planes_CB, :solve_cutting_planes_noCB, :regret_greedy_robust]
    
    for m in robust_methods
        push!(ordered_cols, Symbol(string(m), "_Time"))
        push!(ordered_cols, Symbol(string(m), "_Gap_LB"))
    end

    # On ne garde que les colonnes qui existent réellement dans le DF
    final_cols = [c for c in ordered_cols if c in propertynames(df)]
    df = df[:, final_cols]

    # Sauvegarde
    CSV.write(output_csv, df)
    println("\nTerminé. Résultats sauvegardés dans : $output_csv")
end

# --- Lancement ---
if !isempty(ARGS)
    process_all_data(ARGS[1], "resultats_comparaison.csv")
else
    println("Usage: julia test_eval.jl <dossier_data>")
end