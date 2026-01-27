function distance_matrix(coords::Matrix{Float64})
    n = size(coords, 1)
    D = zeros(Float64, n, n)
    for i in 1:n
        for j in i:n
            d = sqrt((coords[i,1] - coords[j,1])^2 + (coords[i,2] - coords[j,2])^2)
            D[i,j] = d
            D[j,i] = d  # By symmetry
        end
    end
    return D
end

function parse(file::String)
    include(file)
    distances = distance_matrix(coordinates)

    data = Dict(
        "n" => n,
        "L" => L,
        "W" => W,
        "K" => K,
        "B" => B,
        "w_v" => w_v,
        "W_v" => W_v,
        "lh" => lh,
        "coordinates" => coordinates,
        "distances" => distances
    )

    return data
end