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

function parse()
    distances = distance_matrix(coordinates)

    data = (n, L, W, K, B, w_v, W_v, lh, coordinates, distances)

    return data
end