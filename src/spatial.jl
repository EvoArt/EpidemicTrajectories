# Spatial capture-recapture observations: detectors at known or latent
# positions, each active on some occasions, and capture counts per individual,
# occasion and detector. For the observation weights of a LatentHMMData model
# whose individuals have activity centres.

"""
    SpatialCaptures

Detectors and captures for [`capture_loglik`](@ref). Build with
[`spatial_captures`](@ref).
"""
struct SpatialCaptures
    xy::Matrix{Float64}
    latent_col::Vector{Int}           # 0 for a mapped detector, else its column in the latent positions
    latent_detectors::Vector{Int}
    on::Vector{Vector{Int}}           # detectors active on each occasion
    start::Matrix{Int}                # occasion x individual, into caps
    stop::Matrix{Int}
    caps::Vector{Tuple{Int,Int}}      # (detector, count)
end

"""
    spatial_captures(; detector_xy, active, captures, n_individuals, latent_detectors = Int[])

- `detector_xy`: `2 x J` detector positions. Columns of latent detectors are ignored.
- `active`: `J x n_occasions`, whether each detector was trapping on each occasion.
- `captures`: `(individual, occasion, detector, count)` for every detector that
  caught an individual on an occasion, e.g. `zip(df.i, df.occasion, df.detector, df.n)`.
- `latent_detectors`: the detectors whose positions are unknown. Their positions
  are passed to [`capture_loglik`](@ref) as a `2 x n` matrix in this order,
  usually a [`Shared`](@ref) latent.
"""
function spatial_captures(; detector_xy::AbstractMatrix, active::AbstractMatrix, captures,
                          n_individuals::Integer, latent_detectors = Int[])
    J, n_occ = size(active)
    size(detector_xy) == (2, J) || throw(DimensionMismatch("detector_xy should be 2 x $J"))
    col = zeros(Int, J)
    for (c, j) in enumerate(latent_detectors); col[j] = c; end
    rows = sort!([(Int(i), Int(o), Int(j), Int(n)) for (i, o, j, n) in captures if n > 0])
    start = ones(Int, n_occ, n_individuals); stop = zeros(Int, n_occ, n_individuals)
    caps = Tuple{Int,Int}[]
    for (r, (i, o, j, n)) in enumerate(rows)
        (r == 1 || rows[r-1][1:2] != (i, o)) && (start[o, i] = length(caps) + 1)
        push!(caps, (j, n))
        stop[o, i] = length(caps)
    end
    SpatialCaptures(Matrix{Float64}(detector_xy), col, collect(Int, latent_detectors),
                    [findall(view(active, :, o)) for o in 1:n_occ], start, stop, caps)
end

"""
    caught(sc::SpatialCaptures, i, occasion) -> Bool

Whether individual `i` was caught on `occasion`.
"""
caught(sc::SpatialCaptures, i, o) = sc.stop[o, i] >= sc.start[o, i]

"""
    trapping(sc::SpatialCaptures, occasion) -> Bool

Whether any detector was active on `occasion`.
"""
trapping(sc::SpatialCaptures, o) = !isempty(sc.on[o])

@inline function _det_xy(sc::SpatialCaptures, j, latent)
    c = sc.latent_col[j]
    c == 0 ? (sc.xy[1, j], sc.xy[2, j]) : (latent[1, c], latent[2, c])
end

"""
    capture_loglik(sc, i, occasion, centre, sigma, rate; detectors = nothing, form = :hazard)

The log probability of individual `i`'s captures on `occasion`, given its
activity centre `centre = (x, y)`. Each active detector `j` has a half-normal
kernel `k_j = exp(-d_j^2 / (2 sigma^2))`, `d_j` its distance from the centre.

- `form = :hazard`: caught at least once with probability
  `1 - exp(-rate * sum(k))`, each capture at detector `j` with probability
  `k_j / sum(k)`.
- `form = :poisson`: the count at detector `j` is `Poisson(rate * k_j)`.

`detectors`: the positions of the latent detectors (see [`spatial_captures`](@ref)).
"""
function capture_loglik(sc::SpatialCaptures, i, o, centre, sigma, rate; detectors = nothing,
                        form::Symbol = :hazard)
    x, y = centre
    c = inv(2 * sigma^2)
    g = zero(c * x)
    @inbounds for j in sc.on[o]
        dx, dy = _det_xy(sc, j, detectors)
        g += exp(-c * ((x - dx)^2 + (y - dy)^2))
    end
    n = 0; m = zero(x); lf = 0.0
    @inbounds for r in sc.start[o, i]:sc.stop[o, i]
        j, nj = sc.caps[r]
        dx, dy = _det_xy(sc, j, detectors)
        m += nj * ((x - dx)^2 + (y - dy)^2); n += nj
        for k in 2:nj; lf += log(k); end
    end
    if form === :hazard
        n == 0 ? -rate * g : log1mexp(-rate * g) - c * m - n * log(g)
    elseif form === :poisson
        -rate * g + n * log(rate) - c * m - lf
    else
        throw(ArgumentError("capture_loglik: form is :hazard or :poisson, got :$form"))
    end
end

"""
    starting_centres(sc, index, n; bounds, rng = Xoshiro(1)) -> 2 x n x N array

Starting activity centres for a [`Field`](@ref) of `n` elements, where
`index(occasion)` is the element in use on an occasion. Each individual's
centre on an element is the mean position of its captures at mapped detectors,
carried to elements without captures (linear in between, flat at the ends).
Individuals caught only at latent detectors start in the middle of
`bounds = (xmin, xmax, ymin, ymax)`, and individuals never caught at one uniform
position inside them.
"""
function starting_centres(sc::SpatialCaptures, index, n; bounds, rng = Random.Xoshiro(1))
    n_occ, N = size(sc.start)
    S = zeros(2, n, N)
    for i in 1:N
        sx = zeros(n); sy = zeros(n); w = zeros(n)
        for o in 1:n_occ, r in sc.start[o, i]:sc.stop[o, i]
            j, nj = sc.caps[r]
            sc.latent_col[j] == 0 || continue
            e = index(o)
            sx[e] += nj * sc.xy[1, j]; sy[e] += nj * sc.xy[2, j]; w[e] += nj
        end
        have = findall(>(0), w)
        if isempty(have) && any(o -> caught(sc, i, o), 1:n_occ)
            S[1, :, i] .= (bounds[1] + bounds[2]) / 2; S[2, :, i] .= (bounds[3] + bounds[4]) / 2
            continue
        elseif isempty(have)
            S[1, :, i] .= bounds[1] + rand(rng) * (bounds[2] - bounds[1])
            S[2, :, i] .= bounds[3] + rand(rng) * (bounds[4] - bounds[3])
            continue
        end
        mx = sx[have] ./ w[have]; my = sy[have] ./ w[have]
        for e in 1:n
            k = searchsortedlast(have, e)
            if k == 0
                S[1, e, i] = mx[1]; S[2, e, i] = my[1]
            elseif k == length(have) || have[k] == e
                S[1, e, i] = mx[k]; S[2, e, i] = my[k]
            else
                f = (e - have[k]) / (have[k+1] - have[k])
                S[1, e, i] = mx[k] + f * (mx[k+1] - mx[k]); S[2, e, i] = my[k] + f * (my[k+1] - my[k])
            end
        end
    end
    S
end

"""
    starting_detectors(sc, centres, index; bounds) -> 2 x n matrix

Starting positions for the latent detectors: the mean of `centres` (as from
[`starting_centres`](@ref)) over the individuals and elements with a capture
there, kept inside `bounds`. A detector with no captures starts in the middle.
"""
function starting_detectors(sc::SpatialCaptures, centres, index; bounds)
    n_occ, N = size(sc.start)
    X = zeros(2, length(sc.latent_detectors)); w = zeros(length(sc.latent_detectors))
    for i in 1:N, o in 1:n_occ, r in sc.start[o, i]:sc.stop[o, i]
        j, nj = sc.caps[r]
        c = sc.latent_col[j]; c == 0 && continue
        X[1, c] += centres[1, index(o), i]; X[2, c] += centres[2, index(o), i]; w[c] += 1
    end
    for c in axes(X, 2)
        x = w[c] > 0 ? X[1, c] / w[c] : (bounds[1] + bounds[2]) / 2
        y = w[c] > 0 ? X[2, c] / w[c] : (bounds[3] + bounds[4]) / 2
        X[1, c] = clamp(x, bounds[1] + 1e-3, bounds[2] - 1e-3)
        X[2, c] = clamp(y, bounds[3] + 1e-3, bounds[4] - 1e-3)
    end
    X
end
