
#### package code
using InverseFunctions
##### inference

"""
    forward_filter(observation_process, model, data, X, i)

Run the forward pass of a Hidden Markov Model (HMM) filter for individual `i`
over that individual's sampling window.

Algorithm:
- Initializes the filtering distribution at the first sampled time.
- For each subsequent time point `t`, applies one-step prediction using the
  transition matrix and then updates with observation likelihood weights.
- Stores `α_t(s) = p(x_t = s | y_{1:t})` row-wise in `probs`.
- Caches per-time transition matrices in `trans_cache` for reuse by the
  backward sampler.

Notes:
- This implementation follows the standard forward recursion used in HMMs and
  state-space models.
- In this walkthrough, `transition_matrix_at(...)` provides the individual/time
  specific latent dynamics, and `observation_process(...)` provides per-state
  observation likelihood factors.

References:
- Rabiner, L. R. (1989). A tutorial on Hidden Markov Models and selected
  applications in speech recognition. *Proceedings of the IEEE*.
- Cappé, O., Moulines, E., & Rydén, T. (2005). *Inference in Hidden Markov
  Models*. Springer.
"""

function X_init(model,data)
    X = dats.X_init_func(data.T,data.n_individuals)
    for func in data.initialization_functions
        func(model,data,X)
    end
    return X
end

function iFFBS(model,data,X)
    for i in 1:data.n_individuals
        iFFBS_individual(model,data,X,i)
    end
    for func in data.per_iteration_functions
        func(model,data,X)
    end
end


function iFFBS_individual(model,data,X,i)
    start_sampling,end_sampling = data.sampling_period[i]
    xᵢ = @view X[start_sampling:end_sampling,i]
    data.xᵢ_prev = copy(X[start_sampling:end_sampling,i])

    # Erase focal i's contribution so the aggregators represent the rest of the
    # population while we forward-filter i (gives the (I+1)/(M+1) accounting).
    for t in start_sampling:end_sampling, derived_summary in data.derived_summaries
        derived_summary(model, data, X, X[t,i], i, t; reverse=true)
    end

    probs, trans_cache = forward_filter(observation_process,xᵢ,start_sampling,end_sampling,model,data,X,i; rest_contribution = data.rest_contribution, affected_individuals = data.affected_individuals)
    backward_sample(probs, trans_cache,xᵢ,start_sampling,end_sampling,model,data,X,i,rng)

    # Add focal i's newly sampled contribution back into the aggregators.
    for t in start_sampling:end_sampling, derived_summary in data.derived_summaries
        derived_summary(model, data, X, X[t,i], i, t)
    end

    for func in data.per_individual_functions
        func(model,data,X,i)
    end
end

no_rest_contribution(model,data,X,i,t,n_states) = fill(1.0, n_states)

function _affected_ids_at(affected_individuals, t, i)
    # If no sparse map was configured, signal "use default/all" behavior.
    affected_individuals === nothing && return nothing
    # Retrieve only individuals influenced by i at this (t, i).
    return affected_individuals[t, i]
end

function _rest_weights(rest_contribution, model, data, X, i, t, n_states, affected_ids)
    # Prefer the richer callback signature when affected IDs are available.
    if affected_ids !== nothing && applicable(rest_contribution, model, data, X, i, t, n_states, affected_ids)
        return rest_contribution(model, data, X, i, t, n_states, affected_ids)
    end
    # Backward-compatible fallback to the original signature.
    return rest_contribution(model, data, X, i, t, n_states)
end

function forward_filter(observation_process,xᵢ,start_sampling,end_sampling,model,data,X,i; rest_contribution = no_rest_contribution, affected_individuals = nothing)
    # Inclusive time window where this individual's latent path is sampled.
    t = start_sampling

    # View into this individual's latent trajectory over the active window.

    # probs[j, s] stores filtering mass for state s at local index j.
    probs = zeros(length(xᵢ),data.n_states)
    trans_cache = Vector{Any}(undef, length(xᵢ))
    trans_cache[1] = nothing

    # Initialize α_{t0}.
    probs[1,:] .= initialise_forward_filter(model,data,X,i,t)
    for j in 2:length(xᵢ)
        t+=1
        trans = transition_matrix_at(data.trans_mat,model, data, X, i, t)
        trans_cache[j] = trans
        # Fetch optional sparse dependency list for this individual/time.
        affected_ids = _affected_ids_at(affected_individuals, t, i)
        # One full forward recursion step: predict then update.
        probs[j,:] .= forward_filter_step(observation_process,rest_contribution,probs[j-1,:],trans,model,data,X,i,t,affected_ids)
    end    
    return probs, trans_cache
end

function forward_filter_step(observation_process,rest_contribution,prev_probs,trans,model,data,X,i,t,affected_ids=nothing)
    n_states = data.n_states

    # Predictive step: p(x_t | y_{1:t-1}) from previous filtering probs.
    pred_probs = trans' * prev_probs

    # Observation likelihood multipliers for current time.
    obs_weights = observation_process(data, X, i, t)

    # Coupled-HMM correction from all other chains (defaults to ones).
    rest_weights = _rest_weights(rest_contribution, model, data, X, i, t, n_states, affected_ids)
        
    # Bayes update up to proportionality.
    unnorm = pred_probs .* obs_weights .* rest_weights
    z = sum(unnorm)

    if z <= 0
        # Defensive fallback if all weights underflow/are invalid.
        return fill(one(eltype(prev_probs)) / n_states, n_states)
    end

    # Normalize to valid probabilities.
    return unnorm ./ z
end

function initialise_forward_filter(model,data,X,i,t)
data.starting_state(model,data,X,i,t)
end

function _sample_categorical(rng, probs)
    # Inverse-CDF sampling from a categorical distribution.
    u = rand(rng)
    c = zero(eltype(probs))
    @inbounds for k in eachindex(probs)
        c += probs[k]
        if u <= c
            return k
        end
    end
    return lastindex(probs)
end

"""
    backward_sample(observation_process, model, data, X, i, rng)

Sample a full latent trajectory for individual `i` using
Forward-Filtering Backward-Sampling (FFBS).

Algorithm:
1. Run forward filter to obtain `α_t(s) = p(x_t=s | y_{1:t})`.
2. Sample terminal state `x_T ~ Cat(α_T)`.
3. For `t = T-1, ..., 1`, sample
   `p(x_t | x_{t+1}, y_{1:t}) ∝ α_t(x_t) * p(x_{t+1} | x_t)`.
4. Write sampled states back into `X` for this individual and return the view.

This is the classic FFBS smoother for discrete-state HMMs and is equivalent to
drawing from the exact latent-path conditional under the model assumptions.

References:
- Carter, C. K., & Kohn, R. (1994). On Gibbs sampling for state space models.
  *Biometrika*.
- Frühwirth-Schnatter, S. (1994). Data augmentation and dynamic linear models.
  *Journal of Time Series Analysis*.
- Godsill, S. J., Doucet, A., & West, M. (2004). Monte Carlo smoothing for
  nonlinear time series. *Journal of the American Statistical Association*.
"""
function backward_sample(probs, trans_cache,xᵢ,start_sampling,end_sampling,model,data,X,i,rng)
    # Forward pass provides all filtering marginals needed for backward draws.
    n_t = length(xᵢ)

    # Sample terminal latent state from p(x_T | y_{1:T}).
    xᵢ[n_t] = _sample_categorical(rng, probs[n_t,:])

    for j in (n_t - 1):-1:1
        trans = trans_cache[j + 1]
        next_state = xᵢ[j + 1]

        # Backward conditional up to normalization:
        # p(x_t | x_{t+1}, y_{1:t}) ∝ α_t(x_t) * p(x_{t+1}|x_t).
        cond = probs[j,:] .* view(trans,:,next_state)
        z = sum(cond)
        weights = cond ./ z

        # Draw x_t and continue backward.
        xᵢ[j] = _sample_categorical(rng, weights)
    end

end

##### Simulation

##### Simulation

"""
    simulate_forward!(model, data, X, rng)

Simulate latent trajectories forward in time without conditioning on observations.
This reuses the same `model`, `data`, `X`, transition matrix machinery, and
user-provided hook functions used by inference.
"""
function simulate_forward!(model,data,X,rng)
    for i in 1:data.n_individuals
        simulate_forward_individual!(model,data,X,i,rng)

        if hasproperty(data, :per_individual_functions)
            for func in data.per_individual_functions
                func(model,data,X,i)
            end
        end
    end

    if hasproperty(data, :per_iteration_functions)
        for func in data.per_iteration_functions
            func(model,data,X)
        end
    end

    return X
end

function simulate_forward_individual!(model,data,X,i,rng)
    start_sampling,end_sampling = data.sampling_period[i]

    # Ensure initial state is available for this individual's simulation window.
    if X[start_sampling,i] <= 0
        init_probs = initialise_forward_filter(model,data,X,i,start_sampling)
        X[start_sampling,i] = _sample_categorical(rng, init_probs)
    end

    for t in (start_sampling + 1):end_sampling
        trans = transition_matrix_at(data.trans_mat,model,data,X,i,t)
        prev_state = X[t - 1, i]

        # Sample x_t from p(x_t | x_{t-1}) using the corresponding row.
        X[t,i] = _sample_categorical(rng, view(trans, prev_state, :))
    end

    return nothing
end

##### loglik

"""
    build_loglik(observation_process)

Return a callable `loglik(model, data, X)` that evaluates the joint
log-likelihood `log p(y, X | θ)` for a latent trajectory `X`.
"""
function build_loglik(observation_process)
    return (model, data, X) -> loglik(observation_process, model, data, X)
end

"""
    loglik(observation_process, model, data, X)

Compute joint likelihood
`log p(y, X | θ) = log p(X | θ) + log p(y | X, θ)`.

Includes:
- initial-state mass at each individual's first sampled time,
- transition probabilities along the latent path,
- observation likelihood terms from `observation_process`.
"""
function loglik(observation_process, model, data, X)
    ll = 0.0

    for i in 1:data.n_individuals
        start_sampling, end_sampling = data.sampling_period[i]

        init_state = X[start_sampling, i]
        init_probs = initialise_forward_filter(model, data, X, i, start_sampling)
        p_init = init_probs[init_state]
        if p_init <= 0
            error("Invalid initial-state probability in loglik: p_init=$(p_init), individual=$(i), time=$(start_sampling), state=$(init_state). This implies an impossible latent initialization under current model/data.")
        end
        ll += log(p_init)

        for t in start_sampling:end_sampling
            state = X[t, i]

            # Observation term: p(y_{i,t} | x_{i,t}, θ).
            obs_weights = observation_process(model, data, X, i, t)
            p = obs_weights[state]
            if p <= 0
                error("Invalid observation probability in loglik: p_obs=$(p), individual=$(i), time=$(t), state=$(state). This implies an impossible observation-state combination under current model/data.")
            end
            ll += log(p)

            # Transition term: p(x_{i,t} | x_{i,t-1}, θ), for t > start.
            if t > start_sampling
                trans = transition_matrix_at(data.trans_mat, model, data, X, i, t)
                prev_state = X[t - 1, i]
                p_trans = trans[prev_state, state]
                if p_trans <= 0
                    error("Invalid transition probability in loglik: p_trans=$(p_trans), individual=$(i), time=$(t), prev_state=$(prev_state), state=$(state). This implies an impossible latent transition under current model/data.")
                end
                ll += log(p_trans)
            end
        end
    end

    return ll
end


##### helpers

"""
    build_affected_individuals_from_groups(group_membership; include_self=false)

Build a matrix `affected_individuals[t, i]::Vector{Int}` listing which
individuals are affected by individual `i` at time `t`, based on shared
group membership at that time.

Expected layout for `group_membership`: rows are time points, columns are
individuals (`group_membership[t, i]`).

If `include_self=false` (default), `i` is excluded from `affected_individuals[t, i]`.
"""
function build_affected_individuals_from_groups(group_membership; include_self=false)
    n_timepoints, n_individuals = size(group_membership)
    # Output matrix has one vector cell per (time, individual).
    affected_individuals = Matrix{Vector{Int}}(undef, n_timepoints, n_individuals)

    for t in 1:n_timepoints
        # Build group -> members map once per time for efficient lookup.
        members_by_group = Dict{eltype(group_membership), Vector{Int}}()

        for i in 1:n_individuals
            g = group_membership[t, i]
            # Accumulate members of each group at time t.
            if haskey(members_by_group, g)
                push!(members_by_group[g], i)
            else
                members_by_group[g] = [i]
            end
        end

        for i in 1:n_individuals
            g = group_membership[t, i]
            members = members_by_group[g]
            if include_self
                # Optionally keep self in the influenced set.
                affected_individuals[t, i] = copy(members)
            else
                # Default: only other individuals are potentially affected.
                affected_individuals[t, i] = [j for j in members if j != i]
            end
        end
    end

    return affected_individuals
end

"""
    set_affected_individuals_from_groups!(data, group_membership; include_self=false)

Convenience mutator: builds `affected_individuals` from `group_membership`
and stores it in `data.affected_individuals` for use in iFFBS forward coupling.
"""
function set_affected_individuals_from_groups!(data, group_membership; include_self=false)
    # Precompute sparse dependency structure and attach to data for iFFBS use.
    data.affected_individuals = build_affected_individuals_from_groups(group_membership; include_self=include_self)
    return data.affected_individuals
end


@inline function _matrix_at_time_individual(mat, t, i)
    # Prefer time x individual layout; fall back to individual x time.
    if size(mat, 1) >= t && size(mat, 2) >= i
        return mat[t, i]
    end
    return mat[i, t]
end


function make_rest_contribution(;
    normalize=true,
    min_logprob=-1e12,
    affected_ids,
    neighbor_logprob)

    function rest_contribution(model, data, X, i, t, n_states, affected_ids=nothing)
        # Focal's current state, restored after evaluating all candidates.
        current_state = X[t, i]

        # Neighbours whose transition prob depends on focal i at time t.
        ids = affected_ids

        # logw[s] stores log contribution from affected neighbors for candidate state s.
        logw = zeros(Float64, n_states)

        for s in 1:n_states
            # Set focal candidate state, then add its contribution to the aggregators.
            X[t, i] = s
            for derived_summary in data.derived_summaries
                derived_summary(model, data, X, s, i, t)
            end
            acc = 0.0
            for j in ids
                # Neighbour transition log-prob given focal candidate state.
                lp = neighbor_logprob(model, data, X, j, t, i)
                acc += max(lp, min_logprob)
            end
            # Remove the candidate contribution again (aggregators back to rest-only).
            for derived_summary in data.derived_summaries
                derived_summary(model, data, X, s, i, t; reverse=true)
            end
            logw[s] = acc
        end

        # Restore focal's original state.
        X[t, i] = current_state

        if normalize
            # Stabilize before exponentiation; equivalent up to proportionality.
            logw .-= maximum(logw)
        end
        # Forward step expects multiplicative nonnegative weights.
        return exp.(logw)
    end

    return rest_contribution
end
"""
    make_neighbor_logprob_from_transitions(transition_matrix_with_override; eps_prob=1e-12)

Build a `neighbor_logprob` callback from a transition-matrix constructor.
This avoids writing repetitive indexing/log code.

Required callback signature for `transition_matrix_with_override`:
`transition_matrix_with_override(model, data, X, j, t, summary_override, updated_id, candidate_state)`
which must return a transition matrix for individual `j` at time `t`.
"""
function make_neighbor_logprob_from_transitions(transition_matrix_with_override; eps_prob=1e-12)
    return function (model, data, X, j, t, summary_override, updated_id, candidate_state)
        # Build j's transition matrix under the candidate override from chain i.
        trans = transition_matrix_with_override(model, data, X, j, t, summary_override, updated_id, candidate_state)
        # Realized transition of neighbor j at this time slice.
        from_state = X[t, j]
        to_state = X[t + 1, j]
        # Convert probability to log-scale with floor for numerical safety.
        p = trans[from_state, to_state]
        return log(max(p, eps_prob))
    end
end


"""
    @derived_summary name target OP delta

Build a reversible derived-summary updater from one compound-assignment update.
`OP` is auto-inverted so the same call can add a contribution (forward) or
remove it (reverse):

- `+=` ↔ `-=`
- `*=` ↔ `/=`

The generated `name(model, data, X, s, i, t; reverse=false)` applies the forward
op when `reverse=false` and the inverse op when `reverse=true`. `delta` is
evaluated with the candidate state `s`. Available symbols: `model, data, X, s, i, t`.

Power users can skip the macro entirely and put their own functions in
`data.derived_summaries`; each must have the signature
`f(model, data, X, s, i, t; reverse=false)` and undo itself when `reverse=true`.
"""
macro derived_summary(name, expr)
    op = expr.head
    inv_op = op === :(+=) ? :(-=) :
             op === :(-=) ? :(+=) :
             op === :(*=) ? :(/=) :
             op === :(/=) ? :(*=) :
             error("@derived_summary supports += -= *= /= (got $op)")
    forward = expr
    reverse_expr = Expr(inv_op, expr.args[1], expr.args[2])
    quote
        $(esc(name)) = (model, data, X, s, i, t; reverse=false) -> begin
            if reverse
                $(esc(reverse_expr))
            else
                $(esc(forward))
            end
            nothing
        end
    end
end
###############################################

#### user code

function obs_proc(model,data, X, i, t)
    # Per-state observation likelihood multipliers at time t.
    weights = ones(Float64, data.n_states)

    # If individual is known to have been captured after time t,
    # they cannot be in the dead state at time t.
    if hasproperty(data, :last_capture_time) && data.last_capture_time[i] > t
        weights[end] = 0.0
    end

    # Optional capture component (alive detection process).
    # Expected shape/indexing in this walkthrough: capture_mat[t, i].
    # Convention:
    # - capture_mat[t, i] == 1 => seen/captured alive
    # - capture_mat[t, i] == 0 => not seen
    # - capture_mat[t, i] < 0  => missing / no capture datum
    if hasproperty(data, :capture_mat)
        capture_obs = data.capture_mat[t, i]
        if capture_obs >= 0
            p_det_alive = hasproperty(model, :p_detect_alive) ? model.p_detect_alive : 0.8
            if capture_obs > 0
                # Seen alive implies dead state impossible at this time.
                weights[end] = 0.0
                for s in 1:(data.n_states - 1)
                    weights[s] *= p_det_alive
                end
            else
                # Not seen: alive states get miss-detection penalty, dead stays unchanged.
                for s in 1:(data.n_states - 1)
                    weights[s] *= (1 - p_det_alive)
                end
            end
        end
    end

    # In this sketch, states 2 and 3 are treated as "infected-like" for testing.
    infected_states = [2,3]
    for j in 1:data.n_tests
        # Encoded test result for test j and individual i at current time.
        y = data.test_mat[j,i]
        if y > -1
            # Convention: terminal/dead state gets zero weight in this observation model.
            weights[end] = 0.0 # dead state has zero probability
            sens = model.sensitivity[j]
            spec = model.specificity[j]

            # Positive if y > 0; otherwise treated as negative.
            is_positive = y > 0
            for s in 1:data.n_states-1#skip dead state
                # P(test + | state s): sensitivity for infected-like states,
                # false-positive rate for non-infected states.
                p_pos_given_s = s in infected_states ? sens : (1 - spec)

                # Multiply test-specific contribution into total state weight.
                weights[s] *= is_positive ? p_pos_given_s : (1 - p_pos_given_s)
            end
        end
    end
    return weights
end

function count_infections_per_group(model,data, X)
    # Rebuild per-group infected counts from scratch across all times.
    data.n_infected_per_group = zeros(Int, data.n_groups,data.n_timepoints)
    for i in 1:data.n_individuals
        for t in 1:data.n_timepoints
            # Here state code 3 is interpreted as infected.
            if X[i,t] == 3
                data.n_infected_per_group[data.group[t,i], t] += 1
            end
        end
    end
end

function update_infections_per_group(model,data, X, i)

    # Previous sampled trajectory for individual i (before current update).
    xᵢ_prev = data.xᵢ_prev
    start_sampling,end_sampling = data.sampling_period[i]

    # Current sampled trajectory segment for individual i.
    xᵢ = @view X[start_sampling:end_sampling,i]
    for t in start_sampling:end_sampling    
        # Only adjust counts where this individual's latent state changed.
        if xᵢ[t] != xᵢ_prev[t]
            i_group = data.group[t,i]
            if xᵢ[t] == 3
                # Transitioned into infected state.
                data.n_infected_per_group[i_group, t] += 1
            elseif xᵢ_prev[t] == 3
                # Transitioned out of infected state.
                data.n_infected_per_group[i_group, t] -= 1
            end
        end
    end
end

after_X_init_funcs = [count_infections_per_group]
after_individual_update_funcs = [update_infections_per_group]

function infection_func(model, data, i, t)
    g = data.group[i]
    I_minus = count(j -> j != i && data.states[j, t] == 1, data.members(data, g))
    -expm1(-(model.α + model.β * I_minus))     # 1 - exp(-(α + β·I₋))
end

function _age_at(data, i, t)
    # Support either age_mat[t, i] or age_mat[i, t] layout.
    if size(data.age_mat, 1) == data.n_timepoints
        return data.age_mat[t, i]
    end
    return data.age_mat[i, t]
end

function gompertz_makeham(model, data, i, t)
    # Discrete-time death probability from continuous Gompertz-Makeham hazard:
    # μ(a) = A + B * exp(C * a), where a is age.
    # Over interval Δt: q = 1 - exp(-∫ μ(a+s) ds).
    A = model.A
    B = model.B
    C = model.C
    a = _age_at(data, i, t)
    Δt = hasproperty(data, :age_step) ? data.age_step : 1.0

    H = if C == 0
        (A + B) * Δt
    else
        A * Δt + (B / C) * (exp(C * (a + Δt)) - exp(C * a))
    end

    q_death = -expm1(-H)
    return clamp(q_death, 0.0, 1.0)
end

function initial_prob(model, data, i, t)
   # - If born during monitoring, initialize from birth time (typically susceptible).
   # - If born before monitoring, initialize at first sampled time using νE/νI.

   birth_t = data.birth_time[i]
   start_t, _ = data.sampling_period[i]

   # State order: [:S, :E, :I, :D]
   start_dead = 0.0
   nuE_i = 0.0
   nuI_i = 0.0

   if birth_t < start_t
       # Born before this individual's sampling start:
       # use time-specific ν values if available.
       nu_idx = findfirst(==(start_t), data.nuTimes)
       nuE_i = data.nuEs[nu_idx]
       nuI_i = data.nuIs[nu_idx]
   else
       # Born during monitoring: newborn starts susceptible at birth.
       nuE_i = 0.0
       nuI_i = 0.0
   end

   still_susceptible = 1.0 - start_dead - nuE_i - nuI_i
   return [still_susceptible, nuE_i, nuI_i, start_dead]
end

state_space = [:S, :E, :I, :D] # order informs transition matrix etc.

trans_mat = @transitions :individual :auto_self begin
    @survival survival_func death=:D
    S -> E = infection_func
    E -> I = progression_func
end

# equivalent to 

trans_mat = @transitions :individual :auto_self begin
    S -> E = infection_func * survival_func
    E -> I = progression_func * survival_func
    (S,E,I) -> D = 1 - survival_func
end

# equivalent to
trans_mat = @transitions :individual begin
    S -> S = survival_func * (1 - infection_func)
    S -> E = infection_func * survival_func
    E -> E = survival_func * (1 - progression_func)
    E -> I = progression_func * survival_func
    I -> I = survival_func * (1 - 1)
    (S,E,I) -> D = 1 - survival_func
end

# equivalent to
trans_mat = @transitions :individual begin
    S -> S = survival_func * (1 - infection_func)
    S -> E = infection_func * survival_func
    E -> E = survival_func * (1 - progression_func)
    E -> I = progression_func * survival_func
    I -> I = survival_func * (1 - 1)
    (S,E,I) -> D = 1 - survival_func
    D -> D = 1
end

# compilerturn rhs into a function e.g 1 - survival_func

derived_summaries = @aggregate begin
    MperGT[data.group[i,t],t] += 1 if state != 4
    IperGT[data.group[i,t],t] += 1 if state == 3
end 
#or
derived_summaries = @aggregate begin
    count state not 4 MperGT[data.group[i,t],t] 
    count state 3 IperGT[data.group[i,t],t] 
end 
#or
@derived_summary Infcount   data.IperGT[data.group[i,t], t] += (s == 3)
@derived_summary Alivecount  data.MperGT[data.group[i,t], t] += (s != 4)
derived_summaries = [
    Infcount,
    Alivecount,
]
