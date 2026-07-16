
# Desired workflow sketch for the refactor.
# This is intentionally an API guide for implementation, not a finalized runnable script.

using EpidemicTrajectories

# ------------------------------------------------------------
# 1) Define state space + transitions (cattle E. coli style)
# ------------------------------------------------------------

@transitions :individual CattleEcoli begin
    S => I : infection
    I => S : recovery
end

state_space = CattleEcoli()

# Rates are placeholders here; concrete parameterization stays in package code.
rates = (
    infection = (pars, i, t, data) -> begin
        β_within = pars.β_within
        β_env = pars.β_env
        I_group = data.infected_per_group[data.group[i], t]
        β_within * I_group + β_env
    end,
    recovery = (pars, i, t, data) -> pars.γ,
)

tests = (
    sensitivity = 0.95,
    specificity = 0.98,
)

# ------------------------------------------------------------
# 2) Build function handles separately
# ------------------------------------------------------------

# (a) Simulation function
simulate = build_func(
    :simulate;
    state_space,
    rates,
    tests,
)

# (b) Likelihood function
loglik = build_func(
    :loglik;
    state_space,
    rates,
    tests,
)

# ------------------------------------------------------------
# 3) Latent trajectory sampler builder (iFFBS + update hooks)
# ------------------------------------------------------------

# Per-individual updater: update infected count for the individual's group
# after re-sampling that individual's latent trajectory.
function update_infected_per_group!(cache, data, X, i, i_next)
    g = data.group[i]
    @inbounds for t in axes(X, 2)
        was_infected = cache.prev_status_by_individual[i, t] == :I
        now_infected = X[i, t] == :I
        cache.infected_per_group[g, t] += (now_infected - was_infected)
        cache.prev_status_by_individual[i, t] = X[i, t]
    end
    return nothing
end

# Per-total updater: refresh totals from per-group counts.
function update_total_infected!(cache, data, X)
    @inbounds for t in eachindex(cache.total_infected)
        cache.total_infected[t] = sum(view(cache.infected_per_group, :, t))
    end
    return nothing
end

# Post-init updater: initialize totals once cache arrays exist.
function post_init_total_infected!(cache, data, X)
    update_total_infected!(cache, data, X)
    return nothing
end

per_individual_update_funcs = [update_infected_per_group!]
per_total_update_funcs = [update_total_infected!]
post_initialization_funcs = [post_init_total_infected!]

latent_sampler = build_func(
    :latent_sampler;
    algorithm = :iFFBS,
    state_space,
    rates,
    tests,
    per_individual_update_funcs,
    per_total_update_funcs,
    post_initialization_funcs,
)

# ------------------------------------------------------------
# 4) End-to-end workflow shape
# ------------------------------------------------------------

# data = build_data(...)
# X = initialize_latent_trajectory(...)
# pars = (β_within = 0.3, β_env = 0.05, γ = 0.2)
#
# X_sim = simulate(rng, pars, data)
# ll = loglik(pars, X, data)
# X_new = latent_sampler(rng, pars, X, data)



