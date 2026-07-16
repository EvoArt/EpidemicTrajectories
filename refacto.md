
# EpidemicTrajectories refactor: split build func + pluggable latent samplers (NEW PLAN, 2026-07-12)

## Context

EpidemicTrajectories.jl now works (two-state S/I cattle model recovers all 5 params
end-to-end via PracticalBayes, both functional and transition-matrix styles). But
the current `epidemic_model` bundles `loglik`, `simulate`, AND `latent!` into one
struct, with `latent!` hard-wired to iFFBS. Two problems this refactor fixes:

1. **The latent sampler should be pluggable, not baked in.** iFFBS is only ONE
   example latent-trajectory sampler; particle Gibbs, blocked Gibbs, and MH
   move-events must fit the same seam. So the main build func should return only
   `simulate` + `loglik` (+ the reusable rate funcs/transmat), and a SEPARATE
   builder should take that (or raw funcs/transmat) and produce a sampler.
2. **Performance: group statistics are recomputed on the fly every call.** The
   leave-one-out infected count (`_other_infected`, rates.jl:94; and the
   `_coupling_lik_vec` mutate/restore, iffbs.jl:129) rescans `data.states` on
   every `transition_matrix_at`. During inference this must instead READ a
   concrete-typed, incrementally-maintained cache — no per-step type inference
   (first step slow, fast after), mirroring the user's badger `iFFBS_Data`
   (a mutable struct of fully-concrete fields built once, with after-each-individual
   count roll-forward and after-sweep totals refresh).

The user wants: `IndividualBasedSampler` (after-each-individual hooks + after-sweep
hooks) and `PopulationBasedSampler` (after-sweep hooks only), the hooks maintaining
`infected_per_group`, `infected_female_per_group`, etc.

## Settled user decisions (2026-07-12)

- **Bookkeeping = GroupStat spec + auto-assembled concrete NamedTuple.** Each tracked
  statistic is a small declarative `GroupStat(name, init, full, ind_patch, tot_patch)`;
  the package assembles them into one concrete-typed `NamedTuple` of arrays on the
  first sweep, reused thereafter. Built-in `INFECTED_PER_GROUP` makes the common case
  zero user code; `stratified_infected(:infected_female_per_group, mask)` is a one-line
  factory. NOT a hand-written struct (too much boilerplate), NOT auto-type-discovery
  (brittle — mutating hooks return `nothing`).
- **Sampler family chosen by algorithm trait, with override.** `sampler_kind(::IFFBS)
  = :individual` picks the family by default; `kind=:population` overrides.
- **`coupling` is part of the iFFBS algorithm** — `IFFBS(coupling=true)`, off the model
  struct entirely (the coupling term is intrinsic to iFFBS; default `true` = exact).

## The three collaborating objects (in the model=params, X=latent, data=obs+bookkeeping vocabulary)

```
em      = epidemic_model(ss, rates; tests)              # simulate + loglik + reusable pieces (NO latent!)
sampler = latent_sampler(em, IFFBS(coupling=true); group=group)   # pluggable; algorithm picks family
sampler(rng, model_params, X, data) -> X               # callable in the latent_step shape; mutates X in place
```

## File-by-file plan

### `src/interface.jl` — the split
- `EpidemicModel{SS,RB,TT,LK,SM}` drops the `latent!` field and the `coupling` kwarg;
  keeps `state_space, rates, tests, loglik, simulate`. The reusable handles the
  sampler-builder needs (`rates`/`transition_matrix_at`, `state_space`, `tests`) are
  already fields.
- `EpidemicData` becomes a `mutable struct` with an added `bookkeeping::BK` slot
  (`nothing` until a sampler attaches its concrete object). `_pack(X, data, bk)` gains
  a `bookkeeping=bk` passthrough; `loglik`/`simulate` pass `bk=nothing` (they must
  recompute — loglik stays AD-clean and can't read a Float64 cache), the sampler passes
  its concrete `bk`.

### `src/bookkeeping.jl` (NEW) — the crux
- `struct GroupStat{Init,Full,IndPatch,TotPatch}` with fields `name::Symbol`, `init`
  `(data,X,ctx)->Array` (allocates the CONCRETE array, once), `full`
  `(arr,data,X,ctx)->nothing` (recompute at sweep top), `ind_patch`
  `(arr,data,X,i,i_next,ctx)->nothing` (after-each-individual roll-forward, may be
  `nothing`), `tot_patch` `(arr,data,X,ctx)->nothing` (after-whole-sweep, may be
  `nothing`).
- `build_bookkeeping(stats::Tuple, data, X, ctx)` -> `NamedTuple{names}(map(s->s.init(...),
  stats))` — fully concrete field types, fixed forever (reproduces `iFFBS_Data`'s
  "all concrete, built once" property without the hand-written struct).
- Built-ins: `INFECTED_PER_GROUP` (leave-one-out, badger `numInfecMat`) +
  `TOTAL_INFECTED_PER_GROUP` (including-everyone, refreshed by `tot_patch` at sweep
  bottom). Factory `stratified_infected(name, mask)` for sex-stratified variants.
- `ctx` = a small NamedTuple of shared handles (state_space, infected_code, sex vector,
  group->members map, n_groups), computed once in `latent_sampler`.

### `src/samplers/` (NEW) — pluggable latent sampler layer
- `abstract.jl`: `abstract type LatentSamplerAlgorithm` (the tag: IFFBS, particle Gibbs,
  ...) and `abstract type LatentSampler` (the built, callable object). Built sampler is
  callable `(s::LatentSampler)(rng, model_params, X, data) -> X`.
- `hooks.jl`: after-individual `hook!(bk, data, X, i, i_next, ctx)` (rolls `i` IN,
  `i_next` OUT — badger `id->idNext`; `i_next===nothing` on the last individual);
  after-sweep `hook!(bk, data, X, ctx)`. These ARE the `GroupStat.ind_patch`/`tot_patch`
  closures.
- `individual_based.jl`: `mutable struct IndividualBasedSampler{ALG,STATS,CTX,KW,BK}`
  (fields `algorithm, stats, ctx, kwargs, bookkeeping`). Its call: first sweep builds
  `bookkeeping`, then a FUNCTION BARRIER into `_run_individual_sweep!(alg, bk::ConcreteNT,
  stats, ctx, rng, pars, X, data)` so the hot loop is monomorphic despite the
  `Union{Nothing,T}` field. Inside: full recompute at sweep top, per-individual update
  (e.g. `ffbs_individual!`) + every `ind_patch` after each, every `tot_patch` after the
  sweep.
- `population_based.jl`: same struct shape, NO after-each-individual hooks — full
  recompute at top + `tot_patch` after the whole update (population moves at once).
- `iffbs.jl`: move `ffbs_individual!`/`ffbs_sweep!` here; `struct IFFBS <:
  LatentSamplerAlgorithm; coupling::Bool; end` (default `true`). `sampler_kind(::IFFBS)
  = :individual`. `_coupling_lik_vec`/`_other_infected` read
  `bookkeeping.infected_per_group[g,t]` (O(1)) instead of rescanning.
- `builder.jl`: `latent_sampler(em::EpidemicModel, algorithm; kind=sampler_kind(algorithm),
  stats=default_stats(algorithm), tests=em.tests, group, sex=nothing, kwargs...)` plus a
  raw-pieces overload `latent_sampler(state_space, rates, algorithm; ...)`. Builds `ctx`
  from `group`/`sex`/`state_space`. `default_stats(::IFFBS)` = `(INFECTED_PER_GROUP,
  TOTAL_INFECTED_PER_GROUP)`.

### `src/EpidemicTrajectories.jl`
Add includes (`bookkeeping.jl`, `samplers/*`). Export `latent_sampler, LatentSampler,
IndividualBasedSampler, PopulationBasedSampler, IFFBS, GroupStat, INFECTED_PER_GROUP,
TOTAL_INFECTED_PER_GROUP, stratified_infected`. Only user-facing removal:
`EpidemicModel` no longer has `.latent!`.

### How rate functions read the cache (rates.jl)
`transition_matrix_at(::TwoStateSI,...)` changes `I_minus` to:
```julia
I_minus = data.bookkeeping === nothing ?
    _other_infected(data, ss, i, t, 1) :                 # AD/loglik path: recompute (cache-free)
    data.bookkeeping.infected_per_group[data.group[i], t]  # sampler hot path: O(1) read
```
The `=== nothing` branch resolves at compile time (Nothing vs concrete NT). Keep
`_other_infected` as the ground-truth fallback AND as the reference the incremental
patch is validated against.

## Migration / compat
- `build_data`/`EpidemicData` survive (extended with `bookkeeping`); (model, X, data)
  vocabulary unchanged.
- Both tutorials (`docs/literate/cattle_iffbs.jl` in ET, `latent_trajectory.jl` in PB)
  and `examples/cattle_ecoli_iffbs.jl`: the `iFFBSKernel.latent_step` body changes from
  `em.latent!(rng, pars, X, data; ...)` to building `sampler = latent_sampler(em,
  IFFBS(coupling=true); group=group)` ONCE (outside the model, so its cached bookkeeping
  persists across sweeps) and calling `sampler(rng, pars, X, data)` in `latent_step`.
- `test/interface.jl`'s `em.latent!` assertions move to a new `test/samplers.jl`
  targeting `latent_sampler`; `em.loglik`/`em.simulate` tests unchanged. Keep
  `ffbs_sweep!` exported for low-level use.

## Verification
1. **End-to-end recovery** unchanged: `examples/cattle_ecoli_iffbs.jl` recovers all 5
   params within the existing gate, for BOTH `TwoStateSI` and `@transitions :individual`.
2. **Incremental == recompute (fixed RNG)** — NEW gate: one `IFFBS` sweep with the
   incremental bookkeeping vs a reference sweep forcing `data.bookkeeping=nothing` (pure
   `_other_infected`), same `StableRNG` seed → assert IDENTICAL resampled `X`. This
   proves the badger-style roll-forward equals full recompute.
3. **Brute-force conditional** (`test/functional_style.jl` 2×3 enumerate) still passes
   through the new sampler path.
4. **Type-stability**: `@inferred`/`@code_warntype` on `_run_individual_sweep!(alg,
   bk::ConcreteNT, ...)` and on `transition_matrix_at` with concrete bookkeeping — no
   `Union`/`Any` in the hot loop.
5. **First-slow-then-fast**: sweep 2 must not reallocate the bookkeeping (allocation
   check).

## Critical files
- Modify: `src/interface.jl`, `src/rates.jl`, `src/EpidemicTrajectories.jl`,
  `docs/literate/cattle_iffbs.jl`, `examples/cattle_ecoli_iffbs.jl`, `test/interface.jl`
  (+ PB `docs/literate/latent_trajectory.jl`).
- New: `src/bookkeeping.jl`, `src/samplers/{abstract,hooks,individual_based,
  population_based,iffbs,builder}.jl`, `test/samplers.jl`.
- Reference (read-only): badger `JBIIDRjl/Julia/src/{iFFBS_modular.jl (updateGroupStatistics),
  MCMCiFFBS_.jl (iFFBS_Data struct + construction + sweep loop), updaters.jl}`.
