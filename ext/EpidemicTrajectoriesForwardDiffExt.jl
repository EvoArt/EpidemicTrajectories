# Plain values out of Duals, for the latent model's cache keys.
module EpidemicTrajectoriesForwardDiffExt

using EpidemicTrajectories
using ForwardDiff: ForwardDiff

EpidemicTrajectories._strip(x::ForwardDiff.Dual) = EpidemicTrajectories._strip(ForwardDiff.value(x))

end
