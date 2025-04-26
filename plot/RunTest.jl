using MagnetospherePinn3D
using JLD2
using Dates
using Parameters
using DifferentialEquations
using Integrals
using Distributions
using NaturalSort

include("PostProcess.jl")

# Get path to data directory
dirs = filter(dir -> isdir(dir) && startswith(basename(dir), "ff3d") , readdir(abspath("../data"); join=true))
# dirs = filter(dir -> isdir(dir) && startswith(basename(dir), "local") , readdir(abspath("../data"); join=true))

rundirs = sort(filter(dir -> isdir(dir), readdir(abspath(dirs[end]); join=true, sort=false)), lt=natural)

# datadir = joinpath("../data", "hotspot_a0_1.5_l2_n30")
# datadir = joinpath("../data", "ff3d_3145941")
# datadir = dirs[end]
datadir = rundirs[1]
# datadir = dirs[end]

NN, Θ_trained, st, losses, q, μ, ϕ, t, θ, Br1, Bθ1, Bϕ1, α1, ∇B, B∇α, Nr, Nθ, Nϕ, Nα, Bmag1, params = run_test(datadir)

# println("t = ", params.model.t)

# Calculate energy
energy = calculate_energy(t[1], NN, Θ_trained, st, params)
println("Energy = ", energy)

dipole_energy = 1/3 * (1 + 5 / 2 * params.model.M)
excess_energy = (energy - dipole_energy) / dipole_energy
println("Excess energy = ", excess_energy) 

# Calculate max Bϕ ratio
max_Bϕ_ratio = maximum(abs.(Bϕ1[Bmag1 .!= 0]) ./ Bmag1[Bmag1 .!= 0])
println("Max Bϕ ratio = ", max_Bϕ_ratio)

magnetic_virial = calculate_magnetic_virial(t[1], NN, Θ_trained, st, params)
println("Magnetic virial = ", magnetic_virial)

# println("σ = $(params.model.sigma_gs), gamma = $(params.model.gamma), z = $(params.model.z)")


# using MagnetospherePinn3D: α

# function MagnetospherePinn3D.α(
#     q::AbstractArray,
#     μ::AbstractArray,
#     ϕ::AbstractArray,
#     t::AbstractArray,
#     NN::Any,
#     Θ::AbstractArray,
#     st::NamedTuple,
#     params::Params
#     )

#     subnet_α = NN.layers[4]
#     Nα = subnet_α(vcat(q, μ, cos.(ϕ), sin.(ϕ), t), Θ.layer_4, st.layer_4)[1]

#     return @. q * ($α_surface(μ, ϕ, params) + $h_boundary(q, μ, ϕ, t, params) * Nα)
# end
# α

# # @btime α(q, μ, ϕ, t, Nα, params);

# q1 = reshape(q, 1, :)
# μ1 = reshape(μ, 1, :)
# ϕ1 = reshape(ϕ, 1, :)
# t1 = zero(q1)
# # @btime α(q1, μ1, ϕ1, t1, NN, Θ_trained, st, params);

# function calculate_∂α∂q(q, μ, ϕ, t, NN, Θ_trained, st, params)

#     ϵ = ∛(eps()) 
    
#     return  (α(q .+ ϵ, μ, ϕ, t, NN, Θ_trained, st, params) .- α(q .- ϵ, μ, ϕ, t, NN, Θ_trained, st, params)) / (2 * ϵ)

# end

# ∂α∂q = calculate_∂α∂q(q1, μ1, ϕ1, t1, NN, Θ_trained, st, params)
