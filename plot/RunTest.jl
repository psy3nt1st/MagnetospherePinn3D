using MagnetospherePinn3D
using JLD2
using PrettyPrint
using Dates
using Parameters
using DifferentialEquations
using Integrals
using Distributions
include("PostProcess.jl")

# Get path to data directory
dirs = filter(dir -> isdir(dir) && startswith(basename(dir), "ff3d") , readdir(abspath("../data"); join=true))
# dirs = filter(dir -> isdir(dir) && startswith(basename(dir), "local") , readdir(abspath("../data"); join=true))

# datadir = joinpath("../data", "hotspot_a0_1.5_l2_n30")
# datadir = joinpath("../data", "ff3d_3145941")
# datadir = joinpath("../data", "ff3d_3155186", "48")
datadir = dirs[end]

@info "Using data in $datadir"

params = import_params(joinpath(datadir, "config.toml"))

# Create neural network
NN, _, st = create_neural_network(params, test_mode=true)

Θ_trained = load(joinpath(datadir, "trained_model.jld2"), "Θ_trained")
losses = load(joinpath(datadir, "losses_vs_iterations.jld2"), "losses")

# Create test grid
n_q = 80
n_μ = 40
n_ϕ = 80
t1 = 0.1

test_input = create_test_input(n_q, n_μ, n_ϕ, t1, params; use_θ = true)

q, μ, ϕ, t, 
Br1, Bθ1, Bϕ1, α1, 
∇B, B∇α, αS,
Nr, Nθ, Nϕ, Nα = create_test(test_input, NN, Θ_trained, st, params)
Bmag1 = .√(Br1.^2 .+ Bθ1.^2 .+ Bϕ1.^2)

const ϵ = ∛(eps())

q2 = reshape(q, 1, length(q))
μ2 = reshape(μ, 1, length(μ))
ϕ2 = reshape(ϕ, 1, length(ϕ))
t2 = reshape(t, 1, length(t))
qS2 = ones(size(q2))

@info "Test created"

# Calculate energy
energy = calculate_energy(t1, NN, Θ_trained, st, params)
println("Energy = ", energy)

# Calculate max Bϕ ratio
max_Bϕ_ratio = maximum(Bϕ1[Bmag1 .!= 0] ./ Bmag1[Bmag1 .!= 0])
println("Max Bϕ ratio = ", max_Bϕ_ratio)

