using MagnetospherePinn3D
using JLD2
using GLMakie
using PrettyPrint
using Dates
using DelimitedFiles
using LaTeXStrings
using Parameters

include("Functions.jl")
# include("PlottingFunctions.jl")

# Get path to data directory
dirs = filter(dir -> isdir(dir) && startswith(basename(dir), "ff3d") , readdir(abspath("../data"); join=true))
# dirs = filter(dir -> isdir(dir) && startswith(basename(dir), "local") , readdir(abspath("../data"); join=true))
# datadir = joinpath(dirs[end-2], "24")
# datadir = "../data/ff3d_3042596"
# datadir = "../data/local_2024_10_29_16_39_46"
datadir = dirs[end]
# datadir = "../data/current_free_non_axisym/"
@info "Using data in $datadir"

params = import_params(joinpath(datadir, "config.toml"))

# Create neural network
NN, _, st = create_neural_network(params)

Θ_trained = load(joinpath(datadir, "trained_model.jld2"), "Θ_trained")
losses = load(joinpath(datadir, "losses_vs_iterations.jld2"), "losses")

# Create test grid
n_q = 80
n_μ = 40
n_ϕ = 80
t1 = 1

# q, μ, ϕ, Br1, Bθ1, Bϕ1, α1, ∇B, B∇α = create_test(n_q, n_μ, n_ϕ, NN, Θ_trained, st, params, use_θ=true)[1:9]
q, μ, ϕ, t, Br1, Bθ1, Bϕ1, α1, ∇B, B∇α, Nr, Nθ, Nϕ, Nα, Nα_S = create_test(n_q, n_μ, n_ϕ, t1, NN, Θ_trained, st, params, use_θ=true)
# q, θ, ϕ, Br1, Bθ1, Bϕ1, α1, ∇B = read_gradrubin_data()[1:8]
Bmag1 = .√(Br1.^2 .+ Bθ1.^2 .+ Bϕ1.^2)

size(t)


x = @. sqrt(1 - μ^2) / q * cos(ϕ)
y = @. sqrt(1 - μ^2) / q * sin(ϕ)
z = @. μ / q 

# α_surface1 = α_surface(μ, ϕ, t1, Nα_S, params)


function calclulate_dα_dt(q, μ, ϕ, NN, Θ, st, ϵ; use_θ = false)
	
	n_μ = size(q)[2]
	n_ϕ = size(q)[3]

	q = reshape(q[end, :, :], 1, :)
	μ = reshape(μ[end, :, :], 1, :)
	ϕ = reshape(ϕ[end, :, :], 1, :)

	if use_θ
		θ = reshape([θ for ϕ in range(0, 2π, n_ϕ) for θ in range(1e-2, π - 1e-2, n_μ)], 1, :)
		μ = cos.(θ)
	end
	
	NN = pinn(vcat(q, μ, cos.(ϕ), sin.(ϕ)), Θ, st)[1]
	Nr = reshape(NN[1, :], size(q))
	Nθ = reshape(NN[2, :], size(q))
	Nϕ = reshape(NN[3, :], size(q))
	Nα = reshape(NN[4, :], size(q))

	Br1 = Br(q, μ, ϕ, Θ_trained, st, Nr, params)
	Bθ1 = Bθ(q, μ, ϕ, Θ_trained, st, Nθ)
	Bϕ1 = Bϕ(q, μ, ϕ, Θ_trained, st, Nϕ)
	α_surface = α(q, μ, ϕ, Θ_trained, st, Nα, params)
	dBr_dq, dBθ_dq, dBϕ_dq, dα_dq, dBr_dμ, dBθ_dμ, dBϕ_dμ, dα_dμ, dBr_dϕ, dBθ_dϕ, dBϕ_dϕ, dα_dϕ, d2α_dq2, d2α_dμ2, d2α_dϕ2 = calculate_derivatives(q, μ, ϕ, Θ, st, pinn, ϵ)

	∇α = grad(q, μ, dα_dq, dα_dμ, dα_dϕ)
	∇B2 = calculate_gradB2(q, μ, Br1, Bθ1, Bϕ1, dBr_dq, dBθ_dq, dBϕ_dq, dBr_dμ, dBθ_dμ, dBϕ_dμ, dBr_dϕ, dBθ_dϕ, dBϕ_dϕ)
	B2 = @. Br1^2 + Bθ1^2 + Bϕ1^2
	advective_term = scalar_product(∇α[1], ∇α[2], ∇α[3], ∇B2[1], ∇B2[2], ∇B2[3]) ./ B2
	diffusive_term = laplacian(q, μ, dα_dμ, d2α_dq2, d2α_dμ2, d2α_dϕ2)
	dα_dt = diffusive_term .+ advective_term
	# dα_dt[α1 .< 1e-4] .= 0.0

	return dα_dt, diffusive_term, advective_term, α_surface
end

function advance_α_timestep(q, μ, ϕ, pinn, Θ_trained, st, params; use_θ = false, dt = 1e-3)
	
	dα_dt, diffusive_term, advective_term, α_surface = calclulate_dα_dt(q, μ, ϕ, pinn, Θ_trained, st, params, use_θ = use_θ)

	return α_surface .+ dt .* dα_dt	
end

# α_surface_next = advance_α_timestep(q, μ, ϕ, pinn, Θ_trained, st, params, use_θ = true, dt = 1e-3)
# α_surface_next = reshape(α_surface_next, n_μ, n_ϕ)

# length(α_surface_next)

# dα_dt, diffusive_term, advective_term, α_surface = calclulate_dα_dt(q, μ, ϕ, pinn, Θ_trained, st, params, use_θ = true)
# dα_dt = reshape(dα_dt, n_μ, n_ϕ)
# diffusive_term = reshape(diffusive_term, n_μ, n_ϕ)
# advective_term = reshape(advective_term, n_μ, n_ϕ)
# α_surface = reshape(α_surface, n_μ, n_ϕ)
# println(size(da_dt))

# Integrate fieldlines
fieldlines=[]
footprints = find_footprints(α1, Br1, μ, ϕ, α_thres = 0.0, Br1_thres = 0.0, μ_thres = 0.7)
sol = integrate_fieldlines!(fieldlines, footprints, NN, Θ_trained, st, params)
println("Number of fieldlines = ", length(fieldlines))

# Calculate energy
energy = calculate_energy(NN, Θ_trained, st, params)
println("Energy = ", energy)

# Calculate max Bϕ ratio
max_Bϕ_ratio = maximum(Bϕ1[Bmag1 .!= 0] ./ Bmag1[Bmag1 .!= 0])
println("Max Bϕ ratio = ", max_Bϕ_ratio)

