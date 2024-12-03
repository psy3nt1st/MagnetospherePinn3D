using MagnetospherePinn3D
using JLD2
using GLMakie
using PrettyPrint
using Dates
using DelimitedFiles
using LaTeXStrings
using Parameters


include("functions.jl")

dirs = filter(dir -> isdir(dir) && startswith(basename(dir), "ff3d") , readdir(abspath("../data"); join=true))
# datadir = joinpath(dirs[end-2], "24")
# datadir = "../data/ff3d_2709874/3"
# datadir = "data/local_2024_07_25_13_37_33"
datadir = dirs[end]
@info "Using data in $datadir"

params = import_params(joinpath(datadir, "config.toml"))

# Create neural network and load trained parameters
pinn, _, st = create_neural_network(params)
Θ_trained = load(joinpath(datadir,"trained_model.jld2"), "Θ_trained")

# Set test resolution
# n=200
n_q = 80
n_μ = 40
n_ϕ = 80

# Create test grid
q, μ, ϕ, Br1, Bθ1, Bϕ1, α1  = create_test(n_q, n_μ, n_ϕ, pinn, Θ_trained, st, params, use_θ=false)[1:7]
Bmag = .√(Br1.^2 .+ Bθ1.^2 .+ Bϕ1.^2)
# Calculate PDE absolute error
r_eq, θ_eq, ϕ_eq, ∇B, B∇α = calculate_PDE_absolute_error(n_q, n_μ, n_ϕ, pinn, Θ_trained, st, params)  

# Calculate and plot L2 error as a function of each coordinate
l2errors_vs_q = calculate_l2error_vs_q(n_q, n_μ, n_ϕ, pinn, Θ_trained, st, params)
plot_l2error_vs_q(q, l2errors_vs_q)

l2errors_vs_μ = calculate_l2error_vs_μ(n_q, n_μ, n_ϕ, pinn, Θ_trained, st, params)
plot_l2_errors_vs_μ(μ, l2errors_vs_μ)

l2errors_vs_ϕ = calculate_l2errors_vs_ϕ(n_q, n_μ, n_ϕ, pinn, Θ_trained, st, params)
plot_l2_errors_vs_ϕ(ϕ, l2errors_vs_ϕ)


# Plot PDE error on a q-slice
q0 = 1
idx_q0 = findmin(abs.(q .- q0))[2][1]
# idx_q0 = n_q ÷ 4
f = Figure()
ax = Axis(f[1, 1], xlabel="ϕ", ylabel="μ", title="PDE error")
cont = contourf!(ϕ[idx_q0,1,:], μ[idx_q0,2:n_μ-1,1], transpose(log10.(r_eq[idx_q0,2:n_μ-1,:] ./ Bmag[idx_q0,2:n_μ-1,:])),
                 levels=range(0.4, 1, length=100), colormap=:cividis, extendlow=:auto, mode=:relative)
cbar = Colorbar(f[1, 2], cont, scale=log10)
tightlimits!(ax)
display(GLMakie.Screen(), f)

# Plot PDE error on a μ-slice
μ0 = cos(π / 2)
idx_μ0 = findmin(abs.(μ .- μ0))[2][2]
# idx_μ0 = n_μ ÷ 4
f = Figure()
ax = Axis(f[1, 1], xlabel="q", ylabel="ϕ", title="PDE error")
cont = contourf!(q[2:n_q-1,idx_μ0,1], ϕ[1,idx_μ0,:], log10.(r_eq[2:n_q-1,idx_μ0,:]./ Bmag[2:n_q-1,idx_μ0,:]),
                 levels=range(0.4, 1, length=100), colormap=:cividis, extendlow=:auto, mode=:relative)
cbar = Colorbar(f[1, 2], cont, scale=log10)
tightlimits!(ax)
display(GLMakie.Screen(), f)

# Plot PDE error on a ϕ-slice
ϕ0 = π / 2
idx_ϕ0 = findmin(abs.(ϕ .- ϕ0))[2][3]
# idx_ϕ0 = n_ϕ ÷ 4
f = Figure()
ax = Axis(f[1, 1], xlabel="μ", ylabel="q", title="PDE error")
cont = contourf!(μ[1,2:n_μ-1,idx_ϕ0], q[:2:n_q-1,1,idx_ϕ0], transpose(log10.(r_eq[:2:n_q-1,2:n_μ-1,idx_ϕ0]./ Bmag[:2:n_q-1,2:n_μ-1,idx_ϕ0])),
                 levels=range(0.4, 1, length=100), colormap=:cividis, extendlow=:auto, mode=:relative)
cbar = Colorbar(f[1, 2], cont, scale=log10)
tightlimits!(ax)
display(GLMakie.Screen(), f)


# Calculate and plot L2 error vs resolution
resolutions = [20, 40, 60]
l2errors = calculate_l2error(resolutions, pinn, Θ_trained, st, params, use_finite_differences=true)
plot_l2error(resolutions, l2errors)


