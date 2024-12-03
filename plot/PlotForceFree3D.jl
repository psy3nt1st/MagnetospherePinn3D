using MagnetospherePinn3D
using JLD2
using GLMakie
using PrettyPrint
using Dates
using DelimitedFiles
using LaTeXStrings
using Parameters

include("functions.jl")

# Get path to data directory
dirs = filter(dir -> isdir(dir) && startswith(basename(dir), "ff3d") , readdir(abspath("../data"); join=true))
# dirs = filter(dir -> isdir(dir) && startswith(basename(dir), "local") , readdir(abspath("../data"); join=true))
# datadir = joinpath(dirs[end-2], "24")
# datadir = "../data/ff3d_3042596"
# datadir = "../data/local_2024_10_29_16_39_46"
# datadir = dirs[end]
datadir = "../data/potential_non_axisym"
@info "Using data in $datadir"

params = import_params(joinpath(datadir, "config.toml"))

# Create neural network
pinn, _, st = create_neural_network(params)

Θ_trained = load(joinpath(datadir,"trained_model.jld2"), "Θ_trained")
losses = load(joinpath(datadir, "losses_vs_iterations.jld2"), "losses")

# Create test grid
n_q = 160
n_μ = 80
n_ϕ = 160

q, μ, ϕ, Br1, Bθ1, Bϕ1, α1, ∇B, B∇α = create_test(n_q, n_μ, n_ϕ, pinn, Θ_trained, st, params, use_θ=true)[1:9]
Bmag1 = .√(Br1.^2 .+ Bθ1.^2 .+ Bϕ1.^2)


x = @. sqrt(1 - μ^2) / q * cos(ϕ)
y = @. sqrt(1 - μ^2) / q * sin(ϕ)
z = @. μ / q 

# Integrate fieldlines
ϕs = Float64[]
μs = Float64[]
α_thres = 0.0

for k in range(1, n_ϕ)
	for j in range(1, n_μ)
		if α1[end, j, k] >= α_thres && Br1[end, j, k] > 0
			push!(μs, μ[end, j, k])
			push!(ϕs, ϕ[end, j, k])
		end
	end
end


footprints = zip(μs, ϕs)
# footprints = [(μ, ϕ) for μ in range(0.35, 0.9, 10), ϕ in range(0, 2π, 5)]

fieldlines=[]
integrate_fieldlines!(fieldlines, footprints, pinn, Θ_trained, st, params)


# Calculate energy
energy = calculate_energy(pinn, Θ_trained, st, params)
println("Energy = ", energy)

# Calculate max Bϕ ratio
max_Bϕ_ratio = maximum(Bϕ1[Bmag1 .!= 0] ./ Bmag1[Bmag1 .!= 0])
println("Max Bϕ ratio = ", max_Bϕ_ratio)

# Plot surface α
f = Figure()
ax = Axis(f[1, 1], xlabel="ϕ", ylabel="μ", title="Surface α")
cont = contourf!(ϕ[end,1,:], μ[end,:,1], transpose(α1[end,:,:]), levels=100, colormap=reverse(cgrad(:gist_heat)))
cbar = Colorbar(f[1, 2], cont)
tightlimits!(ax)
display(GLMakie.Screen(), f)
save(joinpath("figures", "surface_alpha.png"), f)

# Plot surface B
# f = Figure()
# ax = Axis(f[1, 1], xlabel="ϕ", ylabel="μ", title="Surface B")
# cont = contourf!(ϕ[end,1,:], μ[end,:,1], transpose(Br1[end,:,:]), levels = 100, colormap=:blues)
# cbar = Colorbar(f[1, 2], cont)
# tightlimits!(ax)
# display(GLMakie.Screen(), f)

# Plot ϕ slice
# ϕ0 = n_ϕ ÷ n_ϕ
# f = Figure()
# ax = Axis(f[1, 1], xlabel="x", ylabel="z", ylabelrotation=0, limits=((0, 5), (-2.5, 2.5)))
# surf = surface!(x[:, :, ϕ0], z[:, :, ϕ0], α1[:, :, ϕ0], colormap=reverse(cgrad(:gist_heat)), shading=NoShading)
# cbar = Colorbar(f[1, 2], surf)
# display(GLMakie.Screen(), f)

# Plot 3D fieldlines
f = Figure()
lscene = LScene(f[1,1])
star = mesh!(lscene, Sphere(Point3(0, 0, 0), 1.0)
				 , color=Br1[end, :, :], colormap=reverse(cgrad(:gist_heat, 100)), interpolate=true, colorrange=(minimum(Br1), maximum(Br1)))
cbar = Colorbar(f[1, 2], star)
plot_fieldlines(fieldlines[1:43:end], lscene)
display(GLMakie.Screen(), f)
save(joinpath("figures", "fieldlines.png"), f, update=false)
;


# Br_surf = Br1[end, :, :]
# sph_modes = sph_transform(Br_surf)

# sph_evaluate([0 0; 1 1.])