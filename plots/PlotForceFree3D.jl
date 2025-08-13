using GLMakie
using LaTeXStrings
using JLD2
using NaturalSort
using MagnetospherePinn3D
using OrdinaryDiffEq
using Integrals
using DrWatson
using DataFrames
using PrettyPrint
using OrderedCollections

include("Plotting.jl")
include("PostProcess.jl")

data_dir = datadir("theta1_sequence/")
rundirs = sort(filter(dir -> isdir(dir), readdir(abspath(data_dir); join=true, sort=false)), lt=natural)
rundir = rundirs[24]

config = NamedTuple(load(joinpath(rundir, "config.jld2"), "data"))
NN, _, st = create_neural_network(config, test_mode=true)
Θ = load(joinpath(rundir, "trained_model.jld2"), "Θ_trained")
griddata = load(joinpath(rundir, "griddata.jld2"), "data")
fieldlines = load(joinpath(rundir, "fieldlines.jld2"), "data")

@unpack μ, ϕ, α1 = griddata
target_mu = findnearest(μ, cosd(config.θ1))
chosen_lines = filter(line -> line.μ[1] in [target_mu], fieldlines)
f = plot_magnetosphere_3d(chosen_lines, α1[end, :, :]; use_lscene=true)
# save(joinpath("figures", "twisted_magnetosphere_M=$(params.model.M)_a0=$(params.model.alpha0)_island.png"), f, update=false, size=(750, 600))


# wsave(joinpath(rundir, "griddata.jld2"), "data", griddata, )


# Integrate fieldlines
# footprints = find_footprints(μ, ϕ, α1, μ_interval=0..1, ϕ_interval=0..2π)
# println("Number of fieldlines = ", length(footprints))
# fieldlines = integrate_fieldlines(footprints, NN, Θ, st, config; q_start = 1);
# save(joinpath(rundir, "fieldlines.jld2"), "data", fieldlines)




# fieldline_length = [line.t[end] for line in fieldlines]
# fieldline_α = abs.(mean.(α_lines))
# safety_factor = 4π ./ (fieldline_length .* fieldline_α)

# # fieldline_length_b = [line.t[end] for line in fieldlines_b]
# # fieldline_α_b = abs.(mean.(α_lines_b))
# # safety_factor_b = 4π ./ (fieldline_length_b .* fieldline_α_b)


# αmax = @sprintf "%.2f" params.model.alpha0 
# αmax_b = @sprintf "%.2f" params_b.model.alpha0

# f = Figure()
# ax = Axis(f[1, 1], 
#     yscale=log10, 
#     xlabel=L"\alpha [R^{-1}]", ylabel="Safety factor", 
#     xticklabelsize=20, yticklabelsize=20, xlabelsize=25, ylabelsize=25,
#     )
# sc = scatter!(ax, fieldline_α[fieldline_α .>= 0.5], safety_factor[fieldline_α .>= 0.5], color = :teal, markersize=15, label="Low energy", )
# sc_b = scatter!(ax, fieldline_α_b[fieldline_α_b .>= 0.5], safety_factor_b[fieldline_α_b .>= 0.5], color = "#CC5500", markersize=15, marker = :utriangle, label="High energy")
# hlines!(ax, [1], color=:black, linestyle=:dash, linewidth=4)
# axislegend(ax, position=:rt, labelsize=label_size)
# # Colorbar(f[1, 2], sc, label="Mean α", labelrotation=3π/2, width=20)
# display(GLMakie.Screen(), f)
# # save(joinpath("figures", "safety_factor_vs_mean_alpha_M=$(params.model.M)_axisymmetric.png"), f, size=(800, 600))

# scatter(fieldline_length, fieldline_α, color=safety_factor, colormap=:viridis, markersize=5, xlabel="Fieldline length", ylabel="Mean α", title="Safety factor")

# Br2 = copy(Br1)
# Br2[Br2 .<= 0.0] .= NaN
# plot_at_surface(μ, ϕ, Br2[end, :, :])
# plot_at_surface(μ, ϕ, α1[end, :, :])
;

