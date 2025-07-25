using GLMakie
using LaTeXStrings
using ColorSchemes
using JLD2
using NaturalSort
using MagnetospherePinn3D
using DifferentialEquations
using Integrals

include("Plotting.jl")
include("PostProcess.jl")

# Get path to data directory
dirs = filter(dir -> isdir(dir) && startswith(basename(dir), "ff3d") , readdir(abspath("../data"); join=true))
# datadir = dirs[end-1]
datadir = "../data/alphamax_sequence_3d/"
rundirs = sort(filter(dir -> isdir(dir), readdir(abspath(datadir); join=true, sort=false)), lt=natural)
rundir = rundirs[56]
rundir2 = rundirs[56]
# rundir = dirs[end-3]


NN, Θ_trained, st, losses, q, μ, ϕ, t, θ, Br1, Bθ1, Bϕ1, α1, ∇B, B∇α, Nr, Nθ, Nϕ, Nα, Bmag1, params = load_test(rundir)
NN_b, Θ_trained_b, st_b, losses_b, q_b, μ_b, ϕ_b, t_b, θ_b, Br1_b, Bθ1_b, Bϕ1_b, α1_b, ∇B_b, B∇α_b, Nr_b, Nθ_b, Nϕ_b, Nα_b, Bmag1_b, params_b = load_test(rundir2)
t1 = 0.0

r = 1 ./ q[:, 1, 1]

r_idx = argmin(abs.(r .- 1.5))

# plot_losses(losses)

fig = Figure()
ax = Axis(fig[1, 1], yscale=log10,
    xlabel="Iterations", ylabel="Loss", 
    xticklabelsize=20, yticklabelsize=20, xlabelsize=25, ylabelsize=25)
linestyles = [:solid, :solid, :dash]
for (i,d) in enumerate([39, 45, 49])
    NN, Θ_trained, st, losses, q, μ, ϕ, t, θ, Br1, Bθ1, Bϕ1, α1, ∇B, B∇α, Nr, Nθ, Nϕ, Nα, Bmag1, params = load_test(rundirs[d])
    label = @sprintf "%.2f" params.model.alpha0
    lines!(ax, losses[1][1:8000], linewidth=4, label=L"α_*^{\text{max}} = %$(label) R^{-1}", linestyle=linestyles[i])
end
axislegend(ax, position=:rt, labelsize=25)
display(GLMakie.Screen(), fig)
save(joinpath("figures", "losses_vs_iterations_M=$(params.model.M).png"), fig, size=(800, 600))


# Integrate fieldlines
fieldlines=[]
# fieldlines_b=[]
α_lines=[]
# α_lines_b=[]
# footprints = find_footprints(α1, Br1, μ, ϕ, α_range = [0.5, maximum(α1)], Br1_range = 0.0, μ_range = [0.3, 0.9], ϕ_range = 3π/4:0.04:5π/4, r_idx = r_idx)
# footprints_b = find_footprints(α1_b, Br1_b, μ_b, ϕ_b, α_range = [0.5, maximum(α1_b)], Br1_range = 0.0, μ_range = [0.3, 0.9], ϕ_range = 3π/4:0.04:5π/4, r_idx = r_idx)
footprints = find_footprints(α1, Br1, μ, ϕ, α_range = [0.3, maximum(α1)], Br1_range = 0.0, μ_range = [0.1], ϕ_range = π)
# footprints_b = find_footprints(α1_b, Br1_b, μ_b, ϕ_b, α_range = [0.5, maximum(α1_b)], Br1_range = 0.0, μ_range = [0.1], ϕ_range = π)
# footprints2 = find_footprints(α1, Br1, μ, ϕ, α_range = [0.0, maximum(α1)], Br1_range = 0.0, μ_range = [0.96])
# footprints = vcat(footprints..., footprints2...)
println("Number of fieldlines = ", length(footprints))
sol = integrate_fieldlines!(fieldlines, α_lines, footprints, t1, NN, Θ_trained, st, params, q_start = 0.67)
# sol_b = integrate_fieldlines!(fieldlines_b, α_lines_b, footprints_b, t1, NN_b, Θ_trained_b, st_b, params, q_start = 0.67)


include("Plotting.jl")
f = plot_magnetosphere_3d(fieldlines, α1[end, :, :], α_lines, params)
# f = plot_magnetosphere_3d(fieldlines_b, α1_b[end, :, :], α_lines_b, params_b)
save(joinpath("figures", "twisted_magnetosphere_M=$(params.model.M)_a0=$(params.model.alpha0)_island.png"), f, update=false, size=(750, 600))

fieldline_length = [line.t[end] for line in fieldlines]
fieldline_α = abs.(mean.(α_lines))
safety_factor = 4π ./ (fieldline_length .* fieldline_α)

# fieldline_length_b = [line.t[end] for line in fieldlines_b]
# fieldline_α_b = abs.(mean.(α_lines_b))
# safety_factor_b = 4π ./ (fieldline_length_b .* fieldline_α_b)


αmax = @sprintf "%.2f" params.model.alpha0 
αmax_b = @sprintf "%.2f" params_b.model.alpha0

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

