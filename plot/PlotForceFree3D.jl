using GLMakie
using LaTeXStrings

include("Plotting.jl")
include("PostProcess.jl")

# Plot loss function
plot_losses(losses)

# Integrate fieldlines
fieldlines=[]
α_lines=[]
footprints = find_footprints(α1, Br1, μ, ϕ, α_range = [0.0, params.model.alpha0], Br1_range = 0.0, μ_range = [0.7] )
sol = integrate_fieldlines!(fieldlines, α_lines, footprints, t1, NN, Θ_trained, st, params)
println("Number of fieldlines = ", length(fieldlines))

plot_magnetosphere_3d(fieldlines, α1[end, :, :], α_lines, params)

θ = acos.(μ)
plot_surface_α(θ[:,end:-1:1, :], ϕ, α1[end, :, :])

# plot_surface_α(μ, ϕ, αS_eq[end, :, :])
# plot_surface_α(μ, ϕ, B∇α[end, :, :])

# maximum(abs.(B∇α[:, :, :]))
# argmax(abs.(B∇α[:, :, :]))

# maximum(abs.(∇B))
# argmax(abs.(∇B))

# maximum(abs.(αS_eq[:, :, :]))
# argmax(abs.(αS_eq[:, :, :]))
# ;