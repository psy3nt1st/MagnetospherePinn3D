using GLMakie
using LaTeXStrings

include("Plotting.jl")
include("PostProcess.jl")

# Plot loss function
plot_losses(losses)

t1 = 0.0

# Integrate fieldlines
fieldlines=[]
α_lines=[]
footprints = find_footprints(α1, Br1, μ, ϕ, α_range = [0.0, maximum(α1)], Br1_range = 0.0, μ_range = [0.1] )
sol = integrate_fieldlines!(fieldlines, α_lines, footprints, t1, NN, Θ_trained, st, params)
println("Number of fieldlines = ", length(fieldlines))

f = plot_magnetosphere_3d(fieldlines, α1[end, :, :], α_lines, params)

x = @. √(1 - μ^2) / q * cos.(ϕ)
z = @. μ / q
f = Figure()
ax = Axis(f[1, 1], limits=(0, 10, -5, 5))
cntr = surface!(ax, x[:, :, 1], z[:, :, 1], zero(α1[:, :, 1]), color=α1[:, :, 1], colormap=:viridis, shading=false)
display(GLMakie.Screen(), f)

# plot_at_surface(θ[:,end:-1:1, :], ϕ, Bϕ1[end, :, :]; title="Bϕ")



# f1 = Figure()
# ax1 = Makie.Axis(f1[1, 1], xlabel="Iteration", ylabel="B")


# lines!(θ[end,:, 1], Bϕ1[end,:, 1], label="Bϕ")
# lines!(θ[end,:, 1], Bθ1[end,:, 1], label="Bθ")


# axislegend(ax1)
# display(GLMakie.Screen(), f1)
# f2 = plot_surface_α(θ[:,end:-1:1, :], ϕ, α1[end, :, :], params)

;
# save(joinpath("figures", "twisted_magnetosphere_t=$(params.model.t).png"), f, update=false)
# save(joinpath("figures", "surface_alpha_t=$(params.model.t).png"), f2, update=false)



# plot_at_surface(θ[:,end:-1:1, :], ϕ, r_eq[end, :, :] .* sin.(θ[end,:,:]); title="r equation")
# plot_at_surface(θ[:,end:-1:1, :], ϕ, θ_eq[end, :, :]; title="θ equation")
# plot_at_surface(θ[:,end:-1:1, :], ϕ, ϕ_eq[end, :, :]; title="ϕ equation")
# plot_at_surface(θ[:,end:-1:1, :], ϕ, ∇B[end, :, :]; title="∇B")
# plot_at_surface(θ[:,end:-1:1, :], ϕ, B∇α[end, :, :]; title="B∇α")

M = 0.25
z = 2 * M * q
Br_newt = @. 2 * cos(θ) * q^3
Br_rel = @. -2 * cos(θ) * (log(1 - 2 * M * q) + 2 * M * q + 2 * M^2 * q^2) * (3 / (8 * M^3))
Bθ_newt = @. sin(θ) * q^3
Bθ_rel = @. -3 * sin(θ) * √(1 - 2 * M * q) * (2 * M * (1/q - M) + 1 / q * (1 / q - 2 * M) * log(1 - 2 * M * q)) / (4 * M^3 * 1 / q * (2 * M - 1 / q))

Br_rel[end, 1, 1] / Br_newt[end, 1, 1]
Bθ_rel[end, 20, 1] / Bθ_newt[end, 20, 1]

Br_newt[end, 1, 1] / Bθ_newt[end, 20, 1]
Br_rel[end, 1, 1] / Bθ_rel[end, 20, 1]

μ_idx = 1
fig = Figure()
ax = Axis(fig[1, 1], xlabel="q", ylabel=L"B_r")
lines!(ax, q[:, μ_idx, 1], Br_rel[:, μ_idx, 1], label="relativistic ")
lines!(ax, q[:, μ_idx, 1], Br1[:, μ_idx, 1], label="PINN")
lines!(ax, q[:, μ_idx, 1], Br_newt[:, μ_idx, 1], label="newtonian")
axislegend(ax, position=:lt)
display(GLMakie.Screen(), fig)

lines(q[:,1,1], Br_rel[:, 1, 1] ./ Br1[:, 1, 1])





;