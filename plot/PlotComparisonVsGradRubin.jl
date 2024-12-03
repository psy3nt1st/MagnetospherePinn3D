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
# datadir = joinpath(dirs[end], "24")
# datadir = "../data/ff3d_2709874/3"
# datadir = "data/local_2024_07_25_13_37_33"
datadir = dirs[end]
@info "Using data in $datadir"

params = import_params(joinpath(datadir, "config.toml"))

# Create neural network and load trained parameters
pinn, _, st = create_neural_network(params)
Θ_trained = load(joinpath(datadir,"trained_model.jld2"), "Θ_trained")


# Load Grad-Rubin data
q_gr, θ_gr, ϕ_gr, Br_gr, Bθ_gr, Bϕ_gr, α_gr, ∇B_gr = read_gradrubin_data()
μ_gr = cos.(θ_gr)
Bmag_gr = sqrt.(Br_gr.^2 .+ Bθ_gr.^2 .+ Bϕ_gr.^2)

# Evaluate PINN on Grad-Rubin grid
Br1, Bθ1, Bϕ1, α1, ∇B1  = evaluate_on_gradrubin_grid(q_gr, μ_gr, ϕ_gr, pinn, Θ_trained, st, params, use_θ=false)
Bmag = .√(Br1.^2 .+ Bθ1.^2 .+ Bϕ1.^2)

n_q, n_μ, n_ϕ = size(q_gr)
interior = CartesianIndices((2:size(q_gr)[1]-1, 2:size(q_gr)[2]-1, 2:size(q_gr)[3]-1))






∂Br_∂q = ∂_∂q(Br1, q_gr)
∂Br_∂q_gr = ∂_∂q(Br_gr, q_gr)
println(L2_error(∂Br_∂q[interior], ∂Br_∂q_gr[interior], relative=true))

∂Br_∂θ = ∂_∂θ(Br1, θ_gr)
∂Br_∂θ_gr = ∂_∂θ(Br_gr, θ_gr)
println(L2_error(∂Br_∂θ[interior], ∂Br_∂θ_gr[interior], relative=true))

∂Br_∂ϕ = ∂_∂ϕ(Br1, ϕ_gr)
∂Br_∂ϕ_gr = ∂_∂ϕ(Br_gr, ϕ_gr)
println(L2_error(∂Br_∂ϕ[interior], ∂Br_∂ϕ_gr[interior], relative=true))

∂Bθ_∂q = ∂_∂q(Bθ1, q_gr)
∂Bθ_∂q_gr = ∂_∂q(Bθ_gr, q_gr)
println(L2_error(∂Bθ_∂q[interior], ∂Bθ_∂q_gr[interior], relative=true))

∂Bθ_∂θ = ∂_∂θ(Bθ1, θ_gr)
∂Bθ_∂θ_gr = ∂_∂θ(Bθ_gr, θ_gr)
println(L2_error(∂Bθ_∂θ[interior], ∂Bθ_∂θ_gr[interior], relative=true))

∂Bθ_∂ϕ = ∂_∂ϕ(Bθ1, ϕ_gr)
∂Bθ_∂ϕ_gr = ∂_∂ϕ(Bθ_gr, ϕ_gr)
println(L2_error(∂Bθ_∂ϕ[interior], ∂Bθ_∂ϕ_gr[interior], relative=true))

∂Bϕ_∂q = ∂_∂q(Bϕ1, q_gr)
∂Bϕ_∂q_gr = ∂_∂q(Bϕ_gr, q_gr)
println(L2_error(∂Bϕ_∂q[interior], ∂Bϕ_∂q_gr[interior], relative=true))

∂Bϕ_∂θ = ∂_∂θ(Bϕ1, θ_gr)
∂Bϕ_∂θ_gr = ∂_∂θ(Bϕ_gr, θ_gr)
println(L2_error(∂Bϕ_∂θ[interior], ∂Bϕ_∂θ_gr[interior], relative=true))

∂Bϕ_∂ϕ = ∂_∂ϕ(Bϕ1, ϕ_gr)
∂Bϕ_∂ϕ_gr = ∂_∂ϕ(Bϕ_gr, ϕ_gr)
println(L2_error(∂Bϕ_∂ϕ[interior], ∂Bϕ_∂ϕ_gr[interior], relative=true))


∂α_∂q = ∂_∂q(α1, q_gr)
∂α_∂q_gr = ∂_∂q(α_gr, q_gr)
println(L2_error(∂α_∂q[interior], ∂α_∂q_gr[interior], relative=true))

∂α_∂θ = ∂_∂θ(α1, θ_gr)
∂α_∂θ_gr = ∂_∂θ(α_gr, θ_gr)
println(L2_error(∂α_∂θ[interior], ∂α_∂θ_gr[interior], relative=true))

∂α_∂ϕ = ∂_∂ϕ(α1, ϕ_gr)
∂α_∂ϕ_gr = ∂_∂ϕ(α_gr, ϕ_gr)
println(L2_error(∂α_∂ϕ[interior], ∂α_∂ϕ_gr[interior], relative=true))

r_eq = @. q_gr * ∂Bϕ_∂θ + q_gr * cos(θ_gr) / sin(θ_gr) * Bϕ1 - q_gr / sin(θ_gr) * ∂Bθ_∂ϕ - α1 * Br1
L2_error(r_eq[interior], zero(r_eq[interior])) 

r_eq_gr = @. q_gr * ∂Bϕ_∂θ_gr + q_gr * cos(θ_gr) / sin(θ_gr) * Bϕ_gr - q_gr / sin(θ_gr) * ∂Bθ_∂ϕ_gr - α_gr * Br_gr
L2_error(r_eq_gr[interior], zero(r_eq_gr[interior]))

L2_error(r_eq[interior], r_eq_gr[interior])

θ_eq = @. q_gr / sin(θ_gr) * ∂Br_∂ϕ - q_gr * Bϕ1 + q_gr^2 * ∂Bϕ_∂q - α1 * Bθ1
L2_error(θ_eq[interior], zero(θ_eq[interior]))

θ_eq_gr = @. q_gr / sin(θ_gr) * ∂Br_∂ϕ_gr - q_gr * Bϕ_gr + q_gr^2 * ∂Bϕ_∂q_gr - α_gr * Bθ_gr
L2_error(θ_eq_gr[interior], zero(θ_eq_gr[interior]))

L2_error(θ_eq[interior], θ_eq_gr[interior])

ϕ_eq = @. q_gr * Bθ1 - q_gr^2 * ∂Bθ_∂q - q_gr * ∂Br_∂θ - α1 * Bϕ1
L2_error(ϕ_eq[interior], zero(ϕ_eq[interior]))

ϕ_eq_gr = @. q_gr * Bθ_gr - q_gr^2 * ∂Bθ_∂q_gr - q_gr * ∂Br_∂θ_gr - α_gr * Bϕ_gr
L2_error(ϕ_eq_gr[interior], zero(ϕ_eq_gr[interior]))

L2_error(ϕ_eq[interior], ϕ_eq_gr[interior])

∇B_eq = @. 2q_gr * Br1 - q_gr^2 * ∂Br_∂q + q_gr * ∂Bθ_∂θ + q_gr * cos(θ_gr) / sin(θ_gr) * Bθ1 + q_gr / sin(θ_gr) * ∂Bϕ_∂ϕ
L2_error(∇B_eq[interior], zero(∇B_eq[interior]))
L2_error(∇B1[interior], zero(∇B1[interior]))

∇B_eq_gr = @. 2q_gr * Br_gr - q_gr^2 * ∂Br_∂q_gr + q_gr * ∂Bθ_∂θ_gr + q_gr * cos(θ_gr) / sin(θ_gr) * Bθ_gr + q_gr / sin(θ_gr) * ∂Bϕ_∂ϕ_gr
L2_error(∇B_eq_gr[interior], zero(∇B_eq_gr[interior]))  

L2_error(∇B_eq[interior], ∇B_eq_gr[interior])

B∇α_eq = @. -q_gr^2 * Br1 * ∂α_∂q + q_gr * Bθ1 * ∂α_∂θ + q_gr / sin(θ_gr) * Bϕ1 * ∂α_∂ϕ
L2_error(B∇α_eq[interior], zero(B∇α_eq[interior]))
# L2_error(B∇α1[interior], zero(B∇α1[interior]))

B∇α_eq_gr = @. -q_gr^2 * Br_gr * ∂α_∂q_gr + q_gr * Bθ_gr * ∂α_∂θ_gr + q_gr / sin(θ_gr) * Bϕ_gr * ∂α_∂ϕ_gr
L2_error(B∇α_eq_gr[interior], zero(B∇α_eq_gr[interior]))

# ∂Br_∂q_gr = zero(Br_gr)
# ∂Br_∂q_gr[2:end-1, :, :] = (Br_gr[3:end, :, :] .- Br_gr[1:end-2, :, :]) / (2(q_gr[2,1,1] - q_gr[1,1,1]))
# ∂Br_∂q_gr[1, :, :] = (-3Br_gr[1,:,:] + 4Br_gr[2,:,:] - 1Br_gr[3,:,:]) / (2(q_gr[2,1,1] - q_gr[1,1,1]))
# ∂Br_∂q_gr[end, :, :] = (3Br_gr[end,:,:] - 4Br_gr[end-1,:,:] + 1Br_gr[end-2,:,:]) / (2(q_gr[end,1,1] - q_gr[end-1,1,1]))



# ∂Br_∂q = ∂_∂q(q_gr, Br1)
# ∂Br_∂q_gr = ∂_∂q(q_gr, Br_gr)
# sqrt.(sum(abs2, (∂Br_∂q .- ∂Br_∂q_gr)) / (size(∂Br_∂q)[1] * size(∂Br_∂q)[2] * size(∂Br_∂q)[3]))

# function ∂_∂θ(θ, u)

#     return (u[2:end-1, 3:end, 2:end-1] .- u[2:end-1, 1:end-2, 2:end-1]) / (2(θ[1,2,1] - θ[1,1,1]))
# end

# ∂Br_∂μ = ∂_∂θ(θ_gr, Br1)
# ∂Br_∂μ_gr = ∂_∂θ(θ_gr, Br_gr)
# sqrt.(sum(abs2, (∂Br_∂μ .- ∂Br_∂μ_gr)) / (size(∂Br_∂μ)[1] * size(∂Br_∂μ)[2] * size(∂Br_∂μ)[3])) ./ L2_error(∂Br_∂μ, zero(∂Br_∂μ))

# L2_error(∂Br_∂μ, ∂Br_∂μ_gr)
# L2_error(∂Br_∂μ, ∂Br_∂μ_gr) / L2_error(∂Br_∂μ_gr, zero(∂Br_∂μ_gr))

# L2_error(∂Br_∂μ_gr, zero(∂Br_∂μ_gr))
# L2_error(∂Br_∂μ, zero(∂Br_∂μ))

# ∂Br_∂q = zero(Br1)
# ∂Br_∂q[2:end-1, :, :] = (Br1[3:end, :, :] .- Br1[1:end-2, :, :]) / (2(q_gr[2,1,1] - q_gr[1,1,1]))
# ∂Br_∂q[1, :, :] = (-3Br1[1,:,:] + 4Br1[2,:,:] - 1Br1[3,:,:]) / (2(q_gr[2,1,1] - q_gr[1,1,1]))
# ∂Br_∂q[end, :, :] = (3Br1[end,:,:] - 4Br1[end-1,:,:] + 1Br1[end-2,:,:]) / (2(q_gr[end,1,1] - q_gr[end-1,1,1]))

# ∂Br_∂q_gr - ∂Br_∂q
# sqrt.(sum(abs2, (∂Br_∂q[interior] .- ∂Br_∂q_gr[interior])) / ((size(Br1)[1] - 2) * (size(Br1)[2] - 2) * (size(Br1)[3] - 2)))

