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
# datadir = dirs[end-1]
# datadir = "../data/gamma_sequence"
# datadir = "../data/alpha0_sequence"
datadir = "../data/alphamax_sequence_axisymmetric"
# datadir = "../data/alphamax_sequence_3d"

rundirs = sort(filter(dir -> isdir(dir), readdir(abspath(datadir); join=true, sort=false)), lt=natural)

rundir = rundirs[3]
# rundir = dirs[end-3]
for rundir in rundirs
    if !isfile(joinpath(rundir, "run_data.jld2"))
        @info "Running test in $rundir"
        NN, Θ_trained, st, losses, q, μ, ϕ, t, θ, Br1, Bθ1, Bϕ1, α1, ∇B, B∇α, Nr, Nθ, Nϕ, Nα, Bmag1, params = run_test(rundir)
        save_test(rundir, NN, Θ_trained, st, losses, q, μ, ϕ, t, θ, Br1, Bθ1, Bϕ1, α1, ∇B, B∇α, Nr, Nθ, Nϕ, Nα, Bmag1, params)
    else
        @info "Loading test data from $rundir"
        NN, Θ_trained, st, losses, q, μ, ϕ, t, θ, Br1, Bθ1, Bϕ1, α1, ∇B, B∇α, Nr, Nθ, Nϕ, Nα, Bmag1, params = load_test(rundir)
    end
end

# Calculate energy
energy = calculate_energy(t[1], NN, Θ_trained, st, params)
println("Energy = ", energy)

dipole_energy = calculate_dipole_energy(params.model.M)
println("Dipole energy = ", dipole_energy)

excess_energy = (energy - dipole_energy) / dipole_energy
println("Excess energy = ", excess_energy) 

# Calculate max Bϕ ratio
max_Bϕ_ratio = maximum(abs.(Bϕ1[Bmag1 .!= 0]) ./ Bmag1[Bmag1 .!= 0])
println("Max Bϕ ratio = ", max_Bϕ_ratio)

magnetic_virial_surface = calculate_magnetic_virial_surface(t[1], NN, Θ_trained, st, params)
magnetic_virial_volume = calculate_magnetic_virial_volume(t[1], NN, Θ_trained, st, params)
magnetic_virial = magnetic_virial_surface + magnetic_virial_volume
println("Magnetic virial = ", magnetic_virial)

quadrupole_moment = calculate_quadrupole_moment(t[1], NN, Θ_trained, st, params)
println("Quadrupole moment = ", quadrupole_moment)

println("M = $(params.model.M)")
println("α0 = $(params.model.alpha0)")
# println("γ = $(params.model.gamma)")
println("σ = $(params.model.sigma)")
println("θ1 = $(params.model.theta1)")
println("coef = $(params.model.coef)")
println("loss = $(losses[1][end])")
abs(magnetic_virial - energy) / energy
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
