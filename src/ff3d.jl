using MagnetospherePinn3D
using JLD2
using PrettyPrint
using ComponentArrays
using Dates
using Lux

include("UtilitiesForceFree3D.jl")
job_dir = setup_jobdir()
config_file = setup_configfile(job_dir)
const params = import_params(config_file)

NN, Θ, st = create_neural_network(params)

invH = Base.RefValue{AbstractArray{Float64, 2}}()
losses = [Float64[] for _ in 1:6]

@info "Setting up optimization problem"
optprob, result = setup_optprob(Θ, st, NN, params)

@info "Training neural network"
result = train_neural_network!(result, optprob, losses, invH, job_dir, params)

@info "Saving trained model"
Θ_trained = result.u |> Lux.cpu_device()

@save joinpath(job_dir, "trained_model.jld2") Θ_trained
@save joinpath(job_dir, "losses_vs_iterations.jld2") losses


