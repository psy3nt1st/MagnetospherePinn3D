using MagnetospherePinn3D
using JLD2
using PrettyPrint
using Dates

function main()

    # Parameters that change between experiments
    loss_functions = ["MSE", "logMSE"]
    N_layers = [1, 2, 3]
    N_neurons = [20, 40, 80]
    N_points = [20000, 40000, 80000]
    quasiNewton_iters = [500, 1000]
    
    combinations = [loss_functions, N_layers, N_neurons, N_points, quasiNewton_iters]

    job_dir = setup_jobdir()
    config_file = setup_configfile(job_dir; combinations=combinations)
    params = import_params(config_file)

    NN, Θ, st = create_neural_network(params)
    println(size(Θ))

    invH = Base.RefValue{AbstractArray{Float64, 2}}()
    losses = [Float64[] for _ in 1:7]

    @info "Setting up optimization problem"
    optprob, result = setup_optprob(Θ, st, NN, params)

    @info "Training neural network"
    result = train_neural_network!(result, optprob, losses, invH, job_dir, params)

    @info "Saving trained model"
    Θ_trained = result.u |> Lux.cpu_device()

    @save joinpath(job_dir, "trained_model.jld2") Θ_trained
    @save joinpath(job_dir, "losses_vs_iterations.jld2") losses

end

main()

