using MagnetospherePinn3D
using JLD2
using PrettyPrint
using Dates
using ComponentArrays

function main()

    start_time = now()

    # Parameters that change between experiments
    # Pc = range(0.0, 0.5, 11)
    # sigma_gs = [3, 4]
    # s = range(0.0, 1.0, 11)

    gamma = range(0, 0.3, 41)
    rng_seed = rand(1:1000, 10)


    combinations = [gamma, rng_seed]

    job_dir = setup_jobdir()
    config_file = setup_configfile(job_dir, combinations=combinations)
    params = import_params(config_file)

    NN, Θ, st = create_neural_network(params)

    # dirs = filter(dir -> isdir(dir) && startswith(basename(dir), "local") , readdir(abspath("data/"); join=true))
    # datadir = dirs[end-1]
    # Θ = load(joinpath(datadir, "trained_model.jld2"), "Θ_trained") |> ComponentArray |> gpu_device() .|> Float64

    # println("Loading trained model from $(datadir)")
    # # println(Θ ≈ Θ1)
    # params.optimization.adam_sets = 0
    


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

    end_time = now()
    elapsed = (end_time-start_time)
    elapsed_seconds = (end_time-start_time) / Millisecond(1000)
    elapsed_readable = Dates.format(convert(DateTime, elapsed), "HH:MM:SS")

    open(joinpath(job_dir, "execution_time.txt"), "w") do io
        # Define fixed column widths (adjust as needed)
        colwidth = 35
    
        # Build the header line and a line with the corresponding values
        header = rpad("Execution time (seconds)", colwidth) * rpad("Execution time (HH:MM:SS)", colwidth)
        values = rpad(elapsed_seconds, colwidth) * rpad(elapsed_readable, colwidth)
    
        # Write header and values to the file
        println(io, header)
        println(io, values)
    end
    
end


main()
