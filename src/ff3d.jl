using MagnetospherePinn3D
using JLD2
using PrettyPrint
using Dates
using ComponentArrays
using NaturalSort


function main()

    start_time = now()

    # gamma = vcat(1.5, range(start=1.35, stop=0.0, step=-0.15))
    # gamma = vcat(0.195, 0.19, 0.185, range(start=0.18, stop=0.0, step=-0.01))
    # gamma = vcat(range(start=0, stop=1.5, step=0.15), range(start=1.5, stop=0, step=-0.15))
    # theta1 = range(start=90, stop=0, step=-5.0) 
    # alpha0 = range(start=0.0, stop=4, step=0.25)
    # sigma = range(start=0.0, stop=0.4, step=0.025)
    coef = [[1.0, 0.5, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0], [1.0, 5.0, 0.0], [1.0, 8.0, 0.0], [1.0, 10.0, 0.0]]
    # M = [0.0, 0.17, 0.25]

    # # rng_seed = rand(1:1000, 10)

    combinations = [coef]
    # combinations = []

    job_dir = setup_jobdir()
    config_file = setup_configfile(job_dir, combinations=combinations)
    params = import_params(config_file)

    NN, Θ, st = create_neural_network(params)
    
    if "SLURM_ARRAY_JOB_ID" in keys(ENV) 
        if ENV["SLURM_ARRAY_TASK_ID"] == "1"
            params.optimization.initialize_weights_from_previous = false
        end
        println("Array job ID: $(ENV["SLURM_ARRAY_TASK_ID"])")
        println("Initializing weights from previous run: $(params.optimization.initialize_weights_from_previous)")

        if params.optimization.initialize_weights_from_previous == true
            datadir = joinpath("./data", "ff3d_$(ENV["SLURM_ARRAY_JOB_ID"])")
            run_dirs = sort(filter(dir -> isdir(dir) , readdir(abspath(datadir); join=true, sort=false)), lt=natural)
            previous_run_dir = run_dirs[end-1]
            println("Loading trained model from $(previous_run_dir)")
            Θ = load(joinpath(previous_run_dir, "trained_model.jld2"), "Θ_trained") |> ComponentArray |> gpu_device() .|> Float64
        end
    end

    if params.optimization.initialize_weights_from_file == true
        if isfile("initial_weights.jld2")
            @info "Loading initial weights from file"
            Θ = load("initial_weights.jld2", "Θ_trained") |> ComponentArray |> gpu_device() .|> Float64
        else
            @warn "Initial weights file not found. Using default initialization."
        end
    end

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
