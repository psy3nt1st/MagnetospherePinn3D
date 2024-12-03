using MagnetospherePinn3D
using JLD2
using PrettyPrint
using ComponentArrays

function compute_combination(idx, combinations)
   
   lengths = map(length, combinations)
   n = length(combinations)

   N_combinations = prod(lengths)
   if N_combinations ≠ parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
      @warn "Number of combinations ($(N_combinations)) does not match number of tasks ($(ENV["SLURM_ARRAY_TASK_COUNT"]))."
   end
   
   # Compute indices of each array in combinations
   remaining_idx = idx - 1  # Use 0-based index
   indices = zeros(Int, n)
   for k in 1:n
       product_of_remaining_lengths = prod(lengths[k+1:end])  # product of lengths from k+1 to n
       indices[k] = div(remaining_idx, product_of_remaining_lengths) + 1  # Calculate index for Aₖ
       
       remaining_idx = mod(remaining_idx, product_of_remaining_lengths)  # Update remaining index
   end
   
   return indices
end

function setup_configfile(job_dir)
   
   # Parameters that change between experiments
   optimizers = ["BFGS", "SSBFGS", "SSBroyden"]
   layers = [2, 3, 4]
   neurons = [20, 30, 40]
   
   combinations = [optimizers, layers, neurons]

   # Set up config file for experiment (if running on SLURM cluster)
   if "SLURM_ARRAY_TASK_ID" in keys(ENV)
      config_file_template = joinpath(job_dir, "config_template.toml")
      params1 = import_params(config_file_template)
      
      task_id = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
      indices = compute_combination(task_id, combinations)

      params1.optimization.quasiNewton_method = optimizers[indices[1]]
      params1.architecture.N_layers = layers[indices[2]]
      params1.architecture.N_neurons = neurons[indices[3]]

      export_params(params1, joinpath(job_dir, "config.toml"))
      config_file = joinpath(job_dir, "config.toml")
   else
      # Set up config file for experiment (if running on local machine)
      cp(joinpath(job_dir, "config_template.toml"), joinpath(job_dir, "config.toml"))
      config_file = joinpath(job_dir, "config.toml")
   end

   return config_file
end

job_dir = setup_jobdir()
config_file = setup_configfile(job_dir)
const params = import_params(config_file)

# vars = Variables([zeros(1, params.optimization.N_points) |> params.general.dev .|> Float64 for _ in 1:fieldcount(Variables)]...)

# Create neural network
NN, Θ, st = create_neural_network(params)



# if "SLURM_ARRAY_TASK_ID" in keys(ENV)
#    task_id = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
#    if task_id !=1
#       println("Loading data from previous task")
#       dirs = filter(isdir, readdir("data"; join=true))
#       datadir = joinpath(dirs[end], "$(task_id-1)")
#       Θ = load(joinpath(datadir,"trained_model.jld2"), "Θ_trained")
#       Θ = params.architecture.use_gpu ? Θ |> ComponentArray |> gpu_device() .|> Float64 : Θ |> ComponentArray .|> Float64
#    end
# end

# iteration = Base.RefValue{Int}(1)
invH = Base.RefValue{AbstractArray{Float64, 2}}()
losses = [Float64[] for _ in 1:6]

optprob, result = setup_optprob(Θ, st, NN, params)

result = train_neural_network!(result, optprob, losses, invH, job_dir, params)

Θ_trained = result.u |> Lux.cpu_device()

@save joinpath(job_dir, "trained_model.jld2") Θ_trained
@save joinpath(job_dir, "losses_vs_iterations.jld2") losses


