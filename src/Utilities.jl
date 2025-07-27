function setup_jobdir()
    
    # Jobs on the cluster
    if "SLURM_JOB_NAME" in keys(ENV)
        jobid = ENV["SLURM_JOB_ID"]
        jobdir = joinpath("data", "$cluster_$(jobid)")
        mkpath(jobdir)

    # Jobs on local machine
    else
        jobdir = "data/local_$(Dates.format(now(), "yyyy_mm_dd_HH_MM_SS"))"
        mkpath(jobdir)
    end

    return jobdir
end

function setup_jobdir2()
    
    # Jobs ran on the cluster
    if "SLURM_JOB_NAME" in keys(ENV)
        job_name = ENV["SLURM_JOB_NAME"]
        # Array jobs
        if "SLURM_ARRAY_JOB_ID" in keys(ENV)
            job_id = ENV["SLURM_ARRAY_JOB_ID"]
            task_id = ENV["SLURM_ARRAY_TASK_ID"]
            job_dir = joinpath("data", "$(job_name)_$(job_id)", "$(task_id)")
            mkpath(job_dir)
            cp("config_template.toml", joinpath(job_dir, "config_template.toml"))
        # Single jobs
        else
            job_id = ENV["SLURM_JOB_ID"]
            job_dir = joinpath("data", "$(job_name)_$(job_id)")
            mkpath(job_dir)
            cp("config_template.toml", joinpath(job_dir, "config_template.toml"))
        end
    # Jobs on local machine
    else
        job_dir = "data/local_$(Dates.format(now(), "yyyy_mm_dd_HH_MM_SS"))"
        mkpath(job_dir)
        cp("config_template.toml", joinpath(job_dir, "config_template.toml"))
    end

    return job_dir

end

function import_params(filename)

	# Parse parameters from .toml file to dictionary
	d = TOML.parsefile(filename)
	
	# Create Parameters object from dictionary
	params = Configurations.from_dict(Params, d)
	return params

end

function export_params(params, filename)
		
	# Pass parameters to dictionary
	d = Configurations.to_dict(params)
	
	# Write dictionary to .toml file
	open(filename, "w") do io
		TOML.print(io, d)
	end
    
end

function setup_configfile(job_dir; combinations=[])
   
    # Array jobs on the cluster
    if "SLURM_ARRAY_TASK_ID" in keys(ENV)
        config_file_template = joinpath(job_dir, "config_template.toml")
        params = import_params(config_file_template)
        
        task_id = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
        indices = compute_combination(task_id, combinations)

        # params.model.alpha0 = combinations[1][indices[1]]
        # params.model.sigma = combinations[1][indices[1]]
        # params.model.M = combinations[2][indices[2]]
        # params.architecture.rng_seed = combinations[2][indices[2]]
        params.model.coef = combinations[1][indices[1]]

        # params1.architecture.q_distributiion = combinations[1][indices[1]]
        # params1.optimization.loss_function  = combinations[2][indices[2]]
        # params1.optimization.linesearch = combinations[3][indices[3]]

        @info "Running experiment with combination
            coef= $(params.model.coef)
        "

        export_params(params, joinpath(job_dir, "config.toml"))
        config_file = joinpath(job_dir, "config.toml")
    else
        # Single jobs on the cluster or on local machine
        cp(joinpath(job_dir, "config_template.toml"), joinpath(job_dir, "config.toml"))
        config_file = joinpath(job_dir, "config.toml")
    end

    return config_file
end

function compute_combination(idx, combinations)
    
    lengths = map(length, combinations)
    n = length(combinations)

    N_combinations = prod(lengths)
    if N_combinations ≠ parse(Int, ENV["SLURM_ARRAY_TASK_COUNT"])
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
