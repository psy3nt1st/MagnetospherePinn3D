function setup_jobdir()
    
    # Jobs on the cluster
    if "SLURM_JOB_ID" in keys(ENV)
        jobid = ENV["SLURM_JOB_ID"]
        jobname = ENV["SLURM_JOB_NAME"]
        jobdir = joinpath("data", "$(jobname)_$(jobid)")
        mkpath(jobdir)

    # Jobs on local machine
    else
        jobdir = "data/local_$(now())"
        mkpath(jobdir)
    end

    return jobdir
end

function setup_subjobdir(config, jobdir, expanded_keys)
    if isempty(expanded_keys)
        config[:subjobdir] = jobdir
    else
        config[:subjobdir] = joinpath(jobdir, savename(config; accesses=[k for k in expanded_keys if k in keys(config)]))
    end
    wsave(joinpath(config[:subjobdir], "config.jld2"), "data", config)
end

function load_gradrubin_data(output_path::String)

    file = readdir(joinpath(output_path, "silo_output"), join=true)[end]
    @info "Using $file"
    
    f = HDF5.h5open(file, "r")
    r = f[read_attribute(f["r"], "silo").value0][]
    θ = f[read_attribute(f["theta"], "silo").value0][]
    φ = f[read_attribute(f["varphi"], "silo").value0][]
    alpha = f[read_attribute(f["alpha"], "silo").value0][]
    u_q = f[read_attribute(f["u_1"], "silo").value0][]
    u_theta = f[read_attribute(f["u_2"], "silo").value0][]
    u_phi = f[read_attribute(f["u_3"], "silo").value0][]
    B_q = f[read_attribute(f["B_1"], "silo").value0][]
    B_theta = f[read_attribute(f["B_2"], "silo").value0][]
    B_phi = f[read_attribute(f["B_3"], "silo").value0][]

    q = 1 ./ r
    μ = cos.(θ)

    return r, q, θ, μ, φ, alpha, u_q, u_theta, u_phi, B_q, B_theta, B_phi
end