using TOML
using LineSearches
using Distributions
using PrettyPrint
using JLD2
using DrWatson
using MagnetospherePinn3D
using IntervalSets
using OrdinaryDiffEq
using OrderedCollections
using ComponentArrays

include("plots/PostProcess.jl")

"""
Convert a TOML config file with sections to a flat Dict matching Config.jl syntax.
"""
function toml_to_config_dict(filename)
    toml_dict = TOML.parsefile(filename)
    config = OrderedDict{Symbol,Any}()

    # Mapping from TOML keys to config keys
    key_map = Dict(
        "coef" => :multipole_bl,
        "M" => :compactness,
        "alpha0" => :α0,
        "theta1" => :θ1,
        "phi1" => :ϕ1,
        "sigma" => :σ,
        "alpha0_b" => :α0_b,
        "theta1_b" => :θ1_b,
        "phi1_b" => :ϕ1_b,
        "sigma_b" => :σ_b,
        "sigma_gs" => :σ_gs,
        "gamma" => :γ,
        "notation" => :axisym_notation,
        "alpha_bc_mode" => :α_bc_mode,
        "loss_function" => :loss_g,
    )

    value_map = Dict(
        "uniform" => Uniform(),
        "Beta" => Beta(3,1),
        "BackTracking" => BackTracking(),
        "HagerZhang" => HagerZhang(),
        "MoreThuente" => MoreThuente(),
        "MSE" => identity,
        "logMSE" => log,
    )

    # Map TOML sections to flat keys with correct symbol names
    for (section, values) in toml_dict
        for (k, v) in values

            # println("Processing key: $k with value: $v... ")

            if haskey(key_map, k)
                if haskey(value_map, v)
                    config[key_map[k]] = value_map[v]
                else
                    config[key_map[k]] = v
                end

            else
                if haskey(value_map, v)
                    config[Symbol(k)] = value_map[v]
                else
                    config[Symbol(k)] = v
                end
            end
        end
    end

    return config
end

function convert_bias_to_1d(θ)
    # Base case: if θ is a ComponentArray with "bias" field, convert it
    if θ isa ComponentArray && haskey(θ, :bias)
        # Convert 2D bias matrix to 1D vector
        new_bias = vec(θ.bias)
        # Rebuild ComponentArray with converted bias
        return ComponentArray(; (k => k == :bias ? new_bias : θ[k] for k in keys(θ))...)
    end

    # Recursive case: process nested structures
    if θ isa ComponentArray || θ isa NamedTuple
        # Rebuild the structure with converted children
        new_fields = []
        for k in keys(θ)
            field_val = θ[k]
            converted_val = convert_bias_to_1d(field_val)
            push!(new_fields, k => converted_val)
        end
        return θ isa ComponentArray ? 
            ComponentArray(; new_fields...) : 
            NamedTuple(new_fields)
    end

    # Return unchanged if not a convertible type
    return θ
end

function read_traindata_from_dir(dir)
    traindata = OrderedDict{Symbol, Any}()

    # Read losses
    losses_file = joinpath(dir, "losses_vs_iterations.jld2")
    if isfile(losses_file)
        traindata[:losses] = load(losses_file, "losses")
    else
        traindata[:losses] = missing
    end

    # Read trained model
    model_file = joinpath(dir, "trained_model.jld2")
    if isfile(model_file)
        traindata[:Θ] = load(model_file, "Θ_trained")
    else
        traindata[:Θ] = missing
    end

    # Read execution time
    exec_file = joinpath(dir, "execution_time.txt")
    if isfile(exec_file)
        # Try to parse the seconds from the first line of the file
        duration = missing
        duration_readable = missing
        open(exec_file, "r") do io
            for line in eachline(io)
                if occursin("Execution time (seconds)", line)
                    continue  # skip header
                end
                # Try to parse the first value (seconds)
                vals = split(strip(line))
                if !isempty(vals)
                    duration = tryparse(Float64, vals[1])
                    duration_readable = vals[2]
                end
                break
            end
        end
        traindata[:duration] = duration
        traindata[:duration_readable] = duration_readable
    else
        traindata[:duration] = missing
    end

    return traindata
end

function convert_old_data(dirs)
    for dir in dirs

        @info "Processing $dir"

        if isfile(joinpath(dir, "config.toml"))
            config = toml_to_config_dict(joinpath(dir, "config.toml"))
            wsave(joinpath(dir, "config.jld2"), "data", config)
        end

        if !isfile(joinpath(dir, "griddata.jld2"))
            @info "Evaluating on grid"
            config = NamedTuple(load(joinpath(dir, "config.jld2"), "data"))
            NN, _, st = create_neural_network(config, test_mode=true)
            Θ = load(joinpath(dir, "trained_model.jld2"), "Θ_trained")
            Θ = convert_bias_to_1d(Θ)
            griddata = evaluate_on_grid(160, 80, 160, NN, Θ, st, config; use_θ = true, extended=false)
            wsave(joinpath(dir, "griddata.jld2"), "data", griddata)
        end

        if !isfile(joinpath(dir, "fieldlines.jld2"))    
            @info "Integrating fieldlines"
            config = NamedTuple(load(joinpath(dir, "config.jld2"), "data"))
            NN, _, st = create_neural_network(config, test_mode=true)
            Θ = load(joinpath(dir, "trained_model.jld2"), "Θ_trained")
            Θ = convert_bias_to_1d(Θ)
            griddata = load(joinpath(dir, "griddata.jld2"), "data")
            @unpack μ, ϕ, α1 = griddata
            footprints = find_footprints(μ, ϕ, α1, μ_interval=0..1, ϕ_interval=0..2π)
            fieldlines = integrate_fieldlines(footprints, NN, Θ, st, config; q_start = 1);
            wsave(joinpath(dir, "fieldlines.jld2"), "data", fieldlines)
        end

        if !isfile(joinpath(dir, "traindata.jld2"))
            @info "Reading training data"
            traindata = read_traindata_from_dir(dir)
            wsave(joinpath(dir, "traindata.jld2"), "data", traindata)
        end

    end
end

dirs = (filter(dir -> isdir(dir), readdir(abspath("data/theta1_sequence/"); join=true, sort=false)))
convert_old_data(dirs)
