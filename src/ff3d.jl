using DrWatson
using MagnetospherePinn3D
using JLD2
using Dates
using ComponentArrays
using NaturalSort
using OrderedCollections

function main(config)

    @info "Creating neural network"
    NN, Θ, st = create_neural_network(config)

    @info "Setting up optimization problem"
    optresult, optprob, traindata = setup_optprob(NN, Θ, st, config)

    @info "Training neural network"

    traindata, optresult = train_pinn!(optresult, optprob, traindata, config)

    return traindata
end

config = create_config()
jobdir = setup_jobdir(config)

configs = dict_list(config)
expanded_keys = filter(k -> config[k] isa Vector && (length(config[k]) > 1), keys(config))

for c in configs

    setup_subjobdir(c, jobdir, expanded_keys)
    c = NamedTuple(c)
    
    @info "Running main function with configuration: $([k => c[k] for k in expanded_keys if k in keys(config)])"
    # println([k => c[k] for k in expanded_keys if k in keys(config)])
    traindata = main(c)

end






