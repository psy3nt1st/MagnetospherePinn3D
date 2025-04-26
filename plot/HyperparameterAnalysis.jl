using MagnetospherePinn3D
using JLD2
using PrettyPrint
using Dates
using Parameters
using Integrals
using NaturalSort
using GLMakie
using LaTeXStrings
using DelimitedFiles


include("PostProcess.jl")

dirs = filter(dir -> isdir(dir) && startswith(basename(dir), "ff3d") , readdir(abspath("../data"); join=true))
datadir = dirs[end]
# datadir = "../data/ff3d_3155186/"
rundirs = sort(filter(dir -> isdir(dir) , readdir(abspath(datadir); join=true, sort=false)), lt=natural)
run_configs = [joinpath(dir, "config.toml") for dir in rundirs]
run_params = [import_params(config) for config in run_configs]
run_losses = [isfile(joinpath(dir, "losses_vs_iterations.jld2")) ? load(joinpath(dir, "losses_vs_iterations.jld2"), "losses") : [] for dir in rundirs]

run_total_losses = [loss == [] ? [] : loss[1] for loss in run_losses]
run_total_losses_last = [all(isnan.(loss)) ? NaN : filter(!isnan, loss)[end] for loss in run_total_losses]

run_networks = [create_neural_network(params, test_mode=true)[1] for params in run_params]
run_states = [create_neural_network(params, test_mode=true)[2] for params in run_params]
run_Θ_trained = [isfile(joinpath(dir, "trained_model.jld2")) ? load(joinpath(dir, "trained_model.jld2"), "Θ_trained") : [] for dir in rundirs]

function read_time(dir)
    if isfile(joinpath(dir, "execution_time.txt"))
        open((joinpath(dir, "execution_time.txt")), "r") do f
            data = readdlm(f, '\t')
            total_time = parse(Float64, data[2][1:25])
            return total_time
        end
    else
        return 0.0
    end

end

run_total_times = [read_time(dir) for dir in rundirs]
# run_energies = [run_Θ_trained[i] == [] ? 0.0 : calculate_energy(t1, run_networks[i], run_Θ_trained[i], run_states[i], run_params[i]) for i in eachindex(run_networks)]
# println(run_energies)

best_runs = sortperm(run_total_losses_last)
best_total_losses = run_total_losses_last[best_runs]
t1 = 0
best_energies = [run_Θ_trained[i] == [] ? 0.0 : calculate_energy(t1, run_networks[i], run_Θ_trained[i], run_states[i], run_params[i]) for i in best_runs]
best_times = run_total_times[best_runs]
best_time_per_iteration = [run_total_times[i] / length(run_total_losses[i]) for i in best_runs]

f = Figure()
ax = Axis(f[1, 1], xlabel="Run", ylabel="Loss", yscale=log10)
scatter!(best_runs, best_total_losses, marker=:xcross)
display(GLMakie.Screen(), f)

f = Figure()
ax = Axis(f[1,1], xlabel="Run", ylabel="Energy")
scatter!(best_runs, best_energies)
energy_gradrubin = 0.335707
hlines!(energy_gradrubin)
display(GLMakie.Screen(), f)




run_hyperparams = [[params.architecture.q_distributiion,
                    params.optimization.loss_function,
                    params.optimization.linesearch] for params in run_params]

best_hyperparms = run_hyperparams[best_runs]

for i in eachindex(best_hyperparms)

    println("Run: $(best_runs[i]) loss: $(best_total_losses[i]) Energy: $(best_energies[i]) Hyperparameters: $(best_hyperparms[i]) Time per iteration: $(best_time_per_iteration[i])")
end






