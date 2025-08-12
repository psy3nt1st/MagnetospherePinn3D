using MagnetospherePinn3D
using JLD2
using Dates
using Parameters
using Integrals
using NaturalSort
using GLMakie
using LaTeXStrings
using DelimitedFiles
using Printf
using DataFrames
using Configurations

include("PostProcess.jl")
include("Plotting.jl")

label_size = 25
tick_size = 20

dirs = filter(dir -> isdir(dir) && startswith(basename(dir), "ff3d") , readdir(abspath("../data"); join=true))
# datadir = dirs[end-1]
# datadir = "../data/gamma_sequence"
# datadir = "../data/alpha0_sequence"
datadir = "../data/alphamax_sequence_axisymmetric"
# datadir = "../data/alphamax_sequence_3d"
# datadir = "../data/sigma_sequence"
# datadir = "../data/theta1_sequence"

@info "Using data in $datadir"

# results1 = collect_results(jobdir; subfolders=true, 
#     rinclude = [r"config"],
#     load_function = filename -> load(filename, "data"))
# results2 = collect_results(jobdir; subfolders=true, 
#     rinclude = [r"traindata"],
#     load_function = filename -> load(filename, "data"))

# results = hcat(results1, results2, makeunique=true)

# datadir = "../data/ff3d_3155186/"
run_dirs = sort(filter(dir -> isdir(dir) , readdir(abspath(datadir); join=true, sort=false)), lt=natural)

filter!(dir -> isfile(joinpath(dir, "trained_model.jld2")), run_dirs)
filter!(dir -> isfile(joinpath(dir, "losses_vs_iterations.jld2")), run_dirs)
run_params, losses, NNs, sts, Θs, times = load_run_data(run_dirs)

energies, excess_energies, relative_excess_energies,
poloidal_energies, excess_poloidal_energies, relative_excess_poloidal_energies,
toroidal_energies, magnetic_virials_surface, magnetic_virials_volume, magnetic_virials,
quadrupole_moments = calculate_run_quantities(NNs, Θs, sts, run_params)

data = DataFrame([merge(Dict(d["model"]), Dict(d["optimization"]), Dict(d["architecture"])) for d in map(to_dict, run_params)])

data = hcat(data, DataFrame(energies=energies, excess_energies=excess_energies, 
    relative_excess_energies=relative_excess_energies, 
    poloidal_energies=poloidal_energies, 
    excess_poloidal_energies=excess_poloidal_energies, 
    relative_excess_poloidal_energies=relative_excess_poloidal_energies, 
    toroidal_energies=toroidal_energies, 
    magnetic_virials_surface=magnetic_virials_surface,
    magnetic_virials_volume=magnetic_virials_volume,
    magnetic_virials=magnetic_virials,
    quadrupole_moments=quadrupole_moments,
    losses=losses,
    run_dirs=basename.(run_dirs)
))

if all(data.alpha_bc_mode .== "axisymmetric")
    P_max = @. (-3 / (8 * data.M^3)) * (log(1 - 2 * data.M) + 2 * data.M + (2 * data.M)^2 / 2)
    P_max[data.M .== 0] .= 1
    α_max = 2 * .√data.gamma .* P_max.^((data.n .- 1) / 2)
elseif all(data.alpha_bc_mode .== "hotspot")
    α_max = data.alpha0
end

data = hcat(data, DataFrame(alpha_max=α_max))
# data = data[data.gamma .!= 0.195, :] 

# f1 = plot_excess_energy_vs_theta1(filter(row -> row.theta1 > 30, data))
# # save(joinpath("figures", "energy_vs_theta1_3d.png"), f1, size=(800, 600))

# include("Plotting.jl")
# f2 = plot_quadrupole_moment_vs_theta1(filter(row -> row.theta1 >= 30, data))
# save(joinpath("figures", "quadrupole_moment_vs_theta1_C=0.17.png"), f2, size=(800, 600))

# f3 = plot_magnetic_virial_vs_theta1(filter(row -> row.theta1 >= 30, data))
# # save(joinpath("figures", "virial_vs_theta1_3d.png"), f3, size=(800, 600))

selected_columns = [:energies, :relative_excess_energies, :quadrupole_moments, :M, :alpha_max, :gamma, :losses, :run_dirs]
grouped_data = groupby(data, :M)
println(select(grouped_data, selected_columns))
# println(select(grouped_data[1], selected_columns))
# println(select(grouped_data[2], selected_columns))
# println(select(grouped_data[3], selected_columns))

Q22_0 = -2.7746e-01
data.quadrupole_moments ./ -2.5967e-01 ./ (1 + 0.17)^2
Q22_0 / (1+ 0.17^2)


(data.quadrupole_moments .- Q22_0) ./ Q22_0

include("Plotting.jl")
f1 = plot_excess_energy_vs_alphamax(data)
# # save(joinpath("figures", "energy_vs_alphamax_3d.png"), f1, size=(800, 600))

# f2 = plot_magnetic_virial_vs_alphamax(data)
# # save(joinpath("figures", "virial_vs_alphamax_3d.png"), f2, size=(800, 600))

# f3 = plot_quadrupole_moment_vs_alphamax(data)
# save(joinpath("figures", "quadrupole_moment_vs_alpha0.png"), f3, size=(800, 600))

# f4 = plot_losses_vs_alphamax(data)
# # save(joinpath("figures", "loss_vs_alphamax_3d.png"), f4, size=(800, 600))

# f5 = error_vs_alphamax(data)
# # save(joinpath("figures", "error_vs_alphamax_3d.png"), f5, size=(800, 600))


# f1 = plot_excess_energy_vs_sigma(data)
# # save(joinpath("figures", "energy_vs_sigma_3d.png"), f1, size=(800, 600))

# f2 = plot_quadrupole_moment_vs_sigma(data)
# save(joinpath("figures", "quadrupole_moment_vs_sigma_C=0.17.png"), f2, size=(800, 600))

# f3 = plot_magnetic_virial_vs_sigma(data)
# # save(joinpath("figures", "virial_vs_sigma_3d.png"), f3, size=(800, 600))


