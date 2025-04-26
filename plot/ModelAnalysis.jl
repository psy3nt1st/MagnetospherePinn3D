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

include("PostProcess.jl")
include("Plotting.jl")

dirs = filter(dir -> isdir(dir) && startswith(basename(dir), "ff3d") , readdir(abspath("../data"); join=true))
datadir = dirs[end]

@info "Using data in $datadir"

# datadir = "../data/ff3d_3155186/"
run_dirs = sort(filter(dir -> isdir(dir) , readdir(abspath(datadir); join=true, sort=false)), lt=natural)[1:end-2]

filter!(dir -> isfile(joinpath(dir, "trained_model.jld2")), run_dirs)
filter!(dir -> isfile(joinpath(dir, "losses_vs_iterations.jld2")), run_dirs)

run_params, losses, NNs, sts, Θs, times = load_run_data(run_dirs)

energies, excess_energies, relative_excess_energies,
poloidal_energies, excess_poloidal_energies, relative_excess_poloidal_energies,
toroidal_energies, magnetic_virials, Ms, Pcs, ns, gammas = calculate_run_quantities(run_dirs, NNs, Θs, sts, run_params)

raw_data = DataFrame(
    M = Ms,
    Pcs1 = Pcs,
    ns = ns,
    gammas = gammas,
    energies = energies,
    excess_energies = excess_energies,
    relative_excess_energies = relative_excess_energies,
    poloidal_energies = poloidal_energies,
    excess_poloidal_energies = excess_poloidal_energies,
    relative_excess_poloidal_energies = relative_excess_poloidal_energies,
    toroidal_energies = toroidal_energies,
    magnetic_virials = magnetic_virials,
)

data = select(raw_data, :gammas, :energies, :excess_energies, :excess_poloidal_energies, :toroidal_energies, :magnetic_virials)
# data = filter(x -> x.magnetic_virials > 0.0, data)
data

f, ax, sc = scatter(data.gammas, data.excess_energies, markersize=20, label="Total")
scatter!(data.gammas, data.excess_poloidal_energies, marker=:xcross, markersize=20, label="Poloidal")
scatter!(data.gammas, data.toroidal_energies, marker=:star8, markersize=20, label="Toroidal")
display(GLMakie.Screen(), f)
axislegend(ax, position=:lt, merge = false, unique = true)


f, ax, sc = scatter(data.gammas, data.magnetic_virials / (4π) ./ data.energies, markersize=20)
display(GLMakie.Screen(), f)


f = Figure()
ax = Axis(f[1, 1], xlabel="γ", ylabel="Loss", yscale=log10)
scatter!(ax, gammas, losses)
current_figure()



# scatter(data.gammas, data.magnetic_virials, markersize=20)
# scatter(data.gammas, data.excess_energies, markersize=20)

# to_dict(run_params[1])

# using CSV
# writedlm("data.txt", eachrow(data), '\t', header=true)
# writedlm("data.txt", Iterators.flatten(([names(data)], eachrow(data))), "\t\t\t")
# CSV.write("data.csv", data, delim="\t")
# data
# open("data.txt", "w") do io
#     show(io, MIME"text/plain"(), data, allrows=true, allcols=true)
# end
# CSV.write("data.tsv", data, delim="\t")
# using PrettyTables

# # Save the DataFrame to a text file with aligned columns

# open("data.txt", "w") do io
#     pretty_table(io, data2)
# end
# using Configurations
# config = Configurations.to_dict(run_params[1])
# keys(config)
# data2 = DataFrame(config["model"])

# show(data2, allrows=true, allcols=true)
# sprint(showall, data, context = :compact => true)

# s[s .== 0 .&& sigmas .== 4.0 .&& z .== 0.25]
# dipole_energy_relativistic = energies[s .== 0 .&& sigmas .== 4.0 .&& z .== 0.25][1]

# excess_energies_rel[z .== 0.1]
# excess_energies_rel[z .== 0.25]



# function plot_quantities_all_models2(z, sigmas, s,  excess_energies, magnetic_virials)

#     γ = (sigmas .+ 1) .* s.^2 / 2
#     unique_sigmas = unique(sigmas)
#     unique_z = unique(z)
#     f = Figure()
#     ax1 = Axis(f[1, 1], xlabel="γ", ylabel=L"\Delta E / E_0")
#     ax2 = Axis(f[1, 2], xlabel="γ", ylabel="Mag. virial")
#     for sigma in unique_sigmas, z1 in unique_z
#         idx = sigmas .== sigma .&& z .== z1
#         marker = sigma == 3.0 ? :xcross : :rect
#         color = z1 == 0.25 ? :red : :blue
#         label = @sprintf("σ = %d, z = %.2f", Int(sigma), z1)
#         scatter!(ax1, γ[idx], excess_energies[idx], 
#             marker = marker, 
#             markersize = 15,
#             color = color,
#             label = label
#         )
#         scatter!(ax2, γ[idx], magnetic_virials[idx], 
#             marker = marker, 
#             markersize = 15,
#             color = color,
#             label = label
#         )
#     end
#     # Colorbar(f[1,3], label = L"P_c", labelrotation=0)
#     axislegend(ax1, merge = false, unique = true)
#     display(GLMakie.Screen(), f)
# end

# scatter!(s[z .== 0.25], excess_energies_rel[z .== 0.25])



# plot_quantities_all_models2(z, sigmas, s,  excess_energies_rel, magnetic_virials)
# plot_quantities_one_model(s, excess_energies, magnetic_virials; Pc0=0.0)
# plot_quantities_one_model(s, excess_energies, magnetic_virials; Pc0=0.0)



