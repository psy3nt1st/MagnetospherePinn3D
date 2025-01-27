using MagnetospherePinn3D
using JLD2
using GLMakie
using PrettyPrint
using Dates
using DelimitedFiles
using LaTeXStrings
using NaturalSort

# Get path to data directory
dirs = filter(dir -> isdir(dir) && startswith(basename(dir), "ff3d") , readdir(abspath("../data"); join=true))
# dirs = filter(dir -> isdir(dir) && startswith(basename(dir), "local") , readdir(abspath("../data"); join=true))
# datadir = dirs[end]
datadir = "../data/current_free"
# datadir = "../data/ff3d_2762241"
# datadir = "data/local_2024_07_25_13_37_33"
@info "Using data in $datadir"

# Read subdirectories corresponding to different experiments
subdirs = sort(filter(isdir, readdir(datadir; join=true, sort=true)), lt=natural)

if endswith(subdirs[1], "checkpoints")
	
	losses = load(joinpath(datadir, "losses_vs_iterations.jld2"), "losses")
	
	# Plot all components of loss
	f1 = Figure()
	ax1 = Makie.Axis(f1[1, 1], xlabel="Iteration", ylabel="Loss", yscale=log10)
	labels=["Total" L"\hat{r}" L"\hat{θ}" L"\hat{ϕ}" L"∇⋅\textbf{B}" L"\textbf{B}⋅∇α"]
	linewidths = [5, 2, 2, 2, 2, 2]
	for (i, l) in enumerate(losses)
		lines!(ax1, l, label=string(labels[i]), linewidth=linewidths[i])
	end
	axislegend(ax1)
	display(GLMakie.Screen(), f1)

else

	# Load losses and parameter files for all experiments
	losses = Vector{Any}(undef, length(subdirs))
	params1 = Vector{Any}(undef, length(subdirs))
	labels = Vector{Any}(undef, length(subdirs))
	linestyles = Vector{Any}(undef, length(subdirs))
	for (i, subdir) in enumerate(subdirs)
		losses[i] = load(joinpath(subdir, "losses_vs_iterations.jld2"), "losses")
		params1[i] = import_params(joinpath(subdir, "config.toml"))

		# Create labels according to parameters that change between experiments
		labels[i] = params1[i].optimization.quasiNewton_method * "-" * "$(params1[i].optimization.loss_function)" * "-" * 
						"N$(params1[i].architecture.N_points)"
		if params1[i].optimization.quasiNewton_method == "BFGS"
			linestyles[i] = :solid
		elseif params1[i].optimization.quasiNewton_method == "SSBFGS"
			linestyles[i] = :dash
		else
			linestyles[i] = :dot
		end
	end

	# Plot losses of different experiments
	f1 = Figure()
	ax1 = Makie.Axis(f1[1, 1], xlabel="Iteration", ylabel="Loss", title="Comparison", yscale=log10)
	for (i, l) in enumerate(losses)
		# println(l[1])
		# if params1[i].optimization.quasiNewton_method == "SSBroyden"
			lines!(ax1, l[1], label=labels[i], linewidth=2, linestyle=linestyles[i])
		# end
	end
	axislegend(ax1)
	display(GLMakie.Screen(), f1)

	# Plot all components of loss for a particular experiment
	# number of experiment
	e = 24
	f2 = Figure()
	ax1 = Makie.Axis(f2[1, 1], xlabel="Iteration", ylabel="Loss", title="BFGS", yscale=log10)
	labels=["Total" L"\hat{r}" L"\hat{θ}" L"\hat{ϕ}" L"∇⋅\textbf{B}" L"\textbf{B}⋅∇α"]
	linewidths = [5, 2, 2, 2, 2, 2]
	for (i, l) in enumerate(losses[e])
		lines!(ax1, l, label=string(labels[i]), linewidth=linewidths[i])
	end
	axislegend(ax1)
	display(GLMakie.Screen(), f2)

	# Plot all components of loss for all experiments

	# for e in 1:length(subdirs)
	# 	f3 = Figure()
	# 	ax1 = Makie.Axis(f3[1, 1], xlabel="Iteration", ylabel="Loss", title="BFGS", yscale=log10)
	# 	labels=["Total" L"\hat{r}" L"\hat{θ}" L"\hat{ϕ}" L"∇⋅\textbf{B}" L"\textbf{B}⋅∇α"]
	# 	linewidths = [5, 2, 2, 2, 2, 2]
	# 	for (i, l) in enumerate(losses[e])
	# 		lines!(ax1, l, label=string(labels[i]), linewidth=linewidths[i])
	# 	end
	# 	axislegend(ax1)
	# 	display(GLMakie.Screen(), f3)
	# end

end