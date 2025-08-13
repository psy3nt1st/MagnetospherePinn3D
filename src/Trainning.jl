function create_subnet(N_input, N_neurons, N_layers, activation)
	
	return Chain(
				 Dense(N_input, N_neurons, activation),
				 [Dense(N_neurons, N_neurons, activation) for _ in 1:N_layers]...,
				 Dense(N_neurons, 1)
				)
end


function create_neural_network(config; test_mode=false)

	@unpack rng_seed, N_input, N_neurons, N_layers, N_output = config

	# Initialise random number generator
	rng = Random.default_rng()
	Random.TaskLocalRNG()
	if rng_seed === nothing
		Random.seed!(rng)
	elseif rng_seed isa Int
		Random.seed!(rng, rng_seed)
	else
		@warn "Invalid RNG seed: $(rng_seed). Setting seed to 0"
		Random.seed!(rng, 0)
	end
	
    # Create neural network. Separate subnetworks for each output
	subnetworks = [create_subnet(N_input, N_neurons, N_layers, tanh) for _ in 1:N_output]
    # NN = Chain(Parallel(vcat, subnetworks...))
    NN = Parallel(vcat, subnetworks...)
    Θ, st = Lux.setup(rng, NN)
    if test_mode
        Θ = Θ |> ComponentArray .|> Float64
    else
        Θ = Θ |> ComponentArray |> gpu_device() .|> Float64
    end
    
    return NN, Θ, st

end

function generate_input(config)

    @unpack q_distribution, N_points = config

    # Generate random values for `q`, `μ`, and `ϕ`
    q = rand(q_distribution, (1, N_points))
	μ = rand(Uniform(-1, 1), (1, N_points))
	ϕ = rand(Uniform(0, 2π), (1, N_points))

	input = vcat(q, μ, ϕ) |> gpu_device() .|> Float64

	return input
end

function calculate_individual_losses(q, μ, r_eq, θ_eq, ϕ_eq, ∇B, B∇α, B_mag, loss_normalization, N_points)

    # MSE loss
    if loss_normalization == "q"
        l1 = sum(abs2, r_eq ./ q .^ 4)
        l2 = sum(abs2, θ_eq ./ q .^ 4)
        l3 = sum(abs2, ϕ_eq ./ q .^ 4)
        l4 = sum(abs2, ∇B  .* .√(1 .- μ .^ 2) ./ q .^ 4)
        l5 = sum(abs2, B∇α .* .√(1 .- μ .^ 2))
    elseif loss_normalization == "B"
        l1 = sum(abs2, r_eq ./ B_mag)
        l2 = sum(abs2, θ_eq ./ B_mag)
        l3 = sum(abs2, ϕ_eq ./ B_mag)
        l4 = sum(abs2, ∇B .* .√(1 .- μ .^ 2) ./ B_mag)
        l5 = sum(abs2, B∇α .* .√(1 .- μ .^ 2))
    elseif loss_normalization == "none"
        l1 = sum(abs2, r_eq)
        l2 = sum(abs2, θ_eq)
        l3 = sum(abs2, ϕ_eq)
        l4 = sum(abs2, ∇B)
        l5 = sum(abs2, B∇α)
    end

    # Lmax norm
    if loss_normalization == "q"
        l1_max = maximum(r_eq ./ q .^ 4)
        l2_max = maximum(θ_eq ./ q .^ 4)
        l3_max = maximum(ϕ_eq ./ q .^ 4)
        l4_max = maximum(∇B  .* .√(1 .- μ .^ 2) ./ q .^ 4)
        l5_max = maximum(B∇α .* .√(1 .- μ .^ 2))
    elseif loss_normalization == "B"
        l1_max = maximum(r_eq ./ B_mag)
        l2_max = maximum(θ_eq ./ B_mag)
        l3_max = maximum(ϕ_eq ./ B_mag)
        l4_max = maximum(∇B .* .√(1 .- μ .^ 2) ./ B_mag)
        l5_max = maximum(B∇α .* .√(1 .- μ .^ 2))
    elseif loss_normalization == "none"
        l1_max = maximum(r_eq)
        l2_max = maximum(θ_eq)
        l3_max = maximum(ϕ_eq)
        l4_max = maximum(∇B)
        l5_max = maximum(B∇α)
    end

    return [l1, l2, l3, l4, l5] ./ N_points, [l1_max, l2_max, l3_max, l4_max, l5_max]

end

function loss_function(input, NN, Θ, st, temp_state, config)
	
    @unpack loss_normalization, loss_g, N_points = config

    # unpack input
	q = @view input[1:1,:]
	μ = @view input[2:2,:]
	ϕ = @view input[3:3,:]

	# Calculate derivatives
	dBr_dq, dBθ_dq, dBϕ_dq, dα_dq, 
    dBr_dμ, dBθ_dμ, dBϕ_dμ, dα_dμ, 
    dBr_dϕ, dBθ_dϕ, dBϕ_dϕ, dα_dϕ = calculate_derivatives(q, μ, ϕ, NN, Θ, st, config)
	
    Nr, Nθ, Nϕ, Nα = evaluate_subnetworks(q, μ, ϕ, NN, Θ, st)

	Br1 = Br(q, μ, ϕ, Nr, config)
	Bθ1 = Bθ(q, μ, ϕ, Nθ)
	Bϕ1 = Bϕ(q, μ, ϕ, Nϕ)
	α1 = α(q, μ, ϕ, Nα, config)

    B_mag = @. √(Br1 ^ 2 + Bθ1 ^ 2 + Bϕ1 ^ 2)

	r_eq = calculate_r_equation(q, μ, Br1, Bϕ1, α1, dBϕ_dμ, dBθ_dϕ, config)
	θ_eq = calculate_θ_equation(q, μ, Bθ1, Bϕ1, α1, dBϕ_dq, dBr_dϕ, config)
	ϕ_eq = calculate_ϕ_equation(q, μ, Bθ1, Bϕ1, α1, dBθ_dq, dBr_dμ, config)
	∇B = calculate_divergence(q, μ, Br1, Bθ1, dBr_dq, dBθ_dμ, dBϕ_dϕ, config)
	B∇α = calculate_Bdotgradα(q, μ, Br1, Bθ1, Bϕ1, dα_dq, dα_dμ, dα_dϕ, config) 

	# Calculate loss
    ls, ls_max = calculate_individual_losses(q, μ, r_eq, θ_eq, ϕ_eq, ∇B, B∇α, B_mag, loss_normalization, N_points)

    temp_state[:individual_losses] = ls
    temp_state[:Linf_norm] = sum(ls_max)
    temp_state[:individual_Linf_norms] = ls_max
    
    return loss_g(sum(ls))
end

function callback(p, l, prog, invH, temp_state, traindata, config)

    @unpack loss_g = config

	# Exponential of the loss function if log(MSE) is used
	if loss_g == log
		l = exp(l)
	end

    push!(traindata[:losses], l)
    push!(traindata[:individual_losses], temp_state[:individual_losses])
    push!(traindata[:Linf_norm], temp_state[:Linf_norm])
    push!(traindata[:individual_Linf_norms], temp_state[:individual_Linf_norms])

	# Store the last inverse Hessian (only in the quasi-Newton stage)
	if "~inv(H)" ∈ keys(p.original.metadata)
		invH[] = p.original.metadata["~inv(H)"]
	end

	# Update progress bar
	next!(prog, showvalues=[(:iteration, @sprintf("%d", length(traindata[:losses]))) (:Loss, @sprintf("%.4e", l))])
	flush(stdout)
	
	return false
end

function setup_optprob(NN, Θ, st, config)
    temp_state = OrderedDict{Symbol, Any}()

	input = generate_input(config)
	optf = Optimization.OptimizationFunction((Θ, input) -> loss_function(input, NN, Θ, st, temp_state, config), 
        Optimization.AutoZygote())
	optprob = Optimization.OptimizationProblem(optf, Θ, input)
	optresult = Optimization.solve(optprob, Adam(), maxiters = 1)

	return optresult, optprob, temp_state
end
 

function train_pinn!(optresult, optprob, temp_state, config)
    start_time = now()

    @unpack N_sets, adam_sets, adam_iters, quasiNewton_method, quasiNewton_iters, linesearch, subjobdir = config
    
    traindata = OrderedDict{Symbol, Any}()
    traindata[:losses] = Float64[]
    traindata[:individual_losses] = Vector{Float64}[]
    traindata[:Linf_norm] = Float64[]
    traindata[:individual_Linf_norms] = Vector{Float64}[]

    invH = Base.RefValue{AbstractArray{Float64, 2}}()

	# Initialise the inverse Hessian
	initial_invH = nothing   

	for i in 1:N_sets
		
		# Set up the optimizer. Adam for the initial stage, then switch to quasi-Newton
		if i <= adam_sets
			optimizer=OptimizationOptimJL.Adam()
			opt_label = "Adam"
			maxiters = adam_iters
		else
			if quasiNewton_method == "BFGS"
				optimizer=BFGS(linesearch=linesearch, initial_invH=initial_invH)
				opt_label = "BFGS"
			elseif quasiNewton_method == "SSBFGS"
				optimizer=SSBFGS(linesearch=linesearch, initial_invH=initial_invH)
				opt_label = "SSBFGS"
			elseif quasiNewton_method == "SSBroyden"
				optimizer=SSBroyden(linesearch=linesearch, initial_invH=initial_invH)
				opt_label = "SSBroyden"
			end

			maxiters = quasiNewton_iters
		end

		# Set up the progress bar
		prog = Progress(maxiters, desc="Set $i/$(N_sets) $opt_label", dt=0.1, showspeed=true, start=1) 
		
		# Train
		input = generate_input(config)
		optprob = remake(optprob, u0 = optresult.u, p = input)
		optresult = Optimization.solve(
            optprob,
            optimizer,
            callback = (p, l) -> callback(p, l, prog, invH, temp_state, traindata, config),
            maxiters = maxiters,
            extended_trace = true
        )
	
		finish!(prog)

		# Use the last inverse Hessian as the initial guess for the next set (only in the quasi-Newton stage)
		if i > adam_sets
			initial_invH = begin x -> invH[] end
		end

        duration = now()-start_time

        # Store results (losses and other data are stored inside the callback function)
        traindata[:Θ] = optresult.u |> Lux.cpu_device()
        traindata[:start_time] = start_time
        traindata[:duration] = duration / Millisecond(1000)
        traindata[:duration_readable] = Dates.format(convert(DateTime, duration), "HH:MM:SS")

        wsave(joinpath(subjobdir, "traindata.jld2"), "data", traindata)

	end
   
	return traindata
end

function main(config)

    @info "Creating neural network"
    NN, Θ, st = create_neural_network(config)

    @info "Setting up optimization problem"
    optresult, optprob, temp_state = setup_optprob(NN, Θ, st, config)

    @info "Training neural network"

    traindata = train_pinn!(optresult, optprob, temp_state, config)

    return traindata
end