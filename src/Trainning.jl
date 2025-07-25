function create_subnet(N_input, N_neurons, N_layers, activation)
	
	return Chain(
				 Dense(N_input, N_neurons, activation),
				 [Dense(N_neurons, N_neurons, activation) for _ in 1:N_layers]...,
				 Dense(N_neurons, 1)
				)
end


function create_neural_network(params; test_mode=false)

	pa = params.architecture

	# Initialise random number generator
	rng = Random.default_rng()
	Random.TaskLocalRNG()
	if pa.rng_seed === nothing
		Random.seed!(rng)
	elseif pa.rng_seed isa Int
		Random.seed!(rng, pa.rng_seed)
	else
		@warn "Invalid RNG seed: $(pa.rng_seed). Setting seed to 0"
		Random.seed!(rng, 0)
	end
	
	# Set activation function
	if pa.activation == "tanh"
		activation = tanh
	else
		@warn "Invalid activation function: $(pa.activation). Using tanh."
		activation = tanh
	end
	
    # Create neural network. Separate subnetworks for each output
	subnetworks = [create_subnet(pa.N_input, pa.N_neurons, pa.N_layers, activation) for _ in 1:pa.N_output]
    NN = Chain(Parallel(vcat, subnetworks...))
    Θ, st = Lux.setup(rng, NN)
    if !test_mode
        Θ = Θ |> ComponentArray |> gpu_device() .|> Float64
    end
    
    return NN, Θ, st

end

function generate_input(params)

    if params.architecture.q_distributiion == "uniform"
        q_distributiion = Uniform(0, 1)
    elseif params.architecture.q_distributiion == "beta"
        q_distributiion = Beta(3, 1)
	end

    # Generate random values for `q`, `μ`, and `ϕ`
    q = rand(q_distributiion, (1, params.architecture.N_points))
	μ = rand(Uniform(-1, 1), (1, params.architecture.N_points))
	ϕ = rand(Uniform(0, 2π), (1, params.architecture.N_points))
    t = rand(Uniform(0, params.model.t_final), (1, params.architecture.N_points))

	input = vcat(q, μ, ϕ, t) |> gpu_device() .|> Float64

	return input
end

function loss_function(input, Θ, st, NN, params)
	# unpack input
	q = @view input[1:1,:]
	μ = @view input[2:2,:]
	ϕ = @view input[3:3,:]
    t = @view input[4:4,:]

	# Calculate derivatives
	dBr_dq, dBθ_dq, dBϕ_dq, dα_dq, 
    dBr_dμ, dBθ_dμ, dBϕ_dμ, dα_dμ, 
    dBr_dϕ, dBθ_dϕ, dBϕ_dϕ, dα_dϕ = calculate_derivatives(q, μ, ϕ, t, Θ, st, NN, params)
	
    Nr, Nθ, Nϕ, Nα = evaluate_subnetworks(q, μ, ϕ, t, Θ, st, NN)
    # subnet_α = NN.layers[4]
    # Nα_S = subnet_α(vcat(q1, μ, cos.(ϕ), sin.(ϕ), t), Θ.layer_4, st.layer_4)[1]

	Br1 = Br(q, μ, ϕ, Nr, params)
	Bθ1 = Bθ(q, μ, ϕ, Nθ)
	Bϕ1 = Bϕ(q, μ, ϕ, Nϕ)
	α1 = α(q, μ, ϕ, t, Nα, params)
    # αS = α(q1, μ, ϕ, t, Nα_S, params)

    B_mag = @. √(Br1 ^ 2 + Bθ1 ^ 2 + Bϕ1 ^ 2)

	r_eq = calculate_r_equation(q, μ, Br1, Bϕ1, α1, dBϕ_dμ, dBθ_dϕ, params)
	θ_eq = calculate_θ_equation(q, μ, Bθ1, Bϕ1, α1, dBϕ_dq, dBr_dϕ, params)
	ϕ_eq = calculate_ϕ_equation(q, μ, Bθ1, Bϕ1, α1, dBθ_dq, dBr_dμ, params)
	∇B = calculate_divergence(q, μ, Br1, Bθ1, dBr_dq, dBθ_dμ, dBϕ_dϕ, params)
	B∇α = calculate_Bdotgradα(q, μ, Br1, Bθ1, Bϕ1, dα_dq, dα_dμ, dα_dϕ, params) 

    # αS_eq = calculate_αS_equation(μ, ϕ, t, q1, αS, dαS_dt, d2αS_dq2, dαS_dμ, d2αS_dμ2, d2αS_dϕ2)

	# Calculate loss
    if params.optimization.loss_normalization == "q"
        l1 = sum(abs2, r_eq ./ q .^ 4)
        l2 = sum(abs2, θ_eq ./ q .^ 4)
        l3 = sum(abs2, ϕ_eq ./ q .^ 4)
        l4 = sum(abs2, ∇B  .* .√(1 .- μ .^ 2) ./ q .^ 4)
        l5 = sum(abs2, B∇α .* .√(1 .- μ .^ 2))
    elseif params.optimization.loss_normalization == "B"

        l1 = sum(abs2, r_eq ./ B_mag)
        l2 = sum(abs2, θ_eq ./ B_mag)
        l3 = sum(abs2, ϕ_eq ./ B_mag)
        l4 = sum(abs2, ∇B .* .√(1 .- μ .^ 2) ./ B_mag)
        l5 = sum(abs2, B∇α .* .√(1 .- μ .^ 2))

    elseif params.optimization.loss_normalization == "none"
        
        l1 = sum(abs2, r_eq)
        l2 = sum(abs2, θ_eq)
        l3 = sum(abs2, ϕ_eq)
        l4 = sum(abs2, ∇B)
        l5 = sum(abs2, B∇α)
    end

    # if params.model.alpha_bc_mode == "diffusive"
    #     l6 = sum(abs2, αS_eq)
    #     ls = [l1, l2, l3, l4, l5, l6] ./ params.architecture.N_points  
    # else
    ls = [l1, l2, l3, l4, l5] ./ params.architecture.N_points 
    # end

	if params.optimization.loss_function == "MSE"
		g = identity
	elseif params.optimization.loss_function == "logMSE"
		g = log
	else
		# @warn "Unrecognized loss function: $(params.optimization.loss_function). Using MSE."
		g = identity
	end

    # if params.model.alpha_bc_mode == "diffusive"
        # return g((l1 + l2 + l3 + l4 + l5 + l6) / params.architecture.N_points), ls
    # else
    return g((l1 + l2 + l3 + l4 + l5) / params.architecture.N_points), ls
    # end

end

function callback(p, l, ls, losses, prog, invH, params)

	# Exponential of the loss function if logMSE is used
	if params.optimization.loss_function == "logMSE"
		l = exp(l)
	end

	# Store loss history
	push!(losses[1], l)
	for i in eachindex(ls)
		push!(losses[i+1], ls[i])
	end

	# Store the last inverse Hessian (only in the quasi-Newton stage)
	if "~inv(H)" ∈ keys(p.original.metadata)
		invH[] = p.original.metadata["~inv(H)"]
	end

	# Update progress bar
	next!(prog, showvalues=[(:iteration, @sprintf("%d", length(losses[1]))) (:Loss, @sprintf("%.4e", l))])
	flush(stdout)
	
	return false
end

function setup_optprob(Θ, st, NN, params)

	input = generate_input(params)
	optf = Optimization.OptimizationFunction((Θ, input) -> loss_function(input, Θ, st, NN, params), Optimization.AutoZygote())
	optprob = Optimization.OptimizationProblem(optf, Θ, input)
	result = Optimization.solve(optprob, Adam(), maxiters = 1)

	return optprob, result
end
 

function train!(result, optprob, losses, invH, job_dir, params)

	# Initialise the inverse Hessian
	initial_invH = nothing   

    # Select linesearch method
    if params.optimization.linesearch == "HagerZhang"
        linesearch = LineSearches.HagerZhang()
    elseif params.optimization.linesearch == "MoreThuente"
        linesearch = LineSearches.MoreThuente()
    elseif params.optimization.linesearch == "BackTracking"
        linesearch = LineSearches.BackTracking()
    elseif params.optimization.linesearch == "StrongWolfe"
        linesearch = LineSearches.StrongWolfe()
    end

	for i in 1:params.optimization.N_sets
		
		# Set up the optimizer. Adam for the initial stage, then switch to quasi-Newton
		if i <= params.optimization.adam_sets
			optimizer=OptimizationOptimJL.Adam()
			opt_label = "Adam"
			maxiters = params.optimization.adam_iters
		else
			if params.optimization.quasiNewton_method == "BFGS"
				optimizer=BFGS(linesearch=linesearch, initial_invH=initial_invH)
				opt_label = "BFGS"
			elseif params.optimization.quasiNewton_method == "SSBFGS"
				optimizer=SSBFGS(linesearch=linesearch, initial_invH=initial_invH)
				opt_label = "SSBFGS"
			elseif params.optimization.quasiNewton_method == "SSBroyden"
				optimizer=SSBroyden(linesearch=linesearch, initial_invH=initial_invH)
				opt_label = "SSBroyden"
			end

			maxiters = params.optimization.quasiNewton_iters
		end

		# Set up the progress bar
		prog = Progress(maxiters, desc="Set $i/$(params.optimization.N_sets) $opt_label", dt=0.1, showspeed=true, start=1) 
		
		# Train
		input = generate_input(params)
		optprob = remake(optprob, u0 = result.u, p = input)
		result = Optimization.solve(
            optprob,
            optimizer,
            callback = (p, l, ls) -> callback(p, l, ls, losses, prog, invH, params),
            maxiters = maxiters,
            extended_trace = true
        )
	
		finish!(prog)

		# Use the last inverse Hessian as the initial guess for the next set (only in the quasi-Newton stage)
		if i > params.optimization.adam_sets
			initial_invH = begin x -> invH[] end
		end

        # Save check point
        Θ_trained = result.u |> Lux.cpu_device()
        
        @save joinpath(job_dir, "trained_model.jld2") Θ_trained
        @save joinpath(job_dir, "losses_vs_iterations.jld2") losses

        checkpoints_dir = mkpath(joinpath(job_dir, "checkpoints"))
        
        @save joinpath(checkpoints_dir, "trained_model_$i.jld2") Θ_trained
        @save joinpath(checkpoints_dir, "losses_vs_iterations_$i.jld2") losses
	end
   
	return result
end

function train_neural_network!(result, optprob, losses, invH, job_dir, params)

		
    result = train!(result, optprob, losses, invH, job_dir, params)

   	return result
end