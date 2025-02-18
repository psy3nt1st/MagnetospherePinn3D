

function setup_jobdir()
   
    if "SLURM_JOB_NAME" in keys(ENV)
        job_name = ENV["SLURM_JOB_NAME"]
        if "SLURM_ARRAY_JOB_ID" in keys(ENV)
            job_id = ENV["SLURM_ARRAY_JOB_ID"]
            task_id = ENV["SLURM_ARRAY_TASK_ID"]
            job_dir = joinpath("data", "$(job_name)_$(job_id)", "$(task_id)")
            mkpath(job_dir)
            cp("config_template.toml", joinpath(job_dir, "config_template.toml"))
        else
            job_id = ENV["SLURM_JOB_ID"]
            job_dir = joinpath("data", "$(job_name)_$(job_id)")
            mkpath(job_dir)
            cp("config_template.toml", joinpath(job_dir, "config_template.toml"))
        end
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

function create_subnet(N_input, N_neurons, N_layers, activation)
	
	return Chain(
				 Dense(N_input, N_neurons, activation),
				 [Dense(N_neurons, N_neurons, activation) for _ in 1:N_layers]...,
				 Dense(N_neurons, 1)
				)
end


function create_neural_network(params)

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
    Θ = Θ |> ComponentArray |> gpu_device() .|> Float64
    
    return NN, Θ, st

end

function generate_input(params)

	# Generate random values for `q`, `μ`, and `ϕ`
	q = rand(Uniform(0.05, 1), (1, params.architecture.N_points))# |> gpu_device() .|> Float64
	μ = rand(Uniform(-1, 1), (1, params.architecture.N_points))# |> gpu_device() .|> Float64
	ϕ = rand(Uniform(0, 2π), (1, params.architecture.N_points))# |> gpu_device() .|> Float64
	
	input = vcat(q, μ, ϕ, t) |> gpu_device() .|> Float64

	return input
end
 
# function P_surface(μ)

# 	return @. (1 - μ^2) * (coef[1] * 1 + coef[2] * 3 * μ  + coef[3] * (15 * μ^2 - 3) / 2)
# end

function Br_surface(μ, ϕ, params)
	
	@unpack coef = params.model

	if hasproperty(params.model, :br_bc_mode)
		
		if params.model.br_bc_mode == "axisymmetric"
			
			return @. coef[1] * 2 * μ + coef[2] * (3 * μ^2 - 1) + coef[3] * (5 * μ^3 - 3 * μ)
		
		elseif params.model.br_bc_mode == "non axisymmetric"
			
			return @. 0.5 * coef[1] * 2 * μ - 0.5 * coef[1] * 2 * sin(μ) * cos(ϕ)
		end

	else
		
		return @. coef[1] * 2 * μ + coef[2] * (3 * μ^2 - 1) + coef[3] * (5 * μ^3 - 3 * μ)
			
	end
end

function α_surface(μ, ϕ, params)
	
	if params.model.alpha_bc_mode == "axisymmetric"
		
		Pc, rc, s, σ = params.model.Pc, params.model.rc, params.model.s, params.model.sigma_gs

		if params.model.use_rc
			Pc = @. 1 / rc
		end

		return @. s * σ * (1 - μ^2 - Pc)^(σ-1) * ((1 - μ^2) > Pc)
	
	elseif params.model.alpha_bc_mode == "hotspot"
		
		α0, θ1, ϕ1, σ = params.model.alpha0, params.model.theta1 * π / 180, params.model.phi1 * π / 180, params.model.sigma
		
		return @. α0 * exp((-(acos(μ) - θ1)^2 - (ϕ - ϕ1)^2) / (2 * σ^2))
	
	elseif params.model.alpha_bc_mode == "double-hotspot"

		α0, θ1, ϕ1, σ = params.model.alpha0, params.model.theta1 * π / 180, params.model.phi1 * π / 180, params.model.sigma
		α0_b, θ1_b, ϕ1_b, σ_b = params.model.alpha0_b, params.model.theta1_b * π / 180, params.model.phi1_b * π / 180, params.model.sigma_b

		return @. α0 * exp((-(acos(μ) - θ1)^2 - (ϕ - ϕ1)^2) / (2 * σ^2)) + α0_b * exp((-(acos(μ) - θ1_b)^2 - (ϕ - ϕ1_b)^2) / (2 * σ_b^2))
    
    elseif params.model.alpha_bc_mode == "diffusive"


	else
		# @warn "Invalid alpha_bc_mode: $(params.model.alpha_bc_mode). Using zero α"
		
		return @. 0
	end
end

function Br(q, μ, ϕ, Θ, st, Nr, params)
		
	return @. q^3 * $Br_surface(μ, ϕ, params) + q^3 * (1 - q) * Nr
end

function Bθ(q, μ, ϕ, Θ, st, Nθ)
	
	return @. q^3 * Nθ
end

function Bϕ(q, μ, ϕ, Θ, st, Nϕ)

	return @. q^3 * Nϕ
end

function α(q, μ, ϕ, Θ, st, Nα, params)

	return @. q * $α_surface(μ, ϕ, params) + q * ((1 - q) + ($Br_surface(μ, ϕ, params) * ($Br_surface(μ, ϕ, params) < 0)) ^ 2) * Nα
end

function Bmag(q, μ, ϕ, Θ_trained, st, Nr, Nθ, Nϕ, params)
	
	return .√(Br(q, μ, ϕ, Θ_trained, st, Nr, params)[1].^2 .+ Bθ(q, μ, ϕ, Θ_trained, st, Nθ)[1].^2 .+ Bϕ(q, μ, ϕ, Θ_trained, st, Nϕ)[1].^2)
end

const ϵ = ∛(eps()) 
# const ϵ2 = ∜(eps())

function evaluate_subnetworks(q, μ, ϕ, Θ, st, NN, )

    subnet_r, subnet_θ, subnet_ϕ, subnet_α = NN.layers

    Nr = subnet_r(vcat(q, μ, cos.(ϕ), sin.(ϕ)), Θ.layer_1, st.layer_1)[1]
    Nθ = subnet_θ(vcat(q, μ, cos.(ϕ), sin.(ϕ)), Θ.layer_2, st.layer_2)[1]
    Nϕ = subnet_ϕ(vcat(q, μ, cos.(ϕ), sin.(ϕ)), Θ.layer_3, st.layer_3)[1]
    Nα = subnet_α(vcat(q, μ, cos.(ϕ), sin.(ϕ)), Θ.layer_4, st.layer_4)[1]

    return Nr, Nθ, Nϕ, Nα
    
end

function calculate_derivatives(q, μ, ϕ, Θ, st, NN, params)

    Nr, Nθ, Nϕ, Nα = evaluate_subnetworks(q, μ, ϕ, Θ, st, NN)
    Nr_qplus, Nθ_qplus, Nϕ_qplus, Nα_qplus = evaluate_subnetworks(q .+ ϵ, μ, ϕ, Θ, st, NN)
    Nr_qminus, Nθ_qminus, Nϕ_qminus, Nα_qminus = evaluate_subnetworks(q .- ϵ, μ, ϕ, Θ, st, NN)
    Nr_μplus, Nθ_μplus, Nϕ_μplus, Nα_μplus = evaluate_subnetworks(q, μ .+ ϵ, ϕ, Θ, st, NN)
    Nr_μminus, Nθ_μminus, Nϕ_μminus, Nα_μminus = evaluate_subnetworks(q, μ .- ϵ, ϕ, Θ, st, NN)
    Nr_ϕplus, Nθ_ϕplus, Nϕ_ϕplus, Nα_ϕplus = evaluate_subnetworks(q, μ, ϕ .+ ϵ, Θ, st, NN)
    Nr_ϕminus, Nθ_ϕminus, Nϕ_ϕminus, Nα_ϕminus = evaluate_subnetworks(q, μ, ϕ .- ϵ, Θ, st, NN)

	return ((Br(q .+ ϵ, μ, ϕ, Θ, st, Nr_qplus, params) .- Br(q .- ϵ, μ, ϕ, Θ, st, Nr_qminus, params)) ./ (2 .* ϵ),
            (Bθ(q .+ ϵ, μ, ϕ, Θ, st, Nθ_qplus) .- Bθ(q .- ϵ, μ, ϕ, Θ, st, Nθ_qminus)) ./ (2 .* ϵ),
            (Bϕ(q .+ ϵ, μ, ϕ, Θ, st, Nϕ_qplus) .- Bϕ(q .- ϵ, μ, ϕ, Θ, st, Nϕ_qminus)) ./ (2 .* ϵ),
            (α(q .+ ϵ, μ, ϕ, Θ, st, Nα_qplus, params) .- α(q .- ϵ, μ, ϕ, Θ, st, Nα_qminus, params)) ./ (2 .* ϵ),
            (Br(q, μ .+ ϵ, ϕ, Θ, st, Nr_μplus, params) .- Br(q, μ .- ϵ, ϕ, Θ, st, Nr_μminus, params)) ./ (2 .* ϵ),
            (Bθ(q, μ .+ ϵ, ϕ, Θ, st, Nθ_μplus) .- Bθ(q, μ .- ϵ, ϕ, Θ, st, Nθ_μminus)) ./ (2 .* ϵ),
            (Bϕ(q, μ .+ ϵ, ϕ, Θ, st, Nϕ_μplus) .- Bϕ(q, μ .- ϵ, ϕ, Θ, st, Nϕ_μminus)) ./ (2 .* ϵ),
            (α(q, μ .+ ϵ, ϕ, Θ, st, Nα_μplus, params) .- α(q, μ .- ϵ, ϕ, Θ, st, Nα_μminus, params)) ./ (2 .* ϵ),
            (Br(q, μ, ϕ .+ ϵ, Θ, st, Nr_ϕplus, params) .- Br(q, μ, ϕ .- ϵ, Θ, st, Nr_ϕminus, params)) ./ (2 .* ϵ),
            (Bθ(q, μ, ϕ .+ ϵ, Θ, st, Nθ_ϕplus) .- Bθ(q, μ, ϕ .- ϵ, Θ, st, Nθ_ϕminus)) ./ (2 .* ϵ),
            (Bϕ(q, μ, ϕ .+ ϵ, Θ, st, Nϕ_ϕplus) .- Bϕ(q, μ, ϕ .- ϵ, Θ, st, Nϕ_ϕminus)) ./ (2 .* ϵ),
            (α(q, μ, ϕ .+ ϵ, Θ, st, Nα_ϕplus, params) .- α(q, μ, ϕ .- ϵ, Θ, st, Nα_ϕminus, params)) ./ (2 .* ϵ),
            (α(q .+ ϵ, μ, ϕ, Θ, st, Nα_qplus, params) .-2 .* α(q, μ, ϕ, Θ, st, Nα, params) .+ α(q .- ϵ, μ, ϕ, Θ, st, Nα_qminus, params)) ./ ϵ .^ 2,
            (α(q, μ .+ ϵ, ϕ, Θ, st, Nα_μplus, params) .-2 .* α(q, μ, ϕ, Θ, st, Nα, params) .+ α(q, μ .- ϵ, ϕ, Θ, st, Nα_μminus, params)) ./ ϵ .^ 2,
            (α(q, μ, ϕ .+ ϵ, Θ, st, Nα_ϕplus, params) .-2 .* α(q, μ, ϕ, Θ, st, Nα, params) .+ α(q, μ, ϕ .- ϵ, Θ, st, Nα_ϕminus, params)) ./ ϵ .^ 2
		   )
	
end

function grad(q, μ, df_dq, df_dμ, df_dϕ)

	return @. -q^2 * df_dq, -q * √(1 - μ^2) * df_dμ, -q / √(1 - μ^2) * df_dϕ
end

function diver(q, μ, Br, Bθ, dBr_dq, dBθ_dμ, dBϕ_dϕ)

	return @. 2 * q * Br - q^2 * dBr_dq - q * √(1 - μ^2) * dBθ_dμ + q * μ / √(1 - μ^2) * Bθ + q / √(1 - μ^2) * dBϕ_dϕ	
end

function curl(q, μ, Bθ, Bϕ, dBr_dμ, dBθ_dq, dBϕ_dμ)
	
	return @. q * μ / √(1 - μ^2) * Bϕ - q * √(1 - μ^2) * dBϕ_dμ - q / √(1 - μ^2) * dBθ_dϕ, q * Bθ - q^2 * dBθ_dq + q * √(1 - μ^2) * dBr_dμ, q * Bθ - q^2 * dBθ_dq + q * √(1 - μ^2) * dBr_dμ
end

function laplacian(q, μ, df_dμ, d2f_dq2, d2f_dμ2, d2f_dϕ2)

	return @. q^4 * d2f_dq2 - 2 * q^2 * μ * df_dμ + q^2 * (1 - μ^2) * d2f_dμ2 + q^2 / (1 - μ^2) * d2f_dϕ2
end

function scalar_product(v1, v2, v3, u1, u2, u3)

	return @. v1 * u1 + v2 * u2 + v3 * u3
end

function calculate_r_equation(q, μ, ϕ, Br, Bθ, Bϕ, α, dBr_dq, dBθ_dq, dBϕ_dq, dBr_dμ, dBθ_dμ, dBϕ_dμ, dBr_dϕ, dBθ_dϕ, dBϕ_dϕ)
	
	return @. q * μ / √(1 - μ^2) * Bϕ - q * √(1 - μ^2) * dBϕ_dμ - q / √(1 - μ^2) * dBθ_dϕ - α * Br
end

function calculate_θ_equation(q, μ, ϕ, Br, Bθ, Bϕ, α, dBr_dq, dBθ_dq, dBϕ_dq, dBr_dμ, dBθ_dμ, dBϕ_dμ, dBr_dϕ, dBθ_dϕ, dBϕ_dϕ)

	return @. q / √(1 - μ^2) * dBr_dϕ - q * Bϕ + q^2 * dBϕ_dq - α * Bθ
end

function calculate_ϕ_equation(q, μ, ϕ, Br, Bθ, Bϕ, α, dBr_dq, dBθ_dq, dBϕ_dq, dBr_dμ, dBθ_dμ, dBϕ_dμ, dBr_dϕ, dBθ_dϕ, dBϕ_dϕ)
	
	return @. q * Bθ - q^2 * dBθ_dq + q * √(1 - μ^2) * dBr_dμ - α * Bϕ
end

function calculate_divergence(q, μ, ϕ,	Br, Bθ, Bϕ, dBr_dq, dBθ_dq, dBϕ_dq, dBr_dμ, dBθ_dμ, dBϕ_dμ, dBr_dϕ, dBθ_dϕ, dBϕ_dϕ)

	return @. 2 * q * Br - q^2 * dBr_dq - q * √(1 - μ^2) * dBθ_dμ + q * μ / √(1 - μ^2) * Bθ + q / √(1 - μ^2) * dBϕ_dϕ
end

function calculate_Bdotgradα(q, μ, ϕ, Br, Bθ, Bϕ, dα_dq, dα_dμ, dα_dϕ)

	return @. -q^2 * Br * dα_dq - q * √(1 - μ^2) * Bθ * dα_dμ + q / √(1 - μ^2) * Bϕ * dα_dϕ
end

function calculate_gradB2(q, μ, Br, Bθ, Bϕ, dBr_dq, dBθ_dq, dBϕ_dq, dBr_dμ, dBθ_dμ, dBϕ_dμ, dBr_dϕ, dBθ_dϕ, dBϕ_dϕ)

	return @. (-q^2 * (Br * dBr_dq + Bθ * dBθ_dq + Bϕ * dBϕ_dq), 
				 -q * √(1 - μ^2) * (Br * dBr_dμ + Bθ * dBθ_dμ + Bϕ * dBϕ_dμ), 
				  q / √(1 - μ^2) * (Br * dBr_dϕ + Bθ * dBθ_dϕ + Bϕ * dBϕ_dϕ))
end

function calclulate_∂α_∂t(q, μ, ϕ, NN, Θ, st, ϵ)
	NN = pinn(vcat(q, μ, cos.(ϕ), sin.(ϕ)), Θ_trained, st)[1]
	Nr = reshape(NN[1, :], size(q))
	Nθ = reshape(NN[2, :], size(q))
	Nϕ = reshape(NN[3, :], size(q))
	Nα = reshape(NN[4, :], size(q))

	Br1 = Br(q, μ, ϕ, Θ_trained, st, Nr, params)
	Bθ1 = Bθ(q, μ, ϕ, Θ_trained, st, Nθ)
	Bϕ1 = Bϕ(q, μ, ϕ, Θ_trained, st, Nϕ)
	α1 = α(q, μ, ϕ, Θ_trained, st, Nα, params)
	dBr_dq, dBθ_dq, dBϕ_dq, dα_dq, dBr_dμ, dBθ_dμ, dBϕ_dμ, dα_dμ, dBr_dϕ, dBθ_dϕ, dBϕ_dϕ, dα_dϕ, d2α_dq2, d2α_dμ2, d2α_dϕ2 = calculate_derivatives(q, μ, ϕ, Θ, st, NN, ϵ)

	∇2α = laplacian(q, μ, dα_dμ, d2α_dq2, d2α_dμ2, d2α_dϕ2)
	∇α = grad(q, μ, dα_dq, dα_dμ, dα_dϕ)
	∇B2 = calculate_gradB2(q, μ, Br1, Bθ1, Bϕ1, dBr_dq, dBθ_dq, dBϕ_dq, dBr_dμ, dBθ_dμ, dBϕ_dμ, dBr_dϕ, dBθ_dϕ, dBϕ_dϕ)
	B2 = Br1^2 + Bθ1^2 + Bϕ1^2

	da_dt = ∇2α + scalar_product(∇α..., ∇B2...) / B2

	return 
end


function loss_function(input, Θ, st, NN, params)
	# unpack input
	q = @view input[1:1,:]
	μ = @view input[2:2,:]
	ϕ = @view input[3:3,:]

	# Calculate derivatives
	dBr_dq, dBθ_dq, dBϕ_dq, dα_dq, dBr_dμ, dBθ_dμ, dBϕ_dμ, dα_dμ, dBr_dϕ, dBθ_dϕ, dBϕ_dϕ, dα_dϕ  = calculate_derivatives(q, μ, ϕ, Θ, st, NN, params)[1:12]
	
    Nr, Nθ, Nϕ, Nα = evaluate_subnetworks(q, μ, ϕ, Θ, st, NN)

	Br1 = Br(q, μ, ϕ, Θ, st, Nr, params)
	Bθ1 = Bθ(q, μ, ϕ, Θ, st, Nθ)
	Bϕ1 = Bϕ(q, μ, ϕ, Θ, st, Nϕ)
	α1 = α(q, μ, ϕ, Θ, st, Nα, params)

	r_eq = calculate_r_equation(q, μ, ϕ, Br1, Bθ1, Bϕ1, α1, dBr_dq, dBθ_dq, dBϕ_dq, dBr_dμ, dBθ_dμ, dBϕ_dμ, dBr_dϕ, dBθ_dϕ, dBϕ_dϕ)
	θ_eq = calculate_θ_equation(q, μ, ϕ, Br1, Bθ1, Bϕ1, α1, dBr_dq, dBθ_dq, dBϕ_dq, dBr_dμ, dBθ_dμ, dBϕ_dμ, dBr_dϕ, dBθ_dϕ, dBϕ_dϕ)
	ϕ_eq = calculate_ϕ_equation(q, μ, ϕ, Br1, Bθ1, Bϕ1, α1, dBr_dq, dBθ_dq, dBϕ_dq, dBr_dμ, dBθ_dμ, dBϕ_dμ, dBr_dϕ, dBθ_dϕ, dBϕ_dϕ)

	∇B = calculate_divergence(q, μ, ϕ, Br1, Bθ1, Bϕ1, dBr_dq, dBθ_dq, dBϕ_dq, dBr_dμ, dBθ_dμ, dBϕ_dμ, dBr_dϕ, dBθ_dϕ, dBϕ_dϕ)
	B∇α = calculate_Bdotgradα(q, μ, ϕ, Br1, Bθ1, Bϕ1, dα_dq, dα_dμ, dα_dϕ) 

	# Calculate loss
	l1 = sum(abs2, r_eq ./ q .^ 4)
	l2 = sum(abs2, θ_eq ./ q .^ 4)
	l3 = sum(abs2, ϕ_eq ./ q .^ 4)
	l4 = sum(abs2, .√(1 .- μ .^ 2) .* ∇B ./ q .^ 4)
	l5 = sum(abs2, B∇α)
	ls = [l1, l2, l3, l4, l5] ./ params.architecture.N_points 

	if params.optimization.loss_function == "MSE"
		g = identity
	elseif params.optimization.loss_function == "logMSE"
		g = log
	else
		# @warn "Unrecognized loss function: $(params.optimization.loss_function). Using MSE."
		g = identity
	end

	return g((l1 + l2 + l3 + l4 + l5) / params.architecture.N_points), ls

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

	for i in 1:params.optimization.N_sets
		
		# Set up the optimizer. Adam for the initial stage, then switch to quasi-Newton
		if i <= params.optimization.adam_sets
			optimizer=OptimizationOptimJL.Adam()
			opt_label = "Adam"
			maxiters = params.optimization.adam_iters
		else
			if params.optimization.quasiNewton_method == "BFGS"
				optimizer=BFGS(linesearch=LineSearches.MoreThuente(), initial_invH=initial_invH)
				opt_label = "BFGS"
			elseif params.optimization.quasiNewton_method == "SSBFGS"
				optimizer=SSBFGS(linesearch=LineSearches.MoreThuente(), initial_invH=initial_invH)
				opt_label = "SSBFGS"
			elseif params.optimization.quasiNewton_method == "SSBroyden"
				optimizer=SSBroyden(linesearch=LineSearches.MoreThuente(), initial_invH=initial_invH)
				opt_label = "SSBroyden"
			end

			maxiters = params.optimization.quasiNewton_iters
		end

		# println(i, optimizer)
		# Set up the progress bar
		prog = Progress(maxiters, desc="$opt_label set $i / $(params.optimization.N_sets)", dt=0.1, showspeed=true, start=1) 
		
		# Train
		input = generate_input(params)
		optprob = remake(optprob, u0 = result.u, p = input)
		result = Optimization.solve(optprob,
                                    optimizer,
                                    callback = (p, l, ls) -> callback(p, l, ls, losses, prog, invH, params),
                                    maxiters = maxiters,
                                    # store_trace = true,
                                    extended_trace = true
                                   )
	
		finish!(prog)

		# Use the last inverse Hessian as the initial guess for the next set (only in the quasi-Newton stage)
		if i > params.optimization.adam_sets
			initial_invH = begin x -> invH[] end
		end

        # Save check point
        Θ_trained = result.u |> Lux.cpu_device()
        
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