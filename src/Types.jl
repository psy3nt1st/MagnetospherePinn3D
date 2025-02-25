@option "architecture" mutable struct ArchitectureParams
	N_input::Int = 2
	N_layers::Int = 1
	N_neurons::Int = 20
	N_output::Int = 1
	activation::String = "tanh"
	N_points::Int = 1000
	rng_seed::Union{Nothing,Int} = 0
end

@option "optimization" mutable struct OptimizationParams
	# N_points::Int = 1000
	N_sets::Int = 5
	adam_sets::Int = 1
	adam_iters::Int = 500
	quasiNewton_method::String = "BFGS"
	quasiNewton_iters::Int = 500
	loss_function::String = "MSE"
end

@option "model" mutable struct ModelParams
	alpha_bc_mode::String = "hotspot"
	br_bc_mode::String = "axisymmetric"
	coef::Vector{Float64} = [1.0, 0.0, 0.0]
	alpha0::Float64 = 1.5
	theta1::Float64 = 45.0
	phi1::Float64 = 180.0
	sigma::Float64 = 1.0
	alpha0_b::Float64 = 1.5
	theta1_b::Float64 = 45.0   
	phi1_b::Float64 = 90.0
	sigma_b::Float64 = 0.2
	Pc::Float64 = 0.3405074
	s::Float64 = 2.0
	sigma_gs::Float64 = 2.0
	use_rc::Bool = false
	rc::Float64 = 3.35
	t_final::Float64 = 1e-3
end

@option "Params" mutable struct Params
	architecture::ArchitectureParams = ArchitectureParams()
	optimization::OptimizationParams = OptimizationParams()
	model::ModelParams = ModelParams()
end

# @with_kw mutable struct Variables{U<:AbstractArray} @deftype U
# 	q
# 	μ
# 	ϕ
# 	Br
# 	Bθ
# 	Bϕ
# 	α
# 	Nr
# 	Nθ
# 	Nϕ
# 	Nα
# 	Br_S
# 	α_S
# 	∇B
# 	B∇α
# 	∂Br∂q
# 	∂Bθ∂q
# 	∂Bϕ∂q
# 	∂α∂q
# 	∂Br∂μ
# 	∂Bθ∂μ
# 	∂Bϕ∂μ
# 	∂α∂μ
# 	∂Br∂ϕ
# 	∂Bθ∂ϕ
# 	∂Bϕ∂ϕ
# 	∂α∂ϕ
# end



