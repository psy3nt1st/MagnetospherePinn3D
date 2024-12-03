function Br_surface(μ)
	
	return @. 2 * μ
end

function α_surface(μ, ϕ, params)
	
	@unpack Pc, s, σ = params.model.Pc, params.model.s, params.model.sigma_gs
	return @. s * sigma * (1 - μ^2 - Pc)^(sigma-1) * ((1 - μ^2) > Pc)

   # α0, θ1, ϕ1, σ = param s.model.alpha0, params.model.theta1 * π / 180, params.model.phi1 * π / 180, params.model.sigma
	# return @. α0 * exp((-(acos(μ) - θ1)^2 - (ϕ - ϕ1)^2) / (2 * σ^2))
end