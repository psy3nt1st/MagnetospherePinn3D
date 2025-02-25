const ϵ = ∛(eps()) 
const ϵ2 = ∜(eps())


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

function gaussian_hotspot(μ, ϕ, α0, θ1, ϕ1, σ)

    return @. α0 * exp((-(acos(μ) - θ1)^2 - (ϕ - ϕ1)^2) / (2 * σ^2))
end

function h_boundary(q, μ, ϕ, t, params)
    
    if params.model.alpha_bc_mode == "diffusive"

        return @. t^2 + 1 - q + ($Br_surface(μ, ϕ, params) * ($Br_surface(μ, ϕ, params) < 0)) ^ 2
    else

        return @. 1 - q + ($Br_surface(μ, ϕ, params) * ($Br_surface(μ, ϕ, params) < 0)) ^ 2
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
		
		return gaussian_hotspot(μ, ϕ, α0, θ1, ϕ1, σ)
	
	elseif params.model.alpha_bc_mode == "double-hotspot"

		α0, θ1, ϕ1, σ = params.model.alpha0, params.model.theta1 * π / 180, params.model.phi1 * π / 180, params.model.sigma
		α0_b, θ1_b, ϕ1_b, σ_b = params.model.alpha0_b, params.model.theta1_b * π / 180, params.model.phi1_b * π / 180, params.model.sigma_b

		return gaussian_hotspot(μ, ϕ, α0, θ1, ϕ1, σ) .+ gaussian_hotspot(μ, ϕ, α0_b, θ1_b, ϕ1_b, σ_b)
    
    elseif params.model.alpha_bc_mode == "diffusive"
        
        α0, θ1, ϕ1, σ = params.model.alpha0, params.model.theta1 * π / 180, params.model.phi1 * π / 180, params.model.sigma
		
        return gaussian_hotspot(μ, ϕ, α0, θ1, ϕ1, σ)

	else
				
		return @. 0
	end
end

function Br(q, μ, ϕ, Nr, params)
		
	return @. q^3 * ($Br_surface(μ, ϕ, params) + (1 - q) * Nr)
end

function Bθ(q, μ, ϕ, Nθ)
	
	return @. q^3 * Nθ
end

function Bϕ(q, μ, ϕ, Nϕ)

	return @. q^3 * Nϕ
end

function α(q, μ, ϕ, t, Nα, params)

    return @. q * ($α_surface(μ, ϕ, params) + $h_boundary(q, μ, ϕ, t, params) * Nα)
end

function evaluate_subnetworks(q, μ, ϕ, t, Θ, st, NN)

    subnet_r, subnet_θ, subnet_ϕ, subnet_α = NN.layers

    Nr = subnet_r(vcat(q, μ, cos.(ϕ), sin.(ϕ), t), Θ.layer_1, st.layer_1)[1]
    Nθ = subnet_θ(vcat(q, μ, cos.(ϕ), sin.(ϕ), t), Θ.layer_2, st.layer_2)[1]
    Nϕ = subnet_ϕ(vcat(q, μ, cos.(ϕ), sin.(ϕ), t), Θ.layer_3, st.layer_3)[1]
    Nα = subnet_α(vcat(q, μ, cos.(ϕ), sin.(ϕ), t), Θ.layer_4, st.layer_4)[1]

    return Nr, Nθ, Nϕ, Nα
    
end

function calculate_derivatives(q, μ, ϕ, t, q1, Θ, st, NN, params)

    Nr_qplus, Nθ_qplus, Nϕ_qplus, Nα_qplus = evaluate_subnetworks(q .+ ϵ, μ, ϕ, t, Θ, st, NN)
    Nr_qminus, Nθ_qminus, Nϕ_qminus, Nα_qminus = evaluate_subnetworks(q .- ϵ, μ, ϕ, t, Θ, st, NN)
    Nr_μplus, Nθ_μplus, Nϕ_μplus, Nα_μplus = evaluate_subnetworks(q, μ .+ ϵ, ϕ, t, Θ, st, NN)
    Nr_μminus, Nθ_μminus, Nϕ_μminus, Nα_μminus = evaluate_subnetworks(q, μ .- ϵ, ϕ, t, Θ, st, NN)
    Nr_ϕplus, Nθ_ϕplus, Nϕ_ϕplus, Nα_ϕplus = evaluate_subnetworks(q, μ, ϕ .+ ϵ, t, Θ, st, NN)
    Nr_ϕminus, Nθ_ϕminus, Nϕ_ϕminus, Nα_ϕminus = evaluate_subnetworks(q, μ, ϕ .- ϵ, t, Θ, st, NN)

    # Calculate derivative of α wrt to t only at the surface q1
    subnet_α = NN.layers[4]
    # Nα = subnet_α(vcat(q1, μ, cos.(ϕ), sin.(ϕ), t), Θ.layer_4, st.layer_4)[1]
    Nα_tplus = subnet_α(vcat(q1, μ, cos.(ϕ), sin.(ϕ), t .+ ϵ), Θ.layer_4, st.layer_4)[1]
    Nα_tminus = subnet_α(vcat(q1, μ, cos.(ϕ), sin.(ϕ), t .- ϵ), Θ.layer_4, st.layer_4)[1]


	return ((Br(q .+ ϵ, μ, ϕ, Nr_qplus, params) .- Br(q .- ϵ, μ, ϕ, Nr_qminus, params)) ./ (2 .* ϵ),
            (Bθ(q .+ ϵ, μ, ϕ, Nθ_qplus) .- Bθ(q .- ϵ, μ, ϕ, Nθ_qminus)) ./ (2 .* ϵ),
            (Bϕ(q .+ ϵ, μ, ϕ, Nϕ_qplus) .- Bϕ(q .- ϵ, μ, ϕ, Nϕ_qminus)) ./ (2 .* ϵ),
            (α(q .+ ϵ, μ, ϕ, t, Nα_qplus, params) .- α(q .- ϵ, μ, ϕ, t, Nα_qminus, params)) ./ (2 .* ϵ),
            (Br(q, μ .+ ϵ, ϕ, Nr_μplus, params) .- Br(q, μ .- ϵ, ϕ, Nr_μminus, params)) ./ (2 .* ϵ),
            (Bθ(q, μ .+ ϵ, ϕ, Nθ_μplus) .- Bθ(q, μ .- ϵ, ϕ, Nθ_μminus)) ./ (2 .* ϵ),
            (Bϕ(q, μ .+ ϵ, ϕ, Nϕ_μplus) .- Bϕ(q, μ .- ϵ, ϕ, Nϕ_μminus)) ./ (2 .* ϵ),
            (α(q, μ .+ ϵ, ϕ, t, Nα_μplus, params) .- α(q, μ .- ϵ, ϕ, t, Nα_μminus, params)) ./ (2 .* ϵ),
            (Br(q, μ, ϕ .+ ϵ, Nr_ϕplus, params) .- Br(q, μ, ϕ .- ϵ, Nr_ϕminus, params)) ./ (2 .* ϵ),
            (Bθ(q, μ, ϕ .+ ϵ, Nθ_ϕplus) .- Bθ(q, μ, ϕ .- ϵ, Nθ_ϕminus)) ./ (2 .* ϵ),
            (Bϕ(q, μ, ϕ .+ ϵ, Nϕ_ϕplus) .- Bϕ(q, μ, ϕ .- ϵ, Nϕ_ϕminus)) ./ (2 .* ϵ),
            (α(q, μ, ϕ .+ ϵ, t, Nα_ϕplus, params) .- α(q, μ, ϕ .- ϵ, t, Nα_ϕminus, params)) ./ (2 .* ϵ),
            (α(q1, μ, ϕ, t .+ ϵ, Nα_tplus, params) .- α(q1, μ, ϕ, t .- ϵ, Nα_tminus, params)) ./ (2 .* ϵ)
		   )
	
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

function calculate_αS_equation(μ, ϕ, t, q1, αS, dαS_dt)
 
	return @. dαS_dt + αS
end

