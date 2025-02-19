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

function α_surface(μ, ϕ, t, Nα_S, params)
	
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
        
        α0, θ1, ϕ1, σ = params.model.alpha0, params.model.theta1 * π / 180, params.model.phi1 * π / 180, params.model.sigma
		
        α_S0 = @. α0 * exp((-(acos(μ) - θ1)^2 - (ϕ - ϕ1)^2) / (2 * σ^2))

        return @. α_S0 + t * Nα_S

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

function α(q, μ, ϕ, t, Θ, st, Nα, Nα_S, params)

	return @. q * $α_surface(μ, ϕ, t, Nα_S, params) + q * ((1 - q) + ($Br_surface(μ, ϕ, params) * ($Br_surface(μ, ϕ, params) < 0)) ^ 2) * Nα
end

function Bmag(q, μ, ϕ, Θ_trained, st, Nr, Nθ, Nϕ, params)
	
	return .√(Br(q, μ, ϕ, Θ_trained, st, Nr, params)[1].^2 .+ Bθ(q, μ, ϕ, Θ_trained, st, Nθ)[1].^2 .+ Bϕ(q, μ, ϕ, Θ_trained, st, Nϕ)[1].^2)
end


function evaluate_subnetworks(q, μ, ϕ, t, Θ, st, NN, )

    subnet_r, subnet_θ, subnet_ϕ, subnet_α, subnet_αS = NN.layers

    Nr = subnet_r(vcat(q, μ, cos.(ϕ), sin.(ϕ), t), Θ.layer_1, st.layer_1)[1]
    Nθ = subnet_θ(vcat(q, μ, cos.(ϕ), sin.(ϕ), t), Θ.layer_2, st.layer_2)[1]
    Nϕ = subnet_ϕ(vcat(q, μ, cos.(ϕ), sin.(ϕ), t), Θ.layer_3, st.layer_3)[1]
    Nα = subnet_α(vcat(q, μ, cos.(ϕ), sin.(ϕ), t), Θ.layer_4, st.layer_4)[1]
    Nα_S = subnet_αS(vcat(μ, cos.(ϕ), sin.(ϕ), t), Θ.layer_5, st.layer_5)[1]

    return Nr, Nθ, Nϕ, Nα, Nα_S
    
end

function calculate_derivatives(q, μ, ϕ, t, Θ, st, NN, params)

    println("q:", typeof(q), size(q), eltype(q))

    # q2 = CUDA.ones(eltype(q), size(q))

    # println("q2:", typeof(q2), size(q2), eltype(q2))


    q1 = similar(q) 
    fill!(q1, one(eltype(q)))

    println("q1:", typeof(q1), size(q1), eltype(q1))

    Nr, Nθ, Nϕ, Nα, Nα_S = evaluate_subnetworks(q, μ, ϕ, t, Θ, st, NN)
    Nr_qplus, Nθ_qplus, Nϕ_qplus, Nα_qplus, Nα_S_qplus = evaluate_subnetworks(q .+ ϵ, μ, ϕ, t, Θ, st, NN)
    Nr_qminus, Nθ_qminus, Nϕ_qminus, Nα_qminus, Nα_S_qminus = evaluate_subnetworks(q .- ϵ, μ, ϕ, t, Θ, st, NN)
    Nr_μplus, Nθ_μplus, Nϕ_μplus, Nα_μplus, Nα_S_μplus = evaluate_subnetworks(q, μ .+ ϵ, ϕ, t, Θ, st, NN)
    Nr_μminus, Nθ_μminus, Nϕ_μminus, Nα_μminus, Nα_S_μminus = evaluate_subnetworks(q, μ .- ϵ, ϕ, t, Θ, st, NN)
    Nr_ϕplus, Nθ_ϕplus, Nϕ_ϕplus, Nα_ϕplus, Nα_S_ϕplus = evaluate_subnetworks(q, μ, ϕ .+ ϵ, t, Θ, st, NN)
    Nr_ϕminus, Nθ_ϕminus, Nϕ_ϕminus, Nα_ϕminus, Nα_S_ϕminus = evaluate_subnetworks(q, μ, ϕ .- ϵ, t, Θ, st, NN)
    Nr_tplus, Nθ_tplus, Nϕ_tplus, Nα_tplus, Nα_S_tplus = evaluate_subnetworks(q, μ, ϕ, t .+ ϵ, Θ, st, NN)
    Nr_tminus, Nθ_tminus, Nϕ_tminus, Nα_tminus, Nα_S_tminus = evaluate_subnetworks(q, μ, ϕ, t .- ϵ, Θ, st, NN)

	return ((Br(q .+ ϵ, μ, ϕ, Θ, st, Nr_qplus, params) .- Br(q .- ϵ, μ, ϕ, Θ, st, Nr_qminus, params)) ./ (2 .* ϵ),
            (Bθ(q .+ ϵ, μ, ϕ, Θ, st, Nθ_qplus) .- Bθ(q .- ϵ, μ, ϕ, Θ, st, Nθ_qminus)) ./ (2 .* ϵ),
            (Bϕ(q .+ ϵ, μ, ϕ, Θ, st, Nϕ_qplus) .- Bϕ(q .- ϵ, μ, ϕ, Θ, st, Nϕ_qminus)) ./ (2 .* ϵ),
            (α(q .+ ϵ, μ, ϕ, t, Θ, st, Nα_qplus, Nα_S, params) .- α(q .- ϵ, μ, ϕ, t, Θ, st, Nα_qminus, Nα_S , params)) ./ (2 .* ϵ),
            (Br(q, μ .+ ϵ, ϕ, Θ, st, Nr_μplus, params) .- Br(q, μ .- ϵ, ϕ, Θ, st, Nr_μminus, params)) ./ (2 .* ϵ),
            (Bθ(q, μ .+ ϵ, ϕ, Θ, st, Nθ_μplus) .- Bθ(q, μ .- ϵ, ϕ, Θ, st, Nθ_μminus)) ./ (2 .* ϵ),
            (Bϕ(q, μ .+ ϵ, ϕ, Θ, st, Nϕ_μplus) .- Bϕ(q, μ .- ϵ, ϕ, Θ, st, Nϕ_μminus)) ./ (2 .* ϵ),
            (α(q, μ .+ ϵ, ϕ, t, Θ, st, Nα_μplus, Nα_S, params) .- α(q, μ .- ϵ, ϕ, t, Θ, st, Nα_μminus, Nα_S, params)) ./ (2 .* ϵ),
            (Br(q, μ, ϕ .+ ϵ, Θ, st, Nr_ϕplus, params) .- Br(q, μ, ϕ .- ϵ, Θ, st, Nr_ϕminus, params)) ./ (2 .* ϵ),
            (Bθ(q, μ, ϕ .+ ϵ, Θ, st, Nθ_ϕplus) .- Bθ(q, μ, ϕ .- ϵ, Θ, st, Nθ_ϕminus)) ./ (2 .* ϵ),
            (Bϕ(q, μ, ϕ .+ ϵ, Θ, st, Nϕ_ϕplus) .- Bϕ(q, μ, ϕ .- ϵ, Θ, st, Nϕ_ϕminus)) ./ (2 .* ϵ),
            (α(q, μ, ϕ .+ ϵ, t, Θ, st, Nα_ϕplus, Nα_S, params) .- α(q, μ, ϕ .- ϵ, t, Θ, st, Nα_ϕminus, Nα_S , params)) ./ (2 .* ϵ),
            (α(q .+ ϵ2, μ, ϕ, t, Θ, st, Nα_qplus, Nα_S, params) .-2 .* α(q, μ, ϕ, t, Θ, st, Nα, Nα_S , params) .+ α(q .- ϵ2, μ, ϕ, t, Θ, st, Nα_qminus, Nα_S, params)) ./ ϵ2 .^ 2,
            (α(q, μ .+ ϵ2, ϕ, t, Θ, st, Nα_μplus, Nα_S, params) .-2 .* α(q, μ, ϕ, t, Θ, st, Nα, Nα_S , params) .+ α(q, μ .- ϵ2, ϕ, t, Θ, st, Nα_μminus, Nα_S, params)) ./ ϵ2 .^ 2,
            (α(q, μ, ϕ .+ ϵ2, t, Θ, st, Nα_ϕplus, Nα_S, params) .-2 .* α(q, μ, ϕ, t, Θ, st, Nα, Nα_S , params) .+ α(q, μ, ϕ .- ϵ2, t, Θ, st, Nα_ϕminus, Nα_S, params)) ./ ϵ2 .^ 2,
            (α_surface(μ, ϕ, t .+ ϵ, Nα_S_tplus, params) .- α_surface(μ, ϕ, t .- ϵ, Nα_S_tminus, params)) ./ (2 .* ϵ)
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

function calclulate_αS_equation(μ, dα_dt, dα_dμ, d2α_dμ2, d2α_dϕ2)
	
	return @. dα_dt - 2 * μ * dα_dμ + (1 - μ^2) * d2α_dμ2 + 1 / (1 - μ^2) * d2α_dϕ2
end

