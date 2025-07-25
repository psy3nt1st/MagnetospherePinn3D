function Br_surface(μ, ϕ, params)
	
	@unpack coef = params.model

	# if hasproperty(params.model, :br_bc_mode)
		
	# 	if params.model.br_bc_mode == "axisymmetric"
			
	# 		return @. coef[1] * 2 * μ + coef[2] * (3 * μ^2 - 1) + coef[3] * (5 * μ^3 - 3 * μ)
		
	# 	elseif params.model.br_bc_mode == "non axisymmetric"
			
	# 		return @. 0.5 * coef[1] * 2 * μ - 0.5 * coef[1] * 2 * sin(μ) * cos(ϕ)
	# 	end

	# else
    if params.model.M == 0
        
        return @. coef[1] * 2 * μ + coef[2] * (3 * μ^2 - 1) + coef[3] * (5 * μ^3 - 3 * μ)
    
    else

        M = params.model.M

        # return @. coef[1] * (-3 / (8 * M^3)) * 2 * μ * (log(1 - 2 * M) + 2 * M + (2 * M)^2 / 2) + coef[2] * 1 / (16 * M^4) * (3 * μ^2 - 1) * ((4 - 6 * M) * log(1 - 2 * M) + 8 * M - 4 * M^2 - 4 * M^3 / 3)
        return @. coef[1] * (-3 / (8 * M^3)) * 2 * μ * (log(1 - 2 * M) + 2 * M + (2 * M)^2 / 2) + coef[2] * 5 / (8 * M^5) * (3 * μ^2 - 1) * ((6 * M - 4) * log(1 - 2 * M) - 8 * M + 4 * M^2 + 4 * M^3 / 3)

    end
        # end
end

function gaussian_hotspot(μ, ϕ, α0, θ1, ϕ1, σ)

    return @. α0 * exp((-(acos(μ) - θ1)^2 - (ϕ - ϕ1)^2) / (2 * σ^2))
end

function h_boundary(q, μ, ϕ, t, params)
    
    # if params.model.alpha_bc_mode == "diffusive"

    #     return @. t^2 + 1 - q + ($Br_surface(μ, ϕ, params) * ($Br_surface(μ, ϕ, params) < 0)) ^ 2
    # else

    return @. 1 - q + ($Br_surface(μ, ϕ, params) * ($Br_surface(μ, ϕ, params) < 0)) ^ 2
    # end
end

function α_surface(μ, ϕ, params)
	
	if params.model.alpha_bc_mode == "axisymmetric"

        if params.model.notation == "Pc_sigma_s"
		    Pc, σ, s  = params.model.Pc, params.model.sigma_gs, params.model.s
        elseif params.model.notation == "Pc_n_gamma"
            Pc, n, γ = params.model.Pc, params.model.n, params.model.gamma
            σ = (n + 1) / 2
            s = √(2 * γ / (n + 1))
        elseif params.model.notation == "rc_sigma_s"
            rc, σ, s = params.model.rc, params.model.sigma_gs, params.model.s
            Pc = 1 / rc
        end

        if params.model.M == 0.0 
            P = @. 1 - μ^2
        else
            M = params.model.M
            P = @. (-3 / (8 * M^3)) * (1 - μ^2) * (log(1 - 2 * M) + 2 * M + (2 * M)^2 / 2)
        end

		return @. s * σ * (P - Pc)^(σ-1) * (P > Pc)
	
	elseif params.model.alpha_bc_mode == "hotspot"
		
		α0, θ1, ϕ1, σ = params.model.alpha0, params.model.theta1 * π / 180, params.model.phi1 * π / 180, params.model.sigma
		
		return gaussian_hotspot(μ, ϕ, α0, θ1, ϕ1, σ)
	
	elseif params.model.alpha_bc_mode == "double-hotspot"

		α0, θ1, ϕ1, σ = params.model.alpha0, params.model.theta1 * π / 180, params.model.phi1 * π / 180, params.model.sigma
		α0_b, θ1_b, ϕ1_b, σ_b = params.model.alpha0_b, params.model.theta1_b * π / 180, params.model.phi1_b * π / 180, params.model.sigma_b

		return gaussian_hotspot(μ, ϕ, α0, θ1, ϕ1, σ) .+ gaussian_hotspot(μ, ϕ, α0_b, θ1_b, ϕ1_b, σ_b)
    
    elseif params.model.alpha_bc_mode == "diffusive"
        
        α0, θ1, ϕ1, σ, t = params.model.alpha0, params.model.theta1 * π / 180, params.model.phi1 * π / 180, params.model.sigma, params.model.t
		
        return @. α0 * σ^2 / (σ^2 + 2t) * exp(-((acos(μ) - θ1)^2 + (ϕ - ϕ1)^2) / (2 * (σ^2 + 2t)))
        # return gaussian_hotspot(μ, ϕ, α0, θ1, ϕ1, σ)

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

function calculate_derivatives(q, μ, ϕ, t, Θ, st, NN, params)

    ϵ = ∛(eps()) 
    # ϵ2 = ∜(eps())

    Nr_qplus, Nθ_qplus, Nϕ_qplus, Nα_qplus = evaluate_subnetworks(q .+ ϵ, μ, ϕ, t, Θ, st, NN)
    Nr_qminus, Nθ_qminus, Nϕ_qminus, Nα_qminus = evaluate_subnetworks(q .- ϵ, μ, ϕ, t, Θ, st, NN)
    Nr_μplus, Nθ_μplus, Nϕ_μplus, Nα_μplus = evaluate_subnetworks(q, μ .+ ϵ, ϕ, t, Θ, st, NN)
    Nr_μminus, Nθ_μminus, Nϕ_μminus, Nα_μminus = evaluate_subnetworks(q, μ .- ϵ, ϕ, t, Θ, st, NN)
    Nr_ϕplus, Nθ_ϕplus, Nϕ_ϕplus, Nα_ϕplus = evaluate_subnetworks(q, μ, ϕ .+ ϵ, t, Θ, st, NN)
    Nr_ϕminus, Nθ_ϕminus, Nϕ_ϕminus, Nα_ϕminus = evaluate_subnetworks(q, μ, ϕ .- ϵ, t, Θ, st, NN)

	return (
        (Br(q .+ ϵ, μ, ϕ, Nr_qplus, params) .- Br(q .- ϵ, μ, ϕ, Nr_qminus, params)) / (2 * ϵ),
        (Bθ(q .+ ϵ, μ, ϕ, Nθ_qplus) .- Bθ(q .- ϵ, μ, ϕ, Nθ_qminus)) / (2 .* ϵ),
        (Bϕ(q .+ ϵ, μ, ϕ, Nϕ_qplus) .- Bϕ(q .- ϵ, μ, ϕ, Nϕ_qminus)) / (2 .* ϵ),
        (α(q .+ ϵ, μ, ϕ, t, Nα_qplus, params) .- α(q .- ϵ, μ, ϕ, t, Nα_qminus, params)) / (2 * ϵ),
        (Br(q, μ .+ ϵ, ϕ, Nr_μplus, params) .- Br(q, μ .- ϵ, ϕ, Nr_μminus, params)) / (2 * ϵ),
        (Bθ(q, μ .+ ϵ, ϕ, Nθ_μplus) .- Bθ(q, μ .- ϵ, ϕ, Nθ_μminus)) / (2 * ϵ),
        (Bϕ(q, μ .+ ϵ, ϕ, Nϕ_μplus) .- Bϕ(q, μ .- ϵ, ϕ, Nϕ_μminus)) / (2 * ϵ),
        (α(q, μ .+ ϵ, ϕ, t, Nα_μplus, params) .- α(q, μ .- ϵ, ϕ, t, Nα_μminus, params)) / (2 * ϵ),
        (Br(q, μ, ϕ .+ ϵ, Nr_ϕplus, params) .- Br(q, μ, ϕ .- ϵ, Nr_ϕminus, params)) / (2 * ϵ),
        (Bθ(q, μ, ϕ .+ ϵ, Nθ_ϕplus) .- Bθ(q, μ, ϕ .- ϵ, Nθ_ϕminus)) / (2 * ϵ),
        (Bϕ(q, μ, ϕ .+ ϵ, Nϕ_ϕplus) .- Bϕ(q, μ, ϕ .- ϵ, Nϕ_ϕminus)) / (2 * ϵ),
        (α(q, μ, ϕ .+ ϵ, t, Nα_ϕplus, params) .- α(q, μ, ϕ .- ϵ, t, Nα_ϕminus, params)) / (2 * ϵ),
    )
	
end

function calculate_r_equation(q, μ, Br, Bϕ, α, dBϕ_dμ, dBθ_dϕ, params)
	
    if params.model.M == 0.0 
	    
        return @. q * μ / √(1 - μ^2) * Bϕ - q * √(1 - μ^2) * dBϕ_dμ - q / √(1 - μ^2) * dBθ_dϕ - α * Br
    
    else

        N = .√(1 .- 2 * params.model.M * q)

        return @. q * μ / √(1 - μ^2) * Bϕ - q * √(1 - μ^2) * dBϕ_dμ - q / √(1 - μ^2) * dBθ_dϕ - α / N * Br
    end
end

function calculate_θ_equation(q, μ, Bθ, Bϕ, α, dBϕ_dq, dBr_dϕ, params)

    if params.model.M == 0.0 
	    
        
        return @. q / √(1 - μ^2) * dBr_dϕ - q * Bϕ + q^2 * dBϕ_dq - α * Bθ
    else
        
        N = .√(1 .- 2 * params.model.M * q)

        return @. q / √(1 - μ^2) * dBr_dϕ - (1 + N^2) / (2 * N) * q * Bϕ + N * q^2 * dBϕ_dq - α / N * Bθ
    end
end

function calculate_ϕ_equation(q, μ, Bθ, Bϕ, α, dBθ_dq, dBr_dμ, params)
    
    if params.model.M == 0.0 

        return @. q * Bθ - q^2 * dBθ_dq + q * √(1 - μ^2) * dBr_dμ - α * Bϕ
     
    else

        N = .√(1 .- 2 * params.model.M * q)

        return @. (1 + N^2) / (2 * N) * q * Bθ - N * q^2 * dBθ_dq + q * √(1 - μ^2) * dBr_dμ - α / N * Bϕ
    end
end

function calculate_divergence(q, μ, Br, Bθ, dBr_dq, dBθ_dμ, dBϕ_dϕ, params)
    
    if params.model.M == 0.0 

        return @. 2 * q * Br - q^2 * dBr_dq - q * √(1 - μ^2) * dBθ_dμ + q * μ / √(1 - μ^2) * Bθ + q / √(1 - μ^2) * dBϕ_dϕ
    
    else

        N = .√(1 .- 2 * params.model.M * q)

        return @. N * (2 * q * Br - q^2 * dBr_dq) - q * √(1 - μ^2) * dBθ_dμ + q * μ / √(1 - μ^2) * Bθ + q / √(1 - μ^2) * dBϕ_dϕ
    end
end

function calculate_Bdotgradα(q, μ, Br, Bθ, Bϕ, dα_dq, dα_dμ, dα_dϕ, params)
    
    if params.model.M == 0.0 

        return @. -q^2 * Br * dα_dq - q * √(1 - μ^2) * Bθ * dα_dμ + q / √(1 - μ^2) * Bϕ * dα_dϕ
    
    else

        N = .√(1 .- 2 * params.model.M * q)

        return @. -N * q^2 * Br * dα_dq - q * √(1 - μ^2) * Bθ * dα_dμ + q / √(1 - μ^2) * Bϕ * dα_dϕ
    end

end

function calculate_αS_equation(μ, ϕ, t, q1, αS, dαS_dt, d2αS_dq2, dαS_dμ, d2αS_dμ2, d2αS_dϕ2)

    ∇2αS = @. #=q1^4 * d2αS_dq2 =# - 2 * q1^2 * μ * dαS_dμ + q1^2 * (1 - μ^2) * d2αS_dμ2 + q1^2 / (1 - μ^2) * d2αS_dϕ2

    # cond = ∇2αS .== maximum(∇2αS)
    # println("μ = $(μ[cond]), ϕ = $(ϕ[cond]), t = $(t[cond]), q1 = $(q1[cond]), dαS_dt = $(dαS_dt[cond]), d2αS_dq2 = $(d2αS_dq2[cond]), dαS_dμ = $(dαS_dμ[cond]), d2αS_dμ2 = $(d2αS_dμ2[cond]), d2αS_dϕ2 = $(d2αS_dϕ2[cond])")
    # println(maximum(∇2αS), " ",  maximum(dαS_dt), " ",  maximum(d2αS_dq2), " ", maximum(dαS_dμ), " ", maximum(d2αS_dμ2), " ", maximum(d2αS_dϕ2))

	# return @. dαS_dt - ∇2αS
    return @. dαS_dt + αS
end

