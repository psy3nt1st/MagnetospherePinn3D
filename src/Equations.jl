function Br_surface(μ, ϕ, config)
	
	@unpack multipoles_bl, compactness = config
    b1, b2, b3 = only(multipoles_bl)

    if compactness == 0

        return @. b1 * 2 * μ + b2 * (3 * μ^2 - 1) + b3 * (5 * μ^3 - 3 * μ)

    else
        M = compactness

        return @. b1 * (-3 / (8 * M^3)) * 2 * μ * (log(1 - 2 * M) + 2 * M + (2 * M)^2 / 2) \
            + b2 * 5 / (8 * M^5) * (3 * μ^2 - 1) * ((6 * M - 4) * log(1 - 2 * M) - 8 * M + 4 * M^2 + 4 * M^3 / 3)
    end
        # end
end

function gaussian_hotspot(μ, ϕ, α0, θ1, ϕ1, σ)

    return @. α0 * exp((-(acos(μ) - θ1)^2 - (ϕ - ϕ1)^2) / (2 * σ^2))
end

function h_boundary(q, μ, ϕ, config)

    return @. 1 - q + ($Br_surface(μ, ϕ, config) * ($Br_surface(μ, ϕ, config) < 0)) ^ 2
end

function α_surface(μ, ϕ, config)
	
    @unpack α_bc_mode = config

	if α_bc_mode == "axisymmetric"

        @unpack axisym_notation, Pc, σ_gs, s, n, γ, rc, compactness = config

        if axisym_notation == "Pc_sigma_s"

        elseif axisym_notation == "Pc_n_gamma"
            σ_gs = (n + 1) / 2
            s = √(2 * γ / (n + 1))
        elseif axisym_notation == "rc_sigma_s"
            Pc = 1 / rc
        end

        if compactness == 0.0 
            P = @. 1 - μ^2
        else
            M = compactness
            P = @. (-3 / (8 * M^3)) * (1 - μ^2) * (log(1 - 2 * M) + 2 * M + (2 * M)^2 / 2)
        end

		return @. s * σ_gs * (P - Pc)^(σ_gs - 1) * (P > Pc)
	
	elseif α_bc_mode == "hotspot"

        @unpack α0, θ1, ϕ1, σ = config
        θ1 = θ1 * π / 180
        ϕ1 = ϕ1 * π / 180
		
		return gaussian_hotspot(μ, ϕ, α0, θ1, ϕ1, σ)
	
	elseif α_bc_mode == "double-hotspot"

        @unpack α0, θ1, ϕ1, σ, α0_b, θ1_b, ϕ1_b, σ_b = config
        θ1 = θ1 * π / 180
        ϕ1 = ϕ1 * π / 180
        θ1_b = θ1_b * π / 180
        ϕ1_b = ϕ1_b * π / 180

		return gaussian_hotspot(μ, ϕ, α0, θ1, ϕ1, σ) .+ gaussian_hotspot(μ, ϕ, α0_b, θ1_b, ϕ1_b, σ_b)
	
    else
        @warn "Unknown boundary condition for α: $α_bc_mode. Returning zero."
		return @. 0
	end
end

function Br(q, μ, ϕ, Nr, config)
		
	return @. q^3 * ($Br_surface(μ, ϕ, config) + (1 - q) * Nr)
end

function Bθ(q, μ, ϕ, Nθ)
	
	return @. q^3 * Nθ
end

function Bϕ(q, μ, ϕ, Nϕ)

	return @. q^3 * Nϕ
end

function α(q, μ, ϕ, Nα, config)

    return @. q * ($α_surface(μ, ϕ, config) + $h_boundary(q, μ, ϕ, config) * Nα)
end

function evaluate_subnetworks(q, μ, ϕ, NN, Θ, st)

    subnet_r, subnet_θ, subnet_ϕ, subnet_α = NN.layers

    Nr = subnet_r(vcat(q, μ, cos.(ϕ), sin.(ϕ)), Θ.layer_1, st.layer_1)[1]
    Nθ = subnet_θ(vcat(q, μ, cos.(ϕ), sin.(ϕ)), Θ.layer_2, st.layer_2)[1]
    Nϕ = subnet_ϕ(vcat(q, μ, cos.(ϕ), sin.(ϕ)), Θ.layer_3, st.layer_3)[1]
    Nα = subnet_α(vcat(q, μ, cos.(ϕ), sin.(ϕ)), Θ.layer_4, st.layer_4)[1]

    return Nr, Nθ, Nϕ, Nα
    
end

function calculate_derivatives(q, μ, ϕ, NN, Θ, st, config)

    ϵ = ∛(eps()) 
    # ϵ2 = ∜(eps())

    Nr_qplus, Nθ_qplus, Nϕ_qplus, Nα_qplus = evaluate_subnetworks(q .+ ϵ, μ, ϕ, NN, Θ, st)
    Nr_qminus, Nθ_qminus, Nϕ_qminus, Nα_qminus = evaluate_subnetworks(q .- ϵ, μ, ϕ, NN, Θ, st)
    Nr_μplus, Nθ_μplus, Nϕ_μplus, Nα_μplus = evaluate_subnetworks(q, μ .+ ϵ, ϕ, NN, Θ, st)
    Nr_μminus, Nθ_μminus, Nϕ_μminus, Nα_μminus = evaluate_subnetworks(q, μ .- ϵ, ϕ, NN, Θ, st)
    Nr_ϕplus, Nθ_ϕplus, Nϕ_ϕplus, Nα_ϕplus = evaluate_subnetworks(q, μ, ϕ .+ ϵ, NN, Θ, st)
    Nr_ϕminus, Nθ_ϕminus, Nϕ_ϕminus, Nα_ϕminus = evaluate_subnetworks(q, μ, ϕ .- ϵ, NN, Θ, st)

	return (
        (Br(q .+ ϵ, μ, ϕ, Nr_qplus, config) .- Br(q .- ϵ, μ, ϕ, Nr_qminus, config)) / (2 * ϵ),
        (Bθ(q .+ ϵ, μ, ϕ, Nθ_qplus) .- Bθ(q .- ϵ, μ, ϕ, Nθ_qminus)) / (2 .* ϵ),
        (Bϕ(q .+ ϵ, μ, ϕ, Nϕ_qplus) .- Bϕ(q .- ϵ, μ, ϕ, Nϕ_qminus)) / (2 .* ϵ),
        (α(q .+ ϵ, μ, ϕ, Nα_qplus, config) .- α(q .- ϵ, μ, ϕ, Nα_qminus, config)) / (2 * ϵ),
        (Br(q, μ .+ ϵ, ϕ, Nr_μplus, config) .- Br(q, μ .- ϵ, ϕ, Nr_μminus, config)) / (2 * ϵ),
        (Bθ(q, μ .+ ϵ, ϕ, Nθ_μplus) .- Bθ(q, μ .- ϵ, ϕ, Nθ_μminus)) / (2 * ϵ),
        (Bϕ(q, μ .+ ϵ, ϕ, Nϕ_μplus) .- Bϕ(q, μ .- ϵ, ϕ, Nϕ_μminus)) / (2 * ϵ),
        (α(q, μ .+ ϵ, ϕ, Nα_μplus, config) .- α(q, μ .- ϵ, ϕ, Nα_μminus, config)) / (2 * ϵ),
        (Br(q, μ, ϕ .+ ϵ, Nr_ϕplus, config) .- Br(q, μ, ϕ .- ϵ, Nr_ϕminus, config)) / (2 * ϵ),
        (Bθ(q, μ, ϕ .+ ϵ, Nθ_ϕplus) .- Bθ(q, μ, ϕ .- ϵ, Nθ_ϕminus)) / (2 * ϵ),
        (Bϕ(q, μ, ϕ .+ ϵ, Nϕ_ϕplus) .- Bϕ(q, μ, ϕ .- ϵ, Nϕ_ϕminus)) / (2 * ϵ),
        (α(q, μ, ϕ .+ ϵ, Nα_ϕplus, config) .- α(q, μ, ϕ .- ϵ, Nα_ϕminus, config)) / (2 * ϵ),
    )
	
end

function calculate_r_equation(q, μ, Br, Bϕ, α, dBϕ_dμ, dBθ_dϕ, config)
    
    @unpack compactness = config

    if compactness == 0.0

        return @. q * μ / √(1 - μ^2) * Bϕ - q * √(1 - μ^2) * dBϕ_dμ - q / √(1 - μ^2) * dBθ_dϕ - α * Br
    
    else

        N = .√(1 .- 2 * compactness * q)

        return @. q * μ / √(1 - μ^2) * Bϕ - q * √(1 - μ^2) * dBϕ_dμ - q / √(1 - μ^2) * dBθ_dϕ - α / N * Br
    end
end

function calculate_θ_equation(q, μ, Bθ, Bϕ, α, dBϕ_dq, dBr_dϕ, config)

    @unpack compactness = config

    if compactness == 0.0 
	    
        return @. q / √(1 - μ^2) * dBr_dϕ - q * Bϕ + q^2 * dBϕ_dq - α * Bθ
    else
        
        N = .√(1 .- 2 * compactness * q)

        return @. q / √(1 - μ^2) * dBr_dϕ - (1 + N^2) / (2 * N) * q * Bϕ + N * q^2 * dBϕ_dq - α / N * Bθ
    end
end

function calculate_ϕ_equation(q, μ, Bθ, Bϕ, α, dBθ_dq, dBr_dμ, config)

    @unpack compactness = config

    if compactness == 0.0

        return @. q * Bθ - q^2 * dBθ_dq + q * √(1 - μ^2) * dBr_dμ - α * Bϕ
     
    else

        N = .√(1 .- 2 * compactness * q)

        return @. (1 + N^2) / (2 * N) * q * Bθ - N * q^2 * dBθ_dq + q * √(1 - μ^2) * dBr_dμ - α / N * Bϕ
    end
end

function calculate_divergence(q, μ, Br, Bθ, dBr_dq, dBθ_dμ, dBϕ_dϕ, config)
    
    @unpack compactness = config

    if compactness == 0.0 

        return @. 2 * q * Br - q^2 * dBr_dq - q * √(1 - μ^2) * dBθ_dμ + q * μ / √(1 - μ^2) * Bθ + q / √(1 - μ^2) * dBϕ_dϕ
    
    else

        N = .√(1 .- 2 * compactness * q)

        return @. N * (2 * q * Br - q^2 * dBr_dq) - q * √(1 - μ^2) * dBθ_dμ + q * μ / √(1 - μ^2) * Bθ + q / √(1 - μ^2) * dBϕ_dϕ
    end
end

function calculate_Bdotgradα(q, μ, Br, Bθ, Bϕ, dα_dq, dα_dμ, dα_dϕ, config)
    
    @unpack compactness = config

    if compactness == 0.0 

        return @. -q^2 * Br * dα_dq - q * √(1 - μ^2) * Bθ * dα_dμ + q / √(1 - μ^2) * Bϕ * dα_dϕ
    
    else

        N = .√(1 .- 2 * compactness * q)

        return @. -N * q^2 * Br * dα_dq - q * √(1 - μ^2) * Bθ * dα_dμ + q / √(1 - μ^2) * Bϕ * dα_dϕ
    end

end


