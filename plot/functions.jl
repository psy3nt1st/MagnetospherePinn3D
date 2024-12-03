@with_kw struct Derivatives{n_q, n_μ, n_ϕ}
	∂Br∂q::Array{Float64,3} = zeros(Float64, n_q, n_μ, n_ϕ)
	∂Bθ∂q::Array{Float64,3} = zeros(Float64, n_q, n_μ, n_ϕ)
	∂Bϕ∂q::Array{Float64,3} = zeros(Float64, n_q, n_μ, n_ϕ)
	∂α∂q::Array{Float64,3} = zeros(Float64, n_q, n_μ, n_ϕ)
	∂Br∂μ::Array{Float64,3} = zeros(Float64, n_q, n_μ, n_ϕ)
	∂Bθ∂μ::Array{Float64,3} = zeros(Float64, n_q, n_μ, n_ϕ)
	∂Bϕ∂μ::Array{Float64,3} = zeros(Float64, n_q, n_μ, n_ϕ)
	∂α∂μ::Array{Float64,3} = zeros(Float64, n_q, n_μ, n_ϕ)
	∂Br∂ϕ::Array{Float64,3} = zeros(Float64, n_q, n_μ, n_ϕ)
	∂Bθ∂ϕ::Array{Float64,3} = zeros(Float64, n_q, n_μ, n_ϕ)
	∂Bϕ∂ϕ::Array{Float64,3} = zeros(Float64, n_q, n_μ, n_ϕ)
	∂α∂ϕ::Array{Float64,3} = zeros(Float64, n_q, n_μ, n_ϕ)
end

function calculate_fd_derivatives(q, μ, ϕ, Br, Bθ, Bϕ, α, derivs)
	
	n_q = size(q, 1)
	n_μ = size(μ, 2)
	n_ϕ = size(ϕ, 3)

	dq, dμ, dϕ = q[2, 1, 1] - q[1, 1, 1], μ[1, 2, 1] - μ[1, 1, 1], ϕ[1, 1, 2] - ϕ[1, 1, 1]

	@unpack ∂Br∂q, ∂Bθ∂q, ∂Bϕ∂q, ∂α∂q, ∂Br∂μ, ∂Bθ∂μ, ∂Bϕ∂μ, ∂α∂μ, ∂Br∂ϕ, ∂Bθ∂ϕ, ∂Bϕ∂ϕ, ∂α∂ϕ = derivs

	∂Br∂q[2:n_q-1, 2:n_μ-1, 2:n_ϕ-1] .= @. (Br[3:n_q, 2:n_μ-1, 2:n_ϕ-1] - Br[1:n_q-2, 2:n_μ-1, 2:n_ϕ-1]) / (2 * dq)
	∂Bθ∂q[2:n_q-1, 2:n_μ-1, 2:n_ϕ-1] .= @. (Bθ[3:n_q, 2:n_μ-1, 2:n_ϕ-1] - Bθ[1:n_q-2, 2:n_μ-1, 2:n_ϕ-1]) / (2 * dq)
	∂Bϕ∂q[2:n_q-1, 2:n_μ-1, 2:n_ϕ-1] .= @. (Bϕ[3:n_q, 2:n_μ-1, 2:n_ϕ-1] - Bϕ[1:n_q-2, 2:n_μ-1, 2:n_ϕ-1]) / (2 * dq)
	∂α∂q[2:n_q-1, 2:n_μ-1, 2:n_ϕ-1] .= @. (α[3:n_q, 2:n_μ-1, 2:n_ϕ-1] - α[1:n_q-2, 2:n_μ-1, 2:n_ϕ-1]) / (2 * dq)

	∂Br∂μ[2:n_q-1, 2:n_μ-1, 2:n_ϕ-1] .= @. (Br[2:n_q-1, 3:n_μ, 2:n_ϕ-1] - Br[2:n_q-1, 1:n_μ-2, 2:n_ϕ-1]) / (2 * dμ)
	∂Bθ∂μ[2:n_q-1, 2:n_μ-1, 2:n_ϕ-1] .= @. (Bθ[2:n_q-1, 3:n_μ, 2:n_ϕ-1] - Bθ[2:n_q-1, 1:n_μ-2, 2:n_ϕ-1]) / (2 * dμ)
	∂Bϕ∂μ[2:n_q-1, 2:n_μ-1, 2:n_ϕ-1] .= @. (Bϕ[2:n_q-1, 3:n_μ, 2:n_ϕ-1] - Bϕ[2:n_q-1, 1:n_μ-2, 2:n_ϕ-1]) / (2 * dμ)
	∂α∂μ[2:n_q-1, 2:n_μ-1, 2:n_ϕ-1] .= @. (α[2:n_q-1, 3:n_μ, 2:n_ϕ-1] - α[2:n_q-1, 1:n_μ-2, 2:n_ϕ-1]) / (2 * dμ)

	∂Br∂ϕ[2:n_q-1, 2:n_μ-1, 2:n_ϕ-1] .= @. (Br[2:n_q-1, 2:n_μ-1, 3:n_ϕ] - Br[2:n_q-1, 2:n_μ-1, 1:n_ϕ-2]) / (2 * dϕ)
	∂Bθ∂ϕ[2:n_q-1, 2:n_μ-1, 2:n_ϕ-1] .= @. (Bθ[2:n_q-1, 2:n_μ-1, 3:n_ϕ] - Bθ[2:n_q-1, 2:n_μ-1, 1:n_ϕ-2]) / (2 * dϕ)
	∂Bϕ∂ϕ[2:n_q-1, 2:n_μ-1, 2:n_ϕ-1] .= @. (Bϕ[2:n_q-1, 2:n_μ-1, 3:n_ϕ] - Bϕ[2:n_q-1, 2:n_μ-1, 1:n_ϕ-2]) / (2 * dϕ)
	∂α∂ϕ[2:n_q-1, 2:n_μ-1, 2:n_ϕ-1] .= @. (α[2:n_q-1, 2:n_μ-1, 3:n_ϕ] - α[2:n_q-1, 2:n_μ-1, 1:n_ϕ-2]) / (2 * dϕ)

	
	∂Br∂q[1, :, :] .= @. (-3 * Br[1, :, :] + 4 * Br[2, :, :] - Br[3, :, :]) / (2 * dq)
	∂Bθ∂q[1, :, :] .= @. (-3 * Bθ[1, :, :] + 4 * Bθ[2, :, :] - Bθ[3, :, :]) / (2 * dq)
	∂Bϕ∂q[1, :, :] .= @. (-3 * Bϕ[1, :, :] + 4 * Bϕ[2, :, :] - Bϕ[3, :, :]) / (2 * dq)
	∂α∂q[1, :, :] .= @. (-3 * α[1, :, :] + 4 * α[2, :, :] - α[3, :, :]) / (2 * dq)

	∂Br∂μ[:, 1, :] .= @. (-3 * Br[:, 1, :] + 4 * Br[:, 2, :] - Br[:, 3, :]) / (2 * dμ)
	∂Bθ∂μ[:, 1, :] .= @. (-3 * Bθ[:, 1, :] + 4 * Bθ[:, 2, :] - Bθ[:, 3, :]) / (2 * dμ)
	∂Bϕ∂μ[:, 1, :] .= @. (-3 * Bϕ[:, 1, :] + 4 * Bϕ[:, 2, :] - Bϕ[:, 3, :]) / (2 * dμ)
	∂α∂μ[:, 1, :] .= @. (-3 * α[:, 1, :] + 4 * α[:, 2, :] - α[:, 3, :]) / (2 * dμ)

	∂Br∂ϕ[:, :, 1] .= @. (-3 * Br[:, :, 1] + 4 * Br[:, :, 2] - Br[:, :, 3]) / (2 * dϕ)
	∂Bθ∂ϕ[:, :, 1] .= @. (-3 * Bθ[:, :, 1] + 4 * Bθ[:, :, 2] - Bθ[:, :, 3]) / (2 * dϕ)
	∂Bϕ∂ϕ[:, :, 1] .= @. (-3 * Bϕ[:, :, 1] + 4 * Bϕ[:, :, 2] - Bϕ[:, :, 3]) / (2 * dϕ)
	∂α∂ϕ[:, :, 1] .= @. (-3 * α[:, :, 1] + 4 * α[:, :, 2] - α[:, :, 3]) / (2 * dϕ)


	∂Br∂q[n_q, :, :] .= @. (3 * Br[n_q, :, :] - 4 * Br[n_q-1, :, :] + Br[n_q-2, :, :]) / (2 * dq)
	∂Bθ∂q[n_q, :, :] .= @. (3 * Bθ[n_q, :, :] - 4 * Bθ[n_q-1, :, :] + Bθ[n_q-2, :, :]) / (2 * dq)
	∂Bϕ∂q[n_q, :, :] .= @. (3 * Bϕ[n_q, :, :] - 4 * Bϕ[n_q-1, :, :] + Bϕ[n_q-2, :, :]) / (2 * dq)
	∂α∂q[n_q, :, :] .= @. (3 * α[n_q, :, :] - 4 * α[n_q-1, :, :] + α[n_q-2, :, :]) / (2 * dq)

	∂Br∂μ[:, n_μ, :] .= @. (3 * Br[:, n_μ, :] - 4 * Br[:, n_μ-1, :] + Br[:, n_μ-2, :]) / (2 * dμ)
	∂Bθ∂μ[:, n_μ, :] .= @. (3 * Bθ[:, n_μ, :] - 4 * Bθ[:, n_μ-1, :] + Bθ[:, n_μ-2, :]) / (2 * dμ)
	∂Bϕ∂μ[:, n_μ, :] .= @. (3 * Bϕ[:, n_μ, :] - 4 * Bϕ[:, n_μ-1, :] + Bϕ[:, n_μ-2, :]) / (2 * dμ)
	∂α∂μ[:, n_μ, :] .= @. (3 * α[:, n_μ, :] - 4 * α[:, n_μ-1, :] + α[:, n_μ-2, :]) / (2 * dμ)

	∂Br∂ϕ[:, :, n_ϕ] .= @. (3 * Br[:, :, n_ϕ] - 4 * Br[:, :, n_ϕ-1] + Br[:, :, n_ϕ-2]) / (2 * dϕ)
	∂Bθ∂ϕ[:, :, n_ϕ] .= @. (3 * Bθ[:, :, n_ϕ] - 4 * Bθ[:, :, n_ϕ-1] + Bθ[:, :, n_ϕ-2]) / (2 * dϕ)
	∂Bϕ∂ϕ[:, :, n_ϕ] .= @. (3 * Bϕ[:, :, n_ϕ] - 4 * Bϕ[:, :, n_ϕ-1] + Bϕ[:, :, n_ϕ-2]) / (2 * dϕ)
	∂α∂ϕ[:, :, n_ϕ] .= @. (3 * α[:, :, n_ϕ] - 4 * α[:, :, n_ϕ-1] + α[:, :, n_ϕ-2]) / (2 * dϕ)

	return
end

function calculate_PDE_absolute_error(n_q, n_μ, n_ϕ, pinn, Θ_trained, st, params; use_finite_differences=false)

	q, μ, ϕ, Br1, Bθ1, Bϕ1, α1 = create_test(n_q, n_μ, n_ϕ, pinn, Θ_trained, st, params)[1:7]

	if use_finite_differences
		derivs = Derivatives{n_q, n_μ, n_ϕ}()
		calculate_fd_derivatives(q, μ, ϕ, Br1, Bθ1, Bϕ1, α1, derivs)
		@unpack ∂Br∂q, ∂Bθ∂q, ∂Bϕ∂q, ∂α∂q, ∂Br∂μ, ∂Bθ∂μ, ∂Bϕ∂μ, ∂α∂μ, ∂Br∂ϕ, ∂Bθ∂ϕ, ∂Bϕ∂ϕ, ∂α∂ϕ = derivs
	else
		q1 = reshape(q, 1, :)
		μ1 = reshape(μ, 1, :)
		ϕ1 = reshape(ϕ, 1, :)

		∂Br∂q, ∂Bθ∂q, ∂Bϕ∂q, ∂α∂q, ∂Br∂μ, ∂Bθ∂μ, ∂Bϕ∂μ, ∂α∂μ, ∂Br∂ϕ, ∂Bθ∂ϕ, ∂Bϕ∂ϕ, ∂α∂ϕ  = calculate_derivatives(q1, μ1, ϕ1, Θ_trained, st, pinn, params)
		∂Br∂q = reshape(∂Br∂q, n_q, n_μ, n_ϕ)
		∂Bθ∂q = reshape(∂Bθ∂q, n_q, n_μ, n_ϕ)
		∂Bϕ∂q = reshape(∂Bϕ∂q, n_q, n_μ, n_ϕ)
		∂α∂q = reshape(∂α∂q, n_q, n_μ, n_ϕ)
		∂Br∂μ = reshape(∂Br∂μ, n_q, n_μ, n_ϕ)
		∂Bθ∂μ = reshape(∂Bθ∂μ, n_q, n_μ, n_ϕ)
		∂Bϕ∂μ = reshape(∂Bϕ∂μ, n_q, n_μ, n_ϕ)
		∂α∂μ = reshape(∂α∂μ, n_q, n_μ, n_ϕ)
		∂Br∂ϕ = reshape(∂Br∂ϕ, n_q, n_μ, n_ϕ)
		∂Bθ∂ϕ = reshape(∂Bθ∂ϕ, n_q, n_μ, n_ϕ)
		∂Bϕ∂ϕ = reshape(∂Bϕ∂ϕ, n_q, n_μ, n_ϕ)
		∂α∂ϕ = reshape(∂α∂ϕ, n_q, n_μ, n_ϕ)
	end

	r_eq = abs.(calculate_r_equation(q, μ, ϕ, Br1, Bθ1, Bϕ1, α1, ∂Br∂q, ∂Bθ∂q, ∂Bϕ∂q, ∂Br∂μ, ∂Bθ∂μ, ∂Bϕ∂μ, ∂Br∂ϕ, ∂Bθ∂ϕ, ∂Bϕ∂ϕ))
	θ_eq = abs.(calculate_θ_equation(q, μ, ϕ, Br1, Bθ1, Bϕ1, α1, ∂Br∂q, ∂Bθ∂q, ∂Bϕ∂q, ∂Br∂μ, ∂Bθ∂μ, ∂Bϕ∂μ, ∂Br∂ϕ, ∂Bθ∂ϕ, ∂Bϕ∂ϕ))
	ϕ_eq = abs.(calculate_ϕ_equation(q, μ, ϕ, Br1, Bθ1, Bϕ1, α1, ∂Br∂q, ∂Bθ∂q, ∂Bϕ∂q, ∂Br∂μ, ∂Bθ∂μ, ∂Bϕ∂μ, ∂Br∂ϕ, ∂Bθ∂ϕ, ∂Bϕ∂ϕ))
	∇B = abs.(calculate_divergence(q, μ, ϕ, Br1, Bθ1, Bϕ1, ∂Br∂q, ∂Bθ∂q, ∂Bϕ∂q, ∂Br∂μ, ∂Bθ∂μ, ∂Bϕ∂μ, ∂Br∂ϕ, ∂Bθ∂ϕ, ∂Bϕ∂ϕ))
	B∇α = abs.(calculate_Bdotgradα(q, μ, ϕ, Br1, Bθ1, Bϕ1, ∂α∂q, ∂α∂μ, ∂α∂ϕ))

	return r_eq, θ_eq, ϕ_eq, ∇B, B∇α
end


function calculate_l2error(resolutions, pinn, Θ_trained, st, params; use_finite_differences=false)
	
   l2error_total = []
	l2error_r = []
	l2error_θ = []
	l2error_ϕ = []
	l2error_∇B = []
	l2error_B∇α = []
	for n in resolutions
		
		n_q = n_μ = n_ϕ = n

		r_eq, θ_eq, ϕ_eq, ∇B, B∇α = calculate_PDE_absolute_error(n_q, n_μ, n_ϕ, pinn, Θ_trained, st, params, use_finite_differences=use_finite_differences)
		
		push!(l2error_r, sqrt(sum(abs2, r_eq[2:n_q-1, 2:n_μ-1, 2:n_ϕ-1]) / (n-2)^3))
		push!(l2error_θ, sqrt(sum(abs2, θ_eq[2:n_q-1, 2:n_μ-1, 2:n_ϕ-1]) / (n-2)^3))
		push!(l2error_ϕ, sqrt(sum(abs2, ϕ_eq[2:n_q-1, 2:n_μ-1, 2:n_ϕ-1]) / (n-2)^3))
		push!(l2error_∇B, sqrt(sum(abs2, ∇B[2:n_q-1, 2:n_μ-1, 2:n_ϕ-1]) / (n-2)^3))
		push!(l2error_B∇α, sqrt(sum(abs2, B∇α[2:n_q-1, 2:n_μ-1, 2:n_ϕ-1]) / (n-2)^3))
      push!(l2error_total, l2error_r[end] + l2error_θ[end] + l2error_ϕ[end] + l2error_∇B[end] + l2error_B∇α[end])
		println("resolution: ", n, " l2error_total: ", l2error_total[end])

	end

	return l2error_total, l2error_r, l2error_θ, l2error_ϕ, l2error_∇B, l2error_B∇α

end

function calculate_l2error_vs_q(n_q, n_μ, n_ϕ, pinn, Θ_trained, st, params; use_finite_differences=false)
	
	Br1, Bθ1, Bϕ1 = create_test(n_q, n_μ, n_ϕ, pinn, Θ_trained, st, params)[4:6]
	Bmag1 = .√(Br1.^2 .+ Bθ1.^2 .+ Bϕ1.^2)
	
	r_eq, θ_eq, ϕ_eq, ∇B, B∇α = calculate_PDE_absolute_error(n_q, n_μ, n_ϕ, pinn, Θ_trained, st, params, use_finite_differences=use_finite_differences)

	l2_r_vs_q = [sqrt(sum(abs2, r_eq[q, 2:n_μ-1, 2:n_ϕ-1] ./ Bmag1[q, 2:n_μ-1, 2:n_ϕ-1]) / (n_μ * n_ϕ)) for q in 2:n_q-1]
	l2_θ_vs_q = [sqrt(sum(abs2, θ_eq[q, 2:n_μ-1, 2:n_ϕ-1] ./ Bmag1[q, 2:n_μ-1, 2:n_ϕ-1]) / (n_μ * n_ϕ)) for q in 2:n_q-1]
	l2_ϕ_vs_q = [sqrt(sum(abs2, ϕ_eq[q, 2:n_μ-1, 2:n_ϕ-1] ./ Bmag1[q, 2:n_μ-1, 2:n_ϕ-1]) / (n_μ * n_ϕ)) for q in 2:n_q-1]
	l2_B∇α_vs_q = [sqrt(sum(abs2, B∇α[q, 2:n_μ-1, 2:n_ϕ-1] ./ Bmag1[q, 2:n_μ-1, 2:n_ϕ-1]) / (n_μ * n_ϕ)) for q in 2:n_q-1]
	l2_∇B_vs_q = [sqrt(sum(abs2, ∇B[q, 2:n_μ-1, 2:n_ϕ-1] ./ Bmag1[q, 2:n_μ-1, 2:n_ϕ-1]) / (n_μ * n_ϕ)) for q in 2:n_q-1]
	l2_total_vs_q = l2_r_vs_q .+ l2_θ_vs_q .+ l2_ϕ_vs_q .+ l2_B∇α_vs_q .+ l2_∇B_vs_q

	return l2_total_vs_q, l2_r_vs_q, l2_θ_vs_q, l2_ϕ_vs_q, l2_B∇α_vs_q, l2_∇B_vs_q
end

function calculate_l2error_vs_μ(n_q, n_μ, n_ϕ, pinn, Θ_trained, st, params; use_finite_differences=false)

	Br1, Bθ1, Bϕ1 = create_test(n_q, n_μ, n_ϕ, pinn, Θ_trained, st, params)[4:6]
	Bmag1 = .√(Br1.^2 .+ Bθ1.^2 .+ Bϕ1.^2)
	
	r_eq, θ_eq, ϕ_eq, ∇B, B∇α = calculate_PDE_absolute_error(n_q, n_μ, n_ϕ, pinn, Θ_trained, st, params, use_finite_differences=use_finite_differences)

	l2_r_vs_μ = [sqrt(sum(abs2, r_eq[2:n_q-1, μ, 2:n_ϕ-1] ./ Bmag1[2:n_q-1, μ, 2:n_ϕ-1]) / (n_q * n_ϕ)) for μ in 2:n_μ-1]
	l2_θ_vs_μ = [sqrt(sum(abs2, θ_eq[2:n_q-1, μ, 2:n_ϕ-1] ./ Bmag1[2:n_q-1, μ, 2:n_ϕ-1]) / (n_q * n_ϕ)) for μ in 2:n_μ-1]
	l2_ϕ_vs_μ = [sqrt(sum(abs2, ϕ_eq[2:n_q-1, μ, 2:n_ϕ-1] ./ Bmag1[2:n_q-1, μ, 2:n_ϕ-1]) / (n_q * n_ϕ)) for μ in 2:n_μ-1]
	l2_B∇α_vs_μ = [sqrt(sum(abs2, B∇α[2:n_q-1, μ, 2:n_ϕ-1] ./ Bmag1[2:n_q-1, μ, 2:n_ϕ-1]) / (n_q * n_ϕ)) for μ in 2:n_μ-1]
	l2_∇B_vs_μ = [sqrt(sum(abs2, ∇B[2:n_q-1, μ, 2:n_ϕ-1] ./ Bmag1[2:n_q-1, μ, 2:n_ϕ-1]) / (n_q * n_ϕ)) for μ in 2:n_μ-1]
	l2_total_vs_μ = l2_r_vs_μ .+ l2_θ_vs_μ .+ l2_ϕ_vs_μ .+ l2_B∇α_vs_μ .+ l2_∇B_vs_μ

	return l2_total_vs_μ, l2_r_vs_μ, l2_θ_vs_μ, l2_ϕ_vs_μ, l2_B∇α_vs_μ, l2_∇B_vs_μ
	
end

function calculate_l2errors_vs_ϕ(n_q, n_μ, n_ϕ, pinn, Θ_trained, st, params; use_finite_differences=false)

	Br1, Bθ1, Bϕ1 = create_test(n_q, n_μ, n_ϕ, pinn, Θ_trained, st, params)[4:6]
	Bmag1 = .√(Br1.^2 .+ Bθ1.^2 .+ Bϕ1.^2)
	
	r_eq, θ_eq, ϕ_eq, ∇B, B∇α = calculate_PDE_absolute_error(n_q, n_μ, n_ϕ, pinn, Θ_trained, st, params, use_finite_differences=use_finite_differences)

	l2_r_vs_ϕ = [sqrt(sum(abs2, r_eq[2:n_q-1, 2:n_μ-1, ϕ] ./ Bmag1[2:n_q-1, 2:n_μ-1, ϕ]) / (n_q * n_μ)) for ϕ in 2:n_ϕ-1]
	l2_θ_vs_ϕ = [sqrt(sum(abs2, θ_eq[2:n_q-1, 2:n_μ-1, ϕ] ./ Bmag1[2:n_q-1, 2:n_μ-1, ϕ]) / (n_q * n_μ)) for ϕ in 2:n_ϕ-1]
	l2_ϕ_vs_ϕ = [sqrt(sum(abs2, ϕ_eq[2:n_q-1, 2:n_μ-1, ϕ] ./ Bmag1[2:n_q-1, 2:n_μ-1, ϕ]) / (n_q * n_μ)) for ϕ in 2:n_ϕ-1]
	l2_B∇α_vs_ϕ = [sqrt(sum(abs2, B∇α[2:n_q-1, 2:n_μ-1, ϕ] ./ Bmag1[2:n_q-1, 2:n_μ-1, ϕ]) / (n_q * n_μ)) for ϕ in 2:n_ϕ-1]
	l2_∇B_vs_ϕ = [sqrt(sum(abs2, ∇B[2:n_q-1, 2:n_μ-1, ϕ] ./ Bmag1[2:n_q-1, 2:n_μ-1, ϕ]) / (n_q * n_μ)) for ϕ in 2:n_ϕ-1]
	l2_total_vs_ϕ = l2_r_vs_ϕ .+ l2_θ_vs_ϕ .+ l2_ϕ_vs_ϕ .+ l2_B∇α_vs_ϕ .+ l2_∇B_vs_ϕ

	return l2_total_vs_ϕ, l2_r_vs_ϕ, l2_θ_vs_ϕ, l2_ϕ_vs_ϕ, l2_B∇α_vs_ϕ, l2_∇B_vs_ϕ

end


function plot_l2error(resolutions, l2errors)
	# Create the first figure
	fig = Figure(size = (800, 600))
	ax = Axis(fig[1, 1], xlabel = "Resolution", ylabel = "L2 error", xscale = log10, yscale = log10)

	labels = ["Total", L"\hat{r}", L"\hat{\theta}", L"\hat{\phi}", L"\nabla\cdot B", L"B \cdot \nabla \alpha"]

	for (i, l2error) in enumerate(l2errors)
		lines!(ax, resolutions, l2error, label = labels[i])
	end

	scaling_factor = l2errors[1][1] / (resolutions[1]^(-2))
	lines!(ax, resolutions, (resolutions) .^ (-2) .* scaling_factor, label = L"N^{-2}")

	axislegend(ax)
	display(GLMakie.Screen(), fig)
end

function plot_l2error_vs_q(q, l2errors_vs_q)

   n_q = size(q, 1)

	fig = Figure(size = (800, 600))
	ax = Axis(fig[1, 1], xlabel = "q", ylabel = "L2 error", yscale = log10)
	
	labels = ["Total", L"\hat{r}", L"\hat{θ}", L"\hat{ϕ}", L"∇⋅\textbf{B}", L"B \cdot \nabla \alpha"]
	
	for (i, l2) in enumerate(l2errors_vs_q)
		lines!(ax, q[2:n_q-1, 1, 1], l2, label = labels[i], )
	end

	axislegend(ax, position = :lt, merge = true)
	display(GLMakie.Screen(), fig)
end

function plot_l2_errors_vs_μ(μ, l2errors_vs_μ)

	n_μ = size(μ, 2)

	fig = Figure(size = (800, 600))
	ax = Axis(fig[1, 1], xlabel = "μ", ylabel = "L2 error", yscale = log10)
	
	labels = ["Total", L"\hat{r}", L"\hat{θ}", L"\hat{ϕ}", L"∇⋅\textbf{B}", L"B \cdot \nabla \alpha"]
	
	for (i, l2) in enumerate(l2errors_vs_μ)
		lines!(ax, μ[1, 2:n_μ-1, 1], l2, label = labels[i], )
	end

	axislegend(ax, position = :lt, merge = true)
	display(GLMakie.Screen(), fig)
	
end

function plot_l2_errors_vs_ϕ(ϕ, l2errors_vs_ϕ)

	n_ϕ = size(ϕ, 3)

	fig = Figure(size = (800, 600))
	ax = Axis(fig[1, 1], xlabel = "ϕ", ylabel = "L2 error", yscale = log10)
	
	labels = ["Total", L"\hat{r}", L"\hat{θ}", L"\hat{ϕ}", L"∇⋅\textbf{B}", L"B \cdot \nabla \alpha"]
	
	for (i, l2) in enumerate(l2errors_vs_ϕ)
		lines!(ax, ϕ[1, 1, 2:n_ϕ-1], l2, label = labels[i], )
	end	

	axislegend(ax, position = :lt, merge = true)
	display(GLMakie.Screen(), fig)

end

function plot_line(sol, lscene)
   q, μ, ϕ = sol[1,:], sol[2,:], sol[3,:]
   x = @. √abs(1 - μ^2) / q * cos(ϕ)
   y = @. √abs(1 - μ^2) / q * sin(ϕ)
   z = @. μ / q

   lines!(lscene,x, y, z, color=:silver)
end

function plot_fieldlines(fieldlines, lscene)
   for l in eachindex(fieldlines)
      sol = fieldlines[l]
      plot_line(sol, lscene)
   end
end


function read_gradrubin_data()
   data = readdlm("../misc/gradrubin_final.dat")

   q = reshape(data[:,1], 162, 82, 162)
   θ = reshape(data[:,2], 162, 82, 162)
   ϕ = reshape(data[:,3], 162, 82, 162)

   Br = reshape(data[:,7], 162, 82, 162)
   Bθ = reshape(data[:,8], 162, 82, 162)
   Bϕ = reshape(data[:,9], 162, 82, 162)

   α = reshape(data[:,10], 162, 82, 162)

   ∇B = reshape(data[:,12], 162, 82, 162)

   return q, θ, ϕ, Br, Bθ, Bϕ, α, ∇B
end

function evaluate_on_gradrubin_grid(q, μ, ϕ, pinn, Θ_trained, st, params; use_θ = false)
	
	n_q = length(q[:, 1, 1])
	n_μ = length(μ[1, :, 1])
	n_ϕ = length(ϕ[1, 1, :])

	q = reshape(q, 1, :)
	μ = reshape(μ, 1, :)
	ϕ = reshape(ϕ, 1, :)

	NN = pinn(vcat(q, μ, cos.(ϕ), sin.(ϕ)), Θ_trained, st)[1]
	Nr = reshape(NN[1, :], size(q))
	Nθ = reshape(NN[2, :], size(q))
	Nϕ = reshape(NN[3, :], size(q))
	Nα = reshape(NN[4, :], size(q))

	Br1 = Br(q, μ, ϕ, Θ_trained, st, Nr)
	Bθ1 = Bθ(q, μ, ϕ, Θ_trained, st, Nθ)
	Bϕ1 = Bϕ(q, μ, ϕ, Θ_trained, st, Nϕ)
	α1 = α(q, μ, ϕ, Θ_trained, st, Nα, params)
	dBr_dq, dBθ_dq, dBϕ_dq, dα_dq, dBr_dμ, dBθ_dμ, dBϕ_dμ, dα_dμ, dBr_dϕ, dBθ_dϕ, dBϕ_dϕ, dα_dϕ  = calculate_derivatives(q, μ, ϕ, Θ_trained, st, pinn, params)
	∇B = calculate_divergence(q, μ, ϕ, Br1, Bθ1, Bϕ1, dBr_dq, dBθ_dq, dBϕ_dq, dBr_dμ, dBθ_dμ, dBϕ_dμ, dBr_dϕ, dBθ_dϕ, dBϕ_dϕ)
	B∇α = calculate_Bdotgradα(q, μ, ϕ, Br1, Bθ1, Bϕ1, dα_dq, dα_dμ, dα_dϕ)

	α1 = reshape(α1, n_q, n_μ, n_ϕ)
	Br1 = reshape(Br1, n_q, n_μ, n_ϕ)
	Bθ1 = reshape(Bθ1, n_q, n_μ, n_ϕ)
	Bϕ1 = reshape(Bϕ1, n_q, n_μ, n_ϕ)
	∇B = reshape(∇B, n_q, n_μ, n_ϕ)
	B∇α = reshape(B∇α, n_q, n_μ, n_ϕ)

	return Br1, Bθ1, Bϕ1, α1, ∇B, B∇α

end

function L1_error(u1, u2; relative=false)

	if relative
		return sum(abs, u1 .- u2) / sum(abs, u2)
	else
		return sum(abs, u1 .- u2) / length(u2)
	end
	
end

function L2_error(u1, u2; relative=false)

	if relative
		return sqrt(sum(abs2, u1 .- u2)) / sqrt(sum(abs2, u2))
	else
		return sqrt(sum(abs2, u1 .- u2) / length(u2))
	end
end

function L∞_error(u1, u2)
	
	return maximum(abs.(u1 .- u2))
end

function print_pinn_vs_gradrubin_errors(Br1, Bθ1, Bϕ1, α1, ∇B1, Br_gr, Bθ_gr, Bϕ_gr, α_gr, ∇B_gr)
    println("B_r\n",
            "L1 error:                   $(L1_error(Br1, Br_gr))\n",
            "L1 error interior:          $(L1_error(Br1[interior], Br_gr[interior]))\n",
            "L1 error interior relative: $(L1_error(Br1[interior], Br_gr[interior], relative=true))\n",
            "L2 error:                   $(L2_error(Br1, Br_gr))\n",
            "L2 error interior:          $(L2_error(Br1[interior], Br_gr[interior]))\n",
            "L2 error interior relative: $(L2_error(Br1[interior], Br_gr[interior], relative=true))\n",
            "L∞ error:                   $(L∞_error(Br1, Br_gr))\n",
            "L∞ error interior:          $(L∞_error(Br1[interior], Br_gr[interior]))\n"
           )

    println("B_θ\n",
            "L1 error:                   $(L1_error(Bθ1, Bθ_gr))\n",
            "L1 error interior:          $(L1_error(Bθ1[interior], Bθ_gr[interior]))\n",
            "L1 error interior relative: $(L1_error(Bθ1[interior], Bθ_gr[interior], relative=true))\n",
            "L2 error:                   $(L2_error(Bθ1, Bθ_gr))\n",
            "L2 error interior:          $(L2_error(Bθ1[interior], Bθ_gr[interior]))\n",
            "L2 error interior relative: $(L2_error(Bθ1[interior], Bθ_gr[interior], relative=true))\n",
            "L∞ error:                   $(L∞_error(Bθ1, Bθ_gr))\n",
            "L∞ error interior:          $(L∞_error(Bθ1[interior], Bθ_gr[interior]))\n"
           )

    println("B_ϕ\n",
            "L1 error:                   $(L1_error(Bϕ1, Bϕ_gr))\n",
            "L1 error interior:          $(L1_error(Bϕ1[interior], Bϕ_gr[interior]))\n",
            "L1 error interior relative: $(L1_error(Bϕ1[interior], Bϕ_gr[interior], relative=true))\n",
            "L2 error:                   $(L2_error(Bϕ1, Bϕ_gr))\n",
            "L2 error interior:          $(L2_error(Bϕ1[interior], Bϕ_gr[interior]))\n",
            "L2 error interior relative: $(L2_error(Bϕ1[interior], Bϕ_gr[interior], relative=true))\n",
            "L∞ error:                   $(L∞_error(Bϕ1, Bϕ_gr))\n",
            "L∞ error interior:          $(L∞_error(Bϕ1[interior], Bϕ_gr[interior]))\n"
           )

    println("α\n",
            "L1 error:                   $(L1_error(α1, α_gr))\n",
            "L1 error interior:          $(L1_error(α1[interior], α_gr[interior]))\n",
            "L1 error interior relative: $(L1_error(α1[interior], α_gr[interior], relative=true))\n",
            "L2 error:                   $(L2_error(α1, α_gr))\n",
            "L2 error interior:          $(L2_error(α1[interior], α_gr[interior]))\n",
            "L2 error interior relative: $(L2_error(α1[interior], α_gr[interior], relative=true))\n",
            "L∞ error:                   $(L∞_error(α1, α_gr))\n",
            "L∞ error interior:          $(L∞_error(α1[interior], α_gr[interior]))\n"
           )

    println("∇⋅B\n",
            "L1 error:                   $(L1_error(∇B1, ∇B_gr))\n",
            "L1 error interior:          $(L1_error(∇B1[interior], ∇B_gr[interior]))\n",
            "L1 error interior relative: $(L1_error(∇B1[interior], ∇B_gr[interior], relative=true))\n",
            "L2 error:                   $(L2_error(∇B1, ∇B_gr))\n",
            "L2 error interior:          $(L2_error(∇B1[interior], ∇B_gr[interior]))\n",
            "L2 error interior relative: $(L2_error(∇B1[interior], ∇B_gr[interior], relative=true))\n",
            "L∞ error:                   $(L∞_error(∇B1, ∇B_gr))\n",
            "L∞ error interior:          $(L∞_error(∇B1[interior], ∇B_gr[interior]))\n"
           )

end

function ∂_∂q(u, q)

    dq = q[2,1,1] - q[1,1,1]
    ∂u_∂q = zero(u)
    ∂u_∂q[2:end-1, :, :] = (u[3:end, :, :] .- u[1:end-2, :, :]) ./ (2dq)
    ∂u_∂q[1, :, :] = (-3u[1,:,:] + 4u[2,:,:] - 1u[3,:,:]) ./ (2dq)
    ∂u_∂q[end, :, :] = (3u[end,:,:] - 4u[end-1,:,:] + 1u[end-2,:,:]) ./ (2dq)

    return ∂u_∂q
end

function ∂_∂θ(u, θ)

    dθ = θ[1,2,1] - θ[1,1,1]
    ∂u_∂θ = zero(u)
    ∂u_∂θ[:, 2:end-1, :] = (u[:, 3:end, :] .- u[:, 1:end-2, :]) ./ (2dθ)
    ∂u_∂θ[:, 1, :] = (-3u[:, 1,:] + 4u[:, 2,:] - 1u[:, 3,:]) ./ (2dθ)
    ∂u_∂θ[:, end, :] = (3u[:, end,:] - 4u[:, end-1,:] + 1u[:, end-2,:]) ./ (2dθ)

    return ∂u_∂θ
end

function ∂_∂ϕ(u, ϕ)

    dϕ = ϕ[1,1,2] - ϕ[1,1,1]
    ∂u_∂ϕ = zero(u)
    ∂u_∂ϕ[:, :, 2:end-1] = (u[:, :, 3:end] .- u[:, :, 1:end-2]) ./ (2dϕ)
    ∂u_∂ϕ[:, :, 1] = (-3u[:, :, 1] + 4u[:, :, 2] - 1u[:, :, 3]) ./ (2dϕ)
    ∂u_∂ϕ[:, :, end] = (3u[:, :, end] - 4u[:, :, end-1] + 1u[:, :, end-2]) ./ (2dϕ)

    return ∂u_∂ϕ
end