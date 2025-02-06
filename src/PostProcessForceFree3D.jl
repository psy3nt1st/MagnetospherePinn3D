function create_test(n_q, n_μ, n_ϕ, NN, Θ, st, params; use_θ = false)
	
	q = reshape([q for ϕ in range(0, 2π, n_ϕ) for μ in range(-1+1e-5, 1-1e-5, n_μ) for q in range(0, 1, n_q)], 1, :)
	μ = reshape([μ for ϕ in range(0, 2π, n_ϕ) for μ in range(-1+1e-5, 1-1e-5, n_μ) for q in range(0, 1, n_q)], 1, :)
	ϕ = reshape([ϕ for ϕ in range(0, 2π, n_ϕ) for μ in range(-1+1e-5, 1-1e-5, n_μ) for q in range(0, 1, n_q)], 1, :)

	if use_θ
		θ = reshape([θ for ϕ in range(0, 2π, n_ϕ) for θ in range(1e-2, π - 1e-2, n_μ) for q in range(0, 1, n_q)], 1, :)
		μ = cos.(θ)
	end

	# NN = pinn(vcat(q, μ, cos.(ϕ), sin.(ϕ)), Θ_trained, st)[1]
	# Nr = reshape(NN[1, :], size(q))
	# Nθ = reshape(NN[2, :], size(q))
	# Nϕ = reshape(NN[3, :], size(q))
	# Nα = reshape(NN[4, :], size(q))

    Nr = NN[1](vcat(q, μ, cos.(ϕ), sin.(ϕ)), Θ.net1, st[1])[1]
	Nθ = NN[2](vcat(q, μ, cos.(ϕ), sin.(ϕ)), Θ.net2, st[2])[1]
	Nϕ = NN[3](vcat(q, μ, cos.(ϕ), sin.(ϕ)), Θ.net3, st[3])[1]
	Nα = NN[4](vcat(q, μ, cos.(ϕ), sin.(ϕ)), Θ.net4, st[4])[1]

	Br1 = Br(q, μ, ϕ, Θ, st, Nr, params)
	Bθ1 = Bθ(q, μ, ϕ, Θ, st, Nθ)
	Bϕ1 = Bϕ(q, μ, ϕ, Θ, st, Nϕ)
	α1 = α(q, μ, ϕ, Θ, st, Nα, params)
	dBr_dq, dBθ_dq, dBϕ_dq, dα_dq, dBr_dμ, dBθ_dμ, dBϕ_dμ, dα_dμ, dBr_dϕ, dBθ_dϕ, dBϕ_dϕ, dα_dϕ  = calculate_derivatives(q, μ, ϕ, Θ, st, NN, params)[1:12]
	∇B = calculate_divergence(q, μ, ϕ, Br1, Bθ1, Bϕ1, dBr_dq, dBθ_dq, dBϕ_dq, dBr_dμ, dBθ_dμ, dBϕ_dμ, dBr_dϕ, dBθ_dϕ, dBϕ_dϕ)
	B∇α = calculate_Bdotgradα(q, μ, ϕ, Br1, Bθ1, Bϕ1, dα_dq, dα_dμ, dα_dϕ)

	q = reshape(q, n_q, n_μ, n_ϕ)
	μ = reshape(μ, n_q, n_μ, n_ϕ)
	ϕ = reshape(ϕ, n_q, n_μ, n_ϕ)
	α1 = reshape(α1, n_q, n_μ, n_ϕ)
	Br1 = reshape(Br1, n_q, n_μ, n_ϕ)
	Bθ1 = reshape(Bθ1, n_q, n_μ, n_ϕ)
	Bϕ1 = reshape(Bϕ1, n_q, n_μ, n_ϕ)
	∇B = reshape(∇B, n_q, n_μ, n_ϕ)
	B∇α = reshape(B∇α, n_q, n_μ, n_ϕ)
	Nr = reshape(Nr, n_q, n_μ, n_ϕ)
	Nθ = reshape(Nθ, n_q, n_μ, n_ϕ)
	Nϕ = reshape(Nϕ, n_q, n_μ, n_ϕ)
	Nα = reshape(Nα, n_q, n_μ, n_ϕ)

	return q, μ, ϕ, Br1, Bθ1, Bϕ1, α1, ∇B, B∇α, Nr, Nθ, Nϕ, Nα
end

function integrand(x, p)
	# println(x)
	q, μ, ϕ = x
	NN, Θ, st, params = p

	# NN = pinn(vcat(q, μ, cos.(ϕ), sin.(ϕ)), Θ_trained, st)[1]
	# Nr = reshape(NN[1, :], size(q))
	# Nθ = reshape(NN[2, :], size(q))
	# Nϕ = reshape(NN[3, :], size(q))

    Nr = NN[1](vcat(q, μ, cos.(ϕ), sin.(ϕ)), Θ.net1, st[1])[1]
    Nθ = NN[2](vcat(q, μ, cos.(ϕ), sin.(ϕ)), Θ.net2, st[2])[1]
    Nϕ = NN[3](vcat(q, μ, cos.(ϕ), sin.(ϕ)), Θ.net3, st[3])[1]

	return Bmag(q, μ, ϕ, Θ, st, Nr, Nθ, Nϕ, params).^2 ./ q.^4 ./ (8π)
end

function calculate_energy(NN, Θ, st, params)
	
	domain = ([0.0, -1, 0], [1, 1, 2π])
	p = (NN, Θ, st, params)
	prob = IntegralProblem(integrand, domain, p)

	energy = solve(prob, HCubatureJL(), reltol = 1e-7, abstol = 1e-7)

	return energy.u
end

function find_footprints(α1, Br1, μ, ϕ; α_thres = 0.0, Br1_thres = 0.0, μ_thres = 0.7)
	ϕs = Float64[]
	μs = Float64[]

	μ0 = findnearest(μ, μ_thres)

	for k in range(1, size(μ, 3))
		for j in range(1, size(μ, 2))
			if α1[end, j, k] >= α_thres && Br1[end, j, k] > Br1_thres && μ[end, j, k] == μ0
				push!(μs, μ[end, j, k])
				push!(ϕs, ϕ[end, j, k])
			end
		end
	end

	return zip(μs, ϕs)
end

function field_line_equations!(du, u, p, t)
    q, μ, ϕ = u
    NN, Θ, st, params = p
    # println("t = $t, q = $q, μ = $μ, ϕ = $ϕ")
    
    # NN = pinn(vcat(q, μ, cos.(ϕ), sin.(ϕ)), Θ_trained, st)[1]
    # Nr = reshape(NN[1, :], size(q))
    # Nθ = reshape(NN[2, :], size(q))
    # Nϕ = reshape(NN[3, :], size(q))

    Nr = NN[1](vcat(q, μ, cos.(ϕ), sin.(ϕ)), Θ.net1, st[1])[1]
    Nθ = NN[2](vcat(q, μ, cos.(ϕ), sin.(ϕ)), Θ.net2, st[2])[1]
    Nϕ = NN[3](vcat(q, μ, cos.(ϕ), sin.(ϕ)), Θ.net3, st[3])[1]
    # Nα = NN[4](vcat(q, μ, cos.(ϕ), sin.(ϕ)), Θ.net4, st[4])[1]

    Nr = reshape(Nr, size(q))
    Nθ = reshape(Nθ, size(q))
    Nϕ = reshape(Nϕ, size(q))


	Br1 = Br(q, μ, ϕ, Θ, st, Nr, params)
	Bθ1 = Bθ(q, μ, ϕ, Θ, st, Nθ)
	Bϕ1 = Bϕ(q, μ, ϕ, Θ, st, Nϕ)
	B = @. √(Br1^2 + Bθ1^2 + Bϕ1^2)


    du[1] = @. -q^2 * Br1 / B
    du[2] = @. -q * √abs(1-μ^2) * Bθ1 / B
    du[3] = @. q / √abs(1-μ^2) * Bϕ1 / B
	
end

function stop_at_surface(u, t, integrator)
	return u[1] - 1.0
end

function stop_at_large_ϕ(u, t, integrator)
	return abs(u[3]) - 100.0 
end

function stop_at_negative_ϕ(u, t, integrator)

	return u[3] - 0.0
end

function integrate_fieldlines!(fieldlines, footprints, NN, Θ, st, params)
   
   affect!(integrator) = terminate!(integrator)
   cb1 = ContinuousCallback(stop_at_surface, affect!)
   cb2 = ContinuousCallback(stop_at_large_ϕ, affect!)
   cb = CallbackSet(cb1, cb2)

   u0 = [1.0; 0.0; 0.0]
   p = (NN, Θ, st, params)
   tspan = (0.0, 50.0)
   prob = ODEProblem(field_line_equations!, u0, tspan, p)

	for (μ, ϕ) in footprints
      # println("Integrating for μ = $μ, ϕ = $ϕ")
      u0 = [1.0; μ; ϕ]
      prob = remake(prob, u0 = u0)
      sol = solve(prob, alg = Tsit5(), callback=cb, abstol=1e-8, reltol=1e-8)
      push!(fieldlines, sol)

		# if abs(abs(sol.u[end][2]) - abs(μ)) > 1e-1 && sol.t[end] < 50
		# 	println("ϕ = $ϕ, initial μ = $μ, final μ = $(sol.u[end][2]), difference = $(abs(abs(sol.u[end][2]) - abs(μ)))")
		# end
   end

	sol = solve(prob, alg = Tsit5(), callback=cb, abstol=1e-8, reltol=1e-8)

	return sol
end

function findnearest(A::AbstractArray, t) 
   
   return A[findmin(x->abs(x-t), A)[2]]
end