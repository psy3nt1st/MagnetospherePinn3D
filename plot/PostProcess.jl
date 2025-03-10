function create_test_input(n_q, n_μ, n_ϕ, t1, params; use_θ = false)

    q = reshape([q for ϕ in range(0, 2π, n_ϕ) for μ in range(-1+1e-2, 1-1e-2, n_μ) for q in range(1e-2, 1, n_q)], 1, :)
	μ = reshape([μ for ϕ in range(0, 2π, n_ϕ) for μ in range(-1+1e-2, 1-1e-2, n_μ) for q in range(1e-2, 1, n_q)], 1, :)
	ϕ = reshape([ϕ for ϕ in range(0, 2π, n_ϕ) for μ in range(-1+1e-2, 1-1e-2, n_μ) for q in range(1e-2, 1, n_q)], 1, :)
    
    if use_θ
		θ = reshape([θ for ϕ in range(0, 2π, n_ϕ) for θ in range(1e-1, π - 1e-1, n_μ) for q in range(1e-2, 1, n_q)], 1, :)
		μ = cos.(θ)
	end

    if params.model.alpha_bc_mode == "diffusive"
        t = t1 * ones(size(q))
        q1 = ones(size(q))

        return q, μ, ϕ, t, q1
    else

        return q, μ, ϕ
    end
end

function create_test(test_input, NN, Θ, st, params)
	
    if params.model.alpha_bc_mode == "diffusive" 
        
        q, μ, ϕ, t, q1 = test_input
    else

        q, μ, ϕ = test_input
        t = zeros(size(q))
        q1 = ones(size(q))
    end

	Nr, Nθ, Nϕ, Nα = evaluate_subnetworks(q, μ, ϕ, t, Θ, st, NN)

	Br1 = Br(q, μ, ϕ, Nr, params)
	Bθ1 = Bθ(q, μ, ϕ, Nθ)
	Bϕ1 = Bϕ(q, μ, ϕ, Nϕ)
	α1 = α(q, μ, ϕ, t, Nα, params)

	dBr_dq, dBθ_dq, dBϕ_dq, dα_dq, 
    dBr_dμ, dBθ_dμ, dBϕ_dμ, dα_dμ, 
    dBr_dϕ, dBθ_dϕ, dBϕ_dϕ, dα_dϕ, 
    dαS_dt, d2αS_dq2, dαS_dμ, d2αS_dμ2, d2αS_dϕ2  = calculate_derivatives(q, μ, ϕ, t, q1, Θ, st, NN, params)

    ∇B = calculate_divergence(q, μ, ϕ, Br1, Bθ1, Bϕ1, dBr_dq, dBθ_dq, dBϕ_dq, dBr_dμ, dBθ_dμ, dBϕ_dμ, dBr_dϕ, dBθ_dϕ, dBϕ_dϕ)
	B∇α = calculate_Bdotgradα(q, μ, ϕ, Br1, Bθ1, Bϕ1, dα_dq, dα_dμ, dα_dϕ)
    αS = α(q1, μ, ϕ, t, Nα, params)

	q = reshape(q, n_q, n_μ, n_ϕ)
	μ = reshape(μ, n_q, n_μ, n_ϕ)
	ϕ = reshape(ϕ, n_q, n_μ, n_ϕ)
    t = reshape(t, n_q, n_μ, n_ϕ)
	α1 = reshape(α1, n_q, n_μ, n_ϕ)
	Br1 = reshape(Br1, n_q, n_μ, n_ϕ)
	Bθ1 = reshape(Bθ1, n_q, n_μ, n_ϕ)
	Bϕ1 = reshape(Bϕ1, n_q, n_μ, n_ϕ)
	∇B = reshape(∇B, n_q, n_μ, n_ϕ)
	B∇α = reshape(B∇α, n_q, n_μ, n_ϕ)
    αS = reshape(αS, n_q, n_μ, n_ϕ)
	Nr = reshape(Nr, n_q, n_μ, n_ϕ)
	Nθ = reshape(Nθ, n_q, n_μ, n_ϕ)
	Nϕ = reshape(Nϕ, n_q, n_μ, n_ϕ)
	Nα = reshape(Nα, n_q, n_μ, n_ϕ)

	return q, μ, ϕ, t, Br1, Bθ1, Bϕ1, α1, ∇B, B∇α, αS, Nr, Nθ, Nϕ, Nα
end

function load_gradrubin_data(output_path::String)

    file = readdir(joinpath(output_path, "silo_output"), join=true)[end]
    @info "Using $file"
    
    f = HDF5.h5open(file, "r")
    r = f[read_attribute(f["r"], "silo").value0][]
    θ = f[read_attribute(f["theta"], "silo").value0][]
    φ = f[read_attribute(f["varphi"], "silo").value0][]
    alpha = f[read_attribute(f["alpha"], "silo").value0][]
    u_q = f[read_attribute(f["u_1"], "silo").value0][]
    u_theta = f[read_attribute(f["u_2"], "silo").value0][]
    u_phi = f[read_attribute(f["u_3"], "silo").value0][]
    B_q = f[read_attribute(f["B_1"], "silo").value0][]
    B_theta = f[read_attribute(f["B_2"], "silo").value0][]
    B_phi = f[read_attribute(f["B_3"], "silo").value0][]

    q = 1 ./ r
    μ = cos.(θ)

    return r, q, θ, μ, φ, alpha, u_q, u_theta, u_phi, B_q, B_theta, B_phi
end

function Bmag(q, μ, ϕ, Nr, Nθ, Nϕ, params)
	
	return .√(Br(q, μ, ϕ, Nr, params)[1].^2 .+ Bθ(q, μ, ϕ, Nθ)[1].^2 .+ Bϕ(q, μ, ϕ, Nϕ)[1].^2)
end

function integrand(x, p)
	q, μ, ϕ = x
	t1, NN, Θ, st, params = p

    Nr, Nθ, Nϕ, Nα = evaluate_subnetworks(q, μ, ϕ, t1, Θ, st, NN)

	return Bmag(q, μ, ϕ, Nr, Nθ, Nϕ, params).^2 ./ q.^4 ./ (8π)
end

function calculate_energy(t1, NN, Θ, st, params)
	# tt = t1 * ones(size(q))

    domain = ([0.0, -1, 0], [1, 1, 2π])
	p = (t1, NN, Θ, st, params)
	prob = IntegralProblem(integrand, domain, p)

	energy = solve(prob, HCubatureJL(), reltol = 1e-7, abstol = 1e-7)

	return energy.u
end

function find_footprints(α1, Br1, μ, ϕ; α_range = 0.0, Br1_range = 0.0, μ_range = 0.7)
	ϕs = Float64[]
	μs = Float64[]

    if length(μ_range) == 1
	    μ_range = [findnearest(μ, μ_range), findnearest(μ, μ_range)]
    end
    if length(α_range) == 1
        α_range = [findnearest(α1, α_range), findnearest(α1, α_range)]
    end

	for k in range(1, size(μ, 3))
		for j in range(1, size(μ, 2))
			if α_range[1] ≤ α1[end, j, k] ≤ α_range[2] && Br1[end, j, k] > Br1_range && μ_range[1] ≤ μ[end, j, k] ≤ μ_range[2]
				push!(μs, μ[end, j, k])
				push!(ϕs, ϕ[end, j, k])
			end
		end
	end

	return zip(μs, ϕs)
end

function field_line_equations!(du, u, p, t)
    q, μ, ϕ = u
    t1, NN, Θ, st, params = p
    
    Nr, Nθ, Nϕ, Nα = evaluate_subnetworks(q, μ, ϕ, t1, Θ, st, NN)

	Br1 = Br(q, μ, ϕ, Nr[1], params)
	Bθ1 = Bθ(q, μ, ϕ, Nθ[1])
	Bϕ1 = Bϕ(q, μ, ϕ, Nϕ[1])
	B = @. √(Br1^2 + Bθ1^2 + Bϕ1^2)

    # println("t = $t, q = $q, μ = $μ, ϕ = $ϕ, α1 = $α1")

    du[1] = @. -q^2 * Br1 / B
    du[2] = @. -q * √abs(1 - μ^2) * Bθ1 / B
    du[3] = @. q / √abs(1 - μ^2) * Bϕ1 / B
	
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

function integrate_fieldlines!(fieldlines, α_lines, footprints, t1, NN, Θ, st, params)
   
    affect!(integrator) = terminate!(integrator)
    cb1 = ContinuousCallback(stop_at_surface, affect!)
    cb2 = ContinuousCallback(stop_at_large_ϕ, affect!)
    cb3 = ContinuousCallback(stop_at_negative_ϕ, affect!)
    cb = CallbackSet(cb1, cb2, cb3)

    u0 = [1.0; 0.0; 0.0]
    
    # q1 = ones(size(q))
    # tt = t1 * ones(size(q))
    p = (t1, NN, Θ, st, params)
    tspan = (0.0, 50.0)
    prob = ODEProblem(field_line_equations!, u0, tspan, p)

    for (μ, ϕ) in footprints
        # println("Integrating for μ = $μ, ϕ = $ϕ")
        u0 = [1.0; μ; ϕ]
        prob = remake(prob, u0 = u0)
        sol = solve(prob, alg = Tsit5(), callback=cb, abstol=1e-8, reltol=1e-8)
        α_line = caluclate_α_along_line(sol, t1, Θ, st, NN, params)
        
        push!(α_lines, α_line)
        push!(fieldlines, sol)
    end

	sol = solve(prob, alg = Tsit5(), callback=cb, abstol=1e-8, reltol=1e-8)

	return sol
end

function caluclate_α_along_line(sol, t1, Θ, st, NN, params)
    
    q, μ, ϕ = sol[1,:], sol[2,:], sol[3,:]

    n = length(q)

    q = reshape(q, 1, n)
    μ = reshape(μ, 1, n)
    ϕ = reshape(ϕ, 1, n)
    t = t1 * ones(1, n)

    subnet_α = NN.layers[4]
    Nα = subnet_α(vcat(q, μ, cos.(ϕ), sin.(ϕ), t), Θ.layer_4, st.layer_4)[1]
    α1 = α(q, μ, ϕ, t, Nα, params)
    α1 = reshape(α1, n)

    return α1
end

function findnearest(A::AbstractArray, t) 
   
   return A[findmin(x -> abs.(x .- t), A)[2]]
end