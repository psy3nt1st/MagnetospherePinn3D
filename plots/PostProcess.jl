function create_test_input(n_q, n_μ, n_ϕ; use_θ = false)

    q = reshape([q for ϕ in range(0, 2π, n_ϕ) for μ in range(-1+1e-2, 1-1e-2, n_μ) for q in range(1e-2, 1, n_q)], 1, :)
	if use_θ
		θ = reshape([θ for ϕ in range(0, 2π, n_ϕ) for θ in range(1e-1, π - 1e-1, n_μ) for q in range(1e-2, 1, n_q)], 1, :)
		μ = cos.(θ)
    else
        μ = reshape([μ for ϕ in range(0, 2π, n_ϕ) for μ in range(-1+1e-2, 1-1e-2, n_μ) for q in range(1e-2, 1, n_q)], 1, :)
	end
	ϕ = reshape([ϕ for ϕ in range(0, 2π, n_ϕ) for μ in range(-1+1e-2, 1-1e-2, n_μ) for q in range(1e-2, 1, n_q)], 1, :)
    
    return q, μ, ϕ
end

function evaluate_on_grid(n_q, n_μ, n_ϕ, NN, Θ, st, config; use_θ = false, extended = false)
	
    griddata = OrderedDict{Symbol, Any}()

    q, μ, ϕ = create_test_input(n_q, n_μ, n_ϕ; use_θ = use_θ)

	Nr, Nθ, Nϕ, Nα = evaluate_subnetworks(q, μ, ϕ, NN, Θ, st)

	Br1 = Br(q, μ, ϕ, Nr, config)
	Bθ1 = Bθ(q, μ, ϕ, Nθ)
	Bϕ1 = Bϕ(q, μ, ϕ, Nϕ)
	α1 = α(q, μ, ϕ, Nα, config)

    q = reshape(q, n_q, n_μ, n_ϕ)
	μ = reshape(μ, n_q, n_μ, n_ϕ)
	ϕ = reshape(ϕ, n_q, n_μ, n_ϕ)
	Br1 = reshape(Br1, n_q, n_μ, n_ϕ)
	Bθ1 = reshape(Bθ1, n_q, n_μ, n_ϕ)
	Bϕ1 = reshape(Bϕ1, n_q, n_μ, n_ϕ)
    α1 = reshape(α1, n_q, n_μ, n_ϕ)

    @pack! griddata = q, μ, ϕ, Br1, Bθ1, Bϕ1, α1

    if extended

        dBr_dq, dBθ_dq, dBϕ_dq, dα_dq, 
        dBr_dμ, dBθ_dμ, dBϕ_dμ, dα_dμ, 
        dBr_dϕ, dBθ_dϕ, dBϕ_dϕ, dα_dϕ  = calculate_derivatives(q, μ, ϕ, NN, Θ, st, config)

        ∇B = calculate_divergence(q, μ, Br1, Bθ1, dBr_dq, dBθ_dμ, dBϕ_dϕ, config)
        B∇α = calculate_Bdotgradα(q, μ, Br1, Bθ1, Bϕ1, dα_dq, dα_dμ, dα_dϕ, config)

        ∇B = reshape(∇B, n_q, n_μ, n_ϕ)
        B∇α = reshape(B∇α, n_q, n_μ, n_ϕ)
        Nr = reshape(Nr, n_q, n_μ, n_ϕ)
        Nθ = reshape(Nθ, n_q, n_μ, n_ϕ)
        Nϕ = reshape(Nϕ, n_q, n_μ, n_ϕ)
        Nα = reshape(Nα, n_q, n_μ, n_ϕ)

        @pack! griddata = q, μ, ϕ, Br1, Bθ1, Bϕ1, α1, ∇B, B∇α, Nr, Nθ, Nϕ, Nα
	end

	return griddata
end

function Bmag(q, μ, ϕ, Nr, Nθ, Nϕ, config)
	
	return √(Br(q, μ, ϕ, Nr, config)[1]^2 + Bθ(q, μ, ϕ, Nθ)[1]^2 + Bϕ(q, μ, ϕ, Nϕ)[1]^2)
end

function energy_integrand(x, p)
	q, μ, ϕ = x
	NN, Θ, st, config = p

    Nr, Nθ, Nϕ, Nα = evaluate_subnetworks(q, μ, ϕ, NN, Θ, st)

	return (Br(q, μ, ϕ, Nr, config)[1]^2 + Bθ(q, μ, ϕ, Nθ)[1]^2 + Bϕ(q, μ, ϕ, Nϕ)[1]^2) / q^4 / (8π)
end

function calculate_energy(NN, Θ, st, config)

    domain = ([0, -1, 0], [1, 1, 2π])
	p = (NN, Θ, st, config)
	prob = IntegralProblem(energy_integrand, domain, p)

	sol = solve(prob, HCubatureJL(), reltol = 1e-5, abstol = 1e-5, maxiters = 10000)
	return sol.u  
end

function toroidal_energy_integrand(x, p)
	q, μ, ϕ = x
	t1, NN, Θ, st, params = p

    Nr, Nθ, Nϕ, Nα = evaluate_subnetworks(q, μ, ϕ, t1, Θ, st, NN)

	return Bϕ(q, μ, ϕ, Nϕ)[1]^2 / q^4 / (8π)
end

function calculate_toroidal_energy(t1, NN, Θ, st, params)
	# tt = t1 * ones(size(q))

    domain = ([0, -1, 0], [1, 1, 2π])
	p = (t1, NN, Θ, st, params)
	prob = IntegralProblem(toroidal_energy_integrand, domain, p)

	sol = solve(prob, HCubatureJL(), reltol = 1e-5, abstol = 1e-5, maxiters = 10000)

	return sol.u
end

function magnetic_virial_surface_integrand(x, p)
    q = 1
    μ, ϕ = x
	t1, NN, Θ, st, params = p

    Nr, Nθ, Nϕ, Nα = evaluate_subnetworks(q, μ, ϕ, t1, Θ, st, NN)

    gr_factor = √(1 - 2 * params.model.M * q)

	return gr_factor^2 * (Br(q, μ, ϕ, Nr, params)[1]^2 - Bθ(q, μ, ϕ, Nθ)[1]^2 - Bϕ(q, μ, ϕ, Nϕ)[1]^2) / (8π)
end

function calculate_magnetic_virial_surface(t1, NN, Θ, st, params)
    
    domain = ([-1, 0], [1, 2π])
	p = (t1, NN, Θ, st, params)
	prob = IntegralProblem(magnetic_virial_surface_integrand, domain, p)

	sol = solve(prob, HCubatureJL(), reltol = 1e-5, abstol = 1e-5, maxiters = 10000)

	return sol.u
end

function magnetic_virial_volume_integrand(x, p)
    q, μ, ϕ = x
    t1, NN, Θ, st, params = p

    Nr, Nθ, Nϕ, Nα = evaluate_subnetworks(q, μ, ϕ, t1, Θ, st, NN)

    gr_factor = √(1 - 2 * params.model.M * q)

    return (1 - gr_factor^2) * (2 * Br(q, μ, ϕ, Nr, params)[1]^2 + Bθ(q, μ, ϕ, Nθ)[1]^2 + Bϕ(q, μ, ϕ, Nϕ)[1]^2) / (8π * q^4)

end

function calculate_magnetic_virial_volume(t1, NN, Θ, st, params)
    
    domain = ([0, -1, 0], [1, 1, 2π])
    p = (t1, NN, Θ, st, params)
    prob = IntegralProblem(magnetic_virial_volume_integrand, domain, p)

    sol = solve(prob, HCubatureJL(), reltol = 1e-5, abstol = 1e-5, maxiters = 10000)

    return sol.u
end

function quadrupole_moment_integrand(x, p)
	q, μ, ϕ = x
	t1, NN, Θ, st, params = p

    Nr, Nθ, Nϕ, Nα = evaluate_subnetworks(q, μ, ϕ, t1, Θ, st, NN)

	return Bmag(q, μ, ϕ, Nr, Nθ, Nϕ, params)^2 / q^6 / (8π) * (cos(ϕ)^2 * (1 - μ^2) - μ^2)
end

function calculate_quadrupole_moment(t1, NN, Θ, st, params)
	# tt = t1 * ones(size(q))

    domain = ([0, -1, 0], [1, 1, 2π])
	p = (t1, NN, Θ, st, params)
	prob = IntegralProblem(quadrupole_moment_integrand, domain, p)

	# sol = solve(prob, HCubatureJL(), reltol = 1e-5, abstol = 1e-5, maxiters = 10000)
	sol = solve(prob, HCubatureJL(), reltol = 1e-5, abstol = 1e-5, maxiters = 10000)
	return sol.u
end

function find_footprints(μ, ϕ, α1; μ_interval = 0..1, ϕ_interval = 0..2π, α_interval = 0.0..maximum(α1))
	
    μs = Float64[]
    ϕs = Float64[]

	for k in range(1, size(ϕ, 3))
		for j in range(1, size(μ, 2))
			if (μ[end, j, k] ∈ μ_interval &&
                ϕ[end, j, k] ∈ ϕ_interval &&
                α1[end, j, k] ∈ α_interval
            )
				push!(μs, μ[end, j, k])
				push!(ϕs, ϕ[end, j, k])
			end
		end
	end

	return zip(μs, ϕs)
end

function field_line_equations!(du, u, p, t)
    q, μ, ϕ = u
    NN, Θ, st, config = p
    
    if isinf(ϕ)
        ϕ = NaN
    end

    Nr, Nθ, Nϕ = evaluate_subnetworks(q, μ, ϕ, NN, Θ, st)[1:3]

	Br1 = Br(q, μ, ϕ, Nr[1], config)
	Bθ1 = Bθ(q, μ, ϕ, Nθ[1])
	Bϕ1 = Bϕ(q, μ, ϕ, Nϕ[1])
	B = @. √(Br1^2 + Bθ1^2 + Bϕ1^2)

    # println("t = $t, q = $q, μ = $μ, ϕ = $ϕ")

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

function integrate_fieldlines(footprints, NN, Θ, st, config; q_start = 1.0)

    fieldlines = Array{NamedTuple}(undef, length(footprints))

    affect!(integrator) = terminate!(integrator)
    cb1 = ContinuousCallback(stop_at_surface, affect!)
    cb2 = ContinuousCallback(stop_at_large_ϕ, affect!)
    cb3 = ContinuousCallback(stop_at_negative_ϕ, affect!)
    cb = CallbackSet(cb1, cb2, cb3)

    u0 = [q_start; 0.0; 0.0]
    p = (NN, Θ, st, config)
    tspan = (0.0, 150.0)
    prob = ODEProblem(field_line_equations!, u0, tspan, p)

    for (i, (μ, ϕ)) in enumerate(footprints)
        # println("Integrating for μ = $μ, ϕ = $ϕ")
        u0 = [q_start; μ; ϕ]
        prob = remake(prob, u0 = u0)
        sol = solve(prob, alg=Tsit5(), callback=cb, abstol=1e-9, reltol=1e-9, dense=false, maxiters = 10000)
        t = sol.t
        q = sol[1, :]
        μ = sol[2, :]
        ϕ = sol[3, :]
        
        α_line = caluclate_α_along_line(q, μ, ϕ, NN, Θ, st, config)
        fieldlines[i] = (;t = t, q = q, μ = μ, ϕ = ϕ, α = α_line)
    end

	return fieldlines
end

function caluclate_α_along_line(q, μ, ϕ, NN, Θ, st, config)
    
    q, μ, ϕ = reshape(q, 1, :), reshape(μ, 1, :), reshape(ϕ, 1, :)

    subnet_α = NN.layers[4]
    # Temporary fix for the case that the saved output was using 5 inputs (mainly old models that had the unused t variable)
    if size(Θ[:layer_1][:layer_1][:weight])[2] == 5
        Nα = subnet_α(vcat(q, μ, cos.(ϕ), sin.(ϕ), zero(q)), Θ.layer_4, st.layer_4)[1]
    else
        Nα = subnet_α(vcat(q, μ, cos.(ϕ), sin.(ϕ)), Θ.layer_4, st.layer_4)[1]
    end

    return reshape(α(q, μ, ϕ, Nα, config), :)
end

function findnearest(A::AbstractArray, t) 
   
   return A[findmin(x -> abs.(x .- t), A)[2]]
end

function load_losses(dir)
    
    # last_losses = Array{Float64}(undef, length(run_dirs))
    # for dir in run_dirs
        if isfile(joinpath(dir, "losses_vs_iterations.jld2"))
            all_losses = load(joinpath(dir, "losses_vs_iterations.jld2"), "losses")
            if !isempty(all_losses)
                total_loss = all_losses[1]
                if !all(isnan.(total_loss))
                    last_loss = filter(!isnan, total_loss)[end]
                else
                    last_loss = NaN
                end
            else
                last_loss = NaN
            end
        else
            last_loss = NaN
        end
    # end
        
    return last_loss
end

function read_time(dir)
    if isfile(joinpath(dir, "execution_time.txt"))
        open((joinpath(dir, "execution_time.txt")), "r") do f
            data = readdlm(f, '\t')
            total_time = parse(Float64, data[2][1:25])
            return total_time
        end
    else
        return 0.0
    end
end

function load_run_data(run_dirs)

    run_configs = [joinpath(dir, "config.toml") for dir in run_dirs]
    run_params = [import_params(config) for config in run_configs]
    losses = [load_losses(dir) for dir in run_dirs]
    NNs = [create_neural_network(params, test_mode=true)[1] for params in run_params]
    sts = [create_neural_network(params, test_mode=true)[3] for params in run_params]
    Θs = [isfile(joinpath(dir, "trained_model.jld2")) ? load(joinpath(dir, "trained_model.jld2"), "Θ_trained") : [] for dir in run_dirs]
    times = [read_time(dir) for dir in run_dirs]

    return run_params, losses, NNs, sts, Θs, times
end

function calculate_dipole_energy(M)

    if M == 0
        return 1/3
    else
        return (3 / (32 * M^6)) * (2M * (M + 1) + log(1 - 2M)) * (2M * (M - 1) + (2M - 1) * log(1 - 2M))
    end
end

function calculate_run_quantities(NNs, Θs, sts, run_params)
    
    Ms = [params.model.M for params in run_params]
    @info "Calculating energies"
    energies = [calculate_energy(NNs[i], Θs[i], sts[i], run_params[i]) for i in eachindex(run_params)]
    dipole_energies = [calculate_dipole_energy(Ms[i]) for i in eachindex(run_params)]
    # dipole_energies = ifelse.(Ms .== 0.0, 1/3, ifelse.(Ms .== 0.1, 0.4376881279484205, ifelse.(Ms .== 0.25, 0.7438769955308882, NaN)))
    excess_energies = (energies .- dipole_energies)
    relative_excess_energies = excess_energies ./ dipole_energies
    @info "Calculating toroidal energies"
    toroidal_energies = [calculate_toroidal_energy(t1, NNs[i], Θs[i], sts[i], run_params[i]) for i in eachindex(run_params)]
    poloidal_energies = energies .- toroidal_energies
    excess_poloidal_energies = (poloidal_energies .- poloidal_energies[1])
    relative_excess_poloidal_energies = excess_poloidal_energies / poloidal_energies[1]
    @info "Calculating magnetic virials"
    magnetic_virials_surface = [calculate_magnetic_virial_surface(t1, NNs[i], Θs[i], sts[i], run_params[i]) for i in eachindex(run_params)]
    magnetic_virials_volume = [calculate_magnetic_virial_volume(t1, NNs[i], Θs[i], sts[i], run_params[i]) for i in eachindex(run_params)]
    magnetic_virials = magnetic_virials_surface .+ magnetic_virials_volume
    @info "Calculating quadrupole moments"
    quadrupole_moments = [calculate_quadrupole_moment(t1, NNs[i], Θs[i], sts[i], run_params[i]) for i in eachindex(run_params)]

    return (energies, excess_energies, relative_excess_energies,
        poloidal_energies, excess_poloidal_energies, relative_excess_poloidal_energies,
        toroidal_energies, magnetic_virials_surface, magnetic_virials_volume, magnetic_virials,
        quadrupole_moments)
end
