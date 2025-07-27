function create_test_input(n_q, n_μ, n_ϕ, params; use_θ = false)

    q = reshape([q for ϕ in range(0, 2π, n_ϕ) for μ in range(-1+1e-2, 1-1e-2, n_μ) for q in range(1e-2, 1, n_q)], 1, :)
	μ = reshape([μ for ϕ in range(0, 2π, n_ϕ) for μ in range(-1+1e-2, 1-1e-2, n_μ) for q in range(1e-2, 1, n_q)], 1, :)
	ϕ = reshape([ϕ for ϕ in range(0, 2π, n_ϕ) for μ in range(-1+1e-2, 1-1e-2, n_μ) for q in range(1e-2, 1, n_q)], 1, :)
    
    if use_θ
		θ = reshape([θ for ϕ in range(0, 2π, n_ϕ) for θ in range(1e-1, π - 1e-1, n_μ) for q in range(1e-2, 1, n_q)], 1, :)
		μ = cos.(θ)
	end

    return q, μ, ϕ
end

function create_test(n_q, n_μ, n_ϕ, NN, Θ, st, params)
	
    q, μ, ϕ = create_test_input(n_q, n_μ, n_ϕ, params, use_θ = true)
    t = zeros(size(q))

	Nr, Nθ, Nϕ, Nα = evaluate_subnetworks(q, μ, ϕ, t, Θ, st, NN)

	Br1 = Br(q, μ, ϕ, Nr, params)
	Bθ1 = Bθ(q, μ, ϕ, Nθ)
	Bϕ1 = Bϕ(q, μ, ϕ, Nϕ)
	α1 = α(q, μ, ϕ, t, Nα, params)

	dBr_dq, dBθ_dq, dBϕ_dq, dα_dq, 
    dBr_dμ, dBθ_dμ, dBϕ_dμ, dα_dμ, 
    dBr_dϕ, dBθ_dϕ, dBϕ_dϕ, dα_dϕ  = calculate_derivatives(q, μ, ϕ, t, Θ, st, NN, params)

    ∇B = calculate_divergence(q, μ, Br1, Bθ1, dBr_dq, dBθ_dμ, dBϕ_dϕ, params)
	B∇α = calculate_Bdotgradα(q, μ, Br1, Bθ1, Bϕ1, dα_dq, dα_dμ, dα_dϕ, params)
    # αS = α(q1, μ, ϕ, t, Nα, params)

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
	Nr = reshape(Nr, n_q, n_μ, n_ϕ)
	Nθ = reshape(Nθ, n_q, n_μ, n_ϕ)
	Nϕ = reshape(Nϕ, n_q, n_μ, n_ϕ)
	Nα = reshape(Nα, n_q, n_μ, n_ϕ)

	return q, μ, ϕ, t, Br1, Bθ1, Bϕ1, α1, ∇B, B∇α, Nr, Nθ, Nϕ, Nα
end

function run_test(datadir)

    params = import_params(joinpath(datadir, "config.toml"))

    # Create neural network
    NN, _, st = create_neural_network(params, test_mode=true)

    Θ_trained = load(joinpath(datadir, "trained_model.jld2"), "Θ_trained")
    losses = load(joinpath(datadir, "losses_vs_iterations.jld2"), "losses")

    # Create test grid
    n_q = 160
    n_μ = 80
    n_ϕ = 160

    # test_input = create_test_input(n_q, n_μ, n_ϕ, params; use_θ = true)

    q, μ, ϕ, t, 
    Br1, Bθ1, Bϕ1, α1, 
    ∇B, B∇α,
    Nr, Nθ, Nϕ, Nα = create_test(n_q, n_μ, n_ϕ, NN, Θ_trained, st, params)
    
    Bmag1 = .√(Br1.^2 .+ Bθ1.^2 .+ Bϕ1.^2)
    θ = acos.(μ)

    @info "Test created"

    return NN, Θ_trained, st, losses, q, μ, ϕ, t, θ, Br1, Bθ1, Bϕ1, α1, ∇B, B∇α, Nr, Nθ, Nϕ, Nα, Bmag1, params
end

function save_test(rundir, NN, Θ_trained, st, losses, q, μ, ϕ, t, θ, Br1, Bθ1, Bϕ1, α1, ∇B, B∇α, Nr, Nθ, Nϕ, Nα, Bmag1, params)
    save_dict = Dict(
        "NN" => NN,
        "Theta" => Θ_trained,
        "st" => st,
        "losses" => losses,
        "q" => q,
        "mu" => μ,
        "phi" => ϕ,
        "t" => t,
        "theta" => θ,
        "Br" => Br1,
        "Btheta" => Bθ1,
        "Bphi" => Bϕ1,
        "alpha" => α1,
        "divB" => ∇B,
        "BdotGradAlpha" => B∇α,
        "Nr" => Nr,
        "Ntheta" => Nθ,
        "Nphi" => Nϕ,
        "Nalpha" => Nα,
        "Bmag" => Bmag1,
        "params" => params
    )

    jldsave(joinpath(rundir, "run_data.jld2"); save_dict)
end

function load_test(rundir::String)
    dict = load(joinpath(rundir, "run_data.jld2"))["save_dict"]
    return (
        NN = dict["NN"],
        Θ_trained = dict["Theta"],
        st = dict["st"],
        losses = dict["losses"],
        q = dict["q"],
        μ = dict["mu"],
        ϕ = dict["phi"],
        t = dict["t"],
        θ = dict["theta"],
        Br1 = dict["Br"],
        Bθ1 = dict["Btheta"],
        Bϕ1 = dict["Bphi"],
        α1 = dict["alpha"],
        ∇B = dict["divB"],
        B∇α = dict["BdotGradAlpha"],
        Nr = dict["Nr"],
        Nθ = dict["Ntheta"],
        Nϕ = dict["Nphi"],
        Nα = dict["Nalpha"],
        Bmag1 = dict["Bmag"],
        params = dict["params"]
    )
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
	
	return √(Br(q, μ, ϕ, Nr, params)[1]^2 + Bθ(q, μ, ϕ, Nθ)[1]^2 + Bϕ(q, μ, ϕ, Nϕ)[1]^2)
end

function energy_integrand(x, p)
	q, μ, ϕ = x
	t1, NN, Θ, st, params = p

    Nr, Nθ, Nϕ, Nα = evaluate_subnetworks(q, μ, ϕ, t1, Θ, st, NN)

	return Bmag(q, μ, ϕ, Nr, Nθ, Nϕ, params)^2 / q^4 / (8π)
end

function calculate_energy(t1, NN, Θ, st, params)
	# tt = t1 * ones(size(q))

    domain = ([0, -1, 0], [1, 1, 2π])
	p = (t1, NN, Θ, st, params)
	prob = IntegralProblem(energy_integrand, domain, p)

	# sol = solve(prob, HCubatureJL(), reltol = 1e-5, abstol = 1e-5, maxiters = 10000)
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

function find_footprints(α1, Br1, μ, ϕ; α_range = 0.0, Br1_range = 0.0, μ_range = 0.7, ϕ_range = [π], r_idx  = 160)
	ϕs = Float64[]
	μs = Float64[]

    if length(μ_range) == 1
	    μ_range = [findnearest(μ, μ_range), findnearest(μ, μ_range)]
    end
    if length(α_range) == 1
        α_range = [findnearest(α1, α_range), findnearest(α1, α_range)]
    end

    ϕ_range = [findnearest(ϕ, _ϕ) for _ϕ in ϕ_range]

    # r_idx = argmin(abs.(r .- 1.5))
    r_idx = 160
	for k in range(1, size(μ, 3))
		for j in range(1, size(μ, 2))
			if (
                α_range[1] ≤ α1[r_idx, j, k] ≤ α_range[2] &&
                Br1[r_idx, j, k] > Br1_range &&
                μ_range[1] ≤ μ[r_idx, j, k] ≤ μ_range[2] &&
                ϕ[r_idx, j, k] in ϕ_range
            )

				push!(μs, μ[r_idx, j, k])
				push!(ϕs, ϕ[r_idx, j, k])
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

function integrate_fieldlines!(fieldlines, α_lines, footprints, t1, NN, Θ, st, params; q_start = 1.0)
   
    affect!(integrator) = terminate!(integrator)
    cb1 = ContinuousCallback(stop_at_surface, affect!)
    cb2 = ContinuousCallback(stop_at_large_ϕ, affect!)
    cb3 = ContinuousCallback(stop_at_negative_ϕ, affect!)
    cb = CallbackSet(cb1, cb2, cb3)

    u0 = [q_start; 0.0; 0.0]
    
    # q1 = ones(size(q))
    # tt = t1 * ones(size(q))
    p = (t1, NN, Θ, st, params)
    tspan = (0.0, 150.0)
    prob = ODEProblem(field_line_equations!, u0, tspan, p)

    for (μ, ϕ) in footprints
        # println("Integrating for μ = $μ, ϕ = $ϕ")
        u0 = [q_start; μ; ϕ]
        prob = remake(prob, u0 = u0)
        sol = solve(prob, alg = Tsit5(), callback=cb, abstol=1e-12, reltol=1e-12)
        α_line = caluclate_α_along_line(sol, t1, Θ, st, NN, params)
        
        push!(α_lines, α_line)
        push!(fieldlines, sol)
    end

	sol = solve(prob, alg = Tsit5(), callback=cb, abstol=1e-12, reltol=1e-12)

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
