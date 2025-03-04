function test_loss_function(input, Θ, st, NN, params)
	# unpack input
	q = @view input[1:1,:]
	μ = @view input[2:2,:]
	ϕ = @view input[3:3,:]
    t = @view input[4:4,:]
    q1 = @view input[5:5,:]

	# Calculate derivatives
	dBr_dq, dBθ_dq, dBϕ_dq, dα_dq, 
    dBr_dμ, dBθ_dμ, dBϕ_dμ, dα_dμ, 
    dBr_dϕ, dBθ_dϕ, dBϕ_dϕ, dα_dϕ, 
    dαS_dt  = calculate_derivatives(q, μ, ϕ, t, q1, Θ, st, NN, params)
	
    Nr, Nθ, Nϕ, Nα = evaluate_subnetworks(q, μ, ϕ, t, Θ, st, NN)
    subnet_α = NN.layers[4]
    Nα_S = subnet_α(vcat(q1, μ, cos.(ϕ), sin.(ϕ), t), Θ.layer_4, st.layer_4)[1]

	Br1 = Br(q, μ, ϕ, Nr, params)
	Bθ1 = Bθ(q, μ, ϕ, Nθ)
	Bϕ1 = Bϕ(q, μ, ϕ, Nϕ)
	α1 = α(q, μ, ϕ, t, Nα, params)
    αS = α(q1, μ, ϕ, t, Nα_S, params)

	r_eq = calculate_r_equation(q, μ, ϕ, Br1, Bθ1, Bϕ1, α1, dBr_dq, dBθ_dq, dBϕ_dq, dBr_dμ, dBθ_dμ, dBϕ_dμ, dBr_dϕ, dBθ_dϕ, dBϕ_dϕ)
	θ_eq = calculate_θ_equation(q, μ, ϕ, Br1, Bθ1, Bϕ1, α1, dBr_dq, dBθ_dq, dBϕ_dq, dBr_dμ, dBθ_dμ, dBϕ_dμ, dBr_dϕ, dBθ_dϕ, dBϕ_dϕ)
	ϕ_eq = calculate_ϕ_equation(q, μ, ϕ, Br1, Bθ1, Bϕ1, α1, dBr_dq, dBθ_dq, dBϕ_dq, dBr_dμ, dBθ_dμ, dBϕ_dμ, dBr_dϕ, dBθ_dϕ, dBϕ_dϕ)

	∇B = calculate_divergence(q, μ, ϕ, Br1, Bθ1, Bϕ1, dBr_dq, dBθ_dq, dBϕ_dq, dBr_dμ, dBθ_dμ, dBϕ_dμ, dBr_dϕ, dBθ_dϕ, dBϕ_dϕ)
	B∇α = calculate_Bdotgradα(q, μ, ϕ, Br1, Bθ1, Bϕ1, dα_dq, dα_dμ, dα_dϕ) 

    αS_eq = calculate_αS_equation(μ, ϕ, t, q1, αS, dαS_dt)

	# Calculate loss
	l1 = sum(abs2, r_eq)
	l2 = sum(abs2, θ_eq)
	l3 = sum(abs2, ϕ_eq)
	l4 = sum(abs2, ∇B)
	l5 = sum(abs2, B∇α)

    ls = [l1, l2, l3, l4, l5] ./ params.architecture.N_points 

    if params.model.alpha_bc_mode == "diffusive"
        l6 = sum(abs2, αS_eq)
        push!(ls, l6 ./ params.architecture.N_points)  
    end

	if params.optimization.loss_function == "MSE"
		g = identity
	elseif params.optimization.loss_function == "logMSE"
		g = log
	else
		# @warn "Unrecognized loss function: $(params.optimization.loss_function). Using MSE."
		g = identity
	end

    if params.model.alpha_bc_mode == "diffusive"
        return g((l1 + l2 + l3 + l4 + l5 + l6) / params.architecture.N_points), ls, r_eq, θ_eq, ϕ_eq, ∇B, B∇α, αS_eq
    else
        return g((l1 + l2 + l3 + l4 + l5) / params.architecture.N_points), ls
    end

end


test_loss, ls, r_eq, θ_eq, ϕ_eq, ∇B, B∇α, αS_eq = test_loss_function(vcat(test_input...), Θ_trained, st, NN, params)
# idc = argmax(B∇α)

# test_input[1][idc]
# test_input[2][idc]
# test_input[3][idc]

println("test_loss = $test_loss, ls = $ls")
