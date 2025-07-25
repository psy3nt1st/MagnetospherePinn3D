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
    dαS_dt, d2αS_dq2, dαS_dμ, d2αS_dμ2, d2αS_dϕ2  = calculate_derivatives(q, μ, ϕ, t, q1, Θ, st, NN, params)
	
    Nr, Nθ, Nϕ, Nα = evaluate_subnetworks(q, μ, ϕ, t, Θ, st, NN)

	Br1 = Br(q, μ, ϕ, Nr, params)
	Bθ1 = Bθ(q, μ, ϕ, Nθ)
	Bϕ1 = Bϕ(q, μ, ϕ, Nϕ)
	α1 = α(q, μ, ϕ, t, Nα, params)

	r_eq = calculate_r_equation(q, μ, ϕ, Br1, Bθ1, Bϕ1, α1, dBr_dq, dBθ_dq, dBϕ_dq, dBr_dμ, dBθ_dμ, dBϕ_dμ, dBr_dϕ, dBθ_dϕ, dBϕ_dϕ)
	θ_eq = calculate_θ_equation(q, μ, ϕ, Br1, Bθ1, Bϕ1, α1, dBr_dq, dBθ_dq, dBϕ_dq, dBr_dμ, dBθ_dμ, dBϕ_dμ, dBr_dϕ, dBθ_dϕ, dBϕ_dϕ)
	ϕ_eq = calculate_ϕ_equation(q, μ, ϕ, Br1, Bθ1, Bϕ1, α1, dBr_dq, dBθ_dq, dBϕ_dq, dBr_dμ, dBθ_dμ, dBϕ_dμ, dBr_dϕ, dBθ_dϕ, dBϕ_dϕ)

	∇B = calculate_divergence(q, μ, ϕ, Br1, Bθ1, Bϕ1, dBr_dq, dBθ_dq, dBϕ_dq, dBr_dμ, dBθ_dμ, dBϕ_dμ, dBr_dϕ, dBθ_dϕ, dBϕ_dϕ)
	B∇α = calculate_Bdotgradα(q, μ, ϕ, Br1, Bθ1, Bϕ1, dα_dq, dα_dμ, dα_dϕ) 

	# Calculate loss
	l1 = sum(abs2, r_eq)
	l2 = sum(abs2, θ_eq)
	l3 = sum(abs2, ϕ_eq)
	l4 = sum(abs2, ∇B)
	l5 = sum(abs2, B∇α)

    lmax1 = maximum(abs, r_eq .* .√(1 .- μ .^ 2))
    lmax2 = maximum(abs, θ_eq .* .√(1 .- μ .^ 2))
    lmax3 = maximum(abs, ϕ_eq .* .√(1 .- μ .^ 2))
    lmax4 = maximum(abs, ∇B .* .√(1 .- μ .^ 2))
    lmax5 = maximum(abs, B∇α .* .√(1 .- μ .^ 2))

    ls = [l1, l2, l3, l4, l5] ./ params.architecture.N_points 
    lmaxs = [lmax1, lmax2, lmax3, lmax4, lmax5]

    return sum(ls), ls, lmaxs, r_eq, θ_eq, ϕ_eq, ∇B, B∇α

end

test_loss, ls, lmaxs, r_eq, θ_eq, ϕ_eq, ∇B, B∇α = test_loss_function(vcat(test_input...), Θ_trained, st, NN, params)

r_eq = reshape(r_eq, size(q))
θ_eq = reshape(θ_eq, size(q))
ϕ_eq = reshape(ϕ_eq, size(q))
∇B = reshape(∇B, size(q))
B∇α = reshape(B∇α, size(q))

println("test_loss = $test_loss, ls = $ls, lmaxs = $lmaxs")

# indices = [(:, 10, 40), (:, 1, 40), (:, 1, 1), (:, 40, 40), (:, 40, 1), (:, 20, 40)]
# f = Figure()
# ax = Axis(f[1, 1])
# for idx in indices
# 	lines!(ax, q[idx...], abs.(r_eq[idx...]), label="θ = $(acos(μ[end, idx[2], 1]) * 180 / π)°, ϕ = $(ϕ[end, 1, idx[3]])°")
# end
# axislegend(ax, position=:lt)
# display(GLMakie.Screen(), f)


# f, plt = plot_volume(q, μ, ϕ, abs.(r_eq .* .√(1 .- μ .^ 2)), title="r equation")
# f, plt = plot_volume(q, μ, ϕ, abs.(θ_eq .* .√(1 .- μ .^ 2)), title="θ equation")
# f, plt = plot_volume(q, μ, ϕ, abs.(ϕ_eq), title="ϕ equation")
# f, plt = plot_volume(q, μ, ϕ, abs.(∇B .* .√(1 .- μ .^ 2)), title="∇B equation")
# f, plt = plot_volume(q, μ, ϕ, abs.(B∇α .* .√(1 .- μ .^ 2)), title="B∇α equation")
