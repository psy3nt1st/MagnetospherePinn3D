using MagnetospherePinn3D
using JLD2
using PrettyPrint
using Plots
using PyCall
using Dates
using LaTeXStrings
using Parameters

pygui(:qt5)
pyplot()
pygui(true)

dirs = filter(isdir, readdir("data"; join=true))
datadir = dirs[end]
datadir = "data/ff3d_2709874/3"
println("Using data in $datadir")

const params = initialize(joinpath(datadir, "configFF.toml"))
vars = Variables([zeros(1, params.optimization.N_points) |> params.general.dev .|> Float64 for _ in 1:fieldcount(Variables)]...)

# Create neural network
Θ_trained = load(joinpath(datadir,"trained_model.jld2"), "Θ_trained") 
losses = load(joinpath(datadir, "losses_vs_iterations.jld2"), "losses")

NN, _, st = create_neural_network(params)

function create_input(q_fd, μ_fd, ϕ_fd)

	q = hcat([q for q in q_fd, μ in μ_fd, ϕ in ϕ_fd]...)
	μ = hcat([μ for q in q_fd, μ in μ_fd, ϕ in ϕ_fd]...)
	ϕ = hcat([ϕ for q in q_fd, μ in μ_fd, ϕ in ϕ_fd]...)

	return q, μ, ϕ, vcat(q, μ, cos.(ϕ), sin.(ϕ)) 
end

@with_kw struct Derivatives{N1, N2, N3}
	∂Br∂q::Array{Float64,3} = zeros(Float64, N1, N2, N3)
	∂Bθ∂q::Array{Float64,3} = zeros(Float64, N1, N2, N3)
	∂Bϕ∂q::Array{Float64,3} = zeros(Float64, N1, N2, N3)
	∂α∂q::Array{Float64,3} = zeros(Float64, N1, N2, N3)
	∂Br∂μ::Array{Float64,3} = zeros(Float64, N1, N2, N3)
	∂Bθ∂μ::Array{Float64,3} = zeros(Float64, N1, N2, N3)
	∂Bϕ∂μ::Array{Float64,3} = zeros(Float64, N1, N2, N3)
	∂α∂μ::Array{Float64,3} = zeros(Float64, N1, N2, N3)
	∂Br∂ϕ::Array{Float64,3} = zeros(Float64, N1, N2, N3)
	∂Bθ∂ϕ::Array{Float64,3} = zeros(Float64, N1, N2, N3)
	∂Bϕ∂ϕ::Array{Float64,3} = zeros(Float64, N1, N2, N3)
	∂α∂ϕ::Array{Float64,3} = zeros(Float64, N1, N2, N3)
end

function calculate_fd_derivatives(q, μ, ϕ, Br, Bθ, Bϕ, α, derivs)
	
	n1 = size(q, 1)
	n2 = size(q, 2)
	n3 = size(q, 3)

	dq, dμ, dϕ = q[2, 1, 1] - q[1, 1, 1], μ[1, 2, 1] - μ[1, 1, 1], ϕ[1, 1, 2] - ϕ[1, 1, 1]

	@unpack ∂Br∂q, ∂Bθ∂q, ∂Bϕ∂q, ∂α∂q, ∂Br∂μ, ∂Bθ∂μ, ∂Bϕ∂μ, ∂α∂μ, ∂Br∂ϕ, ∂Bθ∂ϕ, ∂Bϕ∂ϕ, ∂α∂ϕ = derivs

	# for k in 2:n3-1
	# 	for j in 2:n2-1
	# 		for i in 2:n1-1
	# 			∂Br∂q[i, j, k] = (Br[i+1, j, k] - Br[i-1, j, k]) / (2 * dq)
	# 			∂Bθ∂q[i, j, k] = (Bθ[i+1, j, k] - Bθ[i-1, j, k]) / (2 * dq)
	# 			∂Bϕ∂q[i, j, k] = (Bϕ[i+1, j, k] - Bϕ[i-1, j, k]) / (2 * dq)
	# 			∂α∂q[i, j, k] = (α[i+1, j, k] - α[i-1, j, k]) / (2 * dq)

	# 			∂Br∂μ[i, j, k] = (Br[i, j+1, k] - Br[i, j-1, k]) / (2 * dμ)
	# 			∂Bθ∂μ[i, j, k] = (Bθ[i, j+1, k] - Bθ[i, j-1, k]) / (2 * dμ)
	# 			∂Bϕ∂μ[i, j, k] = (Bϕ[i, j+1, k] - Bϕ[i, j-1, k]) / (2 * dμ)
	# 			∂α∂μ[i, j, k] = (α[i, j+1, k] - α[i, j-1, k]) / (2 * dμ)

	# 			∂Br∂ϕ[i, j, k] = (Br[i, j, k+1] - Br[i, j, k-1]) / (2 * dϕ)
	# 			∂Bθ∂ϕ[i, j, k] = (Bθ[i, j, k+1] - Bθ[i, j, k-1]) / (2 * dϕ)
	# 			∂Bϕ∂ϕ[i, j, k] = (Bϕ[i, j, k+1] - Bϕ[i, j, k-1]) / (2 * dϕ)
	# 			∂α∂ϕ[i, j, k] = (α[i, j, k+1] - α[i, j, k-1]) / (2 * dϕ)
	# 		end
	# 	end
	# end

	∂Br∂q[2:n1-1, 2:n2-1, 2:n3-1] .= (Br[3:n1, 2:n2-1, 2:n3-1] - Br[1:n1-2, 2:n2-1, 2:n3-1]) / (2 * dq)
	∂Bθ∂q[2:n1-1, 2:n2-1, 2:n3-1] .= (Bθ[3:n1, 2:n2-1, 2:n3-1] - Bθ[1:n1-2, 2:n2-1, 2:n3-1]) / (2 * dq)
	∂Bϕ∂q[2:n1-1, 2:n2-1, 2:n3-1] .= (Bϕ[3:n1, 2:n2-1, 2:n3-1] - Bϕ[1:n1-2, 2:n2-1, 2:n3-1]) / (2 * dq)
	∂α∂q[2:n1-1, 2:n2-1, 2:n3-1] .= (α[3:n1, 2:n2-1, 2:n3-1] - α[1:n1-2, 2:n2-1, 2:n3-1]) / (2 * dq)

	∂Br∂μ[2:n1-1, 2:n2-1, 2:n3-1] .= (Br[2:n1-1, 3:n2, 2:n3-1] - Br[2:n1-1, 1:n2-2, 2:n3-1]) / (2 * dμ)
	∂Bθ∂μ[2:n1-1, 2:n2-1, 2:n3-1] .= (Bθ[2:n1-1, 3:n2, 2:n3-1] - Bθ[2:n1-1, 1:n2-2, 2:n3-1]) / (2 * dμ)
	∂Bϕ∂μ[2:n1-1, 2:n2-1, 2:n3-1] .= (Bϕ[2:n1-1, 3:n2, 2:n3-1] - Bϕ[2:n1-1, 1:n2-2, 2:n3-1]) / (2 * dμ)
	∂α∂μ[2:n1-1, 2:n2-1, 2:n3-1] .= (α[2:n1-1, 3:n2, 2:n3-1] - α[2:n1-1, 1:n2-2, 2:n3-1]) / (2 * dμ)

	∂Br∂ϕ[2:n1-1, 2:n2-1, 2:n3-1] .= (Br[2:n1-1, 2:n2-1, 3:n3] - Br[2:n1-1, 2:n2-1, 1:n3-2]) / (2 * dϕ)
	∂Bθ∂ϕ[2:n1-1, 2:n2-1, 2:n3-1] .= (Bθ[2:n1-1, 2:n2-1, 3:n3] - Bθ[2:n1-1, 2:n2-1, 1:n3-2]) / (2 * dϕ)
	∂Bϕ∂ϕ[2:n1-1, 2:n2-1, 2:n3-1] .= (Bϕ[2:n1-1, 2:n2-1, 3:n3] - Bϕ[2:n1-1, 2:n2-1, 1:n3-2]) / (2 * dϕ)
	∂α∂ϕ[2:n1-1, 2:n2-1, 2:n3-1] .= (α[2:n1-1, 2:n2-1, 3:n3] - α[2:n1-1, 2:n2-1, 1:n3-2]) / (2 * dϕ)

	
	∂Br∂q[1, :, :] .= (-3 * Br[1, :, :] + 4 * Br[2, :, :] - Br[3, :, :]) / (2 * dq)
	∂Bθ∂q[1, :, :] .= (-3 * Bθ[1, :, :] + 4 * Bθ[2, :, :] - Bθ[3, :, :]) / (2 * dq)
	∂Bϕ∂q[1, :, :] .= (-3 * Bϕ[1, :, :] + 4 * Bϕ[2, :, :] - Bϕ[3, :, :]) / (2 * dq)
	∂α∂q[1, :, :] .= (-3 * α[1, :, :] + 4 * α[2, :, :] - α[3, :, :]) / (2 * dq)

	∂Br∂μ[:, 1, :] .= (-3 * Br[:, 1, :] + 4 * Br[:, 2, :] - Br[:, 3, :]) / (2 * dμ)
	∂Bθ∂μ[:, 1, :] .= (-3 * Bθ[:, 1, :] + 4 * Bθ[:, 2, :] - Bθ[:, 3, :]) / (2 * dμ)
	∂Bϕ∂μ[:, 1, :] .= (-3 * Bϕ[:, 1, :] + 4 * Bϕ[:, 2, :] - Bϕ[:, 3, :]) / (2 * dμ)
	∂α∂μ[:, 1, :] .= (-3 * α[:, 1, :] + 4 * α[:, 2, :] - α[:, 3, :]) / (2 * dμ)

	∂Br∂ϕ[:, :, 1] .= (-3 * Br[:, :, 1] + 4 * Br[:, :, 2] - Br[:, :, 3]) / (2 * dϕ)
	∂Bθ∂ϕ[:, :, 1] .= (-3 * Bθ[:, :, 1] + 4 * Bθ[:, :, 2] - Bθ[:, :, 3]) / (2 * dϕ)
	∂Bϕ∂ϕ[:, :, 1] .= (-3 * Bϕ[:, :, 1] + 4 * Bϕ[:, :, 2] - Bϕ[:, :, 3]) / (2 * dϕ)
	∂α∂ϕ[:, :, 1] .= (-3 * α[:, :, 1] + 4 * α[:, :, 2] - α[:, :, 3]) / (2 * dϕ)


	∂Br∂q[n1, :, :] .= (3 * Br[n1, :, :] - 4 * Br[n1-1, :, :] + Br[n1-2, :, :]) / (2 * dq)
	∂Bθ∂q[n1, :, :] .= (3 * Bθ[n1, :, :] - 4 * Bθ[n1-1, :, :] + Bθ[n1-2, :, :]) / (2 * dq)
	∂Bϕ∂q[n1, :, :] .= (3 * Bϕ[n1, :, :] - 4 * Bϕ[n1-1, :, :] + Bϕ[n1-2, :, :]) / (2 * dq)
	∂α∂q[n1, :, :] .= (3 * α[n1, :, :] - 4 * α[n1-1, :, :] + α[n1-2, :, :]) / (2 * dq)

	∂Br∂μ[:, n2, :] .= (3 * Br[:, n2, :] - 4 * Br[:, n2-1, :] + Br[:, n2-2, :]) / (2 * dμ)
	∂Bθ∂μ[:, n2, :] .= (3 * Bθ[:, n2, :] - 4 * Bθ[:, n2-1, :] + Bθ[:, n2-2, :]) / (2 * dμ)
	∂Bϕ∂μ[:, n2, :] .= (3 * Bϕ[:, n2, :] - 4 * Bϕ[:, n2-1, :] + Bϕ[:, n2-2, :]) / (2 * dμ)
	∂α∂μ[:, n2, :] .= (3 * α[:, n2, :] - 4 * α[:, n2-1, :] + α[:, n2-2, :]) / (2 * dμ)

	∂Br∂ϕ[:, :, n3] .= (3 * Br[:, :, n3] - 4 * Br[:, :, n3-1] + Br[:, :, n3-2]) / (2 * dϕ)
	∂Bθ∂ϕ[:, :, n3] .= (3 * Bθ[:, :, n3] - 4 * Bθ[:, :, n3-1] + Bθ[:, :, n3-2]) / (2 * dϕ)
	∂Bϕ∂ϕ[:, :, n3] .= (3 * Bϕ[:, :, n3] - 4 * Bϕ[:, :, n3-1] + Bϕ[:, :, n3-2]) / (2 * dϕ)
	∂α∂ϕ[:, :, n3] .= (3 * α[:, :, n3] - 4 * α[:, :, n3-1] + α[:, :, n3-2]) / (2 * dϕ)

	return
end

l2error_r = []
l2error_θ = []
l2error_ϕ = []
l2error_∇B = []
l2error_B∇α = []
resolutions = [50, 75, 100, 125, 150, 175, 200]
function calculate_l2error(l2error_r, l2error_θ, l2error_ϕ, l2error_∇B, l2error_B∇α, resolutions)
	

	for n in resolutions

		n1 = n2 = n3 = n
		q_fd = range(0, 1, n1)
		μ_fd = range(-1, 1, n2)
		ϕ_fd = range(0, 2π, n3)

		q, μ, ϕ, input = create_input(q_fd, μ_fd, ϕ_fd)

		neuralnet = NN(input, Θ_trained, st)[1]
		Nr = reshape(neuralnet[1, :], size(q))
		Nθ = reshape(neuralnet[2, :], size(q))
		Nϕ = reshape(neuralnet[3, :], size(q))
		Nα = reshape(neuralnet[4, :], size(q))

		q1 = reshape(q, n1, n2, n3)
		μ1 = reshape(μ, n1, n2, n3)
		ϕ1 = reshape(ϕ, n1, n2, n3)

		q1[2, 1, 1] - q1[1, 1, 1]
		q1[:, 1, 1]
		μ1[1, 2, 1] - μ1[1, 1, 1]
		μ1[1, :, 1]
		ϕ1[1, 1, 2] - ϕ1[1, 1, 1]
		ϕ1[1, 1, :]

		Br1 = reshape(Br(q, μ, ϕ, Θ_trained, st, Nr, vars), size(q1))
		Bθ1 = reshape(Bθ(q, μ, ϕ, Θ_trained, st, Nθ, vars), size(q1))
		Bϕ1 = reshape(Bϕ(q, μ, ϕ, Θ_trained, st, Nϕ, vars), size(q1))
		α1 = reshape(α(q, μ, ϕ, Θ_trained, st, Nα, vars, params), size(q1))

		derivs = Derivatives{n1, n2, n3}()
		calculate_fd_derivatives(q1, μ1, ϕ1, Br1, Bθ1, Bϕ1, α1, derivs)

		@unpack ∂Br∂q, ∂Bθ∂q, ∂Bϕ∂q, ∂α∂q, ∂Br∂μ, ∂Bθ∂μ, ∂Bϕ∂μ, ∂α∂μ, ∂Br∂ϕ, ∂Bθ∂ϕ, ∂Bϕ∂ϕ, ∂α∂ϕ = derivs

		r_eq = calculate_r_equation(q1, μ1, ϕ1, Br1, Bθ1, Bϕ1, α1, ∂Br∂q, ∂Bθ∂q, ∂Bϕ∂q, ∂Br∂μ, ∂Bθ∂μ, ∂Bϕ∂μ, ∂Br∂ϕ, ∂Bθ∂ϕ, ∂Bϕ∂ϕ)
		θ_eq = calculate_θ_equation(q1, μ1, ϕ1, Br1, Bθ1, Bϕ1, α1, ∂Br∂q, ∂Bθ∂q, ∂Bϕ∂q, ∂Br∂μ, ∂Bθ∂μ, ∂Bϕ∂μ, ∂Br∂ϕ, ∂Bθ∂ϕ, ∂Bϕ∂ϕ)
		ϕ_eq = calculate_ϕ_equation(q1, μ1, ϕ1, Br1, Bθ1, Bϕ1, α1, ∂Br∂q, ∂Bθ∂q, ∂Bϕ∂q, ∂Br∂μ, ∂Bθ∂μ, ∂Bϕ∂μ, ∂Br∂ϕ, ∂Bθ∂ϕ, ∂Bϕ∂ϕ)
		∇B = calculate_divergence(q1, μ1, ϕ1, Br1, Bθ1, Bϕ1, ∂Br∂q, ∂Bθ∂q, ∂Bϕ∂q, ∂Br∂μ, ∂Bθ∂μ, ∂Bϕ∂μ, ∂Br∂ϕ, ∂Bθ∂ϕ, ∂Bϕ∂ϕ)
		B∇α = calculate_Bdotgradα(q1, μ1, ϕ1, Br1, Bθ1, Bϕ1, ∂α∂q, ∂α∂μ, ∂α∂ϕ)
		
		# println(r_eq[2:49, 2:49, 2:49])
		push!(l2error_r, sqrt(sum(abs2, r_eq[2:n1-1, 2:n2-1, 2:n3-1]) / (n-2)^3))
		push!(l2error_θ, sqrt(sum(abs2, θ_eq[2:n1-1, 2:n2-1, 2:n3-1]) / (n-2)^3))
		push!(l2error_ϕ, sqrt(sum(abs2, ϕ_eq[2:n1-1, 2:n2-1, 2:n3-1]) / (n-2)^3))
		push!(l2error_∇B, sqrt(sum(abs2, ∇B[2:n1-1, 2:n2-1, 2:n3-1]) / (n-2)^3))
		push!(l2error_B∇α, sqrt(sum(abs2, B∇α[2:n1-1, 2:n2-1, 2:n3-1]) / (n-2)^3))
		println("resolution: ", n, " l2error_r: ", l2error_r[end])
	end
end

calculate_l2error(l2error_r, l2error_θ, l2error_ϕ, l2error_∇B, l2error_B∇α, resolutions)
l2error_total = l2error_r .+ l2error_θ .+ l2error_ϕ #=.+ l2error_∇B=# .+ l2error_B∇α

plot(resolutions, l2error_total, scale=:log10, label="Total") |> display
plot!(resolutions, l2error_r, scale=:log10, label=L"\hat{r}") |> display
plot!(resolutions, l2error_θ, scale=:log10, label=L"\hat{\theta}") |> display
plot!(resolutions, l2error_ϕ, scale=:log10, label=L"\hat{\phi}") |> display
# plot!(resolutions, l2error_∇B, scale=:log10, label=L"\nabla\cdot \boldsymbol{B}") |> display
plot!(resolutions, l2error_B∇α, scale=:log10, label=L"\boldsymbol{B} \cdot \alpha") |> display
plot!(resolutions, (resolutions) .^ (-2) .+ log10(2), scale=:log10, label=L"N^{-2}") |> display

plot1 = plot(losses, 
				 labels=["Total" L"\hat{r}" L"\hat{θ}" L"\hat{ϕ}" L"∇⋅\boldsymbol{B}" L"\boldsymbol{B}⋅∇α"], 
				 lw=[5 1 1 1 1 1],
				 legendfontsize=10,
				 yaxis=:log) |> display
# ∂Br∂q, ∂Bθ∂q, ∂Bϕ∂q, ∂α∂q, dBr_dμ, dBθ_dμ, dBϕ_dμ, dα_dμ, dBr_dϕ, dBθ_dϕ, dBϕ_dϕ, dα_dϕ  = calculate_derivatives(q, μ, ϕ, Θ_trained, st, NN, vars, params)

# ∂α∂q = reshape(∂α∂q, size(q1))
# sum(abs2, (derivs.∂α∂q[2:49, 2:49, 2:49] .- (∂α∂q[2:49, 2:49, 2:49]))) / √(length(q1[2:49, 2:49, 2:49]))

plot([50, 100, 200, 400, 800, 1600], [50, 100, 200, 400, 800, 1600] .^ (-2), scale=:log10) |> display