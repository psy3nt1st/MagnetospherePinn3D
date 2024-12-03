# Convert entries when appropriate
d["architecture"]["activation"] = getfield(Main, Symbol(d["architecture"]["activation"]))
d["optimization"]["adam"]["optimizer"] = getfield(Main, Symbol(d["optimization"]["adam"]["optimizer"]))
d["optimization"]["bfgs"]["optimizer"] = getfield(Main, Symbol(d["optimization"]["bfgs"]["optimizer"]))

# ================================================================================

function initialize(config_file)

 
	params = import_params(config_file)

	return params
end

# Convert entries when appropriate
d["architecture"]["activation"] = string(d["architecture"]["activation"])
d["general"]["dev"] = string(d["general"]["dev"])
d["general"]["rng"] = string(d["general"]["rng"])
d["optimization"]["adam"]["optimizer"] = string(d["optimization"]["adam"]["optimizer"])
d["optimization"]["bfgs"]["optimizer"] = string(d["optimization"]["bfgs"]["optimizer"])


# =================================================================================

using MagnetospherePinn3D
using PrettyPrint
using Plots
using PyCall
using JLD2
using DelimitedFiles
using Printf

pygui(:qt5)
pyplot()
pygui(true)


const params = initialize()

# Create neural network
NN, Θ, st = create_neural_network(params)
Θ = load("trained_model.jld2", "Θ_trained")

N_1 = 400
N_2 = 200

coef = params.solution.coef

r2 = range(1, 11, N_1)' .* ones(N_2)
θ2 = range(0, 2π, N_2) .* ones(N_1)'
rc2 = 8

q2 = 1 ./ r2
μ2 = cos.(θ2)
qc2 = 1 ./ rc2

P2 = reshape([P(q, μ, qc2, Θ, st, coef, NN)[1] for q in q2[1,:] for μ in μ2[:,1]], N_2, N_1)
Pc2 = P(qc2, 0., qc2, Θ, st, coef, NN)[1]
T2 = reshape(T(P2, Pc2, params), N_2, N_1)
T2[T2 .> 0]

Aϕ2 = @. P2 / (r2 * sin(θ2))
Bϕ2 = @. T2 / (r2 * sin(θ2))

Aϕ2[isnan.(Aϕ2)] .= 0
Bϕ2[isnan.(Bϕ2)] .= 0


x = @.  r2 * sin(θ2)
z = @.  r2 * cos(θ2)


# open("data.txt", "w") do f
   
#    writedlm(f, [vec(r2) vec(θ2) vec(Aϕ2) vec(Bϕ2)])
# end

cntr = contour(x, z 
         , [T2, P2]
         , levels=[100 10]
         , fill=[true false]
         # , cbar=false
         , xlims=(0, 1 / qc2 + 0.5), ylims=(-1 / qc2 - 0.5, 1 / qc2 + 0.5)
         , linewidths = 0
        )

# contour!(x, z 
# 		   , P2
#          , levels=[0,Pc2]
# 		   , cmap=:grays
#          # , cbar=false
# 		  ) |> display

# colorbar = cntr.cbar()


# q_test = range(0, 1, N_1)' .* ones(N_2)
# μ_test = range(-1, 1, N_2) .* ones(N_1)'
qc_test = range(params.solution.qcmin, 1, 100)
Pc_test = [P(qc, 0., qc, Θ, st, coef, NN)[1] for qc in qc_test]


# q1 = reshape(q_test, 1, N_1 * N_2)
# μ1 = reshape(μ_test, 1, N_1 * N_2)
# P1 = reshape([P(q, μ, qc_test[qc_idx], Θ, st, coef, NN)[1] for q in q_test[1,:] for μ in μ_test[:,1]], N_2, N_1)
# Pc = Pc_test[qc_idx]
# T1 = reshape(T(P1, Pc, params), N_2, N_1)

# x = @.  sqrt(1 - μ_test^2) / q_test
# z = @.  μ_test / q_test
# P_exact = @. (1 - μ_test^2) * (coef[1] * 1 * q_test + coef[2] * 3 * μ_test * q_test^2 + coef[3] * (15 * μ_test^2 - 3) / 2 * q_test^3)


# open("data.txt", "w") do f
   
#    writedlm(f, [vec(q_test) vec(μ_test) vec(P1) vec(T1)])
# end


# plot2 = contour(x, z 
# 					 , T1
# 					 , levels=50
# 					 , fill=true
#                 , xlims=(0, 1/qc_test[qc_idx] + 0.5), ylims=(-1/qc_test[qc_idx] - 0.5, 1/qc_test[qc_idx] + 0.5)
#                 , linewidths = 0
# 					) |> display

# contour!(x, z 
# 		   , P1
#          , levels=[Pc_test[qc_idx]]
# 		   , cmap=:grays
# 		  ) |> display

plot(Pc_test, 1 ./ qc_test, yaxis=:log, xlims=(0.2, 1), ylim=(1, 11)) |> display
vline!([Pc]) |> display


# 1 ./ qc_test[abs.(Pc .- Pc_test) .<= 3e-3]


using Roots


qc0 = 1 ./ find_zeros(qc -> P(qc, 0., qc, Θ, st, coef, NN)[1] - 0.3716543, params.solution.qcmin-0.01, 1, M=A42)

P(qc0, 0., qc0, Θ, st, coef, NN)[1]

# ---------------------------------------------------------------------------- #

