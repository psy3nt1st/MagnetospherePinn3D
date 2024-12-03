using MagnetospherePinn3D
using Distributions
using GLMakie



dirs = filter(dir -> isdir(dir) && startswith(basename(dir), "ff3d") , readdir(abspath("../data"); join=true))
datadir = dirs[end]
@info "Using data in $datadir"
params = import_params(joinpath(datadir, "config.toml"))
params.architecture.N_points = 1000

input = generate_input(params)

q = input[1,:]
μ = input[2,:]
ϕ = input[3,:]


p2 = q .* q' + μ .* μ'

p = (q .^ 4 + μ .^ 4) ./ sum(q .^ 4 + μ .^ 4)
dist = Distributions.Categorical(p)
sampled_indices = rand(dist, size(q))
q_resampled = q[sampled_indices]
μ_resampled = μ[sampled_indices]

f = Figure()
lscene = LScene(f[1, 1])
surface!(lscene, q, μ, p2, alpha = 0.1)
scatter!(lscene, q_resampled, μ_resampled, ϕ, color=:black)
display(GLMakie.Screen(), f)

x = @. sqrt(1 - μ^2) / q * cos(ϕ)
y = @. sqrt(1 - μ^2) / q * sin(ϕ)
z = @. μ / q 

f = Figure()
lscene = LScene(f[1, 1])
scatter!(lscene, q, μ, ϕ)
display(GLMakie.Screen(), f)

f = Figure()
lscene = LScene(f[1, 1])
star = mesh!(lscene, Sphere(Point3(0, 0, 0), 1.0))
scatter!(lscene, x, y, z, alpha = 0.1, color=:black)
display(GLMakie.Screen(), f)


# using Distributions

# # Define the domain and probabilities
# x = [1.0, 2.0, 3.0, 4.0, 5.0]
# p = [0.1, 0.2, 0.3, 0.2, 0.2]
# p = x .^ 4 / sum(x .^ 4)                     

# # Create the categorical distribution
# dist = Categorical(p)

# # Sample 10 points based on `p`
# n = 10
# sampled_indices = rand(dist, n)
# sampled_points = x[sampled_indices]

# println("Sampled points: ", sampled_points)
