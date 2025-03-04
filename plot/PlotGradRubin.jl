using GLMakie
using LaTeXStrings
using HDF5

include("Plotting.jl")
include("PostProcess.jl")

output_path = "/home/petros/phys/ff_magnetosphere/modules/gradrubin/Data/results/model_0"

r_gr, q_gr, θ_gr, μ_gr, ϕ_gr, α_gr, u_q_gr, u_theta_gr, u_phi_gr, Br_gr, Bθ_gr, Bϕ_gr = load_gradrubin_data(output_path)

function plot_magnetosphere_3d_gr(α1)
	f = Figure()
	lscene = LScene(f[1,1], show_axis=false)

	star = mesh!(lscene, Sphere(Point3(0, 0, 0), 1.0)
				 , color=abs.(α1), colormap=reverse(cgrad(:gist_heat, 100)), interpolate=true
				#  , colorrange=(0, params.model.alpha0)
				 )
	cbar = Colorbar(f[1, 2], star)

    # Adjust viewing angle
    zoom!(lscene.scene, cameracontrols(lscene.scene), 0.95)
    rotate_cam!(lscene.scene, Vec3f(0.5, 2.2, 0.0))
	display(GLMakie.Screen(), f, update=false)
	save(joinpath("figures", "fieldlines.png"), f, update=false)
end

plot_magnetosphere_3d_gr(α_gr[end-1, :, :])
plot_surface_α(θ_gr[:,end:-1:1,:], ϕ_gr, α_gr[end-1, :, :])


