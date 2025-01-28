using MagnetospherePinn3D
using JLD2
using GLMakie
using PrettyPrint
using Dates
using DelimitedFiles
using LaTeXStrings
using Parameters

include("PlottingFunctions.jl")


plot_surface_α(μ, ϕ, α_surface)
plot_surface_α(μ, ϕ, α_surface_next)


function plot_magnetosphere_3d(fieldlines, α_surface)
	f = Figure()
	lscene = LScene(f[1,1])
	star = mesh!(lscene, Sphere(Point3(0, 0, 0), 1.0)
				 , color=abs.(α_surface), colormap=reverse(cgrad(:gist_heat, 100)), interpolate=true
				#  , colorrange=(0, maximum(α1))
				 )
				#  , color=abs.(Br1[end, :, :]), colormap=reverse(cgrad(:gist_heat, 100)), interpolate=true, colorrange=(0, maximum(Br1)))
	cbar = Colorbar(f[1, 2], star)
	# plot_fieldlines(fieldlines, lscene)
	display(GLMakie.Screen(), f)
	save(joinpath("figures", "fieldlines.png"), f, update=false)
end

# plot_surface_Br(μ, ϕ, Br1)
plot_magnetosphere_3d(fieldlines, α_surface)
plot_magnetosphere_3d(fieldlines, α_surface_next)


# plot_surface_dα_dt(μ, ϕ, dα_dt)
# plot_surface_diffusive_term(μ, ϕ, diffusive_term)
# plot_surface_advective_term(μ, ϕ, advective_term)


function plot_surface_diffusive_advective_ratio(μ, ϕ, diffusive_term, advective_term)
	
	ratio = abs.(diffusive_term ./ advective_term)
	ratio[α_surface .< 1e-3] .= 0.0
	
	f = Figure()
	ax = Axis(f[1, 1], xlabel="ϕ", ylabel="μ", title=L"Surface $\frac{\nabla^2 \alpha}{\frac{\nabla (B^2) \cdot \nabla \alpha}{B^2}}$")
	
	
	
	cont = contourf!(ϕ[end,1,:], μ[end,2:end-2,1]
						  , transpose(ratio[2:end-2,:])
						  , levels=range(-20,20,100)
						  , colormap=:balance
						  , extendlow=:auto
						  , extendhigh=:auto
						#   , coloscale=:log10
						 )

	cont1 = contour!(ϕ[end,1,:], μ[end,:,1], transpose(α_surface), levels=10, color=:black)
	
	cbar = Colorbar(f[1, 2], cont
						#  , ticks=range(minimum(diffusive_term[end,2:end-2,:]), maximum(diffusive_term[end,2:end-2,:]), 5)
						#  , tickformat = "{:.0f}"
						 )
	
	tightlimits!(ax)
	display(GLMakie.Screen(), f)
	save(joinpath("figures", "surface_diffusive_term.png"), f)
end


# plot_surface_diffusive_advective_ratio(μ, ϕ, diffusive_term, advective_term)


minimum(dα_dt[end,2:end-2,:])
maximum(dα_dt[end,2:end-2,:])

minimum(advective_term[end,2:end-2,:])
maximum(advective_term[end,2:end-2,:])