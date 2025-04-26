function plot_losses(losses)

    labels=["Total" L"\hat{r}" L"\hat{θ}" L"\hat{ϕ}" L"∇⋅\textbf{B}" L"\textbf{B}⋅∇α" L"\alpha_S"]
	linewidths = [5, 2, 2, 2, 2, 2, 2]

    f1 = Figure()
	ax1 = Makie.Axis(f1[1, 1], xlabel="Iteration", ylabel="Loss", yscale=log10)
	
	for (i, l) in enumerate(losses)
		lines!(ax1, l, label=string(labels[i]), linewidth=linewidths[i])
	end
    
	axislegend(ax1)
	display(GLMakie.Screen(), f1)
    save(joinpath("figures", "losses.png"), f1)
end

function plot_line(lscene, sol, α_line, α_max, cmap, params)
    
    q, μ, ϕ = sol[1,:], sol[2,:], sol[3,:]
    
    x = @. √(abs(1 - μ^2)) / q * cos(ϕ)
    y = @. √(abs(1 - μ^2)) / q * sin(ϕ)
    z = @. μ / q

    lines!(lscene, x, y, z
        , color=α_line, colormap=cmap
        # , colorrange=(0.01, params.model.alpha0)
        , colorrange=(0.01, α_max)
        , lowclip=:silver, linewidth = 3)
   
end

function plot_fieldlines(lscene, fieldlines, α_lines, α_max, cmap, params)
   for l in eachindex(fieldlines)
      sol = fieldlines[l]
      α_line = α_lines[l]
      plot_line(lscene, sol, α_line, α_max, cmap, params)
   end
end

function plot_magnetosphere_3d(fieldlines, α1, α_lines, params; plot_lines = true)
    α_max = maximum(α1)

    f = Figure()
	lscene = LScene(f[1,1], show_axis=false)

    cmap = reverse(cgrad(:gist_heat, 100))
    # cmap = cgrad(:beach, rev=true)
	star = mesh!(lscene, Sphere(Point3(0, 0, 0), 1.0)
				 , color=α1, colormap=cmap, interpolate=true
				 , colorrange=(0, α_max)
				 )
	cbar = Colorbar(f[1, 2], star)
	if plot_lines
		plot_fieldlines(lscene, fieldlines, α_lines, α_max, cmap, params)  
    end
    # Adjust viewing angle
    zoom!(lscene.scene, cameracontrols(lscene.scene), 1.4)
    rotate_cam!(lscene.scene, Vec3f(0.5, 2.2, 0.0))
	display(GLMakie.Screen(), f, update=false)
	save(joinpath("figures", "twisted_magnetosphere.png"), f, update=false)
    return f
end

function plot_at_surface(μ, ϕ, u; title="", cmap=:plasma)
	f = Figure()
	ax = Axis(f[1, 1], xlabel="ϕ", ylabel="μ", title=title)
	cont = contourf!(ϕ[end,1,:], μ[end,:,1], transpose(u), levels=100, colormap=cmap)
	cbar = Colorbar(f[1, 2], cont)
	tightlimits!(ax)
	display(GLMakie.Screen(), f)
	# save(joinpath("figures", "surface_alpha.png"), f)
end

function plot_surface_α(μ, ϕ, α_surface, params)
	α_max = maximum(α_surface)
    
    f = Figure()
	ax = Axis(f[1, 1], xlabel="ϕ", ylabel="μ", title="Surface α")
	cont = contourf!(ϕ[end,1,:], μ[end,:,1], transpose(α_surface)
        , levels=range(0, α_max, 100)
        , colormap=reverse(cgrad(:gist_heat))
    )
	cbar = Colorbar(f[1, 2], cont)
	tightlimits!(ax)
	display(GLMakie.Screen(), f)
    # save(joinpath("figures", "surface_alpha.png"), f)
    return f
end

function plot_surface_dα_dt(μ, ϕ, dα_dt)
	f = Figure()
	ax = Axis(f[1, 1], xlabel="ϕ", ylabel="μ", title=L"Surface $\frac{\partial\alpha}{\partial t}$")
	cont = contourf!(ϕ[end,1,:], μ[end,2:end-2,1]
						  , transpose(dα_dt[2:end-2,:])
						  , levels=100
						  , colormap=:plasma)
	cbar = Colorbar(f[1, 2], cont
					    , ticks=range(minimum(dα_dt[2:end-2,:]), maximum(dα_dt[2:end-2,:]), 5)
						 , tickformat = "{:.0f}"
						 )
	tightlimits!(ax)
	display(GLMakie.Screen(), f)
	# save(joinpath("figures", "surface_da_dt.png"), f)
end


function plot_volume(q, μ, ϕ, u; title = "", cmap=:plasma)
	    
    f = Figure(title=title)
	lscene = LScene(f[1,1], show_axis=true)
    Label(f[0, 1], title, fontsize=20, tellwidth=false)

    # cmap = reverse(cgrad(:gist_heat, 100))

    sgrid = SliderGrid(f[2, 1],
                        (label = "yz plane - x axis", range = 1:length(q[:, 1, 1])),
                        (label = "xz plane - y axis", range = 1:length(μ[end, :, 1])),
                        (label = "xy plane - z axis", range = 1:length(ϕ[end, 1, :])),
                      )

    lo = sgrid.layout
    nc = ncols(lo)

    plt = volumeslices!(lscene, q[:, 1, 1], μ[end, :, 1], ϕ[end, 1, :], u, colormap=cmap, interpolate=true)
	cbar = Colorbar(f[1, 2], plt)

    vol = volume!(lscene, 0..1, -1..1, 0..2π, u, colormap=cmap, alpha=0.5)
    sl_yz, sl_xz, sl_xy = sgrid.sliders

    on(sl_yz.value) do v; plt[:update_yz][](v) end
    on(sl_xz.value) do v; plt[:update_xz][](v) end
    on(sl_xy.value) do v; plt[:update_xy][](v) end

    set_close_to!(sl_yz, .5length(q[:, 1, 1]))
    set_close_to!(sl_xz, .5length(μ[end, :, 1]))
    set_close_to!(sl_xy, .5length(ϕ[end, 1, :]))

    # add toggles to show/hide heatmaps
    hmaps = [plt[Symbol(:heatmap_, s)][] for s ∈ (:yz, :xz, :xy)]
    toggles = [Toggle(lo[i, nc + 1], active = true) for i ∈ 1:length(hmaps)]

    map(zip(hmaps, toggles)) do (h, t)
    connect!(h.visible, t.active)
    end
	display(GLMakie.Screen(), f, update=false)
    

    return f, plt
end

function plot_quantities_all_models(Pcs, sigmas, s,  excess_energies, magnetic_virials)

    unique_sigmas = unique(sigmas)
    f = Figure()
    ax1 = Axis(f[1, 1], xlabel="s", ylabel=L"\Delta E / E_0")
    ax2 = Axis(f[1, 2], xlabel="s", ylabel="Mag. virial")
    for sigma in unique_sigmas
        idx = sigmas .== sigma# .&& Pcs .== 0.1
        marker = sigma == 3.0 ? :xcross : :rect
        label = "σ = $(Int(sigma))"
        scatter!(ax1, s[idx], excess_energies[idx], 
            marker = marker, 
            markersize = 15,
            color = Pcs[idx],
            label = label
        )
        scatter!(ax2, s[idx], magnetic_virials[idx], 
            marker = marker, 
            markersize = 15,
            color = Pcs[idx],
            label = label
        )
    end
    Colorbar(f[1,3], label = L"P_c", labelrotation=0)
    axislegend(ax1, merge = false, unique = true)
    display(GLMakie.Screen(), f)
end

function plot_quantities_one_model(s, excess_energies, magnetic_virials; Pc0=0.0)
    
    selection1 = (Pcs .== Pc0) .&& (sigmas .== 3.0)
    selection2 = (Pcs .== Pc0) .&& (sigmas .== 4.0)

    f = Figure()
    ax = Axis(f[1, 1], xlabel="s", ylabel=L"\Delta E / E_0")
    scatter!(s[selection1], excess_energies[selection1], marker=:xcross, markersize=15, label="σ = 3")
    scatter!(s[selection2], excess_energies[selection2], marker=:xcross, markersize=15, label="σ = 4")
    display(GLMakie.Screen(), f)
    axislegend(ax, position=:lt)

    f = Figure()
    ax = Axis(f[1, 1], xlabel="s", ylabel="Mag. virial")
    scatter!(s[selection1], magnetic_virials[selection1], marker=:xcross, markersize=15, label="σ = 3")
    scatter!(s[selection2], magnetic_virials[selection2], marker=:xcross, markersize=15, label="σ = 4")
    display(GLMakie.Screen(), f)
    axislegend(ax, position=:lb)
end
