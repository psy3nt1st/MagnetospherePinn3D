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
    # save(joinpath("figures", "losses.png"), f1)
end

function plot_line(ax, line, α_max, cmap)
    
    # q, μ, ϕ = sol[1,:], sol[2,:], sol[3,:]
    @unpack q, μ, ϕ, α = line
    
    x = @. √(abs(1 - μ^2)) / q * cos(ϕ)
    y = @. √(abs(1 - μ^2)) / q * sin(ϕ)
    z = @. μ / q

    lines!(ax, x, y, z
        , color=α, colormap=cmap
        , colorrange=(0.05, α_max)
        , lowclip=:silver
        , linewidth = 5)
   
end

function plot_fieldlines(ax, fieldlines, α_max, cmap)
    for line in fieldlines
        plot_line(ax, line, α_max, cmap)
   end
end

function plot_magnetosphere_3d(fieldlines, α1; plot_lines = true, use_lscene=false)
    α_max = maximum(α1)
    # α_max = 3.5

    f = Figure()
	
    if use_lscene
        ax = LScene(f[1,1], show_axis=true,)
        zoom!(ax.scene, cameracontrols(ax.scene), 1.4)
        rotate_cam!(ax.scene, Vec3f(0.5, 2.2, 0.0))
    else

        ax = Axis3(f[1, 1], aspect = :data, 
            limits=((-2,2), (-2, 2), (-2,2)),
            # limits=((-20, 15), (-20, 10), (-9, 7)),
            ytickformat = values -> ["$(Int(-value))" for value in values],
            # azimuth = 3.2, elevation = 1.1e-01,
            azimuth = 2.24, elevation = 5.9e-01,
            xlabelsize = 25, ylabelsize = 25, zlabelsize = 25,
            xticklabelsize = 20, yticklabelsize = 20, zticklabelsize = 20,
            xlabel = L"x \ [R]", ylabel = L"y \ [R]", zlabel = L"z \ [R]",
            # xlabelrotation = π/2
            )
    end

    cmap = cgrad(:gist_heat, 100, rev=true)
    # cmap = cgrad(:bone_1, 100, rev=true)
    # cmap = cgrad(:plasma, rev=false)
	star = mesh!(ax, Sphere(Point3(0, 0, 0), 1.0)
				 , color=α1, colormap=cmap, interpolate=true
				 , colorrange=(0.0, α_max)
                # , lowclip=:silver
				 )
	cbar = Colorbar(f[1, 2], star, 
        label=L"α \ [R^{-1}]", labelsize = 25, labelrotation=3π/2,
        ticklabelsize = 20,
        height = Relative(7/8)
        )
    # rowsize!(f.layout, 1, lscene.scene.px_area[].widths[2])
	if plot_lines
		plot_fieldlines(ax, fieldlines, α_max, cmap)  
    end
    # Adjust viewing angle
    
	display(GLMakie.Screen(), f, update=false)
	# save(joinpath("figures", "twisted_magnetosphere.png"), f, update=false)
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

    plt = volumeslices!(lscene, q[:, 1, 1], μ[end, :, 1], ϕ[end, 1, :], u[:, end:-1:1, :], colormap=cmap, interpolate=true)
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


function plot_excess_energy_vs_alphamax(data)
    
    grouped_data = groupby(data, :M)
    groups_to_plot = sort!([g for g in grouped_data if g.M[1] ∈ (0.0, 0.17, 0.25)], by = g -> g.M[1])

    f = Figure()
    ax = Axis(f[1, 1], xlabel=L"\alpha_*^{\mathrm{max}} \ [R^{-1}]", ylabel=L"E_e", ylabelrotation = 0,
        xticklabelsize=tick_size, yticklabelsize=tick_size, 
        xlabelsize=label_size, ylabelsize=label_size,
        # xscale=log10, yscale=log10,
        # ylabelpadding=-20    
    )
    markers = [:rect, :xcross, :circle]
    for (i, g) in enumerate(groups_to_plot)
        condition = g.alpha_max .<= 4
        C = @sprintf "%.2f" g.M[1]
        scatter!(ax, g.alpha_max[condition], g.relative_excess_energies[condition];
            markersize=20, marker=markers[i], 
            # color=color
            )
        lines!(ax, g.alpha_max[condition], g.relative_excess_energies[condition];
            label = L"\mathcal{C} = %$(C)", linewidth=4, 
            # color=color
            )
    end
    # scatter!(ax, x1, y1, marker=:rect, markersize=20, label=L"M/R = 0.0",)
    # lines!(ax, x1, y1, linewidth=4)
    # scatter!(ax, x2, abs.(y2), marker=:xcross, markersize=20, label=L"M/R = 0.1")
    # lines!(ax, x2, abs.(y2), linewidth=4)
    # scatter!(ax, x3, y3, marker=:circle, markersize=20, label=L"M/R = 0.25")
    # lines!(ax, x3, y3, linewidth=4)
    # vlines!(ax, [max_alpha_stable], color=:black, linestyle=:dash, linewidth=2)
    axislegend(ax, position=:lt, merge = false, unique = true, labelsize=tick_size)
    display(GLMakie.Screen(), f)

    return f
end

function plot_magnetic_virial_vs_alphamax(data)

    max_alpha_stable = 2.5

    grouped_data = groupby(data, :M)
    x1 = grouped_data[1].alpha_max[grouped_data[1].alpha_max .<= 3]
    y1 = grouped_data[1].magnetic_virials[grouped_data[1].alpha_max .<= 3] ./ grouped_data[1].energies[grouped_data[1].alpha_max .<= 3]
    x2 = grouped_data[2].alpha_max[grouped_data[2].alpha_max .<= 3]
    y2 = grouped_data[2].magnetic_virials[grouped_data[2].alpha_max .<= 3] ./ grouped_data[2].energies[grouped_data[2].alpha_max .<= 3]
    x3 = grouped_data[3].alpha_max[grouped_data[3].alpha_max .<= 3]
    y3 = grouped_data[3].magnetic_virials[grouped_data[3].alpha_max .<= 3] ./ grouped_data[3].energies[grouped_data[3].alpha_max .<= 3]

    f = Figure()
    ax = Axis(f[1, 1], xlabel=L"α_*^{\mathrm{max}} \ [R^{-1}]", ylabel=L"\frac{E_{\mathrm{v}}}{E}", ylabelrotation=0,
        xticklabelsize=tick_size, yticklabelsize=tick_size, 
        xlabelsize=label_size, ylabelsize=label_size,
        # ylabelpadding=-40  
    )
    scatter!(ax, x1, y1, marker=:rect, markersize=20, label=L"M/R = 0.0")
    lines!(ax, x1, y1, linewidth=4)
    scatter!(ax, x2, y2, marker=:xcross, markersize=20, label=L"M/R = 0.1")
    lines!(ax, x2, y2, linewidth=4)
    scatter!(ax, x3, y3, marker=:circle, markersize=20, label=L"M/R = 0.25")
    lines!(ax, x3, y3, linewidth=4)
    vlines!(ax, [max_alpha_stable], color=:black, linestyle=:dash, linewidth=2)
    axislegend(ax, position=:lb, merge=false, unique=true, labelsize=label_size)
    display(GLMakie.Screen(), f)

    return f
end

function plot_quadrupole_moment_vs_alphamax(data)
    max_alpha_stable = 2.5
    grouped_data = groupby(data, :M)
    groups_to_plot = sort!([g for g in grouped_data if g.M[1] ∈ (0.0, 0.17, 0.25)], by = g -> g.M[1])

    f = Figure()
    ax = Axis(f[1, 1], xlabel=L"α_0 \ [R^{-1}]", ylabel=L"\frac{Q^{22}}{Q^{22}_0}", ylabelrotation=0,
        xticklabelsize=tick_size, yticklabelsize=tick_size,
        xlabelsize=label_size, ylabelsize=label_size,
    )

    markers = [:rect, :xcross, :circle]
    # markers = [:xcross]
    # color = :orange
    for (i, g) in enumerate(groups_to_plot)
        condition = g.alpha_max .<= 2.5
        C = @sprintf "%.2f" g.M[1]
        scatter!(ax, g.alpha_max[condition], g.quadrupole_moments[condition] ./ g.quadrupole_moments[1];
            markersize=20, marker=markers[i], 
            # color=color
            )
        lines!(ax, g.alpha_max[condition], g.quadrupole_moments[condition] ./ g.quadrupole_moments[1];
            label = L"\mathcal{C} = %$(C)", linewidth=4, 
            # color=color
            )
    end
    # vlines!(ax, [max_alpha_stable], color=:black, linestyle=:dash, linewidth=2)
    axislegend(ax, position=:lb, merge=false, unique=true, labelsize=label_size)
    display(GLMakie.Screen(), f)

    return f
end


function losses_vs_alphamax(data)
    max_alpha_stable = 2.5
    grouped_data = groupby(data, :M)

    x1 = grouped_data[1].alpha_max[grouped_data[1].alpha_max .<= 3][2:end]
    y1 = grouped_data[1].losses[grouped_data[1].alpha_max .<= 3][2:end]
    x2 = grouped_data[2].alpha_max[grouped_data[2].alpha_max .<= 3][2:end]
    y2 = grouped_data[2].losses[grouped_data[2].alpha_max .<= 3][2:end]
    x3 = grouped_data[3].alpha_max[grouped_data[3].alpha_max .<= 3][2:end]
    y3 = grouped_data[3].losses[grouped_data[3].alpha_max .<= 3][2:end]

    f = Figure()
    ax = Axis(f[1, 1], xlabel=L"\alpha_*^{\mathrm{max}} \ [R^{-1}]", ylabel="Loss", yscale=log10,
        xticklabelsize=tick_size, yticklabelsize=tick_size,
        xlabelsize=label_size, ylabelsize=label_size
    )
    scatter!(ax, x1, y1, marker=:rect, markersize=20, label=L"M/R = 0.0")
    lines!(ax, x1, y1, linewidth=4)
    scatter!(ax, x2, y2, marker=:xcross, markersize=20, label=L"M/R = 0.1")
    lines!(ax, x2, y2, linewidth=4)
    scatter!(ax, x3, y3, marker=:circle, markersize=20, label=L"M/R = 0.25")
    lines!(ax, x3, y3, linewidth=4)
    vlines!(ax, [max_alpha_stable], color=:black, linestyle=:dash, linewidth=2)
    axislegend(ax, position=:lt, merge = false, unique = true, labelsize=label_size)
    display(GLMakie.Screen(), f)

    return f
end

function error_vs_alphamax(data)
    max_alpha_stable = 2.5
    grouped_data = groupby(data, :M)

    x1 = grouped_data[1].alpha_max[grouped_data[1].alpha_max .<= 3][2:end]
    y1 = .√grouped_data[1].losses[grouped_data[1].alpha_max .<= 3][2:end]
    x2 = grouped_data[2].alpha_max[grouped_data[2].alpha_max .<= 3][2:end]
    y2 = .√grouped_data[2].losses[grouped_data[2].alpha_max .<= 3][2:end]
    x3 = grouped_data[3].alpha_max[grouped_data[3].alpha_max .<= 3][2:end]
    y3 = .√grouped_data[3].losses[grouped_data[3].alpha_max .<= 3][2:end]

    f = Figure()
    ax = Axis(f[1, 1], xlabel=L"\alpha_*^{\mathrm{max}} \ [R^{-1}]", ylabel=L"\varepsilon", yscale=log10, ylabelrotation=0,
        xticklabelsize=tick_size, yticklabelsize=tick_size,
        xlabelsize=label_size, ylabelsize=label_size
    )
    scatter!(ax, x1, y1, marker=:rect, markersize=20, label=L"M/R = 0.0")
    lines!(ax, x1, y1, linewidth=4)
    scatter!(ax, x2, y2, marker=:xcross, markersize=20, label=L"M/R = 0.1")
    lines!(ax, x2, y2, linewidth=4)
    scatter!(ax, x3, y3, marker=:circle, markersize=20, label=L"M/R = 0.25")
    lines!(ax, x3, y3, linewidth=4)
    vlines!(ax, [max_alpha_stable], color=:black, linestyle=:dash, linewidth=2)
    axislegend(ax, position=:lt, merge = false, unique = true, labelsize=label_size)
    display(GLMakie.Screen(), f)

    return f
end

function plot_excess_energy_vs_sigma(data)
    max_alpha_stable = 2.5

    grouped_data = groupby(data, :M)

    x1 = grouped_data[1].sigma
    y1 = grouped_data[1].relative_excess_energies
    x2 = grouped_data[2].sigma
    y2 = grouped_data[2].relative_excess_energies
    x3 = grouped_data[3].sigma
    y3 = grouped_data[3].relative_excess_energies

    f = Figure()
    ax = Axis(f[1, 1], xlabel=L"\sigma", ylabel=L"E_e", ylabelrotation = 0,
        xticklabelsize=tick_size, yticklabelsize=tick_size, 
        xlabelsize=label_size, ylabelsize=label_size,   
    )
    scatter!(ax, x1, y1, marker=:rect, markersize=20, label=L"M/R = 0.0",)
    lines!(ax, x1, y1, linewidth=4)
    scatter!(ax, x2, y2, marker=:rect, markersize=20, label=L"M/R = 0.1",)
    lines!(ax, x2, y2, linewidth=4)
    scatter!(ax, x3, y3, marker=:rect, markersize=20, label=L"M/R = 0.25",)
    lines!(ax, x3, y3, linewidth=4)
    axislegend(ax, position=:lt, merge = false, unique = true, labelsize=tick_size)
    display(GLMakie.Screen(), f)
    return f
end

function plot_quadrupole_moment_vs_sigma(data)
    max_sigma_stable = 0.275
    grouped_data = groupby(data, :M)
    groups_to_plot = sort!([g for g in grouped_data if g.M[1] ∈ (0.17)], by = g -> g.M[1])

    f = Figure()
    ax = Axis(f[1, 1], xlabel=L"\sigma", ylabel=L"\frac{Q^{22}}{Q^{22}_0}", ylabelrotation=0,
        xticklabelsize=tick_size, yticklabelsize=tick_size,
        xlabelsize=label_size, ylabelsize=label_size,
    )

    # markers = [:rect, :xcross, :circle]
    markers = [:xcross]
    color = :orange
    for (i, g) in enumerate(groups_to_plot)
        condition = g.sigma .< 0.3
        C = @sprintf "%.2f" g.M[1]
        scatter!(ax, g.sigma[condition], g.quadrupole_moments[condition] ./ g.quadrupole_moments[1];
            markersize=20, marker=markers[i], color=color
            )
        lines!(ax, g.sigma[condition], g.quadrupole_moments[condition] ./ g.quadrupole_moments[1];
            label = L"\mathcal{C} = %$(C)", linewidth=4, color=color
            )
    end
    # vlines!(ax, [max_sigma_stable], color=:black, linestyle=:dash, linewidth=2)
    axislegend(ax, position=:lb, merge=false, unique=true, labelsize=label_size)
    display(GLMakie.Screen(), f)

    return f
end

function plot_magnetic_virial_vs_sigma(data)
    max_alpha_stable = 2.5

    grouped_data = groupby(data, :M)

    x1 = grouped_data[1].sigma
    y1 = grouped_data[1].magnetic_virials ./ grouped_data[1].energies
    x2 = grouped_data[2].sigma
    y2 = grouped_data[2].magnetic_virials ./ grouped_data[2].energies
    x3 = grouped_data[3].sigma
    y3 = grouped_data[3].magnetic_virials ./ grouped_data[3].energies

    f = Figure()
    ax = Axis(f[1, 1], xlabel=L"\sigma", ylabel=L"\frac{E_v}{E}", ylabelrotation = 0,
        xticklabelsize=tick_size, yticklabelsize=tick_size, 
        xlabelsize=label_size, ylabelsize=label_size,   
    )
    scatter!(ax, x1, y1, marker=:rect, markersize=20, label=L"M/R = 0.0",)
    lines!(ax, x1, y1, linewidth=4)
    scatter!(ax, x2, y2, marker=:rect, markersize=20, label=L"M/R = 0.1",)
    lines!(ax, x2, y2, linewidth=4)
    scatter!(ax, x3, y3, marker=:rect, markersize=20, label=L"M/R = 0.25",)
    lines!(ax, x3, y3, linewidth=4)
    axislegend(ax, position=:lt, merge = false, unique = true, labelsize=tick_size)
    display(GLMakie.Screen(), f)
    return f
end

function plot_excess_energy_vs_theta1(data)
    max_alpha_stable = 2.5

    grouped_data = groupby(data, :M)

    x1 = grouped_data[1].theta1
    y1 = grouped_data[1].relative_excess_energies
    x2 = grouped_data[2].theta1
    y2 = grouped_data[2].relative_excess_energies
    x3 = grouped_data[3].theta1
    y3 = grouped_data[3].relative_excess_energies

    f = Figure()
    ax = Axis(f[1, 1], xlabel=L"\theta_1", ylabel=L"E_e", ylabelrotation = 0,
        xticklabelsize=tick_size, yticklabelsize=tick_size, 
        xlabelsize=label_size, ylabelsize=label_size,   
    )
    scatter!(ax, x1, y1, marker=:rect, markersize=20, label=L"M/R = $(grouped_data[1].M[1])",)
    lines!(ax, x1, y1, linewidth=4)
    scatter!(ax, x2, y2, marker=:rect, markersize=20, label=L"M/R = $(grouped_data[2].M[1])",)
    lines!(ax, x2, y2, linewidth=4)
    scatter!(ax, x3, y3, marker=:rect, markersize=20, label=L"M/R = $(grouped_data[3].M[1])",)
    lines!(ax, x3, y3, linewidth=4)
    axislegend(ax, position=:lt, merge = false, unique = true, labelsize=tick_size)
    display(GLMakie.Screen(), f)
    return f
end

function plot_quadrupole_moment_vs_theta1(data)
    max_theta1_stable = 40
    grouped_data = groupby(data, :M)
    groups_to_plot = sort!([g for g in grouped_data if g.M[1] ∈ (0.17)], by = g -> g.M[1])

    f = Figure()
    ax = Axis(f[1, 1], xlabel=L"\theta_1", ylabel=L"\frac{Q^{22}}{Q^{22}_0}", ylabelrotation=0,
        xticklabelsize=tick_size, yticklabelsize=tick_size,
        xlabelsize=label_size, ylabelsize=label_size,
    )

    # markers = [:rect, :xcross, :circle]
    markers = [:xcross]
    color = :orange
    for (i, g) in enumerate(groups_to_plot)
        condition = g.theta1 .>= 40
        C = @sprintf "%.2f" g.M[1]
        scatter!(ax, g.theta1[condition], g.quadrupole_moments[condition] ./ g.quadrupole_moments[1];
            markersize=20, marker=markers[i], color=color
            )
        lines!(ax, g.theta1[condition], g.quadrupole_moments[condition] ./ g.quadrupole_moments[1];
            label = L"\mathcal{C} = %$(C)", linewidth=4, color=color)
    end
    # vlines!(ax, [max_theta1_stable], color=:black, linestyle=:dash, linewidth=2)
    axislegend(ax, position=:rb, merge=false, unique=true, labelsize=label_size)
    display(GLMakie.Screen(), f)

    return f
end

function plot_magnetic_virial_vs_theta1(data)
    max_alpha_stable = 2.5

    grouped_data = groupby(data, :M)

    x1 = grouped_data[1].theta1
    y1 = grouped_data[1].magnetic_virials ./ grouped_data[1].energies
    x2 = grouped_data[2].theta1
    y2 = grouped_data[2].magnetic_virials ./ grouped_data[2].energies
    x3 = grouped_data[3].theta1
    y3 = grouped_data[3].magnetic_virials ./ grouped_data[3].energies

    f = Figure()
    ax = Axis(f[1, 1], xlabel=L"\theta_1", ylabel=L"\frac{E_v}{E}", ylabelrotation = 0,
        xticklabelsize=tick_size, yticklabelsize=tick_size, 
        xlabelsize=label_size, ylabelsize=label_size,   
    )
    scatter!(ax, x1, y1, marker=:rect, markersize=20, label=L"M/R = 0.0",)
    lines!(ax, x1, y1, linewidth=4)
    scatter!(ax, x2, y2, marker=:rect, markersize=20, label=L"M/R = 0.17",)
    lines!(ax, x2, y2, linewidth=4)
    scatter!(ax, x3, y3, marker=:rect, markersize=20, label=L"M/R = 0.25",)
    lines!(ax, x3, y3, linewidth=4)
    axislegend(ax, position=:lt, merge = false, unique = true, labelsize=tick_size)
    display(GLMakie.Screen(), f)
    return f
end