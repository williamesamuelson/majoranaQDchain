module Calc
using DrWatson
@quickactivate "majoranaQDchain"
using QuantumDots
using LinearAlgebra
using Plots
using MajoranaFunctions
using LaTeXStrings
using Roots
using BlackBoxOptim

function scan1d(scan_params, fix_params, particle_ops, ham_fun, points, sites)
    d = particle_ops
    gaps = zeros(Float64, points)
    mps = zeros(Float64, points)
    dρs = zeros(Float64, points)
    params = merge(scan_params, fix_params)
    for i = 1:points
        for (p, val) in scan_params
            params[p] = val[i]
        end
        gaps[i], mps[i], dρs[i] = measures(d, ham_fun, params)
    end
    return gaps, mps, dρs
end

function scan2d(xparams, yparams, fix_params, particle_ops, ham_fun, points, sites)
    d = particle_ops
    gaps = zeros(Float64, points, points)
    mps = zeros(Float64, points, points)
    dρs = zeros(Float64, points, points)
    params = merge(xparams, yparams, fix_params)
    for i = 1:points
        for (p, val) in yparams
            params[p] = val[i]
        end
        for j = 1:points
            for (p, val) in xparams
                params[p] = val[j]
            end
            gaps[i, j], mps[i, j], dρs[i, j] = measures(d, ham_fun, params)
        end
    end
    return gaps, mps, dρs
end

function kitaevtoakhmerovparams(t, Δ, α) 
    λ = atan.(Δ*tan(2α)/t)
    w = t./(cos.(λ)*sin(2α))
    return w, λ
end

function particleenergyfunc(Δind, V, U)
    function f(μ)
        β = √(μ^2+Δind^2)
        return -V + β - U*(β-μ)/(2*β)
    end
    return f
end

function μguess(Δind, Vz, U)
    func = particleenergyfunc(Δind, Vz, U)
    μguess = √(Vz^2-Δind^2)
    μval = find_zero(func, μguess)
end

function create_sweetspot_optfunc(particle_ops, params, t, α)
    function sweetspotfunc(x)
        Δ = x[1]
        μ = x[2]
        w, λ = kitaevtoakhmerovparams(t, Δ, α)
        params["μ"], params["w"], params["λ"] = μ, w, λ
        gap, mp, dρsq = measures(particle_ops, localpairingham, params)
        return dρsq + gap^2
    end
    return sweetspotfunc
end

function optimizesweetspot(particle_ops, params, t, α, range, guess, maxtime)
    opt_func = create_sweetspot_optfunc(particle_ops, params, t, α)
    res = bboptimize(opt_func, guess, SearchRange=range, TraceMode=:silent,
                     MaxTime=maxtime)
    Δ, μ = best_candidate(res)
    w, λ = kitaevtoakhmerovparams(t, Δ, α)
    params["μ"], params["w"], params["λ"] = μ, w, λ
    gap, mp, dρsq = measures(particle_ops, localpairingham, params)
    return [Δ, μ], gap, dρsq
end

function initializeplot()
	plot_font = "Computer Modern"
	default(fontfamily=plot_font, linewidth=2, framestyle=:box, grid=false)
    scalefontsizes()
    scalefontsizes(1.3)
end

function twodimscan()
    sites = 2
    d = FermionBasis((1:sites), (:↑, :↓))
    points = 10
    # Kitaev params
    t = 1
    Δ = collect(range(-2t, 2t, points))
    # Akhmerov params
    α = 0.2*π
    w, λ = kitaevtoakhmerovparams(t, Δ, α)
    Φ = 0
    U = 1
    Vz = 3
    Δind = Vz*cos(2α)
    dμ = 2*t
    μval = μguess(Δind, Vz, U)
    μ = collect(range(μval-dμ, μval+dμ, points))
    xparams = Dict("λ"=>λ, "w"=>w)
    yparams = Dict("μ"=>μ)
    fix_params = Dict("Δind"=>Δind, "Φ"=>Φ, "U"=>U, "Vz"=>Vz) 
    @time gap, mp, dρsq = scan2d(xparams, yparams, fix_params, d, localpairingham, points, sites)
    opt_params = Dict("μ"=>0, "w"=>0, "λ"=>0)
    ssguess = [t, μval]
    @time sweetspot, gapss, dρsqss = optimizesweetspot(d, merge(fix_params, opt_params), t, α,
                                                       [(Δ[end ÷ 2], Δ[end]), (μ[1], μ[end])], ssguess, 100)
    initializeplot()
    p = heatmap(Δ, μ.-μval, dρsq, c=:acton)
    contourlvl = 0.05
    lvls = [[-contourlvl], [0.0], [contourlvl]]
    clrs = [:green4, :lightgreen, :green4]
    for i in 1:length(lvls)
        contour!(p, Δ, μ.-μval, gap, levels=lvls[i], c=clrs[i], colorbar_entry=false)
    end
    scatter!(p, [sweetspot[1]], [sweetspot[2] - μval], c=:cyan, legend=false)
    # scatter!(p, [t], [0], c=:red)
    display(plot(p, xlabel=L"\Delta_{Kitaev}(w, \lambda)/t_{Kitaev}", ylabel=L"(\mu-\mu_{ss})/t_{Kitaev}",
                 colorbar_title=L"||\delta \rho_R||^2", title=L"N=%$sites, U=%$U, V_z=%$Vz"))
    # params = @strdict sites U Vz
    # save = "2dscan"*savename(params)
    # png(plotsdir("2dscans", save))

end

function sweetspotzeeman(sites, params, Vz::Vector, t, α)
    points = length(Vz)
    d = FermionBasis((1:sites), (:↑, :↓))
    gaps = zeros(points)
    dρs = zeros(points)
    for j in 1:points
        Δind = Vz[j]*cos(2α)
        ssguess = [t, μguess(Δind, Vz[j], params["U"])]
        params["Vz"], params["Δind"] = Vz[j], Δind
        diffs = [3t, 2t]
        range = [(ssguess[i] - diffs[i], ssguess[i] + diffs[i]) for i in 1:2]
        _, gaps[j], dρs[j] = optimizesweetspot(d, params, t, α, range, ssguess, 20)
    end
    return gaps, dρs
end

function calc()
    sites = 2
    # Kitaev params
    t = 1
    # Akhmerov params
    α = 0.2*π
    Φ = 0
    U = 1
    Vz = [3, 10, 1e2, 1e3, 1e4]
    Δind = Vz*cos(2α)
    params = Dict("μ"=>0, "w"=>0, "λ"=>0, "Δind"=>Δind, "Φ"=>Φ, "U"=>U, "Vz"=>0) 
    gaps, dρs = sweetspotzeeman(sites, params, Vz, t, α)
    initializeplot()
    p = plot(Vz, [gaps.^2 dρs], label=[L"\delta E" L"\delta \rho"], xscale=:log10, yscale=:log10, 
             markershape=:auto, xlabel=L"V_z/t_{Kitaev}", title=L"N=%$sites, U=%$U")
    # params = @strdict sites U
    # save = "ssvarzeeman"*savename(params)
    # png(plotsdir("ssvarzeeman", save))
end
end
