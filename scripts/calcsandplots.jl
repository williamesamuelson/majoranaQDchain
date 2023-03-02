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
using NLsolve

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

function zeroenergy_condition(μ, Δind, Vz, U)
    β = √(μ^2+Δind^2)
    return -Vz + β - U/2*(1 - μ/β)
end

function μguess(Δind, Vz, U)
    f(μ) = zeroenergy_condition(μ, Δind, Vz, U)
    μinit = √(Vz^2-Δind^2)
    return find_zero(f, μinit)
end

#Now I choose the t=-Δ solution (?)
function get_sweetspot_nlsolvefunc(λ, Vz, U)
    function f!(F, x)
        μ = x[1]
        Δind = x[2]
        β = √(μ^2 + Δind^2)
        F[1] = μ*cos(λ) - Δind*sin(λ)
        F[2] = zeroenergy_condition(μ, Δind, Vz, U)
    end
end

function get_sweetspot_nlsolvejac(λ, Vz, U)
    function j!(J, x)
        μ = x[1]
        Δind = x[2]
        β = √(μ^2 + Δind^2)
        J[1, 1] = cos(λ)
        J[1, 2] = -sin(λ)
        J[2, 1] = 1/β * (μ + U*Δind^2/(2*β^2))
        J[2, 2] = Δind/β * (1 - U*μ/(2*β^2))
    end
end

function μΔind_init(λ, Vz, U)
    f = get_sweetspot_nlsolvefunc(λ, Vz, U)
    J = get_sweetspot_nlsolvejac(λ, Vz, U)
    sol = nlsolve(f, J, [Vz*sin(λ); Vz*cos(λ)], show_trace=true)
    return sol.zero
end

function create_sweetspot_optfunc(particle_ops, params, scansyms, weight, ϵ)
    function sweetspotfunc(x)
        for i in 1:length(scansyms)
            params[scansyms[i]] = x[i]
        end
        gap, mp, dρsq = measures(particle_ops, localpairingham, params)
        return dρsq + weight*(abs(gap) - ϵ)^2
    end
    return sweetspotfunc
end

# should make a function for the middle rows
function optimizesweetspot(particle_ops, params::Dict{Symbol, T}, scansyms::NTuple{M, Symbol}, guess, range, maxtime) where {T,M}
    opt_func = create_sweetspot_optfunc(particle_ops, params, scansyms, 1e7, 1e-8)
    res = bboptimize(opt_func, guess, SearchRange=range, TraceMode=:compact,
                     MaxTime=maxtime)
    point = best_candidate(res)
    for i in 1:length(scansyms)
        params[scansyms[i]] = point[i]
    end
    gap, mp, dρsq = measures(particle_ops, localpairingham, params)
    return point, gap, dρsq
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
    points = 100
    w = 1.0
    λ = π/3
    Φ = 0w
    U = 10w
    Vz = 1e4w
    μval, Δindval = μΔind_init(λ, Vz, U)
    dμ = 5w
    dΔind = 5w
    μ = collect(range(0.1, 1.2Vz, points))
    Δind = collect(range(0.1Vz, 1.2Vz, points))
    # μ = collect(range(μval - dμ, μval + dμ, points))
    # Δind = collect(range(Δindval - dΔind, Δindval + dΔind, points))
    xparams = Dict(:Δind=>Δind)
    yparams = Dict(:μ=>μ)
    fix_params = Dict(:w=>w, :λ=>λ, :Φ=>Φ, :U=>U, :Vz=>Vz) 
    @time gap, mp, dρsq = scan2d(xparams, yparams, fix_params, d, localpairingham, points, sites)
    opt_params = merge(fix_params, Dict(:μ=>0, :Δind=>0))
    range_opt = [(Δind[1], Δind[end]), (μ[1], μ[end])]
    # sweetspot, gapss, dρsqss = optimizesweetspot(d, opt_params, (:Δind, :μ), [Δindval, μval], range_opt, 20)
    initializeplot()
    p = heatmap(Δind, μ, dρsq, c=:acton)
    contourlvl = 0.05
    lvls = [[-contourlvl], [0.0], [contourlvl]]
    clrs = [:green4, :lightgreen, :green4]
    for i in 1:length(lvls)
        contour!(p, Δind, μ, gap, levels=lvls[i], c=clrs[i], colorbar_entry=false)
    end
    # scatter!(p, [sweetspot[1]], [sweetspot[2]], c=:cyan, legend=false)
    scatter!(p, [Δindval], [μval], c=:red, legend=false)
    display(plot(p, xlabel=L"\Delta_{ind}/w", ylabel=L"\mu/w",
                 colorbar_title=L"\delta \rho", title=L"N=%$sites, U=%$U, V_z=%$Vz"))
    # params = @strdict sites U Vz
    # save = "2dscan"*savename(params)
    # png(plotsdir("2dscans", save))

end

function sweetspotzeeman(sites, params, Vz::Vector)
    points = length(Vz)
    d = FermionBasis((1:sites), (:↑, :↓))
    gaps = zeros(points)
    dρs = zeros(points)
    for j in 1:points
        params[:Vz] = Vz[j]
        init = μΔind_init(params[:λ], params[:Vz], params[:U])
        diffs = (5*params[:w], 5*params[:w])
        range = [(init[i] - diffs[i], init[i] + diffs[i]) for i in 1:2]
        _, gaps[j], dρs[j] = optimizesweetspot(d, params, (:Δind, :μ), init, range, 20)
    end
    return gaps, dρs
end

function calc()
    sites = 2
    w = 1.0
    Φ = 0w
    λ = π/4
    U = 0w
    Vz = [1, 3, 10, 1e2].*w
    params = Dict(:μ=>0w, :w=>w, :λ=>λ, :Δind=>0w, :Φ=>Φ, :U=>U, :Vz=>0w) 
    gaps, dρs = sweetspotzeeman(sites, params, Vz)
    initializeplot()
    plot(Vz, abs.(gaps), label=L"|\delta E|", xscale=:log10, yscale=:log10, 
             markershape=:auto, xlabel=L"V_z/w", title=L"N=%$sites, U=%$U")
    p = plot!(twinx(), Vz, dρs, markershape=:circle)
    display(plot(p))
    # params = @strdict sites U
    # save = "ssvarzeeman"*savename(params)
    # png(plotsdir("ssvarzeeman", save))
end


function test()
    sites = 2
    d = FermionBasis((1:sites), (:↑, :↓))
    μ = 10
    w = 1.0
    λ = π/4
    Δind = 10
    Φ = 0w
    U = 1w
    Vz = 1e2w
    params = Dict(:μ=>μ, :w=>w, :λ=>λ, :Δind=>Δind, :Φ=>Φ, :U=>U, :Vz=>Vz) 
    measures(d, localpairingham, params)
end

end
