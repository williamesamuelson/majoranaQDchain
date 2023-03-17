module Calc
using DrWatson
@quickactivate "majoranaQDchain"
using QuantumDots
using LinearAlgebra
using Plots
using MajoranaFunctions
using LaTeXStrings
using BlackBoxOptim
using Printf

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

function optimizesweetspot(particle_ops, params::Dict{Symbol, T}, scansyms::NTuple{M, Symbol}, guess, range, maxtime) where {T,M}
    weights = [1, 1e4, 1e9]
    for w in weights
        opt_func = create_sweetspot_optfunc(particle_ops, params, scansyms, w, 1e-7)
        res = bboptimize(opt_func, guess, SearchRange=range, TraceMode=:compact,
                         MaxTime=maxtime/length(weights))
        guess = best_candidate(res)
    end
    point = guess
    for i in 1:length(scansyms)
        params[scansyms[i]] = point[i]
    end
    gap, mp, dρsq = measures(particle_ops, localpairingham, params)
    return point, gap, dρsq, mp
end

function initializeplot()
	plot_font = "Computer Modern"
	default(fontfamily=plot_font, linewidth=2, framestyle=:box, grid=false)
    scalefontsizes()
    scalefontsizes(1.3)
end

function twodimscan()
    sites = 2
    d = FermionBasis((1:sites), (:↑, :↓), qn=QuantumDots.parity)
    points = 50
    w = 1.0
    λ = π/4
    Φ = 0w
    U = 0w
    Vz = 100w
    μval, Δindval = MajoranaFunctions.μΔind_init(λ, Vz, U)
    dμ = 10w
    dΔind = 10w
    # μ = collect(range(0.1, 1.2Vz, points))
    # Δind = collect(range(0.1, 1.2Vz, points))
    μ = collect(range(μval - dμ, μval + dμ, points))
    Δind = collect(range(Δindval - dΔind, Δindval + dΔind, points))
    xparams = Dict(:Δind=>Δind)
    yparams = Dict(:μ=>μ)
    fix_params = Dict(:w=>w, :λ=>λ, :Φ=>Φ, :U=>U, :Vz=>Vz) 
    @time gap, mp, dρ = scan2d(xparams, yparams, fix_params, d, localpairingham, points, sites)
    # opt_params = merge(fix_params, Dict(:μ=>0, :Δind=>0))
    # range_opt = [(Δind[1], Δind[end]), (μ[1], μ[end])]
    # sweetspot, gapss, dρsqss = optimizesweetspot(d, opt_params, (:Δind, :μ), [Δindval, μval], range_opt, 1000)
    initializeplot()
    p = heatmap(Δind, μ, dρ, c=:acton)
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
    # println(gapss)
    # println(dρsqss)
    # params = @strdict sites U Vz
    # save = "2dscan"*savename(params)
    # png(plotsdir("2dscans", save))

end

function sweetspotzeeman(simparams, maxtime)
    @unpack w, λ, Φ, Vz, U, sites = simparams
    d = FermionBasis((1:sites), (:↑, :↓), qn=QuantumDots.parity)
    points = length(Vz)
    gaps = zeros(Float64, points)
    dρs = zeros(Float64, points)
    mps = zeros(Float64, points)
    for j in 1:points
        μ, Δind = MajoranaFunctions.μΔind_init(λ, Vz[j], U)
        params = Dict(:μ=>μ, :w=>w, :λ=>λ, :Δind=>Δind, :Φ=>Φ, :U=>U, :Vz=>Vz[j])
        diffs = (10w, 10w)
        range = [(init[i] - diffs[i], init[i] + diffs[i]) for i in 1:2]
        _, gaps[j], dρs[j], mps[j] = optimizesweetspot(d, params, (:Δind, :μ), init, range, maxtime)
    end
    fulld = copy(simparams)
    fulld["gap"], fulld["dρ"], fulld["mp"] = gaps, dρs, mps
    return fulld
end

function calc()
    sites = 3
    w = 1.0
    Φ = 0w
    λ = π/4
    U = [0, 5].*w
    Vz = collect(10.0 .^ range(-1.5, 3, 10))
    maxtime = 1000
    simparams = Dict("w"=>w, "λ"=>λ, "Φ"=>Φ, "Vz"=>[Vz], "U"=>U, "sites"=>sites)
    dicts = dict_list(simparams)
    for d in dicts
        res = sweetspotzeeman(d, maxtime)
        @tagsave(datadir("sims", savename(d, "jld2")), res)
    end
    readdir(datadir("sims"))
end

function plotzeeman()
	firstsim = readdir(datadir("sims"))[1]
    dict = wload(datadir("sims", firstsim))
    gap, dρ, mp, Vz, sites, U, λ = dict["gap"], dict["dρ"], dict["mp"], dict["Vz"],
                                   dict["sites"], dict["U"], dict["λ"]
    initializeplot()
    plot(Vz, abs.(gap), label=L"|\delta E|", yscale=:log10, ylims=[1e-8, 1],
         markershape=:circle, xlabel=L"V_z/w", title=L"N=%$sites, U=%$U, λ/\pi=%$(λ/π)", 
            lc=:blue, ylabel=L"\delta E")
    plot!(Vz, NaN.*(1:length(Vz)), label=L"\delta \rho", markershape=:diamond, 
             lc=:orange, markercolor=:orange)
    p = plot!(twinx(), Vz, dρ, markershape=:diamond, markercolor=:orange, lc=:orange, legend=false,
             ylabel=L"\delta\rho")
    plot!(Vz, NaN.*(1:length(Vz)), label=L"MP", markershape=:cross, 
             lc=:green, markercolor=:green)
    p = plot!(twinx(), Vz, abs.(mp), markershape=:cross, markercolor=:green, lc=:green, legend=false,
             ylabel=L"MP")
    display(plot(p, xscale=:log10))
    # params = @strdict sites U λ
    # save = "ssvarzeeman"*savename(params)
    # png(plotsdir("ssvarzeeman", save))
end

beta(μ, Δind) = √(μ^2 + Δind^2)

function bdgparticles(particle_ops, site, μ, Δind)
    c = particle_ops
    β = beta(μ, Δind)
    a = (√(β - μ)*c[site,:↑]' - √(β + μ)*c[site,:↓])/√(2β)
    b = (√(β - μ)*c[site,:↓]' + √(β + μ)*c[site,:↑])/√(2β)
    return a,b
end

function test()
    ϵ = 0
    t = Δ = 1
    paramsk = Dict(:ϵ=>ϵ, :t=>t, :Δ=>Δ)
    λ = π/4
    w = 1
    Vz = 1e6w
    U = 10w
    μ, Δind = MajoranaFunctions.μΔind_init(λ, Vz, U)
    Φ = 0w
    params = Dict(:μ=>μ, :w=>w, :λ=>λ, :Δind=>Δind, :Φ=>Φ, :U=>U, :Vz=>Vz) 
    sitesvec = collect(2:6)
    localp = zeros(length(sitesvec), 3)
    kitaevres = zeros(length(sitesvec), 3)
    for (j, sites) in enumerate(sitesvec)
        d = FermionBasis((1:sites), (:↑, :↓), qn=QuantumDots.parity)
	    c = FermionBasis(1:sites, qn=QuantumDots.parity)
        localp[j, :] .= measures(d, localpairingham, params, sites)
        kitaevres[j, :] .= measures(c, kitaev, paramsk, sites)
    end
    initializeplot()
    display(plot(sitesvec, [abs.(localp[:, 3]) abs.(kitaevres[:,3])], c=[:green :blue], ylabel=L"\delta\rho", dpi=300,
                 labels=["Local pairing" "Kitaev"], xlabel="Sites", title="At sweet spot (U=$U)", shape=:circle))
    params = @strdict U λ Vz
    save = "drhoofsites"*savename(params)
    # png(plotsdir("drhoofsites", save))

end
end
