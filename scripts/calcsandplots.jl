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

function create_sweetspot_optfunc(particle_ops, params, scansyms, weight, sites, ham)
    function sweetspotfunc(x)
        for i in 1:length(scansyms)
            params[scansyms[i]] = x[i]
        end
        gap, mp, dρ = measures(particle_ops, ham, params, sites)
        return dρ^2 + weight*(abs(gap) - 1e-7)^2
    end
    return sweetspotfunc
end

function optimizesweetspot(particle_ops, params::Dict{Symbol, T}, scansyms::NTuple{M, Symbol}, guess, range, maxtime, sites, ham) where {T,M}
    weights = [1, 1e4, 1e9]
    times = [0.6, 0.2, 0.2].*maxtime
    for (w, t) in zip(weights, times)
        opt_func = create_sweetspot_optfunc(particle_ops, params, scansyms, w, sites, ham)
        res = bboptimize(opt_func, guess, SearchRange=range, TraceMode=:compact,
                         MaxTime=t)
        guess = best_candidate(res)
    end
    point = guess
    for i in 1:length(scansyms)
        params[scansyms[i]] = point[i]
    end
    gap, mp, dρ = measures(particle_ops, ham, params, sites)
    return point, gap, dρ, mp
end

beta(μ, Δind) = √(μ^2 + Δind^2)

function bdgparticles(particle_ops, site, μ, Δind)
    c = particle_ops
    β = beta(μ, Δind)
    a = (√(β - μ)*c[site,:↑]' - √(β + μ)*c[site,:↓])/√(2β)
    b = (√(β - μ)*c[site,:↓]' + √(β + μ)*c[site,:↑])/√(2β)
    return a, b
end

function initializeplot()
	plot_font = "Computer Modern"
	default(fontfamily=plot_font, linewidth=2, framestyle=:box, grid=false, markersize=6, legendfontsize=15)
    scalefontsizes()
    scalefontsizes(1.3)
end

function getoptrange(μinit, Δindinit, Vz, U_inter, U, w)
    dμ =  max(2Vz + 2U_inter + U/2, 2w)
    dΔind = Vz + U_inter + U/2
    # if dΔind < w
    #     dΔind = 1.5w
    # end
    return [(0, Δindinit + dΔind), (μinit - dμ, μinit + dμ)]
    # return [(0, Δindinit + dΔind), (0, μinit + dμ)]
end

function twodimscan(simparams; opt=true, maxtime=100)
    if opt
        @unpack scansyms, ham, fix_params, sites, points, scanrange, optrange, init = simparams
    else
        @unpack scansyms, ham, fix_params, sites, points, scanrange = simparams
    end
    if ham == localpairingham
        d = FermionBasis((1:sites), (:↑, :↓), qn=QuantumDots.parity)
    else
        d = FermionBasis(1:sites, qn=QuantumDots.parity)
    end

    if opt
        opt_params = merge(fix_params, Dict(scansyms[1]=>0, scansyms[2]=>0))
        sspoint, ssgap, ssLI, ssmp = optimizesweetspot(d, opt_params, scansyms, init, optrange, maxtime, sites, ham)
    end
    xparams = Dict(scansyms[1]=>scanrange[1])
    yparams = Dict(scansyms[2]=>scanrange[2])
    @time gap, mp, dρ = MajoranaFunctions.scan2d(xparams, yparams, fix_params, d, ham, points, sites)
    fulld = copy(simparams)
    fulld["gap"], fulld["dρ"], fulld["mp"] = gap, dρ, mp
    if opt
        fulld["sspoint"], fulld["ssLI"], fulld["ssgap"], fulld["ssmp"] = sspoint, ssLI, ssgap, ssmp
    end
    return fulld
end

function calctwodimscanlp(;opt=true, save=false)
    sites = 2
    w = 1.0
    λ = π/6
    dΦ = 0
    Φ = collect(range(0,(sites-1)*dΦ, sites))
    U = 0w
    U_inter = 0w
    Vz = 50w
    fix_params = Dict(:w=>w, :λ=>λ, :Φ=>Φ, :U=>U, :Vz=>Vz, :U_inter=>U_inter)
    scansyms = (:Δind, :μ)
    points = 100
    μinit, Δindinit = MajoranaFunctions.μΔind_init(λ, Vz, U)
    init = [Δindinit, μinit]
    simparams = Dict("fix_params"=>fix_params, "sites"=>sites, "ham"=>localpairingham, "points"=>points,
                     "init"=>init, "scansyms"=>scansyms, "opt"=>opt)
    if opt
        optrange = getoptrange(μinit, Δindinit, Vz, U_inter, U, w)
        scanrange = tuple((collect(range(optrange[i][1], optrange[i][2], points)) for i in 1:2)...)
        # scanrange = (collect(range(0.1, 60, points)), collect(range(0.1, 60, points)))
        simparams["optrange"] = optrange
    else
        μscan = [[μ, -μ] for μ in collect(range(0, 60, points))]
        scanrange = (collect(range(0, 60, points)), μscan)
    end
    simparams["scanrange"] = scanrange
    res = twodimscan(simparams, opt=opt, maxtime=100)
    println(fix_params)
    if save
        params = @strdict sites U U_inter Vz λ
        @tagsave(datadir("scans", savename(params, "jld2")), res)
        readdir(datadir("scans"))
    end
    return res
end

function plottwodimscanlp(d; save=false)
    points, scanrange = d["points"], d["scanrange"]
    Δindscan = scanrange[1]
    μscan = scanrange[1]
    Δindinit, μinit = d["init"]
    dρ, gap = d["dρ"], d["gap"]
    fix_params = d["fix_params"]
    sites = d["sites"]
    initializeplot()
    p = heatmap(Δindscan, μscan, dρ, c=:acton, clims=(-1e-9, maximum(dρ)), dpi=300)
    contourlvl = 0.05
    lvls = [[0.0]]
    clrs = [:green4, :lightgreen, :green4]
    for i in 1:length(lvls)
        contour!(p, Δindscan, μscan, gap, levels=lvls[i], c=clrs[i], colorbar_entry=false)
    end
    if d["opt"]
        Δindss, μss = d["sspoint"]
        scatter!(p, [Δindss], [μss], c=:cyan, markershape=:star5, markersize=12, label="Optimized sweet spot")
        println(d["ssLI"])
        println(d["ssgap"])
    end
    scatter!(p, [Δindinit], [μinit], c=:red, label=L"\mathrm{Theoretical\ sweet\ spot\ for\ }V_z \gg w", markersize=6)
    # μinit0, Δindinit0 = MajoranaFunctions.μΔind_init(fix_params[:λ], fix_params[:Vz], 0)
    # scatter!(p, [Δindinit0], [μinit0], c=:black, label="Guess0")
    display(plot(p, xlabel=L"\Delta_\mathrm{ind}/w", ylabel=L"\mu/w",
                 colorbar_title="LD", legendposition=:bottomleft, legendfontsize=11))
    if save
        println(fix_params)
        Φ, U, U_inter, Vz = fix_params[:Φ], fix_params[:U], fix_params[:U_inter], fix_params[:Vz]
        dΦ = Φ[2] - Φ[1]
        params = @strdict sites U U_inter Vz dΦ
        save = "2dscan"*savename(params)
        png(plotsdir("2dscans", save))
    end
end

function calctwodimscankitaev(U_k, save=false)
    sites = 2
    θ = 0
    t = 1.0
    fix_params = Dict(:t=>t, :U_k=>U_k, :θ=>θ)
    scansyms = (:Δ, :ϵ)
    points = 200
    init = [U_k/2+1, -U_k/2]
    dscan = 1.5t + U_k
    optrange = [(0, dscan), (-dscan, dscan)]
    radius = U_k/2 + 1
    opt=true
    scanrange = tuple((collect(range(-dscan, dscan, points)) for i in 1:2)...)
    # scanrange = (collect(range(-radius - 1, radius+1, points)), collect(range(-U_k/2 - radius - 1, -U_k/2 + radius + 1, points)))
    simparams = Dict("fix_params"=>fix_params, "sites"=>sites, "ham"=>kitaev, "points"=>points,
                    "scanrange"=>scanrange, "optrange"=>optrange, "init"=>init, "scansyms"=>scansyms, "opt"=>opt)
    res = twodimscan(simparams, opt=true, maxtime=10)
    if save
        params = @strdict sites U_k
        @tagsave(datadir("scans", savename(params, "jld2")), res)
        readdir(datadir("scans"))
    end
    return res
end

function plottwodimscankitaev(d; save=false)
    points, scanrange = d["points"], d["scanrange"]
    Δscan = scanrange[1]
    ϵscan = scanrange[2]
    Δinit, ϵinit = d["init"]
    dρ, gap, mp = d["dρ"], d["gap"], d["mp"]
    fix_params = d["fix_params"]
    U_k = fix_params[:U_k]
    sites = d["sites"]
    initializeplot()
    p = heatmap(Δscan, ϵscan, dρ, c=:acton, dpi=300)
    contourlvl = 0.05
    lvls = [[0.0]]
    clrs = [:green4]
    for i in 1:length(lvls)
        contour!(p, Δscan, ϵscan, gap, levels=lvls[i], c=clrs[i], colorbar_entry=false)
    end
    if d["opt"]
        Δss, ϵss = d["sspoint"]
        scatter!(p, [Δss], [ϵss], c=:cyan, markershape=:star5, markersize=10, label="Optimized sweet spot")
        println(d["ssLI"])
        println(d["ssmp"])
    end
    display(plot(p, xlabel=L"\Delta/t", ylabel=L"\epsilon/t",
                 colorbar_title="LD"))
    if save
        params = @strdict sites U_k
        save = "2dscan"*savename(params)
        png(plotsdir("2dscans", save))
    end
end

function sweetspotzeeman(simparams, maxtime)
    @unpack w, λ, Φ, Vz, U, U_inter, sites = simparams
    d = FermionBasis((1:sites), (:↑, :↓), qn=QuantumDots.parity)
    points = length(Vz)
    gaps = zeros(Float64, points)
    dρs = zeros(Float64, points)
    mps = zeros(Float64, points)
    for j in 1:points
        μinit, Δindinit = MajoranaFunctions.μΔind_init(λ, Vz[j], U)
        params = Dict(:μ=>μinit, :w=>w, :λ=>λ, :Δind=>Δindinit, :Φ=>Φ, :U=>U, :U_inter=>U_inter, :Vz=>Vz[j])
        range = getoptrange(μinit, Δindinit, Vz[j], U_inter, U, w)
        _, gaps[j], dρs[j], mps[j] = optimizesweetspot(d, params, (:Δind, :μ), [Δindinit, μinit], range, maxtime, sites,
                                                      localpairingham)
    end
    fulld = copy(simparams)
    fulld["gap"], fulld["dρ"], fulld["mp"] = gaps, dρs, mps
    return fulld
end

function calcvaryingzeeman()
    sites = 4
    w = 1.0
    Φ = 0w
    λ = π/4
    U = 100*w
    U_inter = 0*w
    Vz = collect(10.0 .^ range(-1, 3, 10))
    maxtime = 1200
    simparams = Dict("w"=>w, "λ"=>λ, "Φ"=>Φ, "Vz"=>[Vz], "U"=>U, "U_inter"=>U_inter, "sites"=>sites)
    dicts = dict_list(simparams)
    for d in dicts
        res = sweetspotzeeman(d, maxtime)
        @tagsave(datadir("sims", savename(d, "jld2")), res)
    end
    readdir(datadir("sims"))
end

function plotdρ()
    files = [22, 21, 20]
    # files = [23, 20, 1]
    sims = readdir(datadir("sims"))[files]
    dict = wload(datadir("sims", sims[1]))
    Vz, λ = dict["Vz"], dict["λ"]
    initializeplot()
    p = plot()
    p2 = plot()
    shapes = [:circle, :star5, :diamond]
    legs = ["Interdot & intradot", "Intradot", "No interactions"]
    for (j, sim) in enumerate(sims)
        dict = wload(datadir("sims", sim))
        dρ, U, U_inter, sites, gap = dict["dρ"], dict["U"], dict["U_inter"], dict["sites"], dict["gap"]
        # plot!(p, Vz, dρ, label=legs[j], markershape=shapes[j])
        plot!(p, Vz, dρ, label=L"N=%$sites", markershape=shapes[j])
        plot!(p2, Vz, abs.(gap), label=L"N=%$sites")
    end
    xticks = collect(10.0 .^ range(-1, 3))
    yticks = collect(10.0 .^ range(-1, -7, 7))
    xvec = range(Vz[4], Vz[end], 100)
    yvec = dict["w"] ./ xvec
    plot!(p, xvec, yvec, label=L"w/V_z")
    # display(plot(p, xscale=:log10, yscale=:log10, grid=true, ylabel="LD", xlabel=L"V_z/w", dpi=300,
    #              xticks=xticks, yticks=yticks, legend=true, legendposition=:bottomleft))
    display(plot!(p2, xscale=:log10, yscale=:log10, title=L"λ/\pi=%$(λ/π)", ylabel=L"\delta\rho", xlabel=L"V_z/w"))
    params = @strdict λ 
    save = "LIsitesintra"*savename(params)
    # png(plotsdir("ssvarzeeman", save))
end

function compkitaev1d()
    sites = 2
    d = FermionBasis((1:sites), (:↑, :↓), qn=QuantumDots.parity)
    c = FermionBasis((1:sites), qn=QuantumDots.parity)
    t = 1.0 # Kitaev t
    ϵ = 0
    U_k = 0
    points = 100
    Δ = collect(range(-3t, 3t, points)) # Kitaev Δ
    α = π/4/2
    w, λ = MajoranaFunctions.kitaevtoakhmerovparams(t, Δ, α)
    @assert all(w.*cos.(λ)*sin(2α) .≈ t) # check conversion
    @assert all(w.*sin.(λ)*cos(2α) .≈ Δ)
    Vz = [1, 1e1, 1e2, 1e3]
    Δind = Vz.*cos(2α)
    μ = Vz.*sin(2α)
    Φ = 0w
    U = 0w
    U_inter = 0
    scan_params = Dict(:w=>w, :λ=>λ)
    scan_paramsk = Dict(:Δ=>Δ)
    fix_paramsk = Dict(:ϵ=>ϵ, :t=>t, :U_k=>U_k)
    gapk, mpk, dρk = scan1d(scan_paramsk, fix_paramsk, c, kitaev, points, sites)
    p1 = plot()
    p2 = plot()
    println(w)
    for j in 1:length(Vz)
        fix_params = Dict(:μ=>μ[j], :Δind=>Δind[j], :Φ=>Φ, :U=>U, :Vz=>Vz[j], :U_inter=>U_inter)
        gap, mp, dρ = scan1d(scan_params, fix_params, d, localpairingham, points, sites)
        plot!(p1, Δ, abs.(gap.-gapk), label="", ylabel=L"|\delta E - \delta E_K|")
        plot!(p2, Δ, abs.(dρ.-dρk), label=L"V_z/t = %$(floor(Int64,Vz[j]))", ylabel=L"|\mathrm{LD} - \mathrm{LD}_K|")
    end
    initializeplot()
    display(plot(p1, p2, layout=(1,2), yscale=:log10, xlabel=L"\Delta/t", dpi=300))
    params = @strdict sites α
    save = "compkitaev"*savename(params)
    # png(plotsdir(save))
end
end
