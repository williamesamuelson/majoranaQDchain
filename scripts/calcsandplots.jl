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
        gap, mp, dρsq = measures(particle_ops, ham, params, sites)
        return dρsq + weight*(abs(gap) - 1e-7)^2
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
    gap, mp, dρsq = measures(particle_ops, ham, params, sites)
    return point, gap, dρsq, mp
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
	default(fontfamily=plot_font, linewidth=2, framestyle=:box, grid=false)
    scalefontsizes()
    scalefontsizes(1.3)
end

function getoptrange(μinit, Δindinit, Vz, U_inter, U, w)
    dμ = 2Vz + 2U_inter + U/2
    if dμ < w
        dμ = 1.5w
    end
    dΔind = Vz + U_inter + U/2
    return [(0, Δindinit + dΔind), (μinit - dμ, μinit + dμ)]
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
        sspoint, ssgap, ssLI = optimizesweetspot(d, opt_params, scansyms, init, optrange, maxtime, sites, ham)
    end
    xparams = Dict(scansyms[1]=>scanrange[1])
    yparams = Dict(scansyms[2]=>scanrange[2])
    @time gap, mp, dρ = scan2d(xparams, yparams, fix_params, d, ham, points, sites)
    fulld = copy(simparams)
    fulld["gap"], fulld["dρ"], fulld["mp"] = gap, dρ, mp
    if opt
        fulld["sspoint"], fulld["ssLI"], fulld["ssgap"] = sspoint, ssLI, ssgap 
    end
    return fulld
end

function calctwodimscanlp(save=false)
    sites = 2
    w = 1.0
    λ = π/4
    Φ = 0w
    U = 0w
    U_inter = 0w
    Vz = 20w
    fix_params = Dict(:w=>w, :λ=>λ, :Φ=>Φ, :U=>U, :Vz=>Vz, :U_inter=>U_inter)
    scansyms = (:Δind, :μ)
    points = 100
    μinit, Δindinit = MajoranaFunctions.μΔind_init(λ, Vz, U)
    init = [Δindinit, μinit]
    optrange = getoptrange(μinit, Δindinit, Vz, U_inter, U, w)
    scanrange = tuple((collect(range(optrange[i][1], optrange[i][2], points)) for i in 1:2)...)
    simparams = Dict("fix_params"=>fix_params, "sites"=>sites, "ham"=>localpairingham, "points"=>points,
                    "scanrange"=>scanrange, "optrange"=>optrange, "init"=>init, "scansyms"=>scansyms)
    res = twodimscan(simparams, opt=true) 
    if save
        params = @strdict sites U U_inter Vz λ
        @tagsave(datadir("scans", savename(params, "jld2")), res)
        readdir(datadir("scans"))
    end
    return res
end

function plottwodimscanlp(d, save=false)
    points, scanrange = d["points"], d["scanrange"]
    Δindscan = scanrange[1]
    μscan = scanrange[2]
    Δindinit, μinit = d["init"]
    dρ, gap = d["dρ"], d["gap"]
    Δindss, μss = d["sspoint"]
    fix_params = d["fix_params"]
    initializeplot()
    p = heatmap(Δindscan, μscan.-μinit, dρ, c=:acton, dpi=300)
    contourlvl = 0.05
    lvls = [[-contourlvl], [0.0], [contourlvl]]
    clrs = [:green4, :lightgreen, :green4]
    for i in 1:length(lvls)
        contour!(p, Δindscan, μscan.-μinit, gap, levels=lvls[i], c=clrs[i], colorbar_entry=false)
    end
    scatter!(p, [Δindss], [μss-μinit], c=:cyan, legend=false, markershape=:star5, markersize=7)
    println(d["ssLI"])
    scatter!(p, [Δindinit], [μinit-μinit], c=:red, legend=false)
    display(plot(p, xlabel=L"\Delta_{ind}/t", ylabel=L"(\mu-\mu_\mathrm{init})/t",
                 colorbar_title="LI"))
    if save
        params = @strdict sites U U_inter Vz
        save = "2dscan"*savename(params)
        png(plotsdir("2dscans", save))
    end
end

function calctwodimscankitaev(save=false)
    sites = 2
    t = 1.0
    U_k = 20t
    fix_params = Dict(:t=>t, :U_k=>U_k)
    scansyms = (:Δ, :ϵ)
    points = 100
    init = [t, 0]
    dscan = 1.5t + U_k
    optrange = [(0, dscan), (-dscan, dscan)]
    scanrange = tuple((collect(range(optrange[i][1], optrange[i][2], points)) for i in 1:2)...)
    # scanrange = tuple((collect(range(-2t, 2t, points)) for i in 1:2)...)
    simparams = Dict("fix_params"=>fix_params, "sites"=>sites, "ham"=>kitaev, "points"=>points,
                    "scanrange"=>scanrange, "optrange"=>optrange, "init"=>init, "scansyms"=>scansyms)
    res = twodimscan(simparams, opt=true, maxtime=20)
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
    Δss, ϵss = d["sspoint"]
    fix_params = d["fix_params"]
    U_k = fix_params[:U_k]
    sites = d["sites"]
    initializeplot()
    p = heatmap(Δscan, ϵscan, mp, c=:acton, dpi=300)
    contourlvl = 0.05
    lvls = [[-contourlvl], [0.0], [contourlvl]]
    clrs = [:green4, :lightgreen, :green4]
    for i in 1:length(lvls)
        contour!(p, Δscan, ϵscan, gap, levels=lvls[i], c=clrs[i], colorbar_entry=false)
    end
    scatter!(p, [Δss], [ϵss], c=:cyan, legend=false, markershape=:star5, markersize=7)
    println("LI = $(d["ssLI"])")
    println("SS = ($Δss, $ϵss)")
    println("SSguess = ($(U_k/2+1), -$(U_k/2))")
    scatter!(p, [Δinit], [ϵinit], c=:red, legend=false)
    display(plot(p, xlabel=L"\Delta/t", ylabel=L"\epsilon/t",
                 colorbar_title="LI", title=L"N=%$sites, U=%$U_k"))
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
        dμ = 2Vz[j] + 2U_inter + U/2
        if dμ < w
            dμ = 1.5w
        end
        dΔind = Vz[j] + U_inter + U/2
        range = [(μinit - dμ, μinit + dμ), (0, Δindinit + dΔind)]
        _, gaps[j], dρs[j], mps[j] = optimizesweetspot(d, params, (:μ, :Δind), [μinit, Δindinit], range, maxtime, sites)
    end
    fulld = copy(simparams)
    fulld["gap"], fulld["dρ"], fulld["mp"] = gaps, dρs, mps
    return fulld
end

function calcvaryingzeeman()
    sites = [2,3,4]
    w = 1.0
    Φ = 0w
    λ = π/4
    U = 0w
    U_inter = 0w
    Vz = collect(10.0 .^ range(-1, 3, 10))
    maxtime = 500
    simparams = Dict("w"=>w, "λ"=>λ, "Φ"=>Φ, "Vz"=>[Vz], "U"=>U, "U_inter"=>U_inter, "sites"=>sites)
    dicts = dict_list(simparams)
    for d in dicts
        res = sweetspotzeeman(d, maxtime)
        @tagsave(datadir("sims", savename(d, "jld2")), res)
    end
    readdir(datadir("sims"))
end

function plotdρ()
    files = [1, 2, 3]
    sims = readdir(datadir("sims"))[files]
    dict = wload(datadir("sims", sims[1]))
    Vz, λ = dict["Vz"], dict["λ"]
    initializeplot()
    p = plot()
    p2 = plot()
    for (j, sim) in enumerate(sims)
        dict = wload(datadir("sims", sim))
        dρ, U, U_inter, sites, gap = dict["dρ"], dict["U"], dict["U_inter"], dict["sites"], dict["gap"]
        # plot!(p, Vz, dρ, label=L"U_\mathrm{intra}=%$U, U_\mathrm{inter}=%$U_inter", markershape=:circle)
        plot!(p, Vz, dρ, label=L"N=%$sites", markershape=:circle)
        plot!(p2, Vz, abs.(gap), label="N=$sites, U=$U, U_int=$U_inter")
    end
    xticks = collect(10.0 .^ range(-1, 3))
    yticks = collect(10.0 .^ range(-1, -7, 7))
    xvec = range(Vz[3], Vz[end], 100)
    yvec = 1 ./ xvec.^2
    plot!(p, xvec, yvec, label="")
    display(plot!(p, xscale=:log10, yscale=:log10, grid=true, title="No interactions", ylabel=L"LI", xlabel=L"B/t", dpi=300,
                 xticks=xticks, yticks=yticks, legend=:bottomleft))
    # display(plot!(p2, xscale=:log10, yscale=:log10, title=L"λ/\pi=%$(λ/π)", ylabel=L"\delta\rho", xlabel=L"V_z/w"))
    params = @strdict λ
    save = "LInoints"*savename(params)
    # png(plotsdir("ssvarzeeman", save))
end


function kitaevUinter(simparams, maxtime)
    @unpack t, ϵ, Δ, U_k, sites = simparams
    d = FermionBasis(1:sites, qn=QuantumDots.parity)
    points = length(U_k)
    gaps = zeros(Float64, points)
    dρs = zeros(Float64, points)
    mps = zeros(Float64, points)
    for j in 1:points
        ϵinit, Δinit = 0, t
        params = Dict(:ϵ=>ϵ, :t=>t, :Δ=>Δ, :U_k=>U_k)
        dμ = 2Vz[j] + 2U_inter + U/2
        if dμ < w
            dμ = 1.5w
        end
        dΔind = Vz[j] + U_inter + U/2
        range = [(μinit - dμ, μinit + dμ), (0, Δindinit + dΔind)]
        _, gaps[j], dρs[j], mps[j] = optimizesweetspot(d, params, (:μ, :Δind), [μinit, Δindinit], range, maxtime, sites)
    end
    fulld = copy(simparams)
    fulld["gap"], fulld["dρ"], fulld["mp"] = gaps, dρs, mps
    return fulld
end

function sweetspotvarsites()
    ϵ = 0
    t = Δ = 1
    paramsk = Dict(:ϵ=>ϵ, :t=>t, :Δ=>Δ)
    λ = π/4
    w = 1
    Vz = 1e4w
    U = 0w
    μ, Δind = MajoranaFunctions.μΔind_init(λ, Vz, U)
    Φ = 0w
    U_i = 0w
    params = Dict(:μ=>μ, :w=>w, :λ=>λ, :Δind=>Δind, :Φ=>Φ, :U=>U, :Vz=>Vz, :U_inter=>U_i)
    sitesvec = collect(2:4)
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
    # display(plot(sitesvec, abs.(kitaevres[:,3]), c=[:green :blue], ylabel=L"\delta\rho", dpi=300,
    #              labels=["Local pairing" "Kitaev"], xlabel="Sites", title="At sweet spot (U=$U)", shape=:circle))
    params = @strdict U λ Vz
    save = "drhoofsites"*savename(params)
    # png(plotsdir("drhoofsites", save))
end

function test()
    sites = 3
    d = FermionBasis((1:sites), (:↑, :↓), qn=QuantumDots.parity)
    λ = π/4
    w = 1.0
    Vz = 1e6w
    U = 0w
    μ, Δind = MajoranaFunctions.μΔind_init(λ, Vz, U)
    Φ = 0w
    U_inter = 0
    params = Dict(:μ=>μ, :w=>w, :λ=>λ, :Δind=>Δind, :Φ=>Φ, :U=>U, :Vz=>Vz, :U_inter=>U_inter)
    site = 3
    a,b = bdgparticles(d, site, μ, Δind)
    pairing = Δind*d[site, :↑]'d[site, :↓]'
    pairing += pairing'
    pairing2 = Δind/Vz*(Δind*(b'b-a*a') - μ*(a*b +b'a'))
    odd, even = MajoranaFunctions.groundstates(d, localpairingham, params)
    ops = Dict("n_a"=>Δind^2/Vz*a*a', "n_b"=>Δind^2/Vz*b'b, "ab"=>Δind*μ/Vz*a*b, "b'a'"=>Δind*μ/Vz*b'*a', 
              "a"=>a, "b"=>b)
    for (name, state) in Dict("odd"=>odd, "even"=>even)
        println(name)
        for (label, op) in pairs(ops)
            println("$label: $(real(dot(state, op, state)))")
        end
        println("<pairing> $(dot(state, pairing, state))")
        println("Expected value for inf Vz $(-Δind^2/(2*Vz))")
        println(norm(b*state))
        println()
    end
    println(norm(pairing-pairing2))
end

function testgap()
    sites = 3
    d = FermionBasis((1:sites), (:↑, :↓), qn=QuantumDots.parity)
    points = 100
    w = 1.0
    λ = π/4
    Φ = 0w
    U = 0w
    U_inter = 0w
    Vz = 20w
    μinit, Δindinit = MajoranaFunctions.μΔind_init(λ, Vz, U)
    fix_params = Dict(:w=>w, :λ=>λ, :Φ=>Φ, :U=>U, :Vz=>Vz, :U_inter=>U_inter)
    dscan = Vz
    # μscan = collect(range(μinit - dscan, μinit + dscan, points))
    μscan = collect(range(-1.2Vz, 1.2Vz, points))
    Δindscan = collect(range(-1.2Vz, 1.2Vz, points))
    xparams = Dict(:Δind=>Δindscan)
    yparams = Dict(:μ=>μscan)
    @time gap, mp, dρ = scan2d(xparams, yparams, fix_params, d, localpairingham, points, sites)
    initializeplot()
    # p = contour(Δindscan, μscan, gap, levels=[0])
    p = heatmap(Δindscan, μscan, gap, c=:balance)
    display(plot(p, xlabel=L"\Delta_{ind}/w", ylabel=L"\mu/w",
                 colorbar_title=L"\delta E", title=L"N=%$sites, U=%$U, V_z=%$Vz"))
end
end
