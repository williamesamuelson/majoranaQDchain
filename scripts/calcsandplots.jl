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
    dμ = 2Vz + 2U_inter + U/2
    if dμ < w
        dμ = 1.5w
    end
    dΔind = Vz + U_inter + U/2
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
    @time gap, mp, dρ = scan2d(xparams, yparams, fix_params, d, ham, points, sites)
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
    points = 200
    μinit, Δindinit = MajoranaFunctions.μΔind_init(λ, Vz, U)
    # Δindinit = Vz*cos(λ)
    # μinit = MajoranaFunctions.μguess(Δindinit, Vz, U)
    init = [Δindinit, μinit]
    simparams = Dict("fix_params"=>fix_params, "sites"=>sites, "ham"=>localpairingham, "points"=>points,
                     "init"=>init, "scansyms"=>scansyms, "opt"=>opt)
    if opt
        optrange = getoptrange(μinit, Δindinit, Vz, U_inter, U, w)
        # scanrange = tuple((collect(range(optrange[i][1], optrange[i][2], points)) for i in 1:2)...)
        scanrange = (collect(range(0.1, 60, points)), collect(range(0.1, 60, points)))
        simparams["optrange"] = optrange
    else
        scanrange = (collect(range(0, 60, points)), collect(range(0, 60, points)))
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
    μscan = scanrange[2]
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
    scatter!(p, [Δindinit], [μinit], c=:red, label=L"\mathrm{Sweet\ spot\ for\ }V_z \rightarrow \infty", markersize=6)
    # μinit0, Δindinit0 = MajoranaFunctions.μΔind_init(fix_params[:λ], fix_params[:Vz], 0)
    # scatter!(p, [Δindinit0], [μinit0], c=:black, label="Guess0")
    display(plot(p, xlabel=L"\Delta_\mathrm{ind}/w", ylabel=L"\mu/w",
                 colorbar_title="LD", legendposition=:bottomleft))
    if save
        println(fix_params)
        Φ, U, U_inter, Vz = fix_params[:Φ], fix_params[:U], fix_params[:U_inter], fix_params[:Vz]
        dΦ = Φ[2] - Φ[1]
        params = @strdict sites U U_inter Vz dΦ
        save = "2dscan"*savename(params)
        png(plotsdir("2dscans", save))
    end
end

function calctwodimscankitaev(save=false)
    sites = 2
    θ = rand(1)
    t = 1.0
    U_k = 0t
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
    # scatter!(p, [Δinit], [ϵinit], c=:red, label="Guess", markersize=6)
    # xticks = collect(-6.0:2:6.0)
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
    maxtime = 1500
    simparams = Dict("w"=>w, "λ"=>λ, "Φ"=>Φ, "Vz"=>[Vz], "U"=>U, "U_inter"=>U_inter, "sites"=>sites)
    dicts = dict_list(simparams)
    for d in dicts
        res = sweetspotzeeman(d, maxtime)
        @tagsave(datadir("sims", savename(d, "jld2")), res)
    end
    readdir(datadir("sims"))
end

function plotdρ()
    # files = [25, 24, 23]
    files = [23, 20, 1]
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
        plot!(p, Vz, dρ, label=legs[j], markershape=shapes[j])
        # plot!(p, Vz, dρ, label=L"N=%$sites", markershape=shapes[j])
        plot!(p2, Vz, abs.(gap), label=L"N=%$sites")
    end
    xticks = collect(10.0 .^ range(-1, 3))
    yticks = collect(10.0 .^ range(-1, -7, 7))
    xvec = range(Vz[4], Vz[end], 100)
    yvec = dict["w"] ./ xvec
    plot!(p, xvec, yvec, label=L"w/V_z")
    display(plot(p, xscale=:log10, yscale=:log10, grid=true, ylabel="LD", xlabel=L"V_z/w", dpi=300,
                 xticks=xticks, yticks=yticks, legend=true, legendposition=:bottomleft))
    # display(plot!(p2, xscale=:log10, yscale=:log10, title=L"λ/\pi=%$(λ/π)", ylabel=L"\delta\rho", xlabel=L"V_z/w"))
    params = @strdict λ 
    save = "LIsitesintrainter"*savename(params)
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

function compkitaev0d()
    sites = 2
    d = FermionBasis((1:sites), (:↑, :↓), qn=QuantumDots.parity)
    c = FermionBasis((1:sites), qn=QuantumDots.parity)
    t = 1.0 # Kitaev t
    ϵ = 0
    U_k = 0
    points = 100
    Δ = t
    α = π/4/2
    w, λ = MajoranaFunctions.kitaevtoakhmerovparams(t, Δ, α)
    @assert all(w.*cos.(λ)*sin(2α) .≈ t) # check conversion
    @assert all(w.*sin.(λ)*cos(2α) .≈ Δ)
    Vz = [1, 1e1, 1e2, 1e3, 1e4, 1e5]
    Δind = Vz.*cos(2α)
    μ = Vz.*sin(2α)
    Φ = 0w
    U = 0w
    U_inter = 0
    paramsk = Dict(:ϵ=>ϵ, :t=>t, :Δ=>Δ, :U_k=>U_k, :θ=>0)
    gapk, mpk, dρk = measures(c, kitaev, paramsk, sites)
    dgap = zeros(length(Vz))
    dLD = zeros(length(Vz))
    for j in 1:length(Vz)
        params = Dict(:μ=>μ[j], :w=>w, :λ=>λ, :Δind=>Δind[j], :Φ=>Φ, :U=>U, :Vz=>Vz[j], :U_inter=>U_inter)
        gap, mp, dρ = measures(d, localpairingham, params, sites)
        dgap[j] = abs(gap - gapk)
        dLD[j] = abs(dρ - dρk)
    end
    # twin axis
    p1 = plot(Vz, dgap, ylabel=L"|\delta E - \delta E_K|")
    p2 = plot(Vz, dLD, ylabel=L"|\mathrm{LD} - \mathrm{LD}_K|")
    initializeplot()
    display(plot(p1, p2, layout=(1,2), yscale=:log10, xscale=:log10,xlabel=L"V_z", dpi=300))
    params = @strdict sites α
    save = "compkitaev"*savename(params)
    # png(plotsdir(save))
end

function testphase()
    sites = 2
    t = 1.0
    Δ = t
    ϵ = 0
    U_k = 0
    θ = rand(1)
    # θ = [0, pi]
    d = FermionBasis(1:sites, qn=QuantumDots.parity)
    params = Dict(:ϵ=>ϵ, :t=>t, :Δ=>Δ, :U_k=>U_k, :θ=>θ)
    gap, mp, dρ = measures(d, kitaev, params, sites)
end

kitaevt(p, dΦ) = √(p[:μ]^2 + p[:Vz]^2 - p[:Δind]^2*cos(dΦ))/(√(2)p[:Vz]) * cos(p[:λ])

kitaevΔ(p, dΦ) = p[:Δind]/p[:Vz] * sin(p[:λ]) * cos(dΦ/2)


function testphasetuning()
    sites = 2
    d = FermionBasis((1:sites), (:↑, :↓), qn=QuantumDots.parity)
    λ = 0.99π/2
    w = 1.0
    Vz = 1e2w
    U = 0w
    α = 0.9*pi/2
    μ = Vz*sin(α)
    Δind = Vz*cos(α)
    U_inter = 0
    dΦ = collect(range(0,π,50))
    gap = zeros(length(dΦ))
    dρ = zeros(length(dΦ))
    tvec = zeros(length(dΦ))
    Δvec = zeros(length(dΦ))
    for j in 1:length(dΦ)
        params = Dict(:μ=>μ, :w=>w, :λ=>λ, :Δind=>Δind, :Φ=>[0,dΦ[j]], :U=>U, :Vz=>Vz, :U_inter=>U_inter)
        tvec[j] = kitaevt(params, dΦ[j])
        Δvec[j] = kitaevΔ(params, dΦ[j])
    end
    display(plot(dΦ, [tvec Δvec], labels=["t" "Delta"]))

end
end
