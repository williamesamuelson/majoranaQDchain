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

function create_sweetspot_optfunc(particle_ops, params, scansyms, weight, ϵ, sites)
    function sweetspotfunc(x)
        for i in 1:length(scansyms)
            params[scansyms[i]] = x[i]
        end
        gap, mp, dρsq = measures(particle_ops, localpairingham, params, sites)
        return dρsq + weight*(abs(gap) - ϵ)^2
    end
    return sweetspotfunc
end

function optimizesweetspot(particle_ops, params::Dict{Symbol, T}, scansyms::NTuple{M, Symbol}, guess, range, maxtime, sites) where {T,M}
    weights = [1, 1e4, 1e9]
    for w in weights
        opt_func = create_sweetspot_optfunc(particle_ops, params, scansyms, w, 1e-7, sites)
        res = bboptimize(opt_func, guess, SearchRange=range, TraceMode=:compact,
                         MaxTime=maxtime/length(weights))
        guess = best_candidate(res)
    end
    point = guess
    for i in 1:length(scansyms)
        params[scansyms[i]] = point[i]
    end
    gap, mp, dρsq = measures(particle_ops, localpairingham, params, sites)
    return point, gap, dρsq, mp
end

beta(μ, Δind) = √(μ^2 + Δind^2)

function bdgparticles(particle_ops, site, μ, Δind)
    c = particle_ops
    β = beta(μ, Δind)
    a = (√(β - μ)*c[site,:↑]' - √(β + μ)*c[site,:↓])/√(2β)
    b = (√(β - μ)*c[site,:↓]' + √(β + μ)*c[site,:↑])/√(2β)
    return a,b
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
    points = 100
    w = 1.0
    λ = π/4
    Φ = 0w
    U = 5w
    U_inter = 1w
    Vz = 100w
    μinit, Δindinit = MajoranaFunctions.μΔind_init(λ, Vz, U)
    fix_params = Dict(:w=>w, :λ=>λ, :Φ=>Φ, :U=>U, :Vz=>Vz, :U_inter=>U_inter)
    opt_params = merge(fix_params, Dict(:μ=>0, :Δind=>0))
    dμ = 10w
    dΔind = 10w
    μrange = collect(range(μinit - dμ, μinit + dμ, points))
    Δindrange = collect(range(Δindinit - dΔind, Δindinit + dΔind, points))
    range_opt = [(Δindrange[1], Δindrange[end]), (μrange[1], μrange[end])]
    (μss, Δindss), gapss, dρsqss = optimizesweetspot(d, opt_params, (:μ, :Δind), [Δindinit, μinit], range_opt, 500, sites)
    dscan = 10w
    μscan = collect(range(μss - dscan, μss + dscan, points))
    Δindscan = collect(range(Δindss - dscan, Δindss + dscan, points))
    # μscan = collect(range(0.1, 1.2Vz, points))
    # Δindscan = collect(range(0.1, 1.2Vz, points))
    xparams = Dict(:Δind=>Δindscan)
    yparams = Dict(:μ=>μscan)
    @time gap, mp, dρ = scan2d(xparams, yparams, fix_params, d, localpairingham, points, sites)
    initializeplot()
    p = heatmap(Δindscan, μscan, dρ, c=:acton)
    contourlvl = 0.05
    lvls = [[-contourlvl], [0.0], [contourlvl]]
    clrs = [:green4, :lightgreen, :green4]
    for i in 1:length(lvls)
        contour!(p, Δindscan, μscan, gap, levels=lvls[i], c=clrs[i], colorbar_entry=false)
    end
    scatter!(p, [Δindss], [μss], c=:cyan, legend=false)
    scatter!(p, [Δindinit], [μinit], c=:red, legend=false)
    display(plot(p, xlabel=L"\Delta_{ind}/w", ylabel=L"\mu/w",
                 colorbar_title=L"\delta \rho", title=L"N=%$sites, U=%$U, V_z=%$Vz"))
    println(dρsqss)
    # params = @strdict sites U Vz
    # save = "2dscan"*savename(params)
    # png(plotsdir("2dscans", save))

end

function sweetspotzeeman(simparams, maxtime)
    @unpack w, λ, Φ, Vz, U, U_inter, sites = simparams
    d = FermionBasis((1:sites), (:↑, :↓), qn=QuantumDots.parity)
    points = length(Vz)
    gaps = zeros(Float64, points)
    dρs = zeros(Float64, points)
    mps = zeros(Float64, points)
    for j in 1:points
        init = MajoranaFunctions.μΔind_init(λ, Vz[j], U)
        params = Dict(:μ=>init[1], :w=>w, :λ=>λ, :Δind=>init[2], :Φ=>Φ, :U=>U, :U_inter=>U_inter, :Vz=>Vz[j])
        diffs = (500w, 500w)
        range = [(init[i] - diffs[i], init[i] + diffs[i]) for i in 1:2]
        _, gaps[j], dρs[j], mps[j] = optimizesweetspot(d, params, (:Δind, :μ), init, range, maxtime, sites)
    end
    fulld = copy(simparams)
    fulld["gap"], fulld["dρ"], fulld["mp"] = gaps, dρs, mps
    return fulld
end

function calcvaryingzeeman()
    sites = 2
    w = 1.0
    Φ = 0w
    λ = π/4
    # U = [0, 10].*w
    U = 0w
    U_inter = [1, 5, 10].*w
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
    files = [1, 3, 4, 7]
    sims = readdir(datadir("sims"))[files]
    dict = wload(datadir("sims", sims[1]))
    Vz, λ = dict["Vz"], dict["λ"]
    initializeplot()
    p = plot()
    p2 = plot()
    for (j, sim) in enumerate(sims)
        dict = wload(datadir("sims", sim))
        dρ, U, U_inter, sites, gap = dict["dρ"], dict["U"], dict["U_inter"], dict["sites"], dict["gap"]
        plot!(p, Vz, dρ, label="N=$sites, U=$U, U_int=$U_inter")
        plot!(p2, Vz, abs.(gap), label="N=$sites, U=$U, U_int=$U_inter")
    end
    display(plot!(p, xscale=:log10, yscale=:log10, title=L"λ/\pi=%$(λ/π)", ylabel=L"\delta\rho", xlabel=L"V_z/w"))
    # display(plot!(p2, xscale=:log10, yscale=:log10, title=L"λ/\pi=%$(λ/π)", ylabel=L"\delta\rho", xlabel=L"V_z/w"))
    params = @strdict λ
    save = "LIUinter"*savename(params)
    # png(plotsdir("ssvarzeeman", save))
end

function plotzeeman()
	firstsim = readdir(datadir("sims"))[2]
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
    plot!(Vz, NaN.*(1:length(Vz)), label="MP", markershape=:cross, 
             lc=:green, markercolor=:green)
    p = plot!(twinx(), Vz, abs.(mp), markershape=:cross, markercolor=:green, lc=:green, legend=false,
             ylabel=L"MP")
    display(plot(p, xscale=:log10))
    params = @strdict sites U λ
    save = "ssvarzeeman"*savename(params)
    # png(plotsdir("ssvarzeeman", save))
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
end
