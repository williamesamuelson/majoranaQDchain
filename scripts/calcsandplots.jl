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

function scan_gapmp_1d(scan_params::Dict{String, T}, fix_params, ham_fun, points, sites) where T<:Vector
    if ham_fun == kitaev
        d = FermionBasis(1:sites)
    else
        d = FermionBasis((1:sites), (:↑, :↓))
    end
    gaps = zeros(Float64, points)
    mps = zeros(Float64, points)
    formattedscan = format_scan(scan_params, sites)
    params = merge(formattedscan, fix_params)
    for i = 1:points
        for (p, val) in scan_params
            fill!(params[p], val[i])
        end
        H = ham_fun(d, params)
        energies, vecs = eigen!(Hermitian(Matrix(H)))
        even, odd = groundindices(d, eachcol(vecs), energies)
        gaps[i] = energies[even] - energies[odd]
        mps[i] = majoranapolarization(d, vecs[:,odd], vecs[:,even])
    end
    return gaps, mps
end

function scan_gapmp_2d(xscan_params, yscan_params, fix_params, particle_ops, ham_fun, points, sites, labels)
    d = particle_ops
    gaps = zeros(Float64, points, points)
    mps = zeros(Float64, points, points)
    dρs = zeros(Float64, points, points)
    xformattedscan = format_scan(xscan_params, sites)
    yformattedscan = format_scan(yscan_params, sites)
    params = merge(xformattedscan, yformattedscan, fix_params)
    for i = 1:points
        for (p, val) in yscan_params
            fill!(params[p], val[i])
        end
        for j = 1:points
            for (p, val) in xscan_params
                fill!(params[p], val[j])
            end
            H = ham_fun(d, params)
            energies, vecs = eigen!(Hermitian(Matrix(H)))
            even, odd = groundindices(d, eachcol(vecs), energies)
            top_gap = energies[3] - energies[1]
            gaps[i, j] = (energies[even] - energies[odd])/top_gap #normalize to gap
            mps[i, j] = majoranapolarization(d, vecs[:,odd], vecs[:,even])
            dρs[i, j] = dρ_calc(d, vecs[:, odd], vecs[:, even], labels)
        end
    end
    return gaps, mps, dρs
end

function format_scan(scan_params, N)
    lengths = MajoranaFunctions.lengthofparams(N)
    return Dict(p => zeros(lengths[p]) for p in keys(scan_params))
end

function getfunc(Δind, V, U)
    function f(μ)
        β = √(μ^2+Δind^2)
        return -V + β - U*(β-μ)/(2*β)
    end
    return f
end

function localvskitaev()
    sites = 2
    points = 100
    t0 = 1
    t = fill(t0, sites-1)
    Δ = collect(range(-2t0, 2t0, points))
    ϵ = zeros(sites)
    scan_params_kitaev = Dict("Δ"=>Δ)
    fix_params_kitaev = Dict("ϵ"=>ϵ, "t"=>t)
    gapkitaev, mpkitaev = scan_gapmp_1d(scan_params_kitaev, fix_params_kitaev,
                                        kitaev, points, sites)
    α = 0.2*π
    λ = atan.(Δ*tan(2α)/t0)
    w = t0./(cos.(λ)*sin(2α))
    Φ = fill(0, sites)
    Uscal = 10
    U = fill(Uscal*t0, sites)
    Vzscalings = [1e5]
    innerplots = []
	plot_font = "Computer Modern"
	default(fontfamily=plot_font, linewidth=2, framestyle=:box,
            label=nothing, grid=false)
    scalefontsizes()
    scalefontsizes(0.8)
    for Vzscal in Vzscalings
        Vz = fill(Vzscal*t0, sites)
        Δind = Vz*cos(2α)
        func = getfunc(Δind[1], Vz[1], U[1])
        μ = fill(find_zero(func, Vz[1]+U[1]), sites)  
        scan_params = Dict("λ"=>λ, "w"=>w)
        fix_params = Dict("μ"=>μ, "Δind"=>Δind, "Φ"=>Φ, "U"=>U, "Vz"=>Vz) 
        gap, mp = scan_gapmp_1d(scan_params, fix_params, localpairingham, points, sites)
        gapplot = plot(Δ, [gap gapkitaev], yaxis=L"\delta E")
        mpplot = plot(Δ, [abs.(mp) abs.(mpkitaev)], yaxis=L"|MP|", labels=["Local" "Kitaev"])
        push!(innerplots, plot(gapplot, mpplot,
                               xaxis=L"Δ/t", lw=2, title=L"V_z=%$Vzscal t", dpi=300))
    end
    p = plot((pl for pl in innerplots)..., layout=(1,1), plot_title=L"U=%$Uscal t")
    display(plot(p))
    # params = @strdict sites α Uscal
    # save = "localvskitaev"*savename(params)
    # png(plotsdir("localvskitaev", save))
end

function create_sweetspot_optfunc(particle_ops, t, α, Φ, Vz, U, labels)
    sites = length(Vz)
    Δind = Vz*cos(2α)
    function sweetspotfunc(x)
        Δ = x[1]
        μ = fill(x[2], sites)
        λ = atan.(Δ*tan(2α)./t) #sites-1 long
        λ = fill(λ, sites-1)
        w = t./(cos.(λ)*sin(2α)) #sites-1 long
        params = Dict("μ"=>μ, "w"=>w, "λ"=>λ, "Δind"=>Δind, "Φ"=>Φ, "U"=>U, "Vz"=>Vz) 
        H = localpairingham(particle_ops, params)
        energies, vecs = eigen!(Hermitian(Matrix(H)))
        even, odd = groundindices(particle_ops, eachcol(vecs), energies)
        top_gap = energies[3] - energies[1]
        gap = (energies[even] - energies[odd])/top_gap #normalize to gap
        dρsq = dρ_calc(particle_ops, vecs[:, odd], vecs[:, even], labels)
        return dρsq + gap^2
    end
    return sweetspotfunc
end

function getμguess(Δind, Vz, U)
    func = getfunc(Δind, Vz, U)
    μguess = √(Vz^2-Δind^2)
    μval = find_zero(func, μguess)
end

function twodimscan()
    sites = 3
    d = FermionBasis((1:sites), (:↑, :↓))
    labels = collect(((i,σ) for i in 2:sites, σ in (:↑, :↓)))
    labels = ntuple(i->labels[i], length(labels))
    points = 30
    t = 1
    Δ = collect(range(-2t, 2t, points))
    α = 0.2*π
    Φ = fill(0, sites)
    λ = atan.(Δ*tan(2α)/t)
    w = t./(cos.(λ)*sin(2α))
    Uscal = 10
    U = fill(Uscal*t, sites)
    Vzscal = 1e2
    Vz = fill(Vzscal*t, sites)
    Δind = Vz*cos(2α)
    dμ = 3*t
    μval = getμguess(Δind[1], Vz[1], U[1])
    μ = collect(range(μval-dμ, μval+dμ, points))
    xscan_params = Dict("λ"=>λ, "w"=>w)
    yscan_params = Dict("μ"=>μ)
    fix_params = Dict("Δind"=>Δind, "Φ"=>Φ, "U"=>U, "Vz"=>Vz) 
    gap, mp, dρsq = scan_gapmp_2d(xscan_params, yscan_params, fix_params, d, localpairingham, points, sites, labels)
    ssindices = MajoranaFunctions.sweetspot(gap.^2, dρsq)
    ssguess = [Δ[ssindices[2]], μ[ssindices[1]]]
    opt_func = create_sweetspot_optfunc(d, t, α, Φ, Vz, U, labels)
    res = bboptimize(opt_func, ssguess; SearchRange = [(Δ[1], Δ[end]), (μ[1], μ[end])],
                     TraceMode = :silent, MaxTime=30.0, NumDimensions = 2)
    ss = best_candidate(res)
	plot_font = "Computer Modern"
	default(fontfamily=plot_font, linewidth=2, framestyle=:box,
            label=nothing, grid=false)
    scalefontsizes()
    scalefontsizes(1.3)
    p = heatmap(Δ, μ.-μval, dρsq, c=:acton)
    contourlvl = 0.05
    lvls = [[-contourlvl], [0.0], [contourlvl]]
    clrs = [:white, :green4, :white]
    for i in 1:length(lvls)
        contour!(p, Δ, μ.-μval, gap, levels=lvls[i], c=clrs[i], colorbar_entry=false)
    end
    scatter!(p, [ss[1]], [ss[2] - μval], c=:cyan)
    display(plot(p, xlabel=L"\Delta_{eff}(w, \lambda)/t", ylabel=L"(\mu-\mu_{ss})/t",
                 colorbar_title=L"||\delta \rho_R||^2"))
end
end

