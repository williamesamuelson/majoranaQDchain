module Calc
using DrWatson
@quickactivate "majoranaQDchain"
using QuantumDots
using LinearAlgebra
using Plots
using MajoranaFunctions
using LaTeXStrings
using Roots

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
        energies, vecs = eigen!(Matrix{ComplexF64}(H))
        even, odd = groundindices(d, eachcol(vecs), energies)
        gaps[i] = energies[even] - energies[odd]
        mps[i] = majoranapolarization(d, vecs[:,odd], vecs[:,even])
    end
    return gaps, mps
end

function scan_gapmp_2d(xscan_params, yscan_params, fix_params, ham_fun, points, sites)
    if ham_fun == kitaev
        d = FermionBasis(1:sites)
        sites = QuantumDots.nbr_of_fermions(d)
        labels = collect((i in 2:sites))
    else
        d = FermionBasis((1:sites), (:↑, :↓))
        sites = QuantumDots.nbr_of_fermions(d) ÷ 2
        labels = collect(((i,σ) for i in 2:sites, σ in (:↑, :↓)))
    end
    labels = ntuple(i->labels[i], length(labels))
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
            energies, vecs = eigen!(Matrix{ComplexF64}(H))
            even, odd = groundindices(d, eachcol(vecs), energies)
            gaps[i, j] = energies[even] - energies[odd]
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

getfunc(Δind, V, U) = (μ -> -V + √(μ^2+Δind^2) + U*μ/√(μ^2 + Δind^2))

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

function twodimscan()
    sites = 2
    points = 50
    w0 = 1
    t0 = 1
    t = fill(t0, sites-1)
    Δ = collect(range(-2t0, 2t0, points))
    α = 0.3*π
    Φ = fill(0, sites)
    λ = atan.(Δ*tan(2α)/t0)
    w = t0./(cos.(λ)*sin(2α))
    Uscal = 0.1
    U = fill(Uscal*t0, sites)
    Vzscal = 1e3
    Vz = fill(Vzscal*t0, sites)
    Δind = Vz*cos(2α)
    dμ = 2*w0
    func = getfunc(Δind[1], Vz[1], U[1])
    μguess = find_zero(func, Vz[1]+U[1])
    μ = collect(range(μguess-dμ, μguess+dμ, points))
    xscan_params = Dict("λ"=>λ, "w"=>w)
    yscan_params = Dict("μ"=>μ)
    fix_params = Dict("Δind"=>Δind, "Φ"=>Φ, "U"=>U, "Vz"=>Vz) 
    gap, mp, dρsq = scan_gapmp_2d(xscan_params, yscan_params, fix_params, localpairingham, points, sites)
	plot_font = "Computer Modern"
	default(fontfamily=plot_font, linewidth=2, framestyle=:box,
            label=nothing, grid=false)
    scalefontsizes()
    scalefontsizes(1.3)
    p = heatmap(Δ, μ, dρsq, c=:acton, clims=(-1e-9, maximum(dρsq)))
    contour!(p, Δ, μ, gap, levels=[0.0], c=:green, colorbar_entry=false)
    display(plot(p, xlabel=L"\Delta_{eff}(w, \lambda)/t", ylabel=L"\mu/t",
                 colorbar_title=L"||\delta \rho_R||^2"))
                 
end
end

