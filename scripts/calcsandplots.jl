module Calc
using DrWatson
@quickactivate "majoranaQDchain"
using QuantumDots
using LinearAlgebra
using Plots
using MajoranaFunctions
using LaTeXStrings

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

function format_scan(scan_params, N)
    lengths = MajoranaFunctions.lengthofparams(N)
    return Dict(p => zeros(lengths[p]) for p in keys(scan_params))
end

function localvskitaev()
    sites = 4
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
    Uscal = 20
    U = fill(Uscal*t0, sites)
    Vzscalings = [1 5 1e5]
    innerplots = []
	plot_font = "Computer Modern"
	default(fontfamily=plot_font, linewidth=2, framestyle=:box,
            label=nothing, grid=false)
    scalefontsizes()
    scalefontsizes(0.8)
    for Vzscal in Vzscalings
        Vz = fill(Vzscal*t0, sites)
        μ = Vz*sin(2α)
        Δind = Vz*cos(2α)
        scan_params = Dict("λ"=>λ, "w"=>w)
        fix_params = Dict("μ"=>μ, "Δind"=>Δind, "Φ"=>Φ, "U"=>U, "Vz"=>Vz) 
        gap, mp = scan_gapmp_1d(scan_params, fix_params, localpairingham, points, sites)
        gapplot = plot(Δ, [gap gapkitaev], yaxis=L"\delta E")
        mpplot = plot(Δ, [abs.(mp) abs.(mpkitaev)], yaxis=L"|MP|", labels=["Local" "Kitaev"])
        push!(innerplots, plot(gapplot, mpplot,
                               xaxis=L"Δ/t", lw=2, title=L"V_z=%$Vzscal t", dpi=300))
    end
    p = plot((pl for pl in innerplots)..., layout=(4,1), plot_title=L"U=%$Uscal t")
    display(plot(p))
    # params = @strdict sites α Uscal
    # save = "localvskitaev"*savename(params)
    # png(plotsdir("localvskitaev", save))
end

function main()
    sites = 2
    w0 = 0.25
    w = fill(w0, sites-1)
    Vz0 = 1e5*w0
    Vz = fill(Vz0, sites)
    α = 0.2*π
    λ = fill(-1.2566, sites-1)
    μ = fill(Vz0*sin(2*α), sites)
    Δind = fill(Vz0*cos(2*α), sites)
    Φ = fill(0, sites)
    U = fill(0, sites)
    # params = Dict("μ"=>μ, "Δind"=>Δind, "w"=>w, "Φ"=>Φ, "U"=>U, "Vz"=>Vz, "λ"=>λ)
    params = Dict("ϵ"=>μ, "Δ"=>Δind, "t"=>w)
    # d = FermionBasis((1:sites), (:↑, :↓))
    d = FermionBasis(1:sites)
    ham = kitaev(d, params)
    energies, vecs = eigen!(Matrix{ComplexF64}(ham))
    even, odd = groundindices(d, eachcol(vecs), energies)
    majoranapolarization(d, vecs[:, odd], vecs[:, even])
end
end

