module Calc
using DrWatson
@quickactivate "majoranaQDchain"
using QuantumDots
using LinearAlgebra
using Plots
using MajoranaFunctions
using Roots
using LaTeXStrings

function initializeplot()
	plot_font = "Computer Modern"
	default(fontfamily=plot_font, linewidth=2, framestyle=:box, grid=true, markersize=6, legendfontsize=12)
    scalefontsizes()
    scalefontsizes(1.3)
end

function teqΔcondition(tsoq, Δind, Vz, μ0, ϕ, par)
    if par
        return abs(μ0/Vz*cos(ϕ/2) + 1im*sin(ϕ/2)) - tsoq*Δind/Vz*cos(ϕ/2)
    else
        return abs(tsoq*(cos(ϕ/2) - 1im*μ0/Vz * sin(ϕ/2))) - Δind/Vz*sin(ϕ/2)
    end
end

function solve4phase(tsoq, Δind, Vz, μ0, par)
    f(ϕ) = teqΔcondition(tsoq, Δind, Vz, μ0, ϕ, par)
    ϕres = find_zero(f, pi/2)
    return ϕres
end
    
Vzmax(tsoq, Δind, par) = par ? √(1+tsoq^2)*Δind : √(1+1/tsoq^2)*Δind

findμ0(Δind, Vz)= √(Vz^2 - Δind^2)

function findμϕ(tsoq, Δind, Vz, par)
    μ0 = findμ0(Δind, Vz)
    if par
        μ = [μ0, μ0]
    else
        μ = [μ0, -μ0]
    end
    ϕ = solve4phase(tsoq, Δind, Vz, μ0, par)
    ϕvec = [0, ϕ]
    return μ, ϕvec 
end

function measuresvsVz(params, Vzvec, par)
    sites = 2
    tsoq = tan(params[:λ])
    d = FermionBasis((1:sites), (:↑, :↓), qn=QuantumDots.parity)
    LDs = zeros(length((Vzvec)))
    gaps = zeros(length((Vzvec)))
    mps = zeros(length((Vzvec)))
    degs = zeros(length((Vzvec)))
    for j in eachindex(Vzvec)
        params[:Vz] = Vzvec[j]
        params[:μ], params[:Φ] = findμϕ(tsoq, params[:Δind], Vzvec[j], par)
        degs[j], mps[j], LDs[j], gaps[j] = measures(d, localpairingham, params, sites)
    end
    return degs, mps, LDs, gaps
end

function plotmeasuresvsVz(save=false)
    points = 10
    par = false
    Δind = 1.0
    U = 0Δind
    U_inter = 0Δind
    t = [1e-2, 1e-1, 5e-1, 1]*Δind
    tsoq = 0.2
    λ = atan(tsoq)
    Vzm = Vzmax(tsoq, Δind, par)
    Vz = collect(range(1, Vzm, points))*Δind
    initializeplot()
    pLD = plot(ylabel="LD")
    pgap = plot(ylabel=L"$E_g/\Delta_\mathrm{ind}$", legend=false)
    pmp = plot(ylabel="1-MP", legend=false)
    pdeg = plot(ylabel=L"\delta E", legend=false)
    for j in eachindex(t)
        params = Dict{Symbol, Any}(:w=>t[j], :Δind=>Δind, :λ=>λ, :U=>U, :U_inter=>U_inter)
        deg, mp, LD, gap = measuresvsVz(params, Vz, par)
        plot!(pLD, Vz, LD, label=L"$t=%$(t[j])$")
        plot!(pgap, Vz, gap)
        plot!(pmp, Vz, 1 .- mp)
        plot!(pdeg, Vz, abs.(deg))
    end
    p = plot(pLD, pmp, pgap, pdeg, layout=(2,2), yscale=:log10, xlabel=L"$V_z/\Delta_\mathrm{ind}$")
    display(plot(p, dpi=300))
    if save
        params = @strdict U U_inter tsoq par
        filename = "measuresvsVz"*savename(params)
        png(plotsdir("fixDelta", filename))
    end
end

function scanchempotentials(params, points, μ1, μ2)
    sites = 2
    d = FermionBasis((1:sites), (:↑, :↓), qn=QuantumDots.parity)
    LD = zeros(points, points)
    mp = zeros(points, points)
    deg = zeros(points, points)
    for i in 1:points
        for j in 1:points
            params[:μ] = [μ2[i], μ1[j]]
            deg[i,j], mp[i,j], LD[i,j], _ = measures(d, localpairingham, params, sites)
        end
    end
    return deg, mp, LD
end

function plotscanchempotentials(params, points, μ1, μ2, save=false)
    deg, mp, LD = scanchempotentials(params, points, μ1, μ2)
    pdeg = heatmap(μ2, μ1, deg, c=:balance, clims=(-maximum(abs.(deg)), maximum(abs.(deg))),
                   colorbartitle=L"$\delta E$", xlabel=L"$\mu_2$", ylabel=L"$\mu_1$", dpi=300)
    display(plot(pdeg))
    # if save
    #     params = @strdict Vz U U_inter t tsoq ϕ
    #     filename = "scancrossingboth"*savename(params)
    #     png(plotsdir("fixDelta", filename))
    # end
end

function main()
    points = 100
    par = false
    Δind = 1.0
    U = 0Δind
    U_inter = 0Δind
    t = 1e-1Δind
    tsoq = 0.3
    λ = atan(tsoq)
    Vz = 3.3
    μ0 = findμ0(Δind, Vz)
    μ1 = collect(range(-μ0-1, μ0+1, points))
    μ2 = μ1
    ϕ = solve4phase(tsoq, Δind, Vz, μ0, par)
    println(ϕ)
    ϕvec = [0, ϕ]
    params = Dict(:w=>t, :Δind=>Δind, :λ=>λ, :Φ=>ϕvec, :U=>U, :Vz=>Vz, :U_inter=>U_inter)
    plotscanchempotentials(params, points, μ1, μ2)
end

function animate()
    points = 100
    Δind = 1.0
    U = 0Δind
    U_inter = 0Δind
    t = 2e-1Δind
    tsoq = 0.3
    λ = atan(tsoq)
    Vz = 3.3
    μ0 = findμ0(Δind, Vz)
    μ1 = collect(range(-μ0-1, μ0+1, points))
    μ2 = μ1
    ϕvals = collect(range(0.2, pi, 30))
    anim = @animate for ϕ in ϕvals
        ϕvec = [0, ϕ]
        params = Dict(:w=>t, :Δind=>Δind, :λ=>λ, :Φ=>ϕvec, :U=>U, :Vz=>Vz, :U_inter=>U_inter)
        plotscanchempotentials(params, points, μ1, μ2)
    end
    gif(anim, plotsdir("fixDelta", "anim_ap.gif"), fps=5)
end
end
