module Calc
using DrWatson
@quickactivate "majoranaQDchain"
using QuantumDots
using LinearAlgebra
using Plots
using MajoranaFunctions
using LaTeXStrings

function initializeplot()
	plot_font = "Computer Modern"
	default(fontfamily=plot_font, linewidth=2, framestyle=:box, grid=true, markersize=6, legendfontsize=12)
    scalefontsizes()
    scalefontsizes(1.3)
end
    
Vzmax(tsoq, Δind, par) = par ? √(1+tsoq^2)*Δind : √(1+1/tsoq^2)*Δind

function findμϕ(tsoq, Δind, Vz, U, par)
    μ0 = findμ0(Δind, Vz, U, par)
    ϕ = solve4phase(μ0, tsoq, Δind, Vz)
    ϕvec = [0, ϕ]
    return μ0, ϕvec 
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
        params[:μ], params[:Φ] = findμϕ(tsoq, params[:Δind], Vzvec[j], params[:U], par)
        degs[j], mps[j], LDs[j], gaps[j] = measures(d, localpairingham, params, sites)
    end
    return degs, mps, LDs, gaps
end

function plotmeasuresvsVz(save=false)
    points = 10
    par = false
    Δind = 1.0
    U = 0.2Δind
    U_inter = 0Δind
    t = [1e-2, 1e-1, 5e-1, 1]*Δind
    tsoq = 0.2
    λ = atan(tsoq)
    Vzm = Vzmax(tsoq, Δind, par)
    Vz = collect(range(1, Vzm-U/2, points))*Δind
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

function plotscanchempotentials(params, points, μ1, μ2)
    deg, mp, LD = scanchempotentials(params, points, μ1, μ2)
    pdeg = heatmap(μ2, μ1, deg, c=:balance, clims=(-maximum(abs.(deg)), maximum(abs.(deg))),
                   colorbartitle=L"$\delta E$", xlabel=L"$\mu_2$", ylabel=L"$\mu_1$", dpi=300)
    pmp = heatmap(μ2, μ1, mp, c=:acton,
                   colorbartitle="MP", xlabel=L"$\mu_2$", ylabel=L"$\mu_1$", dpi=300)
    return pdeg, pmp
end

function main()
    points = 100
    par = false
    Δind = 1.0
    U = 0.5
    U_inter = 0.05
    t = 5e-2
    tsoq = 0.2
    λ = atan(tsoq)
    Vz = 1.5
    μ0 = findμ0(Δind, Vz, U, par)
    add = t + U/2
    μ1vec = collect(range(μ0[1]-add, μ0[1]+add, points))
    μ2vec = collect(range(μ0[2]-add, μ0[2]+add, points))
    params = Dict{Symbol,Any}(:w=>t, :Δind=>Δind, :λ=>λ, :U=>U, :Vz=>Vz, :U_inter=>U_inter)
    μ1, μ2, ϕ = optimize_sweetspot(params, par, add, 30)
    ϕvec = [0, ϕ]
    params[:Φ] = ϕvec
    # μ1, μ2 = optimize_sweetspot(params, par, add, 30, fixϕ=true)
    pdeg, pmp = plotscanchempotentials(params, points, μ1vec, μ2vec)
    for p in (pdeg, pmp)
        scatter!(p, [μ2], [μ1])
        scatter!(p, [μ0[2]], [μ0[1]])
    end
    display(plot(pdeg, pmp, layout=(1,2), dpi=300))
end
end
