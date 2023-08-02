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
    ϕ = findϕ(μ0, tsoq, Δind, Vz)
    ϕvec = [0, ϕ]
    return μ0, ϕvec 
end

function measuresvsVz(params, Vzvec, par, optimize=false)
    sites = 2
    tsoq = tan(params[:λ])
    d = FermionBasis((1:sites), (:↑, :↓), qn=QuantumDots.parity)
    LDs = zeros(length((Vzvec)))
    gaps = zeros(length((Vzvec)))
    mps = zeros(length((Vzvec)))
    degs = zeros(length((Vzvec)))
    add = params[:w] + params[:U]/2
    for j in eachindex(Vzvec)
        params[:Vz] = Vzvec[j]
        if optimize
            μ1, μ2, ϕ = optimize_sweetspot(params, par, add, 30)
            params[:μ], params[:Φ] = [μ1, μ2], [0, ϕ]
        else
            params[:μ], params[:Φ] = findμϕ(tsoq, params[:Δind], Vzvec[j], params[:U], par)
        end
        degs[j], mps[j], LDs[j], gaps[j] = measures(d, localpairingham, params, sites)
    end
    return degs, mps, LDs, gaps
end

function plotmeasuresvsVz(;optimize=false, save=false)
    points = 20
    par = false
    Δind = 1.0
    U = 5
    U_inter = 0
    t = [1e-1]
    tsoq = 0.2
    λ = atan(tsoq)
    Vzm = Vzmax(tsoq, Δind, par)
    Vz = collect(range(0.1, Vzm, points))
    initializeplot()
    pLD = plot(ylabel="LD")
    pgap = plot(ylabel=L"$E_{gap}/\Delta_\mathrm{ind}$", legend=false)
    pmp = plot(ylabel="1-MP", legend=false)
    pdeg = plot(ylabel=L"\delta E", legend=false)
    for j in eachindex(t)
        params = Dict{Symbol, Any}(:w=>t[j], :Δind=>Δind, :λ=>λ, :U=>U, :U_inter=>U_inter)
        deg, mp, LD, gap = measuresvsVz(params, Vz, par, optimize)
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
            params[:μ] = [μ1[i], μ2[j]]
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
    U = 1
    U_inter = 0
    t = 1e-1
    tsoq = 0.2
    λ = atan(tsoq)
    # Vz = Vzmax(tsoq, Δind, par) - U/2 
    Vz = 1
    μ0 = findμ0(Δind, Vz, U, par)
    add = t+U/2
    params = Dict{Symbol,Any}(:w=>t, :Δind=>Δind, :λ=>λ, :U=>U, :Vz=>Vz, :U_inter=>U_inter)
    μ1, μ2, ϕ = optimize_sweetspot(params, par, add, 10)
    ϕvec = [0, ϕ]
    params[:Φ] = ϕvec
    # μ1, μ2 = optimize_sweetspot(params, par, add, 30, fixϕ=true)
    # μ1vec = collect(range(μ0[1]-add, μ0[1]+add, points))
    # μ2vec = collect(range(μ0[2]-add, μ0[2]+add, points))
    add2 = add + U
    add3 = add + U/2
    μ1vec = collect(range(μ1-add3, μ1+add2, points))
    μ2vec = collect(range(μ2-add3, -μ2+add2, points))
    pdeg, pmp = plotscanchempotentials(params, points, μ1vec, μ2vec)
    for p in (pdeg, pmp)
        scatter!(p, [μ2], [μ1])
        scatter!(p, [μ0[2]], [μ0[1]])
    end
    display(plot(pdeg, pmp, layout=(1,2), dpi=300))
    println(Vz)
end
end
