module Calc
using DrWatson
@quickactivate "majoranaQDchain"
using QuantumDots
using LinearAlgebra
using Plots
using MajoranaFunctions
using Roots
using LaTeXStrings

function teqΔcondition(tsoq, Δind, Vz, μ0, ϕ)
    return abs(tsoq*(cos(ϕ/2) - 1im*μ0/Vz * sin(ϕ/2))) - Δind/Vz*sin(ϕ/2)
end

function solve4phase(tsoq, Δind, Vz, μ0)
    f(ϕ) = teqΔcondition(tsoq, Δind, Vz, μ0, ϕ)
    ϕres = find_zero(f, pi/2)
    return ϕres
end
    
Vzmax(tsoq, Δind) = √(1+(tsoq)^2)/tsoq * Δind

findμ0(Δind, Vz)= √(Vz^2 - Δind^2)

function findμϕ(tsoq, Δind, Vz)
    μ0 = findμ0(Δind, Vz)
    μ = [μ0, -μ0]
    ϕ = solve4phase(tsoq, Δind, Vz, μ0)
    ϕvec = [0, ϕ]
    return μ, ϕvec 
end

function LDgapvsVz(params, Vzvec)
    sites = 2
    tsoq = tan(params[:λ])
    d = FermionBasis((1:sites), (:↑, :↓), qn=QuantumDots.parity)
    LDs = zeros(length((Vzvec)))
    gaps = zeros(length((Vzvec)))
    for j in eachindex(Vzvec)
        params[:Vz] = Vzvec[j]
        params[:μ], params[:Φ] = findμϕ(tsoq, params[:Δind], Vzvec[j])
        deg, mp, LDs[j], gaps[j] = measures(d, localpairingham, params, sites)
    end
    return LDs, gaps
end

function plotLDgapvsVz()
    points = 10
    Δind = 1.0
    U = 0Δind
    U_inter = 0Δind
    t = [1e-3, 1e-2, 1e-1, 5e-1, 1]*Δind
    tsoq = 0.2
    λ = atan(tsoq)
    Vzm = Vzmax(tsoq, Δind)
    Vz = collect(range(1, Vzm, points))*Δind
    pLD = plot(ylabel="LD")
    pgap = plot(ylabel=L"$E_g/\Delta_\mathrm{ind}$", legend=false)
    for j in eachindex(t)
        params = Dict{Symbol, Any}(:w=>t[j], :Δind=>Δind, :λ=>λ, :U=>U, :U_inter=>U_inter)
        LD, gap = LDgapvsVz(params, Vz)
        plot!(pLD, Vz, LD, label=L"$t=%$(t[j])$")
        plot!(pgap, Vz, gap)
    end
    params = @strdict U U_inter tsoq
    save = "LDgapvsVz"*savename(params)
    p = plot(pLD, pgap, layout=(1,2), yscale=:log10, xlabel=L"$V_z/\Delta_\mathrm{ind}$")
    display(plot(p))
    # png(plotsdir("fixDelta", save))
end

function scan2d()
    sites = 2
    points = 100
    d = FermionBasis((1:sites), (:↑, :↓), qn=QuantumDots.parity)
    Δind = 1.0
    U = 0Δind
    U_inter = 0Δind
    t = 5e-2Δind
    tsoq = 0.2
    λ = atan(tsoq)
    Vz = Vzmax(tsoq, Δind)
    μ0 = findμ0(Δind, Vz)
    ϕvec = [0, solve4phase(tsoq, Δind, Vz, μ0)]
    # μ1 = collect(range(μ0-2t, μ0+2t, points))
    # μ2 = -1*reverse(μ1)
    μ1 = collect(range(-Vz/2, Vz/2, points))
    μ2 = μ1
    params = Dict(:w=>t, :Δind=>Δind, :λ=>λ, :Φ=>ϕvec, :U=>U, :Vz=>Vz, :U_inter=>U_inter)
    LD = zeros(points, points)
    deg = zeros(points, points)
    for i in 1:points
        for j in 1:points
            params[:μ] = [μ2[i], μ1[j]]
            deg[i,j], _, LD[i,j], _ = measures(d, localpairingham, params, sites)
        end
    end
    pdeg = heatmap(μ2, μ1, deg, c=:balance, clims=(-maximum(abs.(deg)), maximum(abs.(deg))),
                  colorbartitle=L"$\delta E$")
    scatter!(pdeg, [-μ0], [μ0], legend=false)
    pLD = heatmap(μ2, μ1, LD, c=:magma, colorbartitle="LD")
    display(plot(pdeg, pLD, layout=(1,2), xlabel=L"$\mu_2$", ylabel=L"$\mu_1$"))
    params[:μ] = [-μ0, μ0]
    measures(d, localpairingham, params, sites)
    params = @strdict Vz U U_inter t tsoq 
    save = "scancrossingVzlow2"*savename(params)
    # png(plotsdir("fixDelta", save))
end
end
