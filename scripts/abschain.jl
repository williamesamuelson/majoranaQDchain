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

function phasetuning()
    sites = 2
    points = 10
    d = FermionBasis((1:sites), (:↑, :↓), qn=QuantumDots.parity)
    Δind = 1.0
    U = 0Δind
    U_inter = 0Δind
    t = 1e-2Δind
    tsoq = 0.2
    λ = atan(tsoq)
    Vzm = Vzmax(tsoq, Δind)
    Vz = collect(range(1, Vzm, points))*Δind
    LDs = zeros(points)
    for j in eachindex(Vz)
        μ, ϕvec = findμϕ(tsoq, Δind, Vz[j])
        params = Dict(:w=>t, :μ=>μ, :Δind=>Δind, :λ=>λ, :Φ=>ϕvec, :U=>U, :Vz=>Vz[j], :U_inter=>U_inter)
        deg, mp, LDs[j], gap = measures(d, localpairingham, params, sites)
    end
    display(plot(Vz, LDs, xlabel=L"$V_z/\Delta_\mathrm{ind}$", ylabel="LD"))
end

function scan2d()
    sites = 2
    points = 100
    d = FermionBasis((1:sites), (:↑, :↓), qn=QuantumDots.parity)
    Δind = 1.0
    U = 0Δind
    U_inter = 0Δind
    t = 2e-1Δind
    tsoq = 0.2
    λ = atan(tsoq)
    Vz = Vzmax(tsoq, Δind)
    μ0 = findμ0(Δind, Vz)
    ϕvec = [0, solve4phase(tsoq, Δind, Vz, μ0)]
    μ1 = collect(range(μ0-t, μ0+t, points))
    μ2 = -1*reverse(μ1)
    params = Dict(:w=>t, :μ=>[0, 0], :Δind=>Δind, :λ=>λ, :Φ=>ϕvec, :U=>U, :Vz=>Vz, :U_inter=>U_inter)
    LD = zeros(points, points)
    deg = zeros(points, points)
    for i in 1:points
        for j in 1:points
            params[:μ] = [μ1[i], μ2[j]]
            deg[i,j], _, LD[i,j], _ = measures(d, localpairingham, params, sites)
        end
    end
    pdeg = heatmap(μ1, μ2, deg, c=:balance, clims=(-maximum(abs.(deg)), maximum(abs.(deg))))
    scatter!(pdeg, [μ0], [-μ0])
    pLD = heatmap(μ1, μ2, LD, c=:magma)
    display(plot(pdeg, pLD, layout=(1,2)))
    params[:μ] = [μ0, -μ0]
    measures(d, localpairingham, params, sites)
end
end
