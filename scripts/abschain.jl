module Calc
using DrWatson
@quickactivate "majoranaQDchain"
using QuantumDots
using LinearAlgebra
using Plots
using MajoranaFunctions
using Roots

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

function findparams(tsoq, Δind, Vz)
    μ0 = findμ0(Δind, Vz)
    μ = [μ0, -μ0]
    ϕ = solve4phase(tsoq, Δind, Vz[j], μ0)
    ϕvec = [0, ϕ]
    params = Dict(:w=>t, :μ=>μ, :Δind=>Δind, :λ=>λ, :Φ=>ϕvec, :U=>U, :Vz=>Vz, :U_inter=>U_inter)
    return params
end

function phasetuning()
    sites = 2
    points = 10
    d = FermionBasis((1:sites), (:↑, :↓), qn=QuantumDots.parity)
    Δind = 1.0
    U = 0Δind
    U_inter = 0Δind
    t = 1e-3Δind
    tsoq = 0.2
    λ = atan(tsoq)
    Vzm = Vzmax(tsoq, Δind)
    Vz = collect(range(1, Vzm, points))*Δind
    LDs = zeros(points)
    for j in eachindex(Vz)
        params = findparams(tsoq, Δind, Vz[j])
        deg, mp, LDs[j], gap = measures(d, localpairingham, params, sites)
    end
    println(LDs)
end

function scan2d()
    sites = 2
    points = 100
    d = FermionBasis((1:sites), (:↑, :↓), qn=QuantumDots.parity)
    Δind = 1.0
    U = 0Δind
    U_inter = 0Δind
    t = 1e-1Δind
    tsoq = 0.2
    λ = atan(tsoq)
    Vz = 4Δind
    μ0 = findμ0(Δind, Vz)
    ϕ = solve4phase(tsoq, Δind, Vz, μ0)
    μvec = collect(range(-Vz, Vz, points))
    params = Dict(:w=>t, :μ=>[0, 0], :Δind=>Δind, :λ=>λ, :Φ=>[0, ϕ], :U=>U, :Vz=>Vz, :U_inter=>U_inter)
    LD = zeros(points, points)
    deg = zeros(points, points)
    for i in 1:points
        for j in 1:points
            params[:μ] = [μvec[i], μvec[j]]
            deg[i,j], _, LD[i,j], _ = measures(d, localpairingham, params, sites)
        end
    end
    display(heatmap(μvec, μvec, LD))


end
end
