module FirstTest
using DrWatson
@quickactivate "majoranaQDchain"
using QuantumDots
using LinearAlgebra
using Plots
using MajoranaFunctions

"Calculate energy gap and MP of Kitaev, scanning chemical potential and SC gap"
function scan_gapmp_kitaev_heatmap(ϵvec, Δvec, t, N)
    a = FermionBasis(1:N)
    points = length(ϵ)
    gaps = zeros(Float64, points, points)
    mps = zeros(Float64, points, points)
    for i = 1:points
        for j = 1:points
            ham_kitaev = kitaev(a, Δvec[j], t, ϵvec[i]) 
            energies, vecs = eigen!(Matrix{ComplexF64}(ham_kitaev))
            even, odd = groundindices(a, eachcol(vecs), energies)
            gaps[i, j] = energies[even] - energies[odd]
            mps[i, j] = mpkitaev(a, vecs[:,odd], vecs[:,even])
        end
    end
    return gaps, mps
end

function scan_gapmp_localpairing(μ, Δ, w, λvals, Φ, U, Vz, N)
    d = FermionBasis((1:N), (:↑, :↓))
    points = length(λvals)
    gaps = zeros(Float64, points)
    mps = zeros(Float64, points)
    λ = zeros(N)
    for i = 1:points
        fill!(λ, λvals[i])
        ham = hamiltonian(d, μ, Δ, w, λ, Φ, U, Vz)
        energies, vecs = eigen!(Matrix{ComplexF64}(ham))
        even, odd = groundindices(d, eachcol(vecs), energies)
        gaps[i] = energies[even] - energies[odd]
        mps[i] = mpspinful(d, vecs[:,odd], vecs[:,even])
    end
    return gaps, mps
end

function plot_gapmp_heatmap(gap, mp, ϵ, Δ)
    gapsplot = heatmap(Δ, ϵ, gap, color = :balance)
    mpplot = heatmap(Δ, ϵ, abs.(mp), color = :plasma)
    display(plot(gapsplot, mpplot, layout = 2))
    xlabel!("SC gap")
    ylabel!("energy gap")
end

function plot_gapmp1d(gap, mp, λ)
    gapsplot = plot(λ, gap)
    mpplot = plot(λ, mp)
    display(plot(gapsplot, mpplot, layout = 2))
end

function main()
    N = 3
    points = 100
    w0 = 1
    Vz = 100*w0
    α = 2*pi/5
    w = fill(w0, N-1)
    μ = fill(Vz*sin(2α), N)
    Δind = fill(Vz*cos(2α), N)
    Φ = fill(0, N)
    λ = range(pi/8, pi/4, points)
    U = 0
    gap, mp = scan_gapmp_localpairing(μ, Δind, w, λ, Φ, U, Vz, N)
    plot_gapmp1d(gap, mp, λ)
end
end

