module FirstTest
using DrWatson
@quickactivate "majoranaQDchain"
using QuantumDots
using LinearAlgebra
using Plots
include(srcdir("functions.jl"))

"Calculate energy gap and MP of Kitaev, scanning chemical potential and SC gap"
function calc_gapmp_kitaev_heatmap(ϵ, Δ, t, N)
    a = FermionBasis(1:N)
    points = length(ϵ)
    gaps = zeros(Float64, points, points)
    mps = zeros(Float64, points, points)
    for i = 1:points
        for j = 1:points
            ham_kitaev = kitaev(a, Δ[j], t, ϵ[i]) 
            energies, vecs = eigen!(Matrix{ComplexF64}(ham_kitaev))
            even, odd = groundindices(a, eachcol(vecs), energies)
            gaps[i, j] = energies[even] - energies[odd]
            mps[i, j] = majoranapolarization(a, vecs[:,odd], vecs[:,even])
        end
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

function main()
    points = 100
    t = 1
    ϵ = range(-2t, 2t, points)
    Δ = range(0, 2t, points)
    N = 3
    gaps, mps = calc_gapmp_kitaev_heatmap(ϵ, Δ, t, N)
    plot_gapmp_heatmap(gaps, mps, ϵ, Δ)
end
end

