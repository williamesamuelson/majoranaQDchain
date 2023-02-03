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

# function scan_gapmp_localpairing(μvals, Δvals, w, λvals, Φ, U, Vz, N)
function scan_gapmp_1d(scan_params, fix_params, N, ham, points)
    d = FermionBasis((1:N), (:↑, :↓))
    gaps = zeros(Float64, points)
    mps = zeros(Float64, points)
    params = Dict("μ"=>zeros(N), "Δ"=>zeros(N), "w"=>zeros(N-1), "λ"=>zeros(N-1),
                  "Φ"=>zeros(N), "U"=>0, "Vz"=>0)
    # maybe merge and copy scan_params and fix_params instead of above
    for (p, val) in fix_params
        params[p] = val
    end
    for i = 1:points
        for (p, val) in scan_params
            if params[p] isa Number
                params[p] = val[i]
            else
                fill!(params[p], val[i])
            end
        end
        H = ham(d, params)
        energies, vecs = eigen!(Matrix{ComplexF64}(H))
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

function run_calc_localtokitaev()
    N = 2
    points = 100
    w0 = 1
    w = fill(w0, N-1)
    Vz = 100*w0
    α0 = pi/12
    α = fill(α0, points)
    λ = range(1.8*pi, 1.9*pi, points)
    t = cos(λ[1])*sin(2*α0)
    println(t)
    α[2:end] = [asin(t/cos(λ[i]))/2 for i in 2:points]
    μ = [Vz*sin(2*α[i]) for i in 1:points]
    Δind = [Vz*cos(2*α[i]) for i in 1:points]
    Φ = fill(0, N)
    U = 0
    gap, mp = scan_gapmp_localpairing(μ, Δind, w, λ, Φ, U, Vz, N)
    plot_gapmp1d(gap, mp, λ)
    println([-sin(λ[i])*cos(2α[i]) for i in 1:points])
end

function main()
    N = 3
    points = 100
    w0 = 1
    w = fill(w0, N-1)
    Vz = 100*w0
    α = 0.2*pi
    λ = range(0.1*pi, 0.5*pi, points)
    μ = fill(Vz*sin(2*α), N)
    Δind = fill(Vz*cos(2*α), N)
    Φ = fill(0, N)
    U = 0
    scan_params = Dict("λ"=>λ)
    fix_params = Dict("μ"=>μ, "Δ"=>Δind, "w"=>w, "Φ"=>Φ, "U"=>U, "Vz"=>Vz) 
    gap, mp = scan_gapmp_1d(scan_params, fix_params, N, hamiltonian, points)
    gapplot = plot(λ, gap)
    mpplot = plot(λ, mp)
    display(plot(gapplot, mpplot, layout=2))
end
end

