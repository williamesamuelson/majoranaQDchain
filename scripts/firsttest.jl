module FirstTest
using DrWatson
@quickactivate "majoranaQDchain"
using QuantumDots
using LinearAlgebra
using Plots
using MajoranaFunctions
using LaTeXStrings

function scan_gapmp_1d(scan_params, fix_params, ham_fun, points, sites)
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

function localtokitaev()
    N = 3
    points = 20
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
    gap, mp = scan_gapmp_1d()
    plot_gapmp1d(gap, mp, λ)
    println([-sin(λ[i])*cos(2α[i]) for i in 1:points])
end

function scanlambda()
    sites = 4
    points = 50
    w0 = 1
    w = fill(w0, sites-1)
    Vz0 = 100*w0
    Vz = fill(Vz0, sites)
    α = 0.2*pi
    λ = collect(range(0.01*pi, 1.999*pi, points))
    μ = fill(Vz0*sin(2*α), sites)
    Δind = fill(Vz0*cos(2*α), sites)
    Φ = fill(0, sites)
    U = fill(0, sites)
    scan_params = Dict("λ"=>λ)
    fix_params = Dict("μ"=>μ, "Δind"=>Δind, "w"=>w, "Φ"=>Φ, "U"=>U, "Vz"=>Vz) 
    gap, mp = scan_gapmp_1d(scan_params, fix_params, localpairingham, points, sites)
    Plots.scalefontsizes()
    Plots.scalefontsizes(1.5)
    display(plot(λ/pi, [gap mp], labels = [L"\delta E" L"MP"], lw=2,
                title=L"\alpha = 0.2\pi, N=%$sites"))
    xlabel!(L"\lambda/\pi")
    # params = @strdict N α
    # save = "scanlambda_"*savename(params)
    # png(plotsdir("scans", save))
end

function main()
    sites = 5
    w0 = 1
    w = fill(w0, sites-1)
    Vz0 = 100*w0
    Vz = fill(Vz0, sites)
    α = 0.2*pi
    λ = fill(0.4*pi, sites-1)
    μ = fill(Vz0*sin(2*α), sites)
    Δind = fill(Vz0*cos(2*α), sites)
    Φ = fill(0, sites)
    U = fill(0, sites)
    params = Dict("μ"=>μ, "Δ"=>Δind, "w"=>w, "Φ"=>Φ, "U"=>U, "Vz"=>Vz, "λ"=>λ)
    # params = Dict("μ"=>μ, "Δ"=>Δind, "w"=>w)
    d = FermionBasis((1:sites), (:↑, :↓))
    # d = FermionBasis(1:N)
    ham = localpairingham(d, params)
    energies, vecs = eigen!(Matrix{ComplexF64}(ham))
    even, odd = groundindices(d, eachcol(vecs), energies)
    mp = majoranapolarization(d, vecs[:,odd], vecs[:,even])
end
end

