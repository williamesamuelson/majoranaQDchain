using MajoranaFunctions
using Test, LinearAlgebra, QuantumDots

@testset "Kitaev" begin
    sites = 4
	c = FermionBasis(1:sites, qn=QuantumDots.parity)
    ϵ = 0
    t = Δ = 1
    params = Dict(:ϵ=>ϵ, :t=>t, :Δ=>Δ)
    gap, mp, dρ = measures(c, kitaev, params, sites)
    @test abs(gap) < 1e-5
    @test abs(mp) ≈ 1
    @test dρ < 1e-12  # appearently not...
    
    vals, vecs, oddind, evenind = MajoranaFunctions.handleblocks(c, kitaev, params)
    @test abs(vals[oddind] - vals[evenind]) < 1e-12
    p = parityoperator(c)
    v1, v2 = vecs[:, oddind], vecs[:, evenind]
    @test dot(v1,p,v1)*dot(v2,p,v2) ≈ -1
    w = [dot(v1,f+f',v2) for f in c.dict]
    z = [dot(v1,(f'-f),v2) for f in c.dict]
    @test abs.(w.^2 - z.^2) ≈ [1,0,0,1]
    # change to my mp functions
end

@testset "sweetspot" begin
    sites = 3
    d = FermionBasis((1:sites), (:↑, :↓), qn=QuantumDots.parity)
    w = 1.0
    λ = π/4
    Vz = 1e5w
    μ = Vz*sin(λ)
    Δind = Vz*cos(λ)
    Φ = 0w
    U = 0w
    params = Dict(:μ=>μ, :w=>w, :λ=>λ, :Δind=>Δind, :Φ=>Φ, :U=>U, :Vz=>Vz) 
    gap, mp, dρ = measures(d, localpairingham, params, sites)
    @test abs(gap) < 1e-5
    @test abs(mp) ≈ 1
    # @test dρ < 1e-10 # appearently not...
end

@testset "kitaevlimit" begin
    sites = 3
    d = FermionBasis((1:sites), (:↑, :↓), qn=QuantumDots.parity)
    c = FermionBasis((1:sites), qn=QuantumDots.parity)
    t = 1.0 # Kitaev t
    ϵ = 0
    points = 100
    Δ = collect(range(-2t, 2t, points)) # Kitaev Δ
    α = π/5
    w, λ = MajoranaFunctions.kitaevtoakhmerovparams(t, Δ, α)
    @test all(w.*cos.(λ)*sin(2α) .≈ t) # check conversion
    @test all(w.*sin.(λ)*cos(2α) .≈ Δ)
    Vz = 1e6t
    Δind = Vz*cos(2α)
    μ = Vz*sin(2α)
    Φ = 0w
    U = 0w
    scan_params = Dict(:w=>w, :λ=>λ)
    scan_paramsk = Dict(:Δ=>Δ)
    fix_params = Dict(:μ=>μ, :Δind=>Δind, :Φ=>Φ, :U=>U, :Vz=>Vz) 
    fix_paramsk = Dict(:ϵ=>ϵ, :t=>t)
    gap, mp, dρ = scan1d(scan_params, fix_params, d, localpairingham, points, sites)
    gapk, mpk, dρk = scan1d(scan_paramsk, fix_paramsk, c, kitaev, points, sites)
    @test isapprox(gap, gapk, atol=1e-3)
    @test isapprox(abs.(mp), abs.(mpk), atol=1e-3)
end
