module FirstTest
using DrWatson
@quickactivate "majoranaQDchain"
using QuantumDots
using LinearAlgebra
using Plots

function hamiltonian(particle_ops, Δ, t0, ϵ, λ, U, Vz, Φ)
    d = particle_ops
    N = QuantumDots.nbr_of_fermions(d) ÷ 2
    ham_dot_sp = ((ϵ[j] - eta(σ)*Vz)*d[j,σ]'d[j,σ] for j in 1:N, σ in (:↑, :↓))
    ham_dot_int = (U*d[j,:↑]'d[j,:↑]*d[j,:↓]'d[j,:↓] for j in 1:N)
    ham_tun_normal = (t0[j]*cos(λ[j])*d[j, σ]'d[j+1, σ] for j in 1:N-1, σ in (:↑, :↓))
    ham_tun_flip = (t0[j]*sin(λ[j])*(d[j, :↑]'d[j+1, :↓] - d[j, :↓]'d[j+1, :↑]) for j in 1:N-1) 
    ham_sc = (Δ[j]*exp(1im*Φ[j])*d[j, :↑]'d[j, :↓]' for j in 1:N)
    ham = sum(ham_tun_normal) + sum(ham_tun_flip) + sum(ham_sc)
    # add conjugates
    ham += ham'
    ham += sum(ham_dot_sp) + sum(ham_dot_int)
    return ham
end

function kitaev(particle_ops, Δ, t0, ϵ)
    a = particle_ops
    N = QuantumDots.nbr_of_fermions(a)
    ham_dot = (ϵ*a[j]'a[j] for j in 1:N)
    ham_tun = (t0*a[j+1]'a[j] for j in 1:N-1)
    ham_sc = (Δ*a[j+1]'a[j]' for j in 1:N-1)
    ham = sum(ham_tun) + sum(ham_sc)
    ham += ham'
    ham += sum(ham_dot)
    return ham
end

function eta(spin)
    return spin == :↑ ? -1 : 1
end

function groundindices(basis, vecs, energies)
    parityop = parityoperator(basis)
    parities = [v'parityop*v for v in vecs]
    evenindices = findall(parity -> parity ≈ 1, parities)
    oddindices = setdiff(1:length(energies), evenindices)
    return evenindices[1]::Int, oddindices[1]::Int
end

function majoranapolarization(majoranas, oddstate, evenstate)
    a, b = map(majorana -> oddstate'*majorana*evenstate, majoranas)
    return a^2 - b^2
end

function plot_gapandmp()
    N = 3
    d = FermionBasis(1:N, (:↑, :↓))
    a = FermionBasis(1:N)
    t = 1
    t0 = fill(t, N-1)
    Δind = fill(20*t, N)
    ϵ = zeros(N)
    λ = fill(π/2, N-1)
    U = 0
    Vz = 20*t
    Φ = zeros(N)

    points = 100
    eps_vec = range(-2t, 2t, points)
    delta_vec = range(0, 2t, points)
    gaps = zeros(Float64, points, points)
    mps = zeros(Float64, points, points)
    maj_plus = a[1] + a[1]'
    maj_minus = a[1] - a[1]'
    majoranas = (maj_plus, maj_minus)
    for i = 1:points
        for j = 1:points
            fill!(ϵ, eps_vec[i])
            # ham = hamiltonian(d, Δind, t0, ϵ, λ, U, Vz, Φ)
            ham_kitaev = kitaev(a, delta_vec[j], t, eps_vec[i]) 
            energies, vecs = eigen!(Matrix{ComplexF64}(ham_kitaev))
            even, odd = groundindices(a, eachcol(vecs), energies)
            gaps[i, j] = energies[even] - energies[odd]
            mps[i, j] = majoranapolarization(majoranas, vecs[:,odd], vecs[:,even])
        end
    end
    # display(plot(eps_vec, [gaps, mps], label=["Gap" "MP"]))
    gapsplot = heatmap(delta_vec, eps_vec, gaps, color = :balance)
    mpplot = heatmap(delta_vec, eps_vec, abs.(mps), color = :plasma)
    display(plot(gapsplot, mpplot, layout = 2))
    xlabel!("energy")
    ylabel!("SC gap")
end
end

