module FirstTest
using DrWatson
@quickactivate "majoranaQDchain"
using QuantumDots
using LinearAlgebra
using Plots

function hamiltonian(particle_ops, Δ, t0, ϵ, ϕ, U, Ez)
    # how can I store the hamiltonian terms as a generator?
    d = particle_ops
    N = QuantumDots.nbr_of_fermions(d) ÷ 2
    t_normal = [t0[j]*cos(ϕ[j]/2) for j in 1:N-1]
    t_flip = [t0[j]*sin(ϕ[j]/2) for j in 1:N-1]
    ham_dot_sp = sum(ϵ[j]*d[j,σ]'d[j,σ] for j in 1:N, σ in (:↑, :↓))
    ham_dot_int = sum(U*d[j,:↑]'d[j,:↑]*d[j,:↓]'d[j,:↓] for j in 1:N)
    ham_tun_normal  = sum(t_normal[j]*d[j+1, σ]'d[j, σ] for j in 1:N-1, σ in (:↑, :↓))
    ham_tun_flip = sum(t_flip[j]*d[j+1, σ]'d[j, flip(σ)] for j in 1:N-1, σ in (:↑, :↓))
    ham_sc = sum(Δ[j]*d[j, :↑]'d[j, :↓]' for j in 1:N)
    ham = ham_tun_normal + ham_tun_flip + ham_sc
    # add conjugates
    ham += ham'
    ham += ham_dot_sp + ham_dot_int 
    return ham
end

function eta(spin)
    return spin == :↑ ? -1 : 1
end

function flip(spin)
    return spin == :↑ ? :↓ : :↑
end

function groundindices(basis, vecs, energies)
    parityop = parityoperator(basis)
    parities = [v'parityop*v for v in vecs]
    evenindices = findall(parity -> parity ≈ 1, parities)
    oddindices = setdiff(1:length(energies), evenindices)
    return evenindices[1]::Int, oddindices[1]::Int
end

function majoranapolarization(majoranas, oddstate, evenstate)
    w, z = map(majorana -> oddstate'*majorana*evenstate, majoranas)
    return (w^2 - z^2)/(w^2 + z^2)
end

function plot_gapandmp()
    N = 2
    d = FermionBasis(1:N, (:↑, :↓))
    t0 = fill(4, N)
    Δ = 2*t0
    ϵ = zeros(N)
    ϕ = fill(π/2, N)
    U = 10*t0[1]
    Ez = t0[1]

    points = 100
    eps_vec = range(-2Δ[1], 2Δ[1], points)
    gaps = zeros(Float64, points)
    # mps = zeros(Float64, points)
    # maj_plus = d[1] + d[1]'
    # maj_minus = d[1] - d[1]'
    # majoranas = (maj_plus, maj_minus)
    for i = 1:points
        fill!(ϵ, eps_vec[i])
        ham = hamiltonian(d, Δ, t0, ϵ, ϕ, U, Ez)
        energies, vecs = eigen!(Matrix(ham))
        even, odd = groundindices(d, eachcol(vecs), energies)
        gaps[i] = abs(energies[even] - energies[odd])
        # mps[i] = majoranapolarization(majoranas, vecs[:,odd], vecs[:,even])
    end
    # display(plot(eps_vec, [gaps, mps], label=["Gap" "MP"]))
    display(plot(eps_vec, gaps))
    xlabel!("Dot energies")
end
end

