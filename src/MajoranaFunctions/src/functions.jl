function lengthofparams(N)
    return Dict("μ"=>N, "Δind"=>N, "w"=>N-1, "λ"=>N-1, "Φ"=>N, "U"=>N, "Vz"=>N,
                "ϵ"=>N, "Δ"=>N, "t"=>N-1)
end

function localpairingham(particle_ops, params)
    d = particle_ops
    N = QuantumDots.nbr_of_fermions(d) ÷ 2
    p = params
    μ = p["μ"]
    Δind = p["Δind"]
    w = p["w"]
    λ = p["λ"]
    Φ = p["Φ"]
    U = p["U"]
    Vz = p["Vz"]
    ham_dot_sp = ((μ[j] - eta(σ)*Vz[j])*d[j,σ]'d[j,σ] for j in 1:N, σ in (:↑, :↓))
    ham_dot_int = (U[j]*d[j,:↑]'d[j,:↑]*d[j,:↓]'d[j,:↓] for j in 1:N)
    ham_tun_normal = (w[j]*cos(λ[j])*d[j, σ]'d[j+1, σ] for j in 1:N-1, σ in (:↑, :↓))
    ham_tun_flip = (w[j]*sin(λ[j])*(d[j, :↑]'d[j+1, :↓] - d[j, :↓]'d[j+1, :↑]) for j in 1:N-1) 
    ham_sc = (Δind[j]*exp(1im*Φ[j])*d[j, :↑]'d[j, :↓]' for j in 1:N)
    ham = sum(ham_tun_normal) + sum(ham_tun_flip) + sum(ham_sc)
    # add conjugates
    ham += ham'
    ham += sum(ham_dot_sp) + sum(ham_dot_int)
    return ham
end

eta(spin) = spin == :↑ ? -1 : 1

function kitaev(particle_ops, params)
    d = particle_ops
    N = QuantumDots.nbr_of_fermions(d)
    p = params
    ϵ = p["ϵ"]
    Δ = p["Δ"]
    t = p["t"]
    ham_dot = (ϵ[j]*d[j]'d[j] for j in 1:N)
    ham_tun = (t[j]*d[j+1]'d[j] for j in 1:N-1)
    ham_sc = (Δ[j]*d[j+1]'d[j]' for j in 1:N-1)
    ham = sum(ham_tun) + sum(ham_sc)
    ham += ham'
    ham += sum(ham_dot)
    return ham
end

function groundindices(particle_ops, vecs, energies)
    parityop = parityoperator(particle_ops)
    parities = [v'parityop*v for v in vecs]
    # evenindices = findall(parity -> parity ≈ 1.0, parities)
    atol = 1e-4
    evenindices = findall(parity -> isapprox(parity, 1; atol=atol), parities)
    oddindices = findall(parity -> isapprox(parity, -1; atol=atol), parities)
    # oddindices = setdiff(1:length(energies), evenindices)
    return evenindices[1]::Int, oddindices[1]::Int
end

function calcmp(plusmajoranas, minusmajoranas, oddstate, evenstate)
    plus_matrixelements = (oddstate'*majplus*evenstate for majplus in plusmajoranas)
    minus_matrixelements = (oddstate'*majminus*evenstate for majminus in minusmajoranas)
    aplus = real.(plus_matrixelements)
    aminus = real.(minus_matrixelements)
    bplus = -imag.(plus_matrixelements)
    bminus = -imag.(minus_matrixelements)
    return sum(aplus.^2 .+ aminus.^2 .- bplus.^2 .- bminus.^2)
end

function majoranapolarization(particle_ops::FermionBasis{M, S, T, Sym}, oddstate, evenstate) where {M, S<:Tuple, T, Sym}
    d = particle_ops
    sites = QuantumDots.nbr_of_fermions(d) ÷ 2
    n = sites÷2 + sites % 2  # sum over half of the sites plus middle if odd
    plusmajoranas = (d[j, σ]' + d[j, σ] for j in 1:n, σ in (:↑, :↓))
    minusmajoranas = (1im*(d[j, σ]' - d[j, σ]) for j in 1:n, σ in (:↑, :↓))
    return calcmp(plusmajoranas, minusmajoranas, oddstate, evenstate)
end

function majoranapolarization(particle_ops::FermionBasis{M, S, T, Sym}, oddstate, evenstate) where {M, S<:Number, T, Sym}
    d = particle_ops
    sites = QuantumDots.nbr_of_fermions(d)
    n = sites÷2 + sites % 2  # sum over half of the sites plus middle if odd
    plusmajoranas = (d[j]' + d[j] for j in 1:n)
    minusmajoranas = (1im*(d[j]' - d[j]) for j in 1:n)
    return calcmp(plusmajoranas, minusmajoranas, oddstate, evenstate)
end

function majoranapolarization(particle_ops, ham)
    energies, vecs = eigen!(Matrix{ComplexF64}(ham))
    even, odd = groundindices(particle_ops, eachcol(vecs), energies)
    return majoranapolarization(particle_ops, vecs[:, odd], vecs[:, even])
end

function dρ_calc(particle_ops, oddstate, evenstate, labels)
    ρe, ρo = map(ψ -> QuantumDots.reduced_density_matrix(ψ, labels, particle_ops),
                 (evenstate, oddstate))
    return norm(ρe - ρo)^2
end

sweetspot(gapsq, measure) = argmin(gapsq .+ measure)
