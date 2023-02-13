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

function eta(spin)
    return spin == :↑ ? -1 : 1
end

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
    evenindices = findall(parity -> parity ≈ 1, parities)
    oddindices = setdiff(1:length(energies), evenindices)
    return evenindices[1]::Int, oddindices[1]::Int
end

function majoranapolarization(particle_ops::FermionBasis{M, S, T, Sym}, oddstate, evenstate) where {M, S<:Tuple, T, Sym}
    d = particle_ops
    sites = QuantumDots.nbr_of_fermions(d) ÷ 2
    n = sites÷2 + sites % 2  # sum over half of the sites plus middle if odd
    acoeffs = real.([oddstate'*(d[j, σ]' + d[j, σ])*evenstate for j in 1:n, σ in (:↑, :↓)])
    bcoeffs = -1*imag.([oddstate'*1im*(d[j, σ]' - d[j, σ])*evenstate for j in 1:n, σ in (:↑, :↓)])
    return sum(acoeffs.^2 .- bcoeffs.^2)
end

function majoranapolarization(particle_ops::FermionBasis{M, S, T, Sym}, oddstate, evenstate) where {M, S<:Number, T, Sym}
    d = particle_ops
    N = QuantumDots.nbr_of_fermions(d)
    acoeffs = real.([oddstate'*(d[j]' + d[j])*evenstate for j in 1:N÷2])
    bcoeffs = -1*imag.([oddstate'*1im*(d[j]' - d[j])*evenstate for j in 1:N÷2])
    return sum(acoeffs.^2 .- bcoeffs.^2)
end

function majoranapolarization(particle_ops, ham)
    energies, vecs = eigen!(Matrix{ComplexF64}(ham))
    even, odd = groundindices(particle_ops, eachcol(vecs), energies)
    return majoranapolarization(particle_ops, vecs[:, odd], vecs[:, even])
end
