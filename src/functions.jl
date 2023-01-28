function hamiltonian(particle_ops, μ::Vector, Δ::Vector, w::Vector, λ::Vector, Φ::Vector, U, Vz)
    d = particle_ops
    N = QuantumDots.nbr_of_fermions(d) ÷ 2
    ham_dot_sp = ((μ[j] - eta(σ)*Vz)*d[j,σ]'d[j,σ] for j in 1:N, σ in (:↑, :↓))
    ham_dot_int = (U*d[j,:↑]'d[j,:↑]*d[j,:↓]'d[j,:↓] for j in 1:N)
    ham_tun_normal = (w[j]*cos(λ[j])*d[j, σ]'d[j+1, σ] for j in 1:N-1, σ in (:↑, :↓))
    ham_tun_flip = (w[j]*sin(λ[j])*(d[j, :↑]'d[j+1, :↓] - d[j, :↓]'d[j+1, :↑]) for j in 1:N-1) 
    ham_sc = (Δ[j]*exp(1im*Φ[j])*d[j, :↑]'d[j, :↓]' for j in 1:N)
    ham = sum(ham_tun_normal) + sum(ham_tun_flip) + sum(ham_sc)
    # add conjugates
    ham += ham'
    ham += sum(ham_dot_sp) + sum(ham_dot_int)
    return ham
end

function hamiltonian(particle_ops, μ, Δ, w, λ, Φ, U, Vz)
    # is this slow?
    d = particle_ops
    N = QuantumDots.nbr_of_fermions(d) ÷ 2
    vec_params = [μ, Δ, w, λ, Φ]
    for param in vec_params
        if param isa Number
            param = fill(param, N)
        end
    end
    return hamiltonian(d, μ, Δ, w, λ, Φ, U, Vz)
end

function eta(spin)
    return spin==:↑ ? -1 : 1
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

function groundindices(particle_ops, vecs, energies)
    parityop = parityoperator(particle_ops)
    parities = [v'parityop*v for v in vecs]
    evenindices = findall(parity -> parity ≈ 1, parities)
    oddindices = setdiff(1:length(energies), evenindices) # does this work with ordering?
    return evenindices[1]::Int, oddindices[1]::Int
end

function majoranapolarization(particle_ops, oddstate, evenstate)
    d = particle_ops
    N = QuantumDots.nbr_of_fermions(d)
    acoeffs = real.([oddstate'*(d[j]' + d[j])*evenstate for j in 1:N÷2])
    bcoeffs = -1*imag.([oddstate'*1im*(d[j]' - d[j])*evenstate for j in 1:N÷2])
    return sum(acoeffs.^2 .- bcoeffs.^2)
end
