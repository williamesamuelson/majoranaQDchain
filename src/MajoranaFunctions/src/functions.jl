function lengthofparams(sites)
    return (μ=sites, Δind=sites, w=sites-1, λ=sites-1, Φ=sites, U=sites, Vz=sites,
                ϵ=sites, Δ=sites, t=sites-1)
end

function convert_to_namedtuple(params, sites)
    lop = lengthofparams(sites)
    newparams = Dict(p=>zeros(Float64, lop[p]) for p in keys(params))
    for (p, val) in params
        if val isa Number
            fill!(newparams[p], val)
        else
            newparams[p] = val 
        end
    end
    return (; newparams...)
end

function localpairingham(particle_ops, params::NamedTuple{S, NTuple{7, Vector{Float64}}}) where S
    d = particle_ops
    sites = QuantumDots.nbr_of_fermions(d) ÷ 2
    p = params
    μ = p.μ
    Δind = p.Δind
    w = p.w
    λ = p.λ
    Φ = p.Φ
    U = p.U
    Vz = p.Vz
    ham_dot_sp = ((μ[j] - η(σ)*Vz[j])*d[j,σ]'d[j,σ] for j in 1:sites, σ in (:↑, :↓))
    ham_dot_int = (U[j]*d[j,:↑]'d[j,:↑]*d[j,:↓]'d[j,:↓] for j in 1:sites)
    ham_tun_normal = (w[j]*cos(λ[j])*d[j, σ]'d[j+1, σ] for j in 1:sites-1, σ in (:↑, :↓))
    ham_tun_flip = (w[j]*sin(λ[j])*(d[j, :↑]'d[j+1, :↓] - d[j, :↓]'d[j+1, :↑]) for j in 1:sites-1) 
    ham_sc = (Δind[j]*exp(1im*Φ[j])*d[j, :↑]'d[j, :↓]' for j in 1:sites)
    ham = sum(ham_tun_normal) + sum(ham_tun_flip) + sum(ham_sc)
    # add conjugates
    ham += ham'
    ham += sum(ham_dot_sp) + sum(ham_dot_int)
    return ham
end

function localpairingham(particle_ops, params::Dict{Symbol, T}) where T
    sites = QuantumDots.nbr_of_fermions(particle_ops) ÷ 2
    newparams = convert_to_namedtuple(params, sites)
    return localpairingham(particle_ops, newparams)
end

η(spin) = spin == :↑ ? -1 : 1

function kitaev(particle_ops, params::NamedTuple{S, NTuple{3, Vector{Float64}}}) where S
    d = particle_ops
    sites = QuantumDots.nbr_of_fermions(d)
    p = params
    ϵ = p.ϵ
    Δ = p.Δ
    t = p.t
    ham_dot = (ϵ[j]*d[j]'d[j] for j in 1:sites)
    ham_tun = (t[j]*d[j+1]'d[j] for j in 1:sites-1)
    ham_sc = (Δ[j]*d[j+1]'d[j]' for j in 1:sites-1)
    ham = sum(ham_tun) + sum(ham_sc)
    ham += ham'
    ham += sum(ham_dot)
    return ham
end

function kitaev(particle_ops, params::Dict{Symbol, T}) where T
    sites = QuantumDots.nbr_of_fermions(particle_ops)
    newparams = convert_to_namedtuple(params, sites)
    return localpairingham(particle_ops, newparams)
end

function measures(particle_ops, ham_fun, params)
        H = ham_fun(particle_ops, params)
        energies, vecs = eigen!(Matrix(H))
        even, odd = groundindices(particle_ops, eachcol(vecs), energies)
        top_gap = energies[3] - energies[1]
        gap = (energies[even] - energies[odd])/top_gap #normalize to gap
        mp = majoranapolarization(particle_ops, vecs[:,odd], vecs[:,even])
        dρ = robustness(particle_ops, vecs[:, odd], vecs[:, even])
        return gap, mp, dρ
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

function majoranapolarization(plusmajoranas, minusmajoranas, oddstate, evenstate)
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
    return majoranapolarization(plusmajoranas, minusmajoranas, oddstate, evenstate)
end

function majoranapolarization(particle_ops::FermionBasis{M, S, T, Sym}, oddstate, evenstate) where {M, S<:Number, T, Sym}
    d = particle_ops
    sites = QuantumDots.nbr_of_fermions(d)
    n = sites÷2 + sites % 2  # sum over half of the sites plus middle if odd
    plusmajoranas = (d[j]' + d[j] for j in 1:n)
    minusmajoranas = (1im*(d[j]' - d[j]) for j in 1:n)
    return majoranapolarization(plusmajoranas, minusmajoranas, oddstate, evenstate)
end

function majoranapolarization(particle_ops, ham)
    energies, vecs = eigen!(Matrix(ham))
    even, odd = groundindices(particle_ops, eachcol(vecs), energies)
    return majoranapolarization(particle_ops, vecs[:, odd], vecs[:, even])
end

function robustness(particle_ops, oddstate, evenstate, sitelabels)
    dρ = 0
    # maybe one parameter with indices corrresponding to correct sitelabels also
    sites = length(sitelabels)
    for j in 1:sites
        keeplabels = tuple(sitelabels[j]...)
        println(setdiff(sitelabels, keeplabels))
        ρe, ρo = map(ψ -> QuantumDots.reduced_density_matrix(ψ, remove_labels, particle_ops),
                    (evenstate, oddstate))
        dρ += norm(ρe - ρo)^2
    end
    return dρ/sites
end

function robustness(particle_ops::FermionBasis{M, S, T, Sym}, oddstate, evenstate) where {M, S<:Tuple, T, Sym}
    sites = QuantumDots.nbr_of_fermions(particle_ops) ÷ 2
    sitelabels = [tuple(((i, σ) for σ in (:↑, :↓))...) for i in 1:sites]
    return robustness(particle_ops, oddstate, evenstate, sitelabels)
end

robustness(particle_ops::FermionBasis{M, S, T, Sym}, oddstate, evenstate) where {M, S<:Number, T, Sym} = robustness(particle_ops, oddstate, evenstate, keys(particle_ops.dict))
