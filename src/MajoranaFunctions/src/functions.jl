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
    return QuantumDots.blockdiagonal(Matrix(ham), d)
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
    return QuantumDots.blockdiagonal(Matrix(ham), d)
end

function kitaev(particle_ops, params::Dict{Symbol, T}) where T
    sites = QuantumDots.nbr_of_fermions(particle_ops)
    newparams = convert_to_namedtuple(params, sites)
    return kitaev(particle_ops, newparams)
end

function kitaevtoakhmerovparams(t, Δ, α) 
    λ = atan.(Δ*tan(2α)/t)
    w = t./(cos.(λ)*sin(2α))
    return w, λ
end

function handleblocks(particle_ops, ham_fun, params)
    H = ham_fun(particle_ops, params)
    energies, blockvecs = QuantumDots.BlockDiagonals.eigen_blockwise(H)
    vecs = Matrix(blockvecs)
    oddind = 1
    evenind = 2^(QuantumDots.nbr_of_fermions(particle_ops) - 1) + 1
    return energies, vecs, oddind, evenind
end

function groundstates(particle_ops, ham_fun, params)
    _, vecs, oddind, evenind = handleblocks(particle_ops, ham_fun, params)
    return vecs[:, oddind], vecs[:, evenind]
end

function measures(particle_ops, ham_fun, params, sites)
    energies, vecs, oddind, evenind = handleblocks(particle_ops, ham_fun, params)
    gap = (energies[evenind] - energies[oddind])
    sort!(energies)
    top_gap = energies[3] - energies[1]
    gap /= top_gap # normalize by topological gap
    mp = majoranapolarization(particle_ops, vecs[:,oddind], vecs[:,evenind], sites)
    dρ = robustness(particle_ops, vecs[:, oddind], vecs[:, evenind], sites)
    return gap, mp, dρ
end

function majoranacoeffs(particle, oddstate, evenstate)
    d = particle
    plus_matrixelement = dot(oddstate, d'+d, evenstate)
    minus_matrixelement = dot(oddstate, 1im*(d'-d), evenstate)
    aplus = real(plus_matrixelement)
    aminus = real(minus_matrixelement)
    bplus = -imag(plus_matrixelement)
    bminus = -imag(minus_matrixelement)
    return aplus, aminus, bplus, bminus
end

function constructmajoranas(particle_ops, oddstate, evenstate)
    γplus = 0*first(particle_ops.dict)
    γminus = copy(γplus)
    for (label, op) in pairs(particle_ops.dict)
        γjk_plus = op' + op
        γjk_minus = 1im*(op'-op)
        aplus, aminus, bplus, bminus = majoranacoeffs(particle_ops[label], oddstate, evenstate)
        γplus += aplus*γjk_plus + aminus*γjk_minus
        γminus += bplus*γjk_plus + bminus*γjk_minus
    end
    return γplus, γminus
end

function majoranapolarization(particle_ops, oddstate, evenstate, sites)
    n = sites÷2 + sites%2  # sum over half of the sites plus middle if odd
    mp = 0
    for (label, op) in pairs(particle_ops.dict)
        if first(label) > n
            continue
        end
        aplus, aminus, bplus, bminus = majoranacoeffs(particle_ops[label], oddstate, evenstate)
        mp += aplus^2 + aminus^2 - bplus^2 - bminus^2
    end
    return mp
end

function robustness(particle_ops, oddstate, evenstate, sites)
    dρ = 0
    for j in 1:sites
        keeplabels = tuple(keys(QuantumDots.cell(j, particle_ops))...)
        ρe, ρo = map(ψ -> QuantumDots.reduced_density_matrix(ψ, keeplabels, particle_ops),
                    (evenstate, oddstate))
        # println("Even, site $j")
        # display(round.(real.(ρe), digits=2))
        # println("Odd, site $j")
        # display(round.(real.(ρo), digits=2))
        dρ += norm(ρe - ρo)^2
    end
    return dρ/sites
end

function scan1d(scan_params, fix_params, particle_ops, ham_fun, points, sites)
    d = particle_ops
    gaps = zeros(Float64, points)
    mps = zeros(Float64, points)
    dρs = zeros(Float64, points)
    params = merge(scan_params, fix_params)
    for i = 1:points
        for (p, val) in scan_params
            params[p] = val[i]
        end
        gaps[i], mps[i], dρs[i] = measures(d, ham_fun, params, sites)
    end
    return gaps, mps, dρs
end

function scan2d(xparams, yparams, fix_params, particle_ops, ham_fun, points, sites)
    d = particle_ops
    gaps = zeros(Float64, points, points)
    mps = zeros(Float64, points, points)
    dρs = zeros(Float64, points, points)
    params = merge(xparams, yparams, fix_params)
    for i = 1:points
        for (p, val) in yparams
            params[p] = val[i]
        end
        for j = 1:points
            for (p, val) in xparams
                params[p] = val[j]
            end
            gaps[i, j], mps[i, j], dρs[i, j] = measures(d, ham_fun, params, sites)
        end
    end
    return gaps, mps, dρs
end
