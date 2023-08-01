findα(μ, Δind) = 1/2*atan.(μ./Δind)

function kitaevtunneling(μ, Δind, Vz, ϕ)
    α = findα(μ, Δind)
    return abs(sin(α[2]+α[1])*cos(ϕ/2) + 1im*cos(α[2]-α[1])*sin(ϕ/2))
end

function kitaevΔ(μ, tsoq, Δind, Vz, ϕ)
    α = findα(μ, Δind)
    return tsoq*abs(cos(α[2]+α[1])*cos(ϕ/2) + 1im*sin(α[2]-α[1])*sin(ϕ/2))
end

function teqΔcondition(μ, tsoq, Δind, Vz, ϕ)
    return kitaevtunneling(μ, Δind, Vz, ϕ) - kitaevΔ(μ, tsoq, Δind, Vz, ϕ)
end

function findϕ(μ, tsoq, Δind, Vz)
    f(ϕ) = teqΔcondition(μ, tsoq, Δind, Vz, ϕ)
    ϕres = find_zero(f, pi/2)
    return ϕres
end

findμ0(Δind, Vz, U, par) = -U/2 .+ [1, (-1)^(Int(!par))]*√((Vz+U/2)^2 - Δind^2)

function init_optim(params, par)
    tsoq = tan(params[:λ])
    μ1, μ2 = findμ0(params[:Δind], params[:Vz], params[:U], par)
    params[:μ] = [μ1, μ2]
    ϕ = findϕ(params[:μ], tsoq, params[:Δind], params[:Vz])
    return [μ1, μ2, ϕ]
end

function create_optfunc(particle_ops, params, fixϕ=false)
    function optfunc(x)
        params[:μ] = [x[1], x[2]]
        if !fixϕ
            params[:Φ] = [0, x[3]]
        end
        deg, mp, _, _ = measures(particle_ops, localpairingham, params, 2)
        return deg^2 + 1-mp
    end
    return optfunc
end

function optimize_sweetspot(params, par, μadd, maxtime; fixϕ=false)
    particle_ops = FermionBasis((1:2), (:↑, :↓), qn=QuantumDots.parity) 
    init = init_optim(params, par)
    ranges = [((init[i] - μadd, init[i] + μadd) for i in 1:2)..., (0, pi)]
    if fixϕ
        ranges = ranges[1:2]
        init = init[1:2]
    end
    optfunc = create_optfunc(particle_ops, params, fixϕ)
    res = bboptimize(optfunc, init, SearchRange=ranges,
                    TraceMode=:compact, MaxTime=maxtime)
    return best_candidate(res)
end
