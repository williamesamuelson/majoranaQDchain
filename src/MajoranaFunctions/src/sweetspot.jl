function zeroenergy_condition(μ, Δind, Vz, U)
    β = √(μ^2+Δind^2)
    return -Vz + β - U/2*(1 - μ/β)
end

function μguess(Δind, Vz, U)
    f(μ) = zeroenergy_condition(μ, Δind, Vz, U)
    μinit = √(Vz^2-Δind^2)
    return find_zero(f, μinit)
end

#Now I choose the t=-Δ solution (?)
function get_sweetspot_nlsolvefunc(λ, Vz, U)
    function f!(F, x)
        μ = x[1]
        Δind = x[2]
        β = √(μ^2 + Δind^2)
        F[1] = μ*cos(λ) - Δind*sin(λ)
        F[2] = zeroenergy_condition(μ, Δind, Vz, U)
    end
end

function get_sweetspot_nlsolvejac(λ, Vz, U)
    function j!(J, x)
        μ = x[1]
        Δind = x[2]
        β = √(μ^2 + Δind^2)
        J[1, 1] = cos(λ)
        J[1, 2] = -sin(λ)
        J[2, 1] = 1/β * (μ + U*Δind^2/(2*β^2))
        J[2, 2] = Δind/β * (1 - U*μ/(2*β^2))
    end
end

function μΔind_init(λ, Vz, U)
    f = get_sweetspot_nlsolvefunc(λ, Vz, U)
    J = get_sweetspot_nlsolvejac(λ, Vz, U)
    sol = nlsolve(f, J, [Vz*sin(λ); Vz*cos(λ)], show_trace=true)
    return sol.zero # μ, Δind
end
