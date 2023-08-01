module MajoranaFunctions
using QuantumDots
using LinearAlgebra
using Roots
using NLsolve
using BlackBoxOptim
export localpairingham, kitaev, measures, findμ0, findϕ, optimize_sweetspot
include("functions.jl")
include("sweetspot.jl")
include("twositess.jl")
end
