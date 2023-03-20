# Majorana bound states in chains of interacting quantum dots

Ongoing repository for my master's thesis. Based on the (unpublished) Julia package "QuantumDots". Most of the code can be found under src/MajoranaFunctions/src.



This code base is using the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> majoranaQDchain

It is authored by William Samuelson.

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:
```julia
using DrWatson
@quickactivate "majoranaQDchain"
```
which auto-activate the project and enable local path handling from DrWatson.
