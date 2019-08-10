# TensorGrad.jl

[![Build Status](https://travis-ci.org/mcabbott/TensorGrad.jl.svg?branch=master)](https://travis-ci.org/mcabbott/TensorGrad.jl)

This package adds gradient definitions for [Zygote.jl](https://github.com/FluxML/Zygote.jl) 
to calculations using [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl).
It provides a macro `@grad` which rewrites an expression like
```julia
@grad @tensor A[i,k] := B[i,j] * C[j,k] * D[l,l]
```
into something equivalent to this:
```julia
fun(b,c,d) = @tensor a[i,k] := b[i,j] * c[j,k] * d[l,l]  # define a function

@adjoint function fun(b,c,d)
    fwd = @tensor a[i,k] := b[i,j] * c[j,k] * d[l,l]     # forward pass
    function back(Δa)
        @tensor Δb[i,j] := Δa[i,k] * c[j,k] * d[l,l]     # reverse pass
        @tensor Δc[j,k] := b[i,j] * Δa[i,k] * d[l,l]
        δ = Diagonal(ones(size(d,1)))
        @tensor Δd[l,l′] := b[i,j] * c[j,k] * Δa[i,k] * δ[l,l′]
        return (Δb, Δc, Δd)
    end
    return (fwd, back)
end

A = fun(B,C,D)                                           # apply this to B, C
```
You may also write `@grad B C @tensor A[i,k] := B[i,j] * C[j,k] * D[l,l]` to specify that
only sensitivities for `B` and `C` are needed, this will remove the calculation 
of `Δd` above. 

### Limitations:

1. The expression must be one term, and scalar factors are not handled yet.
2. Since all the work is done by TensorOperations.jl, it cannot handle more general contractions
  such as `A[i,k] * B[j,k] * C[l,k]`. 
3. It makes no attempt to cache intermediate contractions for re-use, 
  and thus if there are many tensors it will do the same work several times
  (like `b[i,j] * c[j,k]` above, done twice).
4. Requires you to add `@grad` everywhere, so won't work in other people's code.

I can solve 1. For 2, it can equally well call `@einsum` or `@ein` I think. 
But 3 seems hard to solve with this design.

My earlier attempt [TensorTrack.jl](https://github.com/mcabbott/TensorGrad.jl) worked at the level of 
functions `contract!` etc, and thus gets some re-use, 4. 
However with Zygote it doesn't know what sensitivities are needed, and thus computes far too many. 
(With [Tracker.jl](https://github.com/FluxML/Tracker.jl), it could calculate `Δc` only 
when `c::TrackedArray`.) 
But is completely limited by 2, being deeply plugged into TensorOperations.

Another approach is being developed in [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl). 

Note also that [TensorCast.jl](https://github.com/mcabbott/TensorCast.jl) should be almost 
fully differentiable, possibly with [SliceMap.jl](https://github.com/mcabbott/SliceMap.jl). 
It's a terrible way to do contractions but good for other things. 

--- Michael Abbott, August 2019