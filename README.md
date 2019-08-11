# TensorGrad.jl

[![Build Status](https://travis-ci.org/mcabbott/TensorGrad.jl.svg?branch=master)](https://travis-ci.org/mcabbott/TensorGrad.jl)

This package adds gradient definitions for [Zygote.jl](https://github.com/FluxML/Zygote.jl) 
to calculations using [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl).
It exports a macro `@grad` which rewrites an expression like
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

A = fun(B,C,D)                                           # apply this to B, C, D
```
You may also write `@grad B C @tensor A[i,k] := B[i,j] * C[j,k] * D[l,l]` to specify that
only sensitivities for `B` and `C` are needed, this will remove the calculation 
of `Δd` above. 

To see what is being defined, call `TensorGrad.verbose(true)` before the macro 
(rather than using `@macroexpand1`).

If [Tracker.jl](https://github.com/FluxML/Tracker.jl) is loaded, then it will now
define the same gradients for `B::TrackedArray` etc. 

Note that this is a fairly crude experiment, probably not something to rely on.

### Limitations:

1. The expression must be one term, and scalar factors are not handled yet.
2. Since all the work is done by TensorOperations.jl, it cannot handle more general contractions
  such as `A[i,k] * B[j,k] * C[l,k]`. 
3. It makes no attempt to cache intermediate contractions for re-use, 
  and thus if there are many tensors it will do the same work several times
  (like `b[i,j] * c[j,k]` above, done twice).
4. Requires you to add `@grad` everywhere, so won't work in other people's code.

I can solve 1. But 3 seems hard to solve with this design.

For 2, it can equally well be taught to call another macro like `@einsum`, I think. 
Or `@ein` from [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl),
which may be pointless as that has its own gradients built-in. 
Probably you should use that instead! 

An earlier attempt is now [TensorTrack.jl](https://github.com/mcabbott/TensorTrack.jl), which works at the level of 
functions `contract!` etc, and thus gets some re-use, 4. 
But is completely limited by 2, being deeply plugged into TensorOperations.

Note also that [TensorCast.jl](https://github.com/mcabbott/TensorCast.jl) should be almost 
fully differentiable, possibly with [SliceMap.jl](https://github.com/mcabbott/SliceMap.jl). 
It's a terrible way to do contractions but good for other things. 

--- Michael Abbott, August 2019
