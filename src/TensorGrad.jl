module TensorGrad

export @grad

using MacroTools, LinearAlgebra

"""
    @grad @tensor A[i,j,k] := B[i,...] * C[...] * D...

Macro which provides gradient definitions for Zygote.jl & Tracker.jl,
for the given tensor contraction handled by TensorOperations.jl.

By default gradients are defined for all tensors on the right,
but writing for instance `@grad B D @tensor A[...] ...` will treat `C` as a constant.

* Mutation with `@tensor A[...] = ...` rather than `:=` is not supported.
* Applying the macro to a block of code `@tensor begin A[] := ... end` is also not supported.

Not working yet:
* Constants like `a * B[i,j,...]`
* Multiple terms `B[...] * C[...] + D[...] * E[...]`
* Other macros like `@einsum A[i,j] := ...`
"""
macro grad(exs...)
    _grad(exs...; mod=__module__)
end

const VERBOSE = Ref(false)
"""
    TensorGrad.verbose(true)

Turns on printing of `Zygote.@adjoint` definition.
"""
function verbose(x::Bool)
    VERBOSE[] = x
    @show TensorGrad.VERBOSE[]
    nothing
end

function _grad(exs...; mod=Main)
    ex = exs[end]
    @capture(ex, @tensor left_ := right_) || error("don't understand input, expected @tensor A[...] := B[...] * ...")
    @capture(left, A_[leftind__]) || error("don't understand LHS, expected A[i,j,...], got $left")
    if length(exs) == 1
        gradlist = nothing
    else
        gradlist = exs[1:end-1]
    end

    arrays, indices, scalars = [], [], []
    MacroTools.postwalk(right) do x
        if @capture(x, B_[ijk__])
            B isa Symbol || error("don't like $B")
            push!(arrays, B)
            push!(indices, ijk)
            return nothing
        # elseif x isa Symbol # finds all the wrong things!
        #     @show x
        #     push!(scalars, x)
        end
        return x
    end
    inputs = unique(vcat(arrays, scalars))

    isempty(intersect(arrays, scalars)) || error("can't have the same symbol as array & scalar")
    A in inputs && error("output array can't appear on the right too")

    # The same symbol may appear several times in arrays, from same or distinct terms
    # To build up backward function, first define Δ_B = similar(B) (and later can use LRU cache)
    # then add on gradients for each appearance.

    backsteps, backseen = [], []
    for (B,ijk) in zip(arrays, indices)
        gradlist == nothing || B in gradlist || continue

        deltaB = Symbol("_Δ_", B)

        newright, extra, ijk = replace_B_with_Δ(B, ijk, right, leftind)
        append!(backsteps, extra)

        if B in backseen
            addon = :( @tensor $deltaB[$(ijk...)] = $deltaB[$(ijk...)] + $newright )
            push!(backsteps, addon)
        else
            # create = :( $deltaB = similar($B) )
            tup123 = Tuple(1:length(ijk))
            symB = QuoteNode(gensym(string("_Δ_", B, '_', join(ijk), '_')))
            create = :( $deltaB = TensorOperations.cached_similar_from_indices($symB, eltype($B), $tup123, (), $B, :N) )
            infill = :( @tensor $deltaB[$(ijk...)] = $newright )
            append!(backsteps, Any[create, infill])
            push!(backseen, B)
        end
    end
    for β in scalars
        error("scalars not working yet, sorry")
        # Simple way is to replace β with 1 & sew with Δ
        # Efficient way is to calculate A without β first,
        # contract that for gradient, then scale!(A, β)
        # but this must happen per term.
    end
    backtuple = [ B in backseen ? Symbol("_Δ_", B) : nothing for B in inputs ]

    @gensym fun fwd back
    zygote_defn = quote
        $fun($(inputs...)) = $ex

        Zygote.@adjoint function $fun($(inputs...))
            $fwd = $ex
            $back(_Δ::Zygote.FillArrays.Fill) = back(collect(_Δ)) # not sure this works
            function $back(_Δ)
                $(backsteps...)
                return ($(backtuple...),)
            end
            return $fwd, $back
        end
    end

    # Zygote.@adjoint seems to need to be at top level, but this means
    # you can't @macroexpand1 to see...
    # Better just define forward() yourself? But what's Context()?
    # There may be scope issues with where fun() is defined now.
    if VERBOSE[]
        defn_ = MacroTools.alias_gensyms(MacroTools.striplines(zygote_defn))
        @show defn_
    end

    # Now define the same steps for tracker?
    # Would be better to check if any are tracked, not just the first input
    # Likewise needs to be at top level, it says.
    primeinputs = map(B -> Symbol("_",B,"′_"), inputs)
    tracker_defn = quote
        $fun($(inputs[1])::Tracker.TrackedArray, $(inputs[2:end]...)) = Tracker.track($fun, $(inputs...))

        Tracker.@grad function $fun($(primeinputs...))
            ($(inputs...),) = Tracker.data.(($(primeinputs...),))
            $fwd = $ex
            function $back(_Δ)
                $(backsteps...)
                return ($(backtuple...),)
            end
            return $fwd, $back
        end
    end

    fun_defn = :( $fun($(inputs...)) = $ex )

    @eval mod $fun_defn

    if isdefined(mod, :Zygote)
        @eval mod $zygote_defn
    end

    if isdefined(mod, :Tracker)
        @eval mod $tracker_defn
    end

    return esc(:( $A = $fun($(inputs...)) ))
end

function replace_B_with_Δ(B, Bijk, right, leftind)
    # This needs to remove B, and replace with Δ[leftind...]
    # But this has two problems: (1) if B[ijk] occurs twice it's wrong,
    countB = 0

    # and (2) if right has several terms, then it's wrong,
    # although you can fix it in this function without too much effort?
    if @capture(right, term1_ + term2_ ) || @capture(right, term1_ - term2_ )
        error("can't deal with multiple terms yet, sorry")
    end

    extra, deltas = [], []
    newijk = copy(Bijk)
    if !allunique(Bijk)
        n = 1
        for n=1:length(Bijk)
            i = newijk[n]
            m = findfirst(isequal(i), newijk[n+1:end])
            if m != nothing
                j = Symbol('_',i,'′')
                newijk[n] = j
                delta = Symbol("_δ_",i,j)
                push!(extra, :($delta = $Diagonal(similar($B, real(eltype($B)), size($B,$n)) .= 1) ))
                push!(deltas, :( $delta[$i,$j] ))
            end
        end
    end

    out = MacroTools.postwalk(right) do x
        if @capture(x, $B[ijk__]) && ijk == Bijk
            countB += 1
            return :( _Δ[$(leftind...)] )
        else
            return x
        end
    end

    if length(extra) > 0
        out = :( *($out, $(deltas...)) )
    end

    countB > 1 && @warn "can't handle case of $B appearing twice with same indices, sorry"
    # Gradient has indices appearing only on LHS... so you need * ones()[i,j]?
    # Could also multiply by countB to avoid repetition

    return out, extra, newijk
end

end # module
