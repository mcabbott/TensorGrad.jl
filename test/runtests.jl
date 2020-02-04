using Test
using TensorGrad, TensorOperations, Einsum, LinearAlgebra, ForwardDiff, Zygote
Zygote.refresh()

@testset "simple" begin
    m1 = rand(2,3)
    m2 = rand(3,4)
    m3 = randn(4,4)

    f1(x,y) = @grad @tensor z[i,k] := x[i,j] * y[j,k]
    @test f1(m1, m2) ≈ m1 * m2

    g1 = Zygote.gradient((x,y) -> sum(sin, f1(x,y)), m1, m2)
    @test g1[1] ≈ ForwardDiff.gradient(x -> sum(sin, f1(x,m2)), m1)
    @test g1[2] ≈ ForwardDiff.gradient(y -> sum(sin, f1(m1,y)), m2)

    f2(x,y) = @grad @tensor z[i,j] := x[j,i] * y[k,k]
    @test f2(m1,m3) ≈ transpose(m1) .* tr(m3)

    g2 = Zygote.gradient((x,y) -> sum(sin, f2(x,y)), m1, m3)
    @test g2[1] ≈ ForwardDiff.gradient(x -> sum(sin, f2(x,m3)), m1)
    @test g2[2] ≈ ForwardDiff.gradient(y -> sum(sin, f2(m1,y)), m3)

    f3(x,y) = @grad x @tensor z[i,j] := x[j,i] * y[k,k]
    g3 = Zygote.gradient((x,y) -> sum(sin, f3(x,y)), m1, m3)
    @test g3[1] == g2[1]
    @test g3[2] == nothing

    f4(x) = @grad @tensor z[i,j] := x[i,k] * x[j,k]
    @test f4(m1) ≈ m1 * transpose(m1)

    g4 = Zygote.gradient(x -> sum(sin, f4(x)), m1)
    @test g4[1] ≈ ForwardDiff.gradient(x -> sum(sin, f4(x)), m1)

    f5(x) = @grad @tensor z[] := 3 * x[i,k] * x[i,k]
    @test f5(m2)[] ≈ 3 * sum(abs2, m2)

    # g5 = Zygote.gradient(x -> sum(sin, f5(x)), m2) # error @tensor _Δ_x[i, k] = 3 * _Δ[] * _Δ[]

end

@testset "einsum" begin

    # Same as above
    m1 = rand(2,3)
    m2 = rand(3,4)
    f1(x,y) = @grad @einsum z[i,k] := x[i,j] * y[j,k]
    @test f1(m1, m2) ≈ m1 * m2

    g1 = Zygote.gradient((x,y) -> sum(sin, f1(x,y)), m1, m2)
    @test g1[1] ≈ ForwardDiff.gradient(x -> sum(sin, f1(x,m2)), m1)
    @test g1[2] ≈ ForwardDiff.gradient(y -> sum(sin, f1(m1,y)), m2)

    # batch matmul
    t1 = rand(2,2,2);
    t2 = rand(2,2,2);
    f10(x,y) = @grad @einsum z[i,k,b] := x[i,j,b] * y[j,k,b]
    t3 = similar(t1);
    for b=1:2
        t3[:,:,b] .= t1[:,:,b] * t2[:,:,b]
    end
    @test t3 ≈ f10(t1, t2)

    g10 = Zygote.gradient((x,y) -> sum(sin, f10(x,y)), t1, t2)
    @test g10[1] ≈ ForwardDiff.gradient(x -> sum(sin, f10(x,t2)), t1)
    @test g10[2] ≈ ForwardDiff.gradient(y -> sum(sin, f10(t1,y)), t2)

    f11(x,y) = @grad @einsum z[i,b,k] := x[i,b,j] * y[k,b,j]
    g11 = Zygote.gradient((x,y) -> sum(sin, f11(x,y)), t1, t2)
    @test g11[1] ≈ ForwardDiff.gradient(x -> sum(sin, f11(x,t2)), t1)
    @test g11[2] ≈ ForwardDiff.gradient(y -> sum(sin, f11(t1,y)), t2)

end

@testset "errors" begin

    # multiple terms
    @test_throws Exception TensorGrad._grad(:( @tensor S[i,j] := 2 * x[i,j] + 3 * r32[j,i] ))

    # not a contraction
    # @test_throws Exception TensorGrad._grad(:( @einsum A[i,j] := exp(B[i,j]) ))

end

# https://github.com/FluxML/Zygote.jl/blob/master/test/gradcheck.jl

function ngradient(f, xs::AbstractArray...)
  grads = zero.(xs)
  for (x, Δ) in zip(xs, grads), i in 1:length(x)
    δ = sqrt(eps())
    tmp = x[i]
    x[i] = tmp - δ/2
    y1 = f(xs...)
    x[i] = tmp + δ/2
    y2 = f(xs...)
    x[i] = tmp
    Δ[i] = (y2-y1)/δ
  end
  return grads
end

gradcheck(f, xs...) =
  all(isapprox.(ngradient(f, xs...),
                Zygote.gradient(f, xs...), rtol = 1e-5, atol = 1e-5))

gradtest(f, xs::AbstractArray...) = gradcheck((xs...) -> sum(sin.(f(xs...))), xs...)
gradtest(f, dims...) = gradtest(f, rand.(Float64, dims)...)

@testset "january" begin
    # Tests written for my tracing approach, recycled here:

    triv1(x) = @grad @tensor A[i,j] := 2 * x[i,j]
    gradtest(triv1, (2,3))

    r32 = randn(3,2);
    r312 = randn(3,1,2);
#=
    ## add!
    add1(x) = @grad @tensor S[i,j] := 2 * x[i,j] + 3 * r32[j,i]
    @test gradtest(add1, (2,3))

    add2(y) = @grad @tensor S[i,j] := 2 * r32[i,j] + 3 * y[j,i]
    @test gradtest(add2, (2,3))

    add3(x) = @grad @tensor S[k,j,i] := 0.11 * x[i,j,k] - 33 * r312[k,i,j]
    @test gradtest(add3, (1,2,3))
=#

    ## trace!
    tr1(x) = @grad @tensor T[k] := 22 * x[i,i,k]
    @test gradtest(tr1, (3,3,4))

    tr2(x) = @grad @tensor T[k] := 22 * x[i,i,k,j,j]
    @test gradtest(tr2, (3,3,4,7,7))

    # tr3add(x) = prod(@grad @tensor R[i,j] := 5 * x[k,i,k,j] + 7 * r32[j,i])
    # @test gradcheck(tr3add, rand(7,2,7,3))

    ## contract! A
    con1(x) = @grad @tensor C[i,j] := 5 * x[i,k] * r32[k,j]
    @test gradtest(con1, (2,3))

    r22 = rand(2,2);
    # con2add(x) = @grad @tensor C[i,j] := 5 * x[i,k] * r32[k,j] + 7 * r22[j,i]
    # @test gradtest(con2add, (2,3))

    con3(x) = @grad @tensor C[i,j,m,n] := x[i,j,k] * r312[k,m,n]
    @test gradtest(con3, (1,2,3))

    con4(x) = @grad @tensor C[i,m] := x[i,kk,k] * r312[k,m,kk]
    @test gradtest(con4, (1,2,3))

    con5(x) = @grad @tensor C[j,i,n,m] := 44 * x[i,j,k] * r312[k,m,n]
    @test gradtest(con5, (1,2,3))

    r392 = randn(3,9,2);
    con6(x) = @grad @tensor C[n,i,m,j] := x[i,j,k] * r392[k,m,n]
    @test gradtest(con6, (9,2,3))

    con7(x) = @grad @tensor C[m,n,j,i] := 44 * x[i,j,k] * r392[k,m,n]
    @test gradtest(con7, (9,2,3))

    ## contract! B
    con8b(x) = @grad @tensor K[i,j] := 5 * r32[i,k] * x[k,j]
    @test gradtest(con8b, (2,3))

    con9b(x) = @grad @tensor K[i,j,m,n] := r312[i,j,k] * x[m,k,n]
    @test gradtest(con9b, (1,2,3))

    con10b(x) = @grad @tensor K[n,j,m,i] := r392[i,j,k] * x[m,k,n]
    @test gradtest(con10b, (9,2,3))

    r3399 = randn(3,3,9,9);
#=
    con11add(x) = @grad @tensor K[n,j,m,i] := r392[i,j,k] * x[m,k,n] + 7 * r3399[n,i,j,m]
    @test gradtest(con11add, (9,2,3))

    con12add(x) = @grad @tensor K[n,j,m,i] := r392[i,j,k] * x[m,n,k] + 7 * r3399[n,i,j,m] - r3399[i,n,m,j]
    @test gradtest(con12add, (9,3,2))
=#
    con13(x) = @grad @tensor K[i,j] := r3399[s,s,j,k] * x[t,t,k,i]
    @test gradtest(con13, (3,3,9,9))

    r33 = rand(3,3);
    con14(x) = @grad @tensor K[i,j] := r3399[a,b,j,k] * x[b,c,k,i] * r33[a,c]
    @test gradtest(con14, (3,3,9,9))
#=
    con15(x) = @grad @tensor K[i,j] := r3399[a,a,j,k] * x[b,b,k,i] + 3.14 * r33[a,b] * x[a,c,k,i] * x[c,b,j,k]
    @test gradtest(con15, (3,3,9,9))

    con16(x) = @grad @tensor K[i,j] := r3399[a,b,j,k] * x[b,a,k,i] + 3.14 * r33[a,b] * x[a,c,k,i] * x[c,b,j,k]
    @test gradtest(con16, (3,3,9,9))
=#
end
