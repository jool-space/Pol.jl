using Pol: Pol, Shadowed, primal, shadow, checkpoints, scratch, outputs,
    Space, Frame, Arena, Mark, Undef, Similar, alloc, reset!, watermark, mark,
    retract!, carve, scratchspace, release!
using Test

using Republic: public_names, exported_names

# a downstream "primitive" declaring its buffers — pure data, no allocation
dummy_kernel!(y, x; partial) = (partial .= x; copyto!(y, partial))
Pol.outputs(::typeof(dummy_kernel!), x) = (; y = Undef(x))
Pol.checkpoints(::typeof(dummy_kernel!), y, x) = (; stats = Undef{eltype(x)}(1))
Pol.scratch(::typeof(dummy_kernel!), y, x) = (; partial = Undef(x))

# the functional shell: output from the unframed space, scratch in a frame
function dummy_kernel(x; space = Similar(x))
    (; y) = outputs(space, dummy_kernel!, x)
    scratchspace(space) do frame
        dummy_kernel!(y, x; scratch(frame, dummy_kernel!, y, x)...)
    end
    return y
end

# a GPU-like array family: eager reclamation via release!, observably
struct Freeable{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
    freed::Base.RefValue{Bool}
end
Freeable(data::Array) = Freeable(data, Ref(false))
Base.size(f::Freeable) = size(f.data)
Base.getindex(f::Freeable, i::Int...) = f.data[i...]
Base.similar(f::Freeable, ::Type{T}, dims::Dims) where {T} = Freeable(Array{T}(undef, dims))
Pol.release!(f::Freeable) = f.freed[] = true

@testset "Pol.jl" begin
    @testset "Shadowed" begin
        x = rand(3, 4)
        dx = zeros(3, 4)
        d = Shadowed(x, dx)
        @test primal(d) === x
        @test shadow(d) === dx

        # bare arrays and Shadowed thread through forward code identically;
        # a bare array is a constant — no gradient requested, none written
        @test primal(x) === x
        @test shadow(x) === nothing

        # one-argument form: fresh shadow, same type and shape, contents undef
        d2 = Shadowed(x)
        @test typeof(shadow(d2)) == typeof(x)
        @test size(shadow(d2)) == size(x)
        @test shadow(d2) !== x

        # eltype override: narrow primal, wider gradient
        d3 = Shadowed(rand(Float16, 3), Float32)
        @test primal(d3) isa Vector{Float16}
        @test shadow(d3) isa Vector{Float32} && size(shadow(d3)) == (3,)

        # a leading space materializes the shadow from it — a shadow's
        # spec is the primal itself, so no declaration protocol exists
        d4 = Shadowed(Similar(x), x)
        @test typeof(shadow(d4)) == typeof(x) && size(shadow(d4)) == size(x)
        a = Arena(Vector{UInt8}(undef, 4096))
        d5 = Shadowed(a, rand(Float16, 3), Float32)
        @test shadow(d5) isa AbstractVector{Float32} && size(shadow(d5)) == (3,)
        @test a.offset == 3 * sizeof(Float32)      # carved, not similar'd
    end

    @testset "Undef declarations" begin
        x = rand(Float32, 3, 4)
        @test Undef(x) === Undef{Float32}((3, 4))
        @test Undef(x, (5,)) === Undef{Float32}((5,))
        @test Undef{Float64}(2, 3) === Undef{Float64}((2, 3))
        @test Undef(Float64, 2, 3) === Undef{Float64}((2, 3))
    end

    @testset "declaration protocols" begin
        y, x = zeros(4), rand(4)
        spec = scratch(dummy_kernel!, y, x)
        @test spec === (; partial = Undef{Float64}((4,)))
        @test checkpoints(dummy_kernel!, y, x) === (; stats = Undef{Float64}((1,)))
        # a primitive that declares nothing simply has no method
        @test !applicable(checkpoints, identity, x)

        # materialize against an exemplar's family: similar, GC-owned
        sc = @inferred alloc(Similar(x), spec)
        @test sc.partial isa Vector{Float64} && size(sc.partial) == (4,)

        # or against an arena: carved, same spec
        a = Arena(Vector{UInt8}(undef, 4096))
        sc2 = @inferred alloc(a, spec)
        @test sc2.partial isa AbstractVector{Float64} && length(sc2.partial) == 4
        @test a.offset == 4 * sizeof(Float64)      # carved, not similar'd

        # specs nest: buffers bundled under one name stay bundled
        nested = (; scratch = (; tmp = Undef{Float32}(2)), out = Undef{Int}(1))
        nsc = @inferred alloc(Similar(x), nested)
        @test nsc.scratch.tmp isa Vector{Float32} && length(nsc.scratch.tmp) == 2
        @test nsc.out isa Vector{Int}

        # a leading space materializes the declaration on the spot
        out = @inferred outputs(Similar(x), dummy_kernel!, x)
        @test out.y isa Vector{Float64} && size(out.y) == (4,)
        cp = @inferred checkpoints(Similar(x), dummy_kernel!, y, x)
        @test cp.stats isa Vector{Float64} && size(cp.stats) == (1,)

        # scratch's one-step form demands a Frame — proof of an open block;
        # the two-step alloc(space, scratch(f, ...)) remains for the rest
        @test_throws MethodError scratch(Similar(x), dummy_kernel!, y, x)
    end

    @testset "Similar space" begin
        x = rand(Float32, 4)
        @test alloc(Similar(x), Float64, 2, 3) isa Matrix{Float64}
        @test alloc(Similar(x), Undef(x)) isa Vector{Float32}
    end

    @testset "arena: carve mechanics" begin
        a = Arena(Vector{UInt8}(undef, 4096))
        v = alloc(a, Float32, 2, 3)
        @test v isa Matrix{Float32} && size(v) == (2, 3)  # Vector slab: native carve
        @test a.offset == sizeof(Float32) * 6

        # the carve aliases the slab's bytes
        v .= 1.0f0
        @test all(reinterpret(Float32, view(a.slab, 1:24)) .== 1.0f0)

        # alignment: each carve starts at the next 256 boundary
        w = alloc(a, UInt8, 3)
        @test a.offset == 256 + 3
        z = alloc(a, Float64, 2)
        @test a.offset == 512 + 16

        # resetting and re-carving reuses the same bytes (contents are undefined,
        # i.e. whatever was last written there)
        reset!(a)
        v2 = alloc(a, Float32, 6)
        @test v2[1] == 1.0f0

        # a non-Vector slab takes the generic carve: a reinterpreted view,
        # GC-rooted through the slab, still aliasing its bytes
        s = view(Vector{UInt8}(undef, 8192), 1:4096)
        b = Arena(s)
        u = alloc(b, Float32, 2, 3)
        @test u isa AbstractMatrix{Float32} && size(u) == (2, 3)
        u .= 2.0f0
        @test all(reinterpret(Float32, view(s, 1:24)) .== 2.0f0)
    end

    @testset "arena: alignment argument" begin
        # alignment = 1 packs tight, but never below the element type's own
        a = Arena(Vector{UInt8}(undef, 4096); alignment = 1)
        alloc(a, UInt8, 3)
        alloc(a, Float64, 2)                       # starts at 8, not 3
        @test a.offset == 8 + 16
        # page alignment for pinned/O_DIRECT-style consumers
        p = Arena(Vector{UInt8}(undef, 2^14); alignment = 4096)
        alloc(p, UInt8, 1)
        alloc(p, UInt8, 1)
        @test p.offset == 4096 + 1
        @test_throws ArgumentError Arena(Vector{UInt8}(undef, 64); alignment = 3)
    end

    @testset "arena: watermark" begin
        a = Arena(Vector{UInt8}(undef, 4096))
        alloc(a, UInt8, 100)
        m = mark(a)
        alloc(a, UInt8, 700)           # peak: aligned start 256 + 700
        retract!(a, m)
        alloc(a, UInt8, 10)
        @test watermark(a) == 956
        # survives reset — the requirement is the max over all passes
        reset!(a)
        @test watermark(a) == 956
        alloc(a, UInt8, 10)
        @test watermark(a) == 956
    end

    @testset "arena: ceiling, no growth" begin
        a = Arena(Vector{UInt8}(undef, 512))
        alloc(a, UInt8, 200)
        @test_throws ArgumentError alloc(a, UInt8, 300)   # aligned start 256 + 300 > 512
        @test a.offset == 200                              # a failed carve changes nothing
        @test_throws ArgumentError alloc(Arena(Vector{UInt8}(undef, 16)), Float64, 3)
        @test_throws ArgumentError alloc(Arena(Vector{UInt8}(undef, 64)), String, 1)  # bitstype only
    end

    @testset "arena: mark/retract discipline" begin
        a = Arena(Vector{UInt8}(undef, 4096))
        alloc(a, UInt8, 10)                        # a "checkpoint" below the frame
        m = mark(a)
        alloc(a, Float32, 8); alloc(a, Int32, 4)   # a whole frame, two carves
        retract!(a, m)                             # one retraction pops both
        @test a.offset == 10
        retract!(a, m)                             # same mark, nothing in between: no-op
        @test a.offset == 10

        # retract-alloc-retract-again is the double-pullback bug: caught loudly
        alloc(a, UInt8, 300)
        m2 = mark(a)
        retract!(a, m)
        @test_throws ArgumentError retract!(a, m2) # m2 above the current offset

        # marks from before a reset! are stale: caught loudly
        m3 = mark(a)
        reset!(a)
        @test_throws ArgumentError retract!(a, m3)
    end

    @testset "scratchspace: a frame on the space" begin
        x = rand(4)
        # the primitive is a plain function: callable with no Pol in sight
        y0 = zeros(4)
        @test dummy_kernel!(y0, x; partial = similar(x)) == x

        # GC space: the block gets the framed space
        y = zeros(4)
        result = scratchspace(Similar(x)) do frame
            dummy_kernel!(y, x; scratch(frame, dummy_kernel!, y, x)...)
            :done
        end
        @test result === :done
        @test y == x

        # arena: everything carved inside is reclaimed when the block closes
        a = Arena(Vector{UInt8}(undef, 4096))
        alloc(a, UInt8, 32)                        # something below the frame
        y2 = zeros(4)
        scratchspace(a) do frame
            @test frame isa Frame && frame.arena === a
            sc = scratch(frame, dummy_kernel!, y2, x)
            @test sc.partial isa AbstractVector{Float64} && length(sc.partial) == 4
            @test a.offset > 32
            dummy_kernel!(y2, x; sc...)
        end
        @test y2 == x
        @test a.offset == 32                       # frame fully retracted, floor intact

        # frames span any number of allocations and calls
        scratchspace(a) do frame
            alloc(frame, Float32, 8)
            alloc(frame, Int32, 4)
            @test a.offset > 32
        end
        @test a.offset == 32

        # exception-safe: the frame is retracted even on a throw
        @test_throws ErrorException scratchspace(a) do frame
            alloc(frame, Float64, 100)
            error("boom")
        end
        @test a.offset == 32

        # frames nest: an inner frame on an arena frame stacks a new mark
        scratchspace(a) do outer
            alloc(outer, UInt8, 8)
            floor = a.offset
            scratchspace(outer) do inner
                alloc(inner, UInt8, 8)
                @test a.offset > floor
            end
            @test a.offset == floor                # inner popped, outer intact
        end
        @test a.offset == 32

        # the functional shell: output allocated at caller level, before the
        # frame — it survives; the scratch above it is retracted
        z = dummy_kernel(x)
        @test z == x && z isa Vector{Float64}
        z2 = dummy_kernel(x; space = a)
        @test z2 == x
        @test a.offset == 256 + 32                 # output kept, scratch gone
    end

    @testset "scratchspace: frame release" begin
        # non-arena frames track their buffers and release! them on close
        x = Freeable(rand(4))
        local p, q
        scratchspace(Similar(x)) do frame
            p = alloc(frame, Float32, 2)
            q = alloc(frame, (; t = Undef{Int}(3))).t   # nested-spec leaves tracked too
            @test !p.freed[] && !q.freed[]              # alive within the frame
        end
        @test p.freed[] && q.freed[]

        # released even on a throw
        local r
        @test_throws ErrorException scratchspace(Similar(x)) do frame
            r = alloc(frame, Float64, 1)
            error("boom")
        end
        @test r.freed[]

        # nested tracked frames release at their own level, not the outer's
        local outer_buf, inner_buf
        scratchspace(Similar(x)) do outer
            outer_buf = alloc(outer, Float32, 2)
            scratchspace(outer) do inner
                inner_buf = alloc(inner, Float32, 2)
                @test !inner_buf.freed[]
            end
            @test inner_buf.freed[]                # inner released at inner close
            @test !outer_buf.freed[]               # outer's still live
        end
        @test outer_buf.freed[]

        # the unframed space does not track: GC semantics
        s = alloc(Similar(x), Float32, 2)
        @test !s.freed[]

        # the default release! is a no-op — plain GC arrays just lapse
        @test release!(rand(2)) === nothing
    end

    @testset "visibility" begin
        _public = public_names(Pol)
        _exported = exported_names(Pol)
        # the shell vocabulary is exported: what a verb definition spends
        for name in (:Undef, :Similar, :Arena, :scratchspace,
                     :outputs, :checkpoints, :scratch)
            @test name in _exported
        end
        # the machinery is public, not exported: reached qualified
        for name in (:Space, :Frame, :alloc, :Mark, :reset!, :watermark, :mark,
                     :retract!, :carve, :release!, :Shadowed, :primal, :shadow)
            @test name in _public
            @test !(name in _exported)
        end
    end
end
