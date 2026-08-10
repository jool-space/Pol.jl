using Pol: Pol, Shadowed, primal, shadow, checkpoints, scratch, outputs,
    Space, Frame, Arena, Mark, Undef, Similar, alloc, reset!, watermark, mark,
    retract!, carve, scratchspace, release!, Allocating, takes_space,
    withspace, ambientspace
using Test

using Republic
using JLArrays: JLArray, jl   # the GPUArrays reference array: loads GPUArraysExt
import CUDACore               # loads CUDACoreExt; capture tests gate on functional()

# An array on a device that is mid-capture. Stands in for a `CuArray` while a
# stream capture is recording — the real predicate lives in Pol's CUDACoreExt,
# and dispatches on exactly this exemplar, so the doctrine is testable on CPU.
struct Recording{T,N} <: AbstractArray{T,N}
    a::Array{T,N}
end
Base.size(r::Recording) = size(r.a)
Base.getindex(r::Recording, i::Int...) = getindex(r.a, i...)
Base.similar(r::Recording, ::Type{T}, dims::Dims) where {T} = Recording(similar(r.a, T, dims))
Pol.capturing(::Recording) = true

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

# NamedTuple-out verbs for the Allocating form: outputs destructured first;
# one scratchless, one whose defaulted scratch materializes from `space`
square!((; y), x) = (y .= x .* x; nothing)
Pol.outputs(::typeof(square!), x) = (; y = Undef(x))

function scaled!((; y), x; space = Similar(x),
                 scratch = alloc(space, Pol.scratch(scaled!, x)))
    scratch.tmp .= x .* 2
    copyto!(y, scratch.tmp)
    return
end
Pol.outputs(::typeof(scaled!), x) = (; y = Undef(x))
Pol.scratch(::typeof(scaled!), x) = (; tmp = Undef(x))
Pol.@takes_space scaled!

# a space with no leaf method: exercises the ambient catch-all's loop guard
struct NoLeaf <: Pol.Space end

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

    @testset "Allocating" begin
        x = rand(4)

        @test takes_space(square!) == false
        @test takes_space(scaled!) == true
        @test repr(Allocating(square!)) == "Allocating(square!)"

        # scratchless verb: called bare, no frame opened on its behalf
        square = Allocating(square!)
        out = square(Similar(x), x)
        @test out.y == x .* x

        # space-taking verb: defaulted scratch lands in the wrapper's frame
        scaled = Allocating(scaled!)
        @test scaled(Similar(x), x).y == x .* 2

        # on an arena: outputs carve below the frame's mark and survive its
        # retract; the scratch above the mark is reclaimed
        a = Arena(Vector{UInt8}(undef, 4096))
        out = scaled(a, x)
        @test out.y == x .* 2
        @test a.offset == 4 * sizeof(Float64)          # just the output remains
        @test watermark(a) > a.offset                  # scratch was carved, then retracted

        # `space` may itself be a frame: outputs take the caller's frame lifetime
        scratchspace(a) do frame
            @test scaled(frame, x).y == x .* 2
        end
        @test a.offset == 4 * sizeof(Float64)          # outer close reclaimed those too

        # partially-materialized specs: an array in a spec passes through
        spec = (; fresh = Undef(x), owned = x)
        m = alloc(Similar(x), spec)
        @test m.owned === x && m.fresh isa Vector{Float64}

        # output_override: the overridden entry is never allocated
        reset!(a)
        pre = zeros(4)
        out = scaled(a, x; output_override = (; y = pre))
        @test out.y === pre && pre == x .* 2
        @test a.offset == 0                            # y owned, scratch retracted: nothing remains
    end

    @testset "Similar space" begin
        x = rand(Float32, 4)
        @test alloc(Similar(x), Float64, 2, 3) isa Matrix{Float64}
        @test alloc(Similar(x), Undef(x)) isa Vector{Float32}
    end

    @testset "ambient space" begin
        x = rand(4)
        n = 4 * sizeof(Float64)                        # bytes per test vector

        # unscoped: the zero-arg accessor throws, the defaulted one falls back
        @test_throws ArgumentError ambientspace()
        s = Similar(x)
        @test ambientspace(s) === s
        @test_throws ArgumentError alloc(Float64, 2)
        @test_throws ArgumentError Allocating(square!)(x)
        @test_throws ArgumentError scratchspace(() -> nothing)

        # withspace binds for the dynamic extent, and only that extent;
        # every explicit alloc spelling forwards through one catch-all
        withspace(s) do
            @test ambientspace() === s
            @test alloc(Float64, 2, 3) isa Matrix{Float64}
            @test alloc(Undef(x)) isa Vector{Float64}
            nt = alloc((; fresh = Undef(Float64, 2), owned = x))
            @test nt.fresh isa Vector{Float64} && nt.owned === x
            @test Allocating(square!)(x).y == x .* x
        end
        @test_throws ArgumentError ambientspace()

        # a Space-first call that matches no explicit method is a
        # MethodError, not a trip through the ambient form
        @test_throws MethodError alloc(NoLeaf(), Float32, (2,))
        withspace(s) do
            @test_throws MethodError alloc(NoLeaf(), Float32, (2,))
        end

        # zero-arg scratchspace: a frame on the ambient space, itself rebound
        # as ambient — the ambient space is always the innermost open frame
        a = Arena(Vector{UInt8}(undef, 4096); alignment = 8)
        withspace(a) do
            base = alloc(Float64, 4)                   # ambient = the arena itself
            scratchspace() do
                @test ambientspace() isa Pol.Frame
                tmp = alloc(Float64, 4)
                @test a.offset == 2n
                scratchspace() do                      # nesting: innermost wins
                    alloc(Float64, 4)
                    @test a.offset == 3n
                end
                @test a.offset == 2n                   # inner frame retracted
            end
            @test a.offset == n                        # only base survives

            # ambient Allocating ≡ explicit: outputs live in the ambient
            # space, defaulted scratch in the wrapper's frame
            out = Allocating(scaled!)(x)
            @test out.y == x .* 2
            @test a.offset == 2n                       # y kept, scratch retracted
        end

        # the explicit form rebinds too — the frame argument is the
        # concretely-typed spelling of the same space, not a different one
        scratchspace(s) do frame
            @test ambientspace() === frame
        end
        @test_throws ArgumentError ambientspace()
    end

    @testset "arena: carve mechanics" begin
        a = Arena(Vector{UInt8}(undef, 4096))
        v = alloc(a, Float32, 2, 3)
        @test v isa Matrix{Float32} && size(v) == (2, 3)  # Vector slab: native carve
        @test a.offset == sizeof(Float32) * 6

        # the carve aliases the slab's bytes, above the base pad
        v .= 1.0f0
        @test all(reinterpret(Float32, view(a.slab, a.pad .+ (1:24))) .== 1.0f0)

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

    @testset "arena: absolute alignment via the base pad" begin
        # a Vector's base is only 16/64-aligned; the pad makes carves absolute
        a = Arena(Vector{UInt8}(undef, 4096))
        @test a.pad == mod(-Pol.slabbase(a.slab), UInt(256))
        v = alloc(a, Float32, 4)
        w = alloc(a, Float64, 2)
        @test UInt(pointer(v)) % 256 == 0
        @test UInt(pointer(w)) % 256 == 0

        # no readable base: pad is zero, the relative contract stands
        s = view(Vector{UInt8}(undef, 64), 1:64)
        @test Pol.slabbase(s) === nothing
        @test Arena(s).pad == 0

        # the pad rides on top of the ceiling: a slab that holds the bytes
        # but not the pad still throws
        tight = Arena(Vector{UInt8}(undef, 256); alignment = 256)
        if tight.pad > 0
            @test_throws ArgumentError alloc(tight, UInt8, 256 - tight.pad + 1)
        end
        @test_throws ArgumentError alloc(tight, UInt8, 257)
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

    @testset "GPUArrays extension" begin
        @test !isnothing(Base.get_extension(Pol, :GPUArraysExt))

        # release! is eager for AbstractGPUArrays: a frame closing frees
        # device memory immediately, not at the next GC
        x = jl(rand(Float32, 4))
        local b
        scratchspace(Similar(x)) do frame
            b = alloc(frame, Float32, 2)
            @test b isa JLArray{Float32,1}
            b .= 1f0
            @test Array(b) == [1f0, 1f0]           # alive within the frame
        end
        @test_throws ArgumentError Array(b)        # freed at block close
        @test release!(b) === nothing              # double-free is a no-op

        # an arena over a device byte slab: the generic carve derives a
        # native device array (GPUArrays wraps view/reinterpret/reshape),
        # aliasing the slab's bytes
        slab = JLArray{UInt8}(undef, 4096)
        a = Arena(slab)
        u = alloc(a, Float32, 2, 3)
        @test u isa JLArray{Float32,2} && size(u) == (2, 3)
        u .= 2f0
        @test all(Array(reinterpret(Float32, slab[a.pad .+ (1:24)])) .== 2f0)
    end

    @testset "capture: GC spaces refuse, arenas carve" begin
        # A device that is recording a graph capture. The real answer comes
        # from CUDACoreExt (`is_capturing(stream())`); the guard dispatches on
        # the exemplar, so a mock array exercises the whole doctrine on CPU.
        x = Recording(rand(Float32, 8))

        @test !Pol.capturing(rand(Float32, 8))     # a CPU array never captures
        @test !Pol.capturing(jl(rand(Float32, 4))) # nor a backend without an answer
        @test Pol.capturing(x)

        # a GC-owned space refuses: allocating here would record a memory node
        # into the graph instead of failing, which is the whole hazard
        @test_throws Pol.CaptureViolation alloc(Similar(x), Float32, (4,))
        @test_throws Pol.CaptureViolation scratchspace(Similar(x)) do frame
            alloc(frame, Float32, (4,))            # a frame inherits its space's answer
        end

        # the error names the type and shape it refused, and says what to do
        err = try; alloc(Similar(x), Float32, (2, 3)); catch e; e; end
        msg = sprint(showerror, err)
        @test occursin("Float32(2, 3)", msg)
        @test occursin("Arena", msg)

        # an arena carves regardless: offset arithmetic over a slab allocated
        # long before capture began records nothing
        arena = Arena(zeros(UInt8, 1 << 16))
        @test alloc(arena, Float32, (4,)) isa AbstractArray{Float32,1}
        scratchspace(arena) do frame
            @test alloc(frame, Float32, (4,)) isa AbstractArray{Float32,1}
        end

        # and the property replay actually depends on: identical allocation
        # sequences yield identical addresses
        addrs(a) = (reset!(a); [UInt(pointer(alloc(a, Float32, (i,)))) for i in 1:4])
        @test addrs(arena) == addrs(arena)
    end

    # the real device answering what the Recording mock stands in for
    if CUDACore.functional()
        @testset "capture: CUDA ground truth" begin
            x = CUDACore.zeros(Float32, 8)
            @test !Pol.capturing(x)

            # warm up outside capture: a kernel's first launch loads its
            # module, which capture forbids
            let t = similar(x); t .= 1f0; x .+= t end

            # the predicate flips during a live recording, and a GC space
            # refuses exactly there
            CUDACore.capture() do
                @test Pol.capturing(x)
                @test_throws Pol.CaptureViolation alloc(Similar(x), Float32, (8,))
            end
            @test !Pol.capturing(x)

            # an arena-framed step captures and replays: carves record
            # nothing, frames retract, replays accumulate
            a = Arena(CUDACore.zeros(UInt8, 1 << 12))
            step!() = scratchspace(a) do frame
                t = alloc(frame, Float32, (8,))
                t .= 1f0
                x .+= t
            end
            step!(); step!()                       # warm kernels and pools
            x .= 0f0
            exec = CUDACore.instantiate(CUDACore.capture(step!))
            CUDACore.launch(exec); CUDACore.launch(exec)
            CUDACore.synchronize()
            @test Array(x) == fill(2f0, 8)
            @test a.offset == 0                    # every frame retracted

            # the ground truth behind CaptureViolation: a raw pool alloc
            # records a memory node — the graph launches once, then refuses
            x .= 0f0
            g = CUDACore.capture() do
                t = similar(x); t .= 1f0; x .+= t
            end
            leaky = CUDACore.instantiate(g)
            CUDACore.launch(leaky); CUDACore.synchronize()
            @test Array(x) == fill(1f0, 8)         # the launch that lulls
            @test_throws CUDACore.CuError CUDACore.launch(leaky)
        end
    end

    @testset "visibility" begin
        # the shell vocabulary is exported: what a verb definition spends
        for name in (:Undef, :Similar, :Arena, :scratchspace,
                     :outputs, :checkpoints, :scratch)
            @test Republic.isexported(Pol, name)
        end
        for name in (:capturing, :CaptureViolation)
            @test Republic.ispublic(Pol, name)
        end
        # the machinery is public, not exported: reached qualified
        for name in (:Space, :Frame, :alloc, :Mark, :reset!, :watermark, :mark,
                     :retract!, :carve, :release!, :Shadowed, :primal, :shadow)
            @test Republic.ispublic(Pol, name)
            @test !Republic.isexported(Pol, name)
        end
    end
end
