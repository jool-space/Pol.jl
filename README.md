# Pol

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://docs.jool.space/Pol.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://docs.jool.space/Pol.jl/dev/)
[![Build Status](https://github.com/jool-space/Pol.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jool-space/Pol.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/jool-space/Pol.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jool-space/Pol.jl)

Memory management primitives: protocols for kernels to describe the buffers
they need as data, one verb that materializes descriptions into a space of the
caller's choosing, and a bump-allocator with frame-scoped lifetimes.

See https://docs.jool.space/Pol.jl for more details.
