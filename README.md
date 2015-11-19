# CPU Rasterizer

## Specs
Currently only x86 is supported with SSE2 requirement (64bit build is preferred, since more SIMD registers).

## TODO
- Binning of triangles to tiles (currently each tile tests all triangles).
- Add SIMD for triangles (process four triangles once).
- Texture mapping.
- Clip with near and far planes.
- Fill rules.
- Vertex normals.
- Some kind of lighting system.
- Add hierarchical rasterization.
- Add hierarchical Z-Buffering.
- Add guard band clipping.

