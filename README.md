# CPU Rasterizer

## Specs
This currently runs only under x86 and windows. SSE2 is required.

## TODO
- Texture mapping.
- Clip with near and far planes.
- Fill rules.
- Vertex normals.
- Some kind of lighting system.
- Actual threading with a proper tile-system (binning).
- Should probably test 128x128 or 64x64 tiles.
- Improve the API in general.
- Add SIMD for triangles (process four triangles once).
- Add SIMD to pixel processing (process 2x2 blocks of pixels).
- Add more SIMD optimizations.
- Add hierarchical rasterization.
- Add hierarchical Z-Buffering.
- Improve the API in general.
- Add guard band clipping.
