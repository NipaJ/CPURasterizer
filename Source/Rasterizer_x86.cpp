#include "General.h"
#include "Rasterizer.h"

#include "PlatformAPI.h"
#include "Vector.h"
#include "Matrix.h"
#include <emmintrin.h>

#include <string.h>

// Disable this warning, since our template trick relies heavily on conditional constant optimizations.
#pragma warning(disable : 4127)

namespace nmj
{
	// Fixed-point configs for the subpixel accuracy.
	enum { PixelFracBits = 4 };
	enum { PixelFracUnit = 1 << PixelFracBits };

	// Buffer settings
	enum { ColorBytes = 4 };
	enum { DepthBytes = 4 };

	// SIMD block settings
	enum { BlockSizeX = 2 };
	enum { BlockSizeY = 2 };
	enum { ColorBlockBytes = BlockSizeX * BlockSizeY * ColorBytes };
	enum { DepthBlockBytes = BlockSizeX * BlockSizeY * DepthBytes };

	// Tile settings
	enum { MaxTrianglesPerTile = 4096 }; // TODO: Make this dynamic.
	enum { TileSizeX = 32 };
	enum { TileSizeY = 32 };
	enum { TileSizeXInBlocks = TileSizeX / BlockSizeX };
	enum { TileSizeYInBlocks = TileSizeY / BlockSizeY };
	enum { ColorTilePitch = TileSizeXInBlocks * ColorBlockBytes };
	enum { DepthTilePitch = TileSizeXInBlocks * DepthBlockBytes };
	enum { ColorTileBytes = TileSizeX * TileSizeY * ColorBytes };
	enum { DepthTileBytes = TileSizeX * TileSizeY * DepthBytes };

#if 0
	// Triangle bin
	struct TriangleBin
	{
		// Fixed-point pixel coordinates for the triangle.
		//
		// Using structure of arrays instead of array of structures, since it's more
		// SIMD and cache friendly.
		U32 triangle_x[3][MaxTrianglesPerTile];
		U32 triangle_y[3][MaxTrianglesPerTile];

		U32 triangle_count;
	};
#endif

	// Function type for the RasterizeTile function.
	typedef void RasterizeTileFunc(
		U32 tile_x, U32 tile_y,
		U32 screen_width, U32 screen_height,
		void *color_buffer, void *depth_buffer,
		const RasterizerInput &input
	);

	// Multiply two SSE epi32 integer vectors.
	NMJ_FORCEINLINE __m128i MulEpi32(__m128i a, __m128i b)
	{
		__m128i lo = _mm_mul_epu32(a, b);
		__m128i hi = _mm_mul_epu32(_mm_shuffle_epi32(a, _MM_SHUFFLE(1, 3, 1, 1)), _mm_shuffle_epi32(b, _MM_SHUFFLE(1, 3, 1, 1)));
		return _mm_unpacklo_epi32(_mm_shuffle_epi32(lo, _MM_SHUFFLE(0, 0, 2, 0)), _mm_shuffle_epi32(hi, _MM_SHUFFLE(0, 0, 2, 0)));
	}

	// Use template to easily generate multiple functions with different rasterizer state.
	template <bool ColorWrite, bool DepthWrite, bool DepthTest, bool DiffuseMap, bool VertexColor>
	static void RasterizeTile(
		U32 tile_x, U32 tile_y,
		U32 screen_width, U32 screen_height,
		void *color_buffer, void *depth_buffer,
		const RasterizerInput &input)
	{
		__m128 transform_matrix[4] =
		{
			_mm_loadu_ps(input.transform[0].v),
			_mm_loadu_ps(input.transform[1].v),
			_mm_loadu_ps(input.transform[2].v),
			_mm_loadu_ps(input.transform[3].v)
		};

		// Transform Y to point down.
		transform_matrix[1] = _mm_xor_ps(transform_matrix[1], _mm_set1_ps(-0.0f));

		// Screen coordinates.
		S32 scx = screen_width / 2;
		S32 scy = screen_height / 2;
		S32 sx = tile_x * TileSizeX;
		S32 sy = tile_y * TileSizeY;

		// Tile rectangle.
		S32 tile_min_x = sx - scx;
		S32 tile_min_y = sy - scy;
		S32 tile_max_x = Min((sx + TileSizeX) - scx, scx - 1);
		S32 tile_max_y = Min((sy + TileSizeY) - scy, scx - 1);

		// TODO: This could probably be combined with the matrix above? Should think about it.
		float xscale = float(scx << PixelFracBits);
		float yscale = float(scy << PixelFracBits);

		const float3 *vertices = input.vertices;
		const float4 *colors = input.colors;
		// const float2 *texcoords = input.texcoords;
		const U16 *indices = input.indices;

		for (U32 count = input.triangle_count; count--; )
		{
			// Fetch triangle vertex information
			__declspec(align(16)) float4 v[3];
			float4 c[3];
			{
				const U16 i0 = indices[0];
				const U16 i1 = indices[1];
				const U16 i2 = indices[2];

				v[0] = float4(vertices[i0], 1.0f);
				v[1] = float4(vertices[i1], 1.0f);
				v[2] = float4(vertices[i2], 1.0f);

				if (VertexColor)
				{
					c[0] = colors[i0];
					c[1] = colors[i1];
					c[2] = colors[i2];
				}

				indices += 3;
			}

			// Transform vertices
			for (unsigned i = 0; i < 3; ++i)
			{
				__m128 result;
				result = _mm_mul_ps(transform_matrix[0], _mm_set1_ps(v[i].x));
				result = _mm_add_ps(result, _mm_mul_ps(transform_matrix[1], _mm_set1_ps(v[i].y)));
				result = _mm_add_ps(result, _mm_mul_ps(transform_matrix[2], _mm_set1_ps(v[i].z)));
				result = _mm_add_ps(result, _mm_mul_ps(transform_matrix[3], _mm_set1_ps(v[i].w)));
				_mm_store_ps(v[i].v, result);
			}

			// Hack rejection for planes, that cross near plane
			if (v[0].z < 0.0f || v[1].z < 0.0f || v[2].z < 0.0f)
				continue;

			// Convert to clip space coordinates to fixed point screen space coordinates.
			S32 coord[3][2];
			coord[0][0] = S32(v[0].x * xscale / v[0].w);
			coord[0][1] = S32(v[0].y * yscale / v[0].w);
			coord[1][0] = S32(v[1].x * xscale / v[1].w);
			coord[1][1] = S32(v[1].y * yscale / v[1].w);
			coord[2][0] = S32(v[2].x * xscale / v[2].w);
			coord[2][1] = S32(v[2].y * yscale / v[2].w);

			// Some common constants for the barycentric calculations.
			const S32 coord21x = coord[2][0] - coord[1][0];
			const S32 coord21y = coord[2][1] - coord[1][1];
			const S32 coord02x = coord[0][0] - coord[2][0];
			const S32 coord02y = coord[0][1] - coord[2][1];

			// Triangle area * 2
			const S32 triarea_x2 = -((coord02x * coord21y) >> PixelFracBits) + ((coord02y * coord21x) >> PixelFracBits);
			if (triarea_x2 < 0)
				continue;

			// Calculate bounds
			S32 bounds[2][2];
			bounds[0][0] = (Min(Min(coord[0][0], coord[1][0]), coord[2][0]) + (PixelFracUnit - 1)) >> PixelFracBits;
			bounds[0][1] = (Min(Min(coord[0][1], coord[1][1]), coord[2][1]) + (PixelFracUnit - 1)) >> PixelFracBits;
			bounds[1][0] = (Max(Max(coord[0][0], coord[1][0]), coord[2][0]) + (PixelFracUnit - 1)) >> PixelFracBits;
			bounds[1][1] = (Max(Max(coord[0][1], coord[1][1]), coord[2][1]) + (PixelFracUnit - 1)) >> PixelFracBits;

			// Clip off-tile triangles.
			// NOTE: If the binning process would be accurate enough, we could just ignore this.
			if (bounds[0][0] > tile_max_x || bounds[0][1] > tile_max_y)
				continue;
			if (bounds[1][0] < tile_min_x || bounds[1][1] < tile_min_y)
				continue;

			// Make sure that the bounds are block aligned
			bounds[0][0] = Max(Min(bounds[0][0], tile_max_x), tile_min_x) & ~(BlockSizeX - 1);
			bounds[0][1] = Max(Min(bounds[0][1], tile_max_y), tile_min_y) & ~(BlockSizeY - 1);
			bounds[1][0] = (Max(Min(bounds[1][0] + 1, tile_max_x), tile_min_x) + (BlockSizeX - 1)) & ~(BlockSizeX - 1);
			bounds[1][1] = (Max(Min(bounds[1][1] + 1, tile_max_y), tile_min_y) + (BlockSizeY - 1)) & ~(BlockSizeY - 1);

			// Calculate variables for stepping
			__m128i bcoord_row[3], bcoord_xstep[3], bcoord_ystep[3];
			__m128 inv_w_row, inv_w_xstep, inv_w_ystep;
			__m128 z_row, z_xstep, z_ystep;
			__m128 pers_color_row[3], pers_color_xstep[3], pers_color_ystep[3];
			{
				// Barycentric integer coordinates
				{
					__m128i offsetx = _mm_add_epi32(_mm_set1_epi32(bounds[0][0]), _mm_set_epi32(1, 0, 1, 0));
					__m128i offsety = _mm_add_epi32(_mm_set1_epi32(bounds[0][1]), _mm_set_epi32(1, 1, 0, 0));

					// 1x1 block steps
					bcoord_xstep[0] = _mm_set1_epi32(-coord21y);
					bcoord_xstep[1] = _mm_set1_epi32(-coord02y);
					bcoord_xstep[2] = _mm_set1_epi32(coord[0][1] - coord[1][1]);
					bcoord_ystep[0] = _mm_set1_epi32(coord21x);
					bcoord_ystep[1] = _mm_set1_epi32(coord02x);
					bcoord_ystep[2] = _mm_set1_epi32(coord[1][0] - coord[0][0]);

					// Triangle start position
					bcoord_row[0] = _mm_set1_epi32(((coord21x * -coord[1][1]) >> PixelFracBits) - ((coord21y * -coord[1][0]) >> PixelFracBits));
					bcoord_row[0] = _mm_add_epi32(bcoord_row[0], MulEpi32(offsetx, bcoord_xstep[0]));
					bcoord_row[0] = _mm_add_epi32(bcoord_row[0], MulEpi32(offsety, bcoord_ystep[0]));
					bcoord_row[0] = _mm_sub_epi32(bcoord_row[0], _mm_srai_epi32(bcoord_xstep[0], 1));
					bcoord_row[0] = _mm_sub_epi32(bcoord_row[0], _mm_srai_epi32(bcoord_ystep[0], 1));
					bcoord_row[1] = _mm_set1_epi32(((coord02x * -coord[2][1]) >> PixelFracBits) - ((coord02y * -coord[2][0]) >> PixelFracBits));
					bcoord_row[1] = _mm_add_epi32(bcoord_row[1], MulEpi32(offsetx, bcoord_xstep[1]));
					bcoord_row[1] = _mm_add_epi32(bcoord_row[1], MulEpi32(offsety, bcoord_ystep[1]));
					bcoord_row[1] = _mm_sub_epi32(bcoord_row[1], _mm_srai_epi32(bcoord_xstep[1], 1));
					bcoord_row[1] = _mm_sub_epi32(bcoord_row[1], _mm_srai_epi32(bcoord_ystep[1], 1));

					// Change stepping to 2x2 blocks
					bcoord_xstep[0] = _mm_slli_epi32(bcoord_xstep[0], 1);
					bcoord_xstep[1] = _mm_slli_epi32(bcoord_xstep[1], 1);
					bcoord_xstep[2] = _mm_slli_epi32(bcoord_xstep[2], 1);
					bcoord_ystep[0] = _mm_slli_epi32(bcoord_ystep[0], 1);
					bcoord_ystep[1] = _mm_slli_epi32(bcoord_ystep[1], 1);
					bcoord_ystep[2] = _mm_slli_epi32(bcoord_ystep[2], 1);

					bcoord_row[2] = _mm_sub_epi32(_mm_sub_epi32(_mm_set1_epi32(triarea_x2), bcoord_row[0]), bcoord_row[1]);
				}

				// Normalized barycentric coordinates as floating point.
				__m128 inv_triarea_x2f = _mm_rcp_ps(_mm_cvtepi32_ps(_mm_set1_epi32(triarea_x2)));
				__m128 bcoordf_row1 = _mm_mul_ps(_mm_cvtepi32_ps(bcoord_row[1]), inv_triarea_x2f);
				__m128 bcoordf_row2 = _mm_mul_ps(_mm_cvtepi32_ps(bcoord_row[2]), inv_triarea_x2f);
				__m128 bcoordf_xstep1 = _mm_mul_ps(_mm_cvtepi32_ps(bcoord_xstep[1]), inv_triarea_x2f);
				__m128 bcoordf_xstep2 = _mm_mul_ps(_mm_cvtepi32_ps(bcoord_xstep[2]), inv_triarea_x2f);
				__m128 bcoordf_ystep1 = _mm_mul_ps(_mm_cvtepi32_ps(bcoord_ystep[1]), inv_triarea_x2f);
				__m128 bcoordf_ystep2 = _mm_mul_ps(_mm_cvtepi32_ps(bcoord_ystep[2]), inv_triarea_x2f);

				// W interpolation
				__m128 inv_w0 = _mm_rcp_ps(_mm_set1_ps(v[0].w));
				__m128 inv_w1 = _mm_rcp_ps(_mm_set1_ps(v[1].w));
				__m128 inv_w2 = _mm_rcp_ps(_mm_set1_ps(v[2].w));
				__m128 inv_w10 = _mm_sub_ps(inv_w1, inv_w0);
				__m128 inv_w20 = _mm_sub_ps(inv_w2, inv_w0);
				inv_w_row = _mm_add_ps(inv_w0, _mm_add_ps(_mm_mul_ps(inv_w10, bcoordf_row1), _mm_mul_ps(inv_w20, bcoordf_row2)));
				inv_w_xstep = _mm_add_ps(_mm_mul_ps(inv_w10, bcoordf_xstep1), _mm_mul_ps(inv_w20, bcoordf_xstep2));
				inv_w_ystep = _mm_add_ps(_mm_mul_ps(inv_w10, bcoordf_ystep1), _mm_mul_ps(inv_w20, bcoordf_ystep2));

				// Z interpolation
				if (DepthWrite || DepthTest)
				{
					__m128 z0 = _mm_mul_ps(_mm_set1_ps(v[0].z), inv_w0);
					__m128 z10 = _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(v[1].z), inv_w1), z0);
					__m128 z20 = _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(v[2].z), inv_w2), z0);
					z_row = _mm_add_ps(z0, _mm_add_ps(_mm_mul_ps(z10, bcoordf_row1), _mm_mul_ps(z20, bcoordf_row2)));
					z_xstep = _mm_add_ps(_mm_mul_ps(z10, bcoordf_xstep1), _mm_mul_ps(z20, bcoordf_xstep2));
					z_ystep = _mm_add_ps(_mm_mul_ps(z10, bcoordf_ystep1), _mm_mul_ps(z20, bcoordf_ystep2));
				}

				// Color interpolation
				if (ColorWrite && VertexColor)
				{
					__m128 pers_color0x = _mm_mul_ps(_mm_set1_ps(c[0].x), inv_w0);
					__m128 pers_color0y = _mm_mul_ps(_mm_set1_ps(c[0].y), inv_w0);
					__m128 pers_color0z = _mm_mul_ps(_mm_set1_ps(c[0].z), inv_w0);
					__m128 pers_color10x = _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(c[1].x), inv_w1), pers_color0x);
					__m128 pers_color10y = _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(c[1].y), inv_w1), pers_color0y);
					__m128 pers_color10z = _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(c[1].z), inv_w1), pers_color0z);
					__m128 pers_color20x = _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(c[2].x), inv_w2), pers_color0x);
					__m128 pers_color20y = _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(c[2].y), inv_w2), pers_color0y);
					__m128 pers_color20z = _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(c[2].z), inv_w2), pers_color0z);
					pers_color_row[0] = _mm_add_ps(pers_color0x, _mm_add_ps(_mm_mul_ps(pers_color10x, bcoordf_row1), _mm_mul_ps(pers_color20x, bcoordf_row2)));
					pers_color_row[1] = _mm_add_ps(pers_color0y, _mm_add_ps(_mm_mul_ps(pers_color10y, bcoordf_row1), _mm_mul_ps(pers_color20y, bcoordf_row2)));
					pers_color_row[2] = _mm_add_ps(pers_color0z, _mm_add_ps(_mm_mul_ps(pers_color10z, bcoordf_row1), _mm_mul_ps(pers_color20z, bcoordf_row2)));
					pers_color_xstep[0] = _mm_add_ps(_mm_mul_ps(pers_color10x, bcoordf_xstep1), _mm_mul_ps(pers_color20x, bcoordf_xstep2));
					pers_color_xstep[1] = _mm_add_ps(_mm_mul_ps(pers_color10y, bcoordf_xstep1), _mm_mul_ps(pers_color20y, bcoordf_xstep2));
					pers_color_xstep[2] = _mm_add_ps(_mm_mul_ps(pers_color10z, bcoordf_xstep1), _mm_mul_ps(pers_color20z, bcoordf_xstep2));
					pers_color_ystep[0] = _mm_add_ps(_mm_mul_ps(pers_color10x, bcoordf_ystep1), _mm_mul_ps(pers_color20x, bcoordf_ystep2));
					pers_color_ystep[1] = _mm_add_ps(_mm_mul_ps(pers_color10y, bcoordf_ystep1), _mm_mul_ps(pers_color20y, bcoordf_ystep2));
					pers_color_ystep[2] = _mm_add_ps(_mm_mul_ps(pers_color10z, bcoordf_ystep1), _mm_mul_ps(pers_color20z, bcoordf_ystep2));
				}
			}

			// Output buffer
			char *out_color_row;
			char *out_depth_row;
			U32 xcount, ycount;
			{
				S32 tile_begin_x = (bounds[0][0] - tile_min_x) / BlockSizeX;
				S32 tile_begin_y = (bounds[0][1] - tile_min_y) / BlockSizeY;
				S32 tile_end_x = (bounds[1][0] - tile_min_x) / BlockSizeX;
				S32 tile_end_y = (bounds[1][1] - tile_min_y) / BlockSizeY;

				xcount = tile_end_x - tile_begin_x;
				ycount = tile_end_y - tile_begin_y;

				if (ColorWrite)
				{
					out_color_row = (char *)color_buffer;
					out_color_row += tile_begin_y * ColorTilePitch + tile_begin_x * ColorBlockBytes;
				}
				if (DepthWrite || DepthTest)
				{
					out_depth_row = (char *)depth_buffer;
					out_depth_row += tile_begin_y * DepthTilePitch + tile_begin_x * DepthBlockBytes;
				}
			}

			// Sample the bounding box of the triangle and output pixels.
			for (S32 y = ycount; y--; )
			{
				// Setup output buffers
				char *out_color;
				char *out_depth;
				{
					if (ColorWrite)
						out_color = out_color_row;
					if (DepthWrite || DepthTest)
						out_depth = out_depth_row;
				}

				// Setup stepped buffers for row operations.
				__m128i bcoord[3];
				__m128 inv_w;
				__m128 z;
				__m128 pers_color[3];
				{
					bcoord[0] = bcoord_row[0];
					bcoord[1] = bcoord_row[1];
					bcoord[2] = bcoord_row[2];

					inv_w = inv_w_row;

					if (DepthWrite || DepthTest)
						z = z_row;

					if (ColorWrite && VertexColor)
					{
						pers_color[0] = pers_color_row[0];
						pers_color[1] = pers_color_row[1];
						pers_color[2] = pers_color_row[2];
					}
				}

				// X loop
				for (S32 x = xcount; x--; )
				{
					// Generate mask for pixels that overlap the triangle.
					__m128i mask = _mm_cmpgt_epi32(_mm_or_si128(_mm_or_si128(bcoord[0], bcoord[1]), bcoord[2]), _mm_setzero_si128());

					// Skip blocks that don't overlap the triangle.
					if (_mm_movemask_epi8(mask) == 0)
						goto skip_block;

					// Depth buffering
					if (DepthTest || DepthWrite)
					{
						__m128i old_z = _mm_load_si128((__m128i *)out_depth);
						__m128i new_z = _mm_cvtps_epi32(_mm_mul_ps(z, _mm_set1_ps(float(0xFFFFFF))));

						// Apply depth testing.
						if (DepthTest)
						{
							mask = _mm_and_si128(mask, _mm_cmpgt_epi32(old_z, new_z));

							// Skip the block, when depth buffer occludes it completely.
							if (_mm_movemask_epi8(mask) == 0)
								goto skip_block;
						}

						// Write depth output
						if (DepthWrite)
						{
							__m128i result = _mm_or_si128(_mm_andnot_si128(mask, old_z), _mm_and_si128(mask, new_z));
							_mm_store_si128((__m128i *)out_depth, result);
						}
					}

					// Write color output
					if (ColorWrite)
					{
						__m128 w = _mm_rcp_ps(inv_w);

						__m128i old_color = _mm_load_si128((__m128i *)out_color);
						__m128i new_color;

						// Output pixel
						if (VertexColor)
						{
							__m128i x = _mm_cvtps_epi32(_mm_mul_ps(_mm_mul_ps(pers_color[0], w), _mm_set1_ps(255.0f)));
							__m128i y = _mm_cvtps_epi32(_mm_mul_ps(_mm_mul_ps(pers_color[1], w), _mm_set1_ps(255.0f)));
							__m128i z = _mm_cvtps_epi32(_mm_mul_ps(_mm_mul_ps(pers_color[2], w), _mm_set1_ps(255.0f)));

							new_color = _mm_or_si128(_mm_or_si128(x, _mm_slli_epi32(y, 8)), _mm_slli_epi32(z, 16));
						}
						else
						{
							new_color = mask;
						}

						__m128i result = _mm_or_si128(_mm_andnot_si128(mask, old_color), _mm_and_si128(mask, new_color));
						_mm_store_si128((__m128i *)out_color, result);
					}

					// I dislike goto, but it wins the over-nested case above without it.
					skip_block:
					{
						if (ColorWrite)
							out_color += ColorBlockBytes;
						if (DepthWrite || DepthTest)
							out_depth += DepthBlockBytes;

						bcoord[0] = _mm_add_epi32(bcoord[0], bcoord_xstep[0]);
						bcoord[1] = _mm_add_epi32(bcoord[1], bcoord_xstep[1]);
						bcoord[2] = _mm_add_epi32(bcoord[2], bcoord_xstep[2]);

						inv_w = _mm_add_ps(inv_w, inv_w_xstep);

						if (DepthWrite || DepthTest)
							z = _mm_add_ps(z, z_xstep);

						if (ColorWrite && VertexColor)
						{
							pers_color[0] = _mm_add_ps(pers_color[0], pers_color_xstep[0]);
							pers_color[1] = _mm_add_ps(pers_color[1], pers_color_xstep[1]);
							pers_color[2] = _mm_add_ps(pers_color[2], pers_color_xstep[2]);
						}
					}
				} // X loop

				if (ColorWrite)
					out_color_row += ColorTilePitch;
				if (DepthWrite || DepthTest)
					out_depth_row += DepthTilePitch;

				bcoord_row[0] = _mm_add_epi32(bcoord_row[0], bcoord_ystep[0]);
				bcoord_row[1] = _mm_add_epi32(bcoord_row[1], bcoord_ystep[1]);
				bcoord_row[2] = _mm_add_epi32(bcoord_row[2], bcoord_ystep[2]);

				inv_w_row = _mm_add_ps(inv_w_row, inv_w_ystep);

				if (DepthWrite || DepthTest)
					z_row = _mm_add_ps(z_row, z_ystep);

				if (ColorWrite && VertexColor)
				{
					pers_color_row[0] = _mm_add_ps(pers_color_row[0], pers_color_ystep[0]);
					pers_color_row[1] = _mm_add_ps(pers_color_row[1], pers_color_ystep[1]);
					pers_color_row[2] = _mm_add_ps(pers_color_row[2], pers_color_ystep[2]);
				}
			} // Y loop
		} // Triangle loop
	}

	U32 GetRequiredMemoryAmount(const RasterizerOutput &self, bool color, bool depth)
	{
		const U32 width = DivWithRoundUp<U32>(self.width, TileSizeX);
		const U32 height = DivWithRoundUp<U32>(self.height, TileSizeY);

		U32 ret = 16;

		if (color)
		{
			ret = GetAligned(ret, 16u);

			U32 pitch = width * ColorTileBytes;
			ret += pitch * height;
		}

		if (depth)
		{
			ret = GetAligned(ret, 16u);

			U32 pitch = width * DepthTileBytes;
			ret += pitch * height;
		}

		// Bins
#if 0
		{
			ret = GetAligned(ret, 16);

			const U32 x_tiles = DivWithRoundUp<U32>(width, TileSizeX);
			const U32 y_tiles = DivWithRoundUp<U32>(height, TileSizeY);

			ret += x_tiles * y_tiles * sizeof(TriangleBin);
		}
#endif

		return ret;
	}

	void Initialize(RasterizerOutput &self, void *memory, bool color, bool depth)
	{
		const U32 width = DivWithRoundUp<U32>(self.width, TileSizeX);
		const U32 height = DivWithRoundUp<U32>(self.height, TileSizeY);

		char *alloc_stack = (char *)memory;

		if (color)
		{
			alloc_stack = GetAligned(alloc_stack, 16u);

			U32 pitch = width * ColorTileBytes;
			self.color_buffer = alloc_stack;
			alloc_stack += pitch * height;
		}

		if (depth)
		{
			alloc_stack = GetAligned(alloc_stack, 16u);

			U32 pitch = width * DepthTileBytes;
			self.depth_buffer = alloc_stack;
			alloc_stack += pitch * height;
		}

		// Bins
#if 0
		{
			alloc_stack = GetAligned(alloc_stack, 16);

			const U32 x_tiles = DivWithRoundUp<U32>(width, TileSizeX);
			const U32 y_tiles = DivWithRoundUp<U32>(height, TileSizeY);

			self.bin = (TriangleBin *)alloc_stack;
			alloc_stack += x_tiles * y_tiles * sizeof(TriangleBin);
		}
#endif
	}

	void Rasterize(RasterizerState &state, const RasterizerInput *input, U32 input_count, U32 split_index, U32 num_splits)
	{
		// [VertexColor << 4 | DiffuseMap << 3 | ColorWrite << 2 | DepthWrite << 1 | DepthTest]
		static RasterizeTileFunc *pipeline[] =
		{
			&RasterizeTile<0, 0, 0, 0, 0>,
			&RasterizeTile<1, 0, 0, 0, 0>,
			&RasterizeTile<0, 1, 0, 0, 0>,
			&RasterizeTile<1, 1, 0, 0, 0>,
			&RasterizeTile<0, 0, 1, 0, 0>,
			&RasterizeTile<1, 0, 1, 0, 0>,
			&RasterizeTile<0, 1, 1, 0, 0>,
			&RasterizeTile<1, 1, 1, 0, 0>,
			&RasterizeTile<0, 0, 0, 1, 0>,
			&RasterizeTile<1, 0, 0, 1, 0>,
			&RasterizeTile<0, 1, 0, 1, 0>,
			&RasterizeTile<1, 1, 0, 1, 0>,
			&RasterizeTile<0, 0, 1, 1, 0>,
			&RasterizeTile<1, 0, 1, 1, 0>,
			&RasterizeTile<0, 1, 1, 1, 0>,
			&RasterizeTile<1, 1, 1, 1, 0>,
			&RasterizeTile<0, 0, 0, 0, 1>,
			&RasterizeTile<1, 0, 0, 0, 1>,
			&RasterizeTile<0, 1, 0, 0, 1>,
			&RasterizeTile<1, 1, 0, 0, 1>,
			&RasterizeTile<0, 0, 1, 0, 1>,
			&RasterizeTile<1, 0, 1, 0, 1>,
			&RasterizeTile<0, 1, 1, 0, 1>,
			&RasterizeTile<1, 1, 1, 0, 1>,
			&RasterizeTile<0, 0, 0, 1, 1>,
			&RasterizeTile<1, 0, 0, 1, 1>,
			&RasterizeTile<0, 1, 0, 1, 1>,
			&RasterizeTile<1, 1, 0, 1, 1>,
			&RasterizeTile<0, 0, 1, 1, 1>,
			&RasterizeTile<1, 0, 1, 1, 1>,
			&RasterizeTile<0, 1, 1, 1, 1>,
			&RasterizeTile<1, 1, 1, 1, 1>
		};

		// General settings
		const U32 screen_width = state.output->width;
		const U32 screen_height = state.output->height;
		U32 flags = state.flags & 7;

		// Validate buffers
		char *color_buffer = (char *)state.output->color_buffer;
		char *depth_buffer = (char *)state.output->depth_buffer;
		if (color_buffer == NULL)
			flags &= ~RasterizerFlagColorWrite;
		if (depth_buffer == NULL)
			flags &= ~RasterizerFlagDepthWrite;

		// Tile information
		const U32 x_tile_count = DivWithRoundUp<U32>(screen_width, TileSizeX);
		const U32 y_tile_count = DivWithRoundUp<U32>(screen_height, TileSizeY);
		const U32 tile_count = x_tile_count * y_tile_count;

		while (input_count--)
		{
			const RasterizerInput &ri = *input++;

			// Get rasterizer function.
			RasterizeTileFunc *RasterizeTile;
			{
				U32 lookup_index = flags;
				if (ri.colors)
					lookup_index |= 1 << 4;
				if (ri.texcoords)
					lookup_index |= 1 << 3;

				RasterizeTile = pipeline[lookup_index];
			}

			char *out_color = color_buffer + split_index * ColorTileBytes;
			char *out_depth = depth_buffer + split_index * DepthTileBytes;
			for (U32 index = split_index; index < tile_count; index += num_splits)
			{
				RasterizeTile(index % x_tile_count, index / x_tile_count, screen_width, screen_height, out_color, out_depth, ri);

				out_color += ColorTileBytes * num_splits;
				out_depth += DepthTileBytes * num_splits;
			}
		}
	}

	void ClearColor(RasterizerOutput &output, float4 value, U32 split_index, U32 num_splits)
	{
		const U32 x_tile_count = DivWithRoundUp<U32>(output.width, TileSizeX);
		const U32 y_tile_count = DivWithRoundUp<U32>(output.height, TileSizeY);
		const U32 tile_count = x_tile_count * y_tile_count;

		__m128i cv = _mm_set1_epi32(U8(value.x * 255.0f) | U8(value.y * 255.0f) << 8 | U8(value.z * 255.0f) << 16 | U8(value.w * 255.0f) << 24);

		char *out = ((char *)output.color_buffer) + split_index * ColorTileBytes;
		for (U32 index = split_index; index < tile_count; index += num_splits)
		{
			for (U32 count = TileSizeXInBlocks * TileSizeYInBlocks; count--;)
			{
				_mm_store_si128((__m128i *)out, cv);
				out += ColorBlockBytes;
			}

			out += (num_splits - 1) * ColorTileBytes;
		}
	}

	void ClearDepth(RasterizerOutput &output, float value, U32 split_index, U32 num_splits)
	{
		const U32 x_tile_count = DivWithRoundUp<U32>(output.width, TileSizeX);
		const U32 y_tile_count = DivWithRoundUp<U32>(output.height, TileSizeY);
		const U32 tile_count = x_tile_count * y_tile_count;

		__m128i cv = _mm_set1_epi32(U32(value * float(0xFFFFFF)));

		char *out = ((char *)output.depth_buffer) + split_index * DepthTileBytes;
		for (U32 index = split_index; index < tile_count; index += num_splits)
		{
			for (U32 count = TileSizeXInBlocks * TileSizeYInBlocks; count--; )
			{
				_mm_store_si128((__m128i *)out, cv);
				out += DepthBlockBytes;
			}

			out += (num_splits - 1) * DepthTileBytes;
		}
	}

	void Blit(LockBufferInfo &output, RasterizerOutput &input, U32 split_index, U32 num_splits)
	{
		NMJ_STATIC_ASSERT(BlockSizeX == 2 && BlockSizeY == 2, "Update this function.");
		NMJ_ASSERT(output.width % (BlockSizeX * 2) == 0);
		NMJ_ASSERT(output.height % BlockSizeY == 0);
		NMJ_ASSERT(output.width == input.width);
		NMJ_ASSERT(output.height == input.height);

		const U32 width = input.width;
		const U32 height = input.height;

		const U32 x_tile_count = DivWithRoundUp<U32>(width, TileSizeX);
		const U32 y_tile_count = DivWithRoundUp<U32>(height, TileSizeY);
		const U32 tile_count = x_tile_count * y_tile_count;

		__m128i x_mask = _mm_set1_epi32(0x00FF0000);
		__m128i y_mask = _mm_set1_epi32(0x000000FF);
		__m128i zw_mask = _mm_set1_epi32(0xFF00FF00);

		char *in_tile = ((char *)input.color_buffer) + split_index * ColorTileBytes;
		for (U32 index = split_index; index < tile_count; index += num_splits)
		{
			const U32 sx = (index % x_tile_count) * TileSizeX;
			const U32 sy = (index / x_tile_count) * TileSizeY;
			const U32 xcount = Min(width - sx, TileSizeX) / (BlockSizeX * 2);
			const U32 ycount = Min(height - sy, TileSizeY) / BlockSizeY;

			U32 out_pitch = output.pitch * 2;
			char *out_row0 = ((char *)output.data) + sy * output.pitch + sx * 4;
			char *out_row1 = ((char *)output.data) + (sy + 1) * output.pitch + sx * 4;
			char *in_row = in_tile;

			for (U32 y = ycount; y--; )
			{
				char *out0 = out_row0;
				char *out1 = out_row1;
				char *in = in_row;

				for (U32 x = xcount; x--; )
				{
					__m128i simd_x0 = _mm_load_si128((__m128i *)in);
					__m128i simd_x1 = _mm_load_si128((__m128i *)(in + 16));
					__m128i simd_z0 = _mm_and_si128(_mm_srli_epi32(simd_x0, 16), y_mask);
					__m128i simd_z1 = _mm_and_si128(_mm_srli_epi32(simd_x1, 16), y_mask);
					__m128i simd_yw0 = _mm_and_si128(simd_x0, zw_mask);
					__m128i simd_yw1 = _mm_and_si128(simd_x1, zw_mask);
					simd_x0 = _mm_and_si128(_mm_slli_epi32(simd_x0, 16), x_mask);
					simd_x1 = _mm_and_si128(_mm_slli_epi32(simd_x1, 16), x_mask);

					__m128i xyz1 = _mm_or_si128(_mm_or_si128(simd_x0, simd_z0), simd_yw0);
					__m128i xyz2 = _mm_or_si128(_mm_or_si128(simd_x1, simd_z1), simd_yw1);

					_mm_stream_si128((__m128i *)out0, _mm_unpacklo_epi64(xyz1, xyz2));
					_mm_stream_si128((__m128i *)out1, _mm_unpackhi_epi64(xyz1, xyz2));

					out0 += 16;
					out1 += 16;
					in += ColorBlockBytes * 2;
				}

				in_row += ColorTilePitch;
				out_row0 += out_pitch;
				out_row1 += out_pitch;
			}

			in_tile += num_splits * ColorTileBytes;
		}
	}
}

