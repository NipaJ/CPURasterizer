#include "General.h"
#include "Rasterizer.h"

#include "PlatformAPI.h"
#include "Vector.h"
#include "Matrix.h"
#include <emmintrin.h>

// Disable this warning, since our template trick relies heavily on conditional constant optimizations.
#pragma warning(disable : 4127)

namespace nmj
{
	// Fixed-point configs for the subpixel accuracy.
	enum { PixelFracBits = 4 };
	enum { PixelFracUnit = 1 << PixelFracBits };

	// SIMD block settings
	enum { BlockSizeX = 2 };
	enum { BlockSizeY = 2 };
	enum { ColorBlockBytes = 16 };
	enum { DepthBlockBytes = 16 };

	// Tile settings
	enum { MaxTrianglesPerTile = 4096 }; // TODO: Make this dynamic.
	enum { TileSizeX = 32 };
	enum { TileSizeY = 32 };

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

	// Function type for the RasterizeTile function.
	typedef void RasterizeTileFunc(
		U32 tile_x, U32 tile_y,
		U32 screen_width, U32 screen_height,
		void *color_buffer, void *depth_buffer,
		const RasterizerInput &input
	);

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

		// Screen coordinates for center, block position and size.
		S32 scx = screen_width / 2;
		S32 scy = screen_height / 2;
		S32 sx = tile_x * TileSizeX;
		S32 sy = tile_y * TileSizeY;

		// Scissor rect
		S32 scissor_min_x = sx - scx;
		S32 scissor_min_y = sy - scy;
		S32 scissor_max_x = Min((sx + TileSizeX - 1) - scx, scx - 1);
		S32 scissor_max_y = Min((sy + TileSizeY - 1) - scy, scx - 1);

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
			_declspec(align(16)) float4 v[3];
			_declspec(align(16)) float4 c[3];
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

			// Hack rejection for planes, that cross near plane or far-plane
			if (v[0].z > v[0].w || v[1].z > v[1].w || v[2].z > v[2].w)
				continue;
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

			// Precalculate barycentric conversion constants.
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
			if (bounds[0][0] > scissor_max_x || bounds[0][1] > scissor_max_y)
				continue;
			if (bounds[1][0] < scissor_min_x || bounds[1][1] < scissor_min_y)
				continue;

			bounds[0][0] = Max(Min(bounds[0][0], scissor_max_x), scissor_min_x);
			bounds[0][1] = Max(Min(bounds[0][1], scissor_max_y), scissor_min_y);
			bounds[1][0] = Max(Min(bounds[1][0], scissor_max_x), scissor_min_x);
			bounds[1][1] = Max(Min(bounds[1][1], scissor_max_y), scissor_min_y);

			// Calculate variables for stepping
			S32 bcoord_row[3], bcoord_xstep[3], bcoord_ystep[3];
			float	inv_w_row, inv_w_xstep, inv_w_ystep;
			float	z_row, z_xstep, z_ystep;
			float4 pers_color_row, pers_color_xstep, pers_color_ystep;
			{
				// Fixed-point min bounds with 0.5 subtracted from it (we want to sample from the middle of pixel).
				S32 fixed_bounds[2];
				fixed_bounds[0] = (bounds[0][0] << PixelFracBits) - PixelFracUnit / 2;
				fixed_bounds[1] = (bounds[0][1] << PixelFracBits) - PixelFracUnit / 2;

				// Barycentric integer coordinates
				bcoord_row[0] = ((coord21x * (fixed_bounds[1] - coord[1][1])) >> PixelFracBits) - ((coord21y * (fixed_bounds[0] - coord[1][0])) >> PixelFracBits);
				bcoord_row[1] = ((coord02x * (fixed_bounds[1] - coord[2][1])) >> PixelFracBits) - ((coord02y * (fixed_bounds[0] - coord[2][0])) >> PixelFracBits);
				bcoord_row[2] = triarea_x2 - bcoord_row[0] - bcoord_row[1];
				bcoord_xstep[0] = -coord21y;
				bcoord_xstep[1] = -coord02y;
				bcoord_xstep[2] = coord[0][1] - coord[1][1];
				bcoord_ystep[0] = coord21x;
				bcoord_ystep[1] = coord02x;
				bcoord_ystep[2] = coord[1][0] - coord[0][0];

				// Normalized barycentric coordinates as floating point.
				float inv_triarea_x2f = 1.0f / float(triarea_x2);
				float bcoordf_row1 = float(bcoord_row[1]) * inv_triarea_x2f;
				float bcoordf_row2 = float(bcoord_row[2]) * inv_triarea_x2f;
				float bcoordf_xstep1 = float(bcoord_xstep[1]) * inv_triarea_x2f;
				float bcoordf_xstep2 = float(bcoord_xstep[2]) * inv_triarea_x2f;
				float bcoordf_ystep1 = float(bcoord_ystep[1]) * inv_triarea_x2f;
				float bcoordf_ystep2 = float(bcoord_ystep[2]) * inv_triarea_x2f;

				// W interpolation
				float inv_w0 = 1.0f / v[0].w;
				float inv_w1 = 1.0f / v[1].w;
				float inv_w2 = 1.0f / v[2].w;
				float inv_w10 = inv_w1 - inv_w0;
				float inv_w20 = inv_w2 - inv_w0;
				inv_w_row = inv_w0 + inv_w10 * bcoordf_row1 + inv_w20 * bcoordf_row2;
				inv_w_xstep = inv_w10 * bcoordf_xstep1 + inv_w20 * bcoordf_xstep2;
				inv_w_ystep = inv_w10 * bcoordf_ystep1 + inv_w20 * bcoordf_ystep2;

				// Z interpolation
				if (DepthWrite || DepthTest)
				{
					float z0 = v[0].z * inv_w0;
					float z10 = (v[1].z * inv_w1) - z0;
					float z20 = (v[2].z * inv_w2) - z0;
					z_row = z0 + z10 * bcoordf_row1 + z20 * bcoordf_row2;
					z_xstep = z10 * bcoordf_xstep1 + z20 * bcoordf_xstep2;
					z_ystep = z10 * bcoordf_ystep1 + z20 * bcoordf_ystep2;
				}

				// Color interpolation
				if (ColorWrite && VertexColor)
				{
					float4 pers_color0 = c[0] * inv_w0;
					float4 pers_color10 = (c[1] * inv_w1) - pers_color0;
					float4 pers_color20 = (c[2] * inv_w2) - pers_color0;
					pers_color_row = pers_color0 + pers_color10 * bcoordf_row1 + pers_color20 * bcoordf_row2;
					pers_color_xstep = pers_color10 * bcoordf_xstep1 + pers_color20 * bcoordf_xstep2;
					pers_color_ystep = pers_color10 * bcoordf_ystep1 + pers_color20 * bcoordf_ystep2;
				}
			}

			// Output buffer
			char *out_color_row;
			char *out_depth_row;
			{
				if (ColorWrite)
				{
					out_color_row = (char *)color_buffer;
					out_color_row += (bounds[0][1] * S32(screen_width) + bounds[0][0]) * 4;
				}
				if (DepthWrite || DepthTest)
				{
					out_depth_row = (char *)depth_buffer;
					out_depth_row += (bounds[0][1] * S32(screen_width) + bounds[0][0]) * 4;
				}
			}

			// Sample the bounding box of the triangle and output pixels.
			for (S32 y = bounds[0][1]; y <= bounds[1][1]; ++y)
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
				S32 bcoord[3];
				float inv_w;
				float z;
				float4 pers_color;
				{
					bcoord[0] = bcoord_row[0];
					bcoord[1] = bcoord_row[1];
					bcoord[2] = bcoord_row[2];

					inv_w = inv_w_row;

					if (DepthWrite || DepthTest)
						z = z_row;

					if (ColorWrite && VertexColor)
						pers_color = pers_color_row;
				}

				// X loop
				for (S32 x = bounds[0][0]; x <= bounds[1][0]; ++x)
				{
					// When inside triangle, output pixel.
					if ((bcoord[0] | bcoord[1] | bcoord[2]) < 0)
						goto skip_pixel;

					// Interpolated Z
					if (DepthTest || DepthWrite)
					{
						U32 z_unorm = U32(z * float(0xFFFFFF));

						// Apply depth testing.
						if (DepthTest)
						{
							if (*(U32 *)out_depth < z_unorm)
								goto skip_pixel;
						}

						// Write depth output
						if (DepthWrite)
							*(U32 *)out_depth = z_unorm;
					}

					// Interpolated W
					float w = 1.0f / inv_w;

					// Write color output
					if (ColorWrite)
					{
						// Output pixel
						if (VertexColor)
						{
							((U8 *)out_color)[0] = U8(w * pers_color.x * 255.0f);
							((U8 *)out_color)[1] = U8(w * pers_color.y * 255.0f);
							((U8 *)out_color)[2] = U8(w * pers_color.z * 255.0f);
						}
						else
						{
							((U8 *)out_color)[0] = U8(255);
							((U8 *)out_color)[1] = U8(255);
							((U8 *)out_color)[2] = U8(255);
						}
					}

					// I dislike goto, but it wins the over-nested case above without it.
					skip_pixel:
					{
						if (ColorWrite)
							out_color += 4;
						if (DepthWrite || DepthTest)
							out_depth += 4;

						bcoord[0] += bcoord_xstep[0];
						bcoord[1] += bcoord_xstep[1];
						bcoord[2] += bcoord_xstep[2];

						inv_w += inv_w_xstep;

						if (DepthWrite || DepthTest)
							z += z_xstep;

						if (ColorWrite && VertexColor)
							pers_color += pers_color_xstep;
					}
				} // X loop

				if (ColorWrite)
					out_color_row += screen_width * 4;
				if (DepthWrite || DepthTest)
					out_depth_row += screen_width * 4;

				bcoord_row[0] += bcoord_ystep[0];
				bcoord_row[1] += bcoord_ystep[1];
				bcoord_row[2] += bcoord_ystep[2];

				inv_w_row += inv_w_ystep;

				if (DepthWrite || DepthTest)
					z_row += z_ystep;

				if (ColorWrite && VertexColor)
					pers_color_row += pers_color_ystep;

			} // Y loop
		} // Triangle loop
	}

	U32 GetRequiredMemoryAmount(const RasterizerOutput &self, bool color, bool depth)
	{
		const U32 width = (self.width + 1) / 2;
		const U32 height = (self.height + 1) / 2;

		U32 ret = 16;

		if (color)
		{
			ret = GetAligned(ret, 16);

			U32 pitch = width * ColorBlockBytes;
			ret += pitch * height;
		}

		if (depth)
		{
			ret = GetAligned(ret, 16);

			U32 pitch = width * DepthBlockBytes;
			ret += pitch * height;
		}

		// Bins
		{
			ret = GetAligned(ret, 16);

			const U32 x_tiles = RoundUpToUnit<U32>(width, TileSizeX);
			const U32 y_tiles = RoundUpToUnit<U32>(height, TileSizeY);

			ret += x_tiles * y_tiles * sizeof(TriangleBin);
		}

		return ret;
	}

	void Initialize(RasterizerOutput &self, void *memory, bool color, bool depth)
	{
		const U32 width = (self.width + 1) / 2;
		const U32 height = (self.height + 1) / 2;

		char *alloc_stack = (char *)memory;

		if (color)
		{
			alloc_stack = GetAligned(alloc_stack, 16);

			U32 pitch = width * ColorBlockBytes;
			self.color_buffer = alloc_stack;
			alloc_stack += pitch * height;
		}

		if (depth)
		{
			alloc_stack = GetAligned(alloc_stack, 16);

			U32 pitch = width * DepthBlockBytes;
			self.depth_buffer = alloc_stack;
			alloc_stack += pitch * height;
		}

		// Bins
		{
			alloc_stack = GetAligned(alloc_stack, 16);

			const U32 x_tiles = RoundUpToUnit<U32>(width, TileSizeX);
			const U32 y_tiles = RoundUpToUnit<U32>(height, TileSizeY);

			self.bin = (TriangleBin *)alloc_stack;
			alloc_stack += x_tiles * y_tiles * sizeof(TriangleBin);
		}
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

		// Offset buffers to center.
		// TODO: Get rid of this.
		const U32 center_offset = ((screen_height / 2) * screen_width + screen_width / 2);
		color_buffer += center_offset * 4;
		depth_buffer += center_offset * 4;

		// Tile information
		const U32 x_tile_count = RoundUpToUnit<U32>(screen_width, TileSizeX);
		const U32 y_tile_count = RoundUpToUnit<U32>(screen_height, TileSizeY);
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

			for (U32 index = split_index; index < tile_count; index += num_splits)
			{
				RasterizeTile(index % x_tile_count, index / x_tile_count, screen_width, screen_height, color_buffer, depth_buffer, ri);
			}
		}
	}

	void ClearColor(RasterizerOutput &output, float4 value, U32 split_index, U32 num_splits)
	{
		// TODO: Handle splits.
		U32 *out = (U32 *)output.color_buffer;

		U32 cv = U8(value.x * 255.0f) | U8(value.y * 255.0f) << 8 | U8(value.z * 255.0f) << 16 | U8(value.w * 255.0f) << 24;
		U32 size = output.width * output.height;
		while (size--)
			*out++ = cv;
	}

	void ClearDepth(RasterizerOutput &output, float value, U32 split_index, U32 num_splits)
	{
		// TODO: Handle splits.
		U32 *out = (U32 *)output.depth_buffer;

		U32 cv = U32(value * float(0xFFFFFF));
		U32 size = output.width * output.height;
		while (size--)
			*out++ = cv;
	}

	void Blit(LockBufferInfo &output, RasterizerOutput &input, U32 split_index, U32 num_splits)
	{
		U8 *in = (U8 *)input.color_buffer;
		char *out = (char *)output.data;

		__m128i x_mask = _mm_set1_epi32(0x00FF0000);
		__m128i y_mask = _mm_set1_epi32(0x000000FF);
		__m128i zw_mask = _mm_set1_epi32(0xFF00FF00);

		for (U32 y = input.height; y--; )
		{
			U8 *p = (U8 *)out;
			for (U32 x = input.width; x != 0; x -= 4)
			{
				__m128i simd_x = _mm_load_si128((__m128i *)in);
				__m128i simd_z = simd_x;
				__m128i simd_yw = simd_x;

				simd_x = _mm_and_si128(_mm_slli_epi32(simd_x, 16), x_mask);
				simd_z = _mm_and_si128(_mm_srli_epi32(simd_z, 16), y_mask);
				simd_yw = _mm_and_si128(simd_yw, zw_mask);

				_mm_storeu_si128((__m128i *)p, _mm_or_si128(_mm_or_si128(simd_x, simd_z), simd_yw));
				p += 16;
				in += 16;
			}

			out += output.pitch;
		}
	}
}

