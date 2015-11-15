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

	enum { ColorBlockBytes = 16 };
	enum { DepthBlockBytes = 16 };

	// Function type for the RasterizeTile function.
	typedef void RasterizeTileFunc(
		U32 block_x, U32 block_y, U32 block_width, U32 block_height,
		U32 screen_width, U32 screen_height,
		void *color_buffer, void *depth_buffer,
		const RasterizerInput &input
	);

	// Use template to easily generate multiple functions with different rasterizer state.
	template <bool ColorWrite, bool DepthWrite, bool DepthTest, bool DiffuseMap, bool VertexColor>
	static void RasterizeTile(
		U32 block_x, U32 block_y, U32 block_width, U32 block_height,
		U32 screen_width, U32 screen_height,
		void *color_buffer, void *depth_buffer,
		const RasterizerInput &input)
	{
		__m128 transform_matrix[4] =
		{
			_mm_load_ps(input.transform[0].v),
			_mm_xor_ps(_mm_load_ps(input.transform[1].v), _mm_castsi128_ps(_mm_set1_epi32(0x80000000))),
			_mm_load_ps(input.transform[2].v),
			_mm_load_ps(input.transform[3].v)
		};

		const float3 *vertices = input.vertices;
		const float4 *colors = input.colors;
		// const float2 *texcoords = input.texcoords;
		const U16 *indices = input.indices;

		S32 half_width = screen_width / 2;
		S32 half_height = screen_height / 2;

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

			// Reject off-screen triangles.
			if (v[0].x > v[0].w && v[0].y > v[0].w && v[0].z > v[0].w)
				continue;
			if (v[0].x < -v[0].w && v[0].y < -v[0].w && v[0].z < 0.0f)
				continue;
			if (v[1].x > v[1].w && v[1].y > v[1].w && v[1].z > v[1].w)
				continue;
			if (v[1].x < -v[1].w && v[1].y < -v[1].w && v[1].z < 0.0f)
				continue;
			if (v[2].x > v[2].w && v[2].y > v[2].w && v[2].z > v[2].w)
				continue;
			if (v[2].x < -v[2].w && v[2].y < -v[2].w && v[2].z < 0.0f)
				continue;

			// Hack rejection for planes, that cross near plane
			if (v[0].z < 0.0f || v[1].z < 0.0f || v[2].z < 0.0f)
				continue;

			// Convert to pixel coordinates as fixed point.
			// Also figure out pixel bounds as integers.
			_declspec(align(16)) S32 coord[4][2];
			{
				__m128 v0 = _mm_load_ps(v[0].v);
				__m128 v1 = _mm_load_ps(v[1].v);
				__m128 v2 = _mm_load_ps(v[2].v);

				__m128 v01xy = _mm_shuffle_ps(v0, v1, _MM_SHUFFLE(1, 0, 1, 0));
				__m128 v22xy = _mm_shuffle_ps(v2, v2, _MM_SHUFFLE(1, 0, 1, 0));
				__m128 v01ww = _mm_shuffle_ps(v0, v1, _MM_SHUFFLE(3, 3, 3, 3));
				__m128 v22ww = _mm_shuffle_ps(v2, v2, _MM_SHUFFLE(3, 3, 3, 3));
				v01xy = _mm_div_ps(v01xy, v01ww);
				v22xy = _mm_div_ps(v22xy, v22ww);

				__m128 res = _mm_cvtepi32_ps(_mm_set_epi32(half_height, half_width, half_height, half_width));
				__m128 unit_scale = _mm_mul_ps(res, _mm_set1_ps(float(PixelFracUnit)));
				v01xy = _mm_mul_ps(v01xy, unit_scale);
				v22xy = _mm_mul_ps(v22xy, unit_scale);
				_mm_store_si128((__m128i *)coord[0], _mm_cvttps_epi32(v01xy));
				_mm_store_si128((__m128i *)coord[2], _mm_cvttps_epi32(v22xy));
			}

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
			bounds[0][0] = Max(Min(bounds[0][0], half_width - 1), -half_width);
			bounds[0][1] = Max(Min(bounds[0][1], half_height - 1), -half_height);
			bounds[1][0] = Max(Min(bounds[1][0], half_width - 1), -half_width);
			bounds[1][1] = Max(Min(bounds[1][1], half_height - 1), -half_height);

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
			U32 *out_color_row;
			U16 *out_depth_row;
			{
				if (ColorWrite)
				{
					out_color_row = (U32 *)color_buffer;
					out_color_row += (bounds[0][1] * S32(screen_width) + bounds[0][0]);
				}
				if (DepthWrite || DepthTest)
				{
					out_depth_row = (U16 *)depth_buffer;
					out_depth_row += (bounds[0][1] * S32(screen_width) + bounds[0][0]);
				}
			}

			// Sample the bounding box of the triangle and output pixels.
			for (S32 y = bounds[0][1]; y <= bounds[1][1]; ++y)
			{
				// Setup output buffers
				U8 *out_color;
				U16 *out_depth;
				{
					if (ColorWrite)
						out_color = (U8 *)out_color_row;
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

					// Interpolated W
					float w = 1.0f / inv_w;

					// Interpolated Z
					U16 z_unorm;
					if (DepthWrite || DepthTest)
						z_unorm = U16(z * float(0xFFFF));

					// Apply depth testing.
					if (DepthTest)
					{
						if (*out_depth < z_unorm)
							goto skip_pixel;
					}

					// Write color output
					if (ColorWrite)
					{
						// Output pixel
						if (VertexColor)
						{
							out_color[0] = U8(w * pers_color.x * 255.0f);
							out_color[1] = U8(w * pers_color.y * 255.0f);
							out_color[2] = U8(w * pers_color.z * 255.0f);
							// out_color[3] = U8(w * pers_color.w * 255.0f);
						}
						else
						{
							out_color[0] = U8(255);
							out_color[1] = U8(255);
							out_color[2] = U8(255);
							// out_color[3] = U8(255);
						}
					}

					// Write depth output
					if (DepthWrite)
						*out_depth = z_unorm;

					// I dislike goto, but it wins the over-nested case above without it.
					skip_pixel:
					{
						if (ColorWrite)
							out_color += 4;
						if (DepthWrite || DepthTest)
							out_depth += 1;

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
					out_color_row += screen_width;
				if (DepthWrite || DepthTest)
					out_depth_row += screen_width;

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
			self.color_pitch = pitch;
			alloc_stack += pitch * height;
		}

		if (depth)
		{
			alloc_stack = GetAligned(alloc_stack, 16);

			U32 pitch = width * DepthBlockBytes;
			self.depth_buffer = alloc_stack;
			self.depth_pitch = pitch;
			alloc_stack += pitch * height;
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

		U32 width = state.output->width;
		U32 height = state.output->height;
		U32 flags = state.flags & 7;

		char *color_buffer = (char *)state.output->color_buffer;
		if (color_buffer)
			color_buffer += ((height / 2) * width + width / 2) * 4;
		else
			flags &= ~RasterizerFlagColorWrite;

		char *depth_buffer = (char *)state.output->depth_buffer;
		if (depth_buffer)
			depth_buffer += ((height / 2) * width + width / 2) * 2;
		else
			flags &= ~RasterizerFlagDepthWrite;

		while (input_count--)
		{
			const RasterizerInput &ri = *input++;

			U32 lookup_index = flags;

			if (ri.colors)
				lookup_index |= 1 << 4;

			if (ri.texcoords)
				lookup_index |= 1 << 3;

			(*pipeline[lookup_index])(
				0, 0, (width + 1) / 2, (height + 1) / 2,
				width, height,
				color_buffer, depth_buffer,
				ri
			);
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
		U16 *out = (U16 *)output.depth_buffer;

		U16 cv = U16(value * float(0xFFFF));
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

