/**
 * Software rasterization API.
 */
#pragma once
#include "Vector.h"

namespace nmj
{
	struct LockBufferInfo;

	enum
	{
		/* Enable color writes. */
		RasterizerFlagColorWrite = 0x00000001,

		/* Enable depth writes. */
		RasterizerFlagDepthWrite = 0x00000002,

		/* Enable depth testing. */
		RasterizerFlagDepthTest = 0x00000004,
	};

	/**
	 * Rasterizer state.
	 */
	struct RasterizerState
	{
		_declspec(align(16)) float4 transform[4];
		U32 flags;
	};

	/**
	 * Rasterizer output data.
	 *
	 * Buffers should point to an array with size of (width * height).
	 */
	struct RasterizerOutput
	{
		/**
		 * 32bit RGBA color buffer.
		 * 16 byte alignment required.
		 */
		U32 *color_buffer;

		/**
		 * 16bit unsigned normalized depth buffer.
		 * 16 byte alignment required.
		 */
		U16 *depth_buffer;

		/* Output resolution. */
		U16 width, height;
	};

	/**
	 * Rasterizer input data.
	 */
	struct RasterizerInput
	{
		/* Rasterization state. */
		const RasterizerState *state;

		/**
		 * Per vertex information.
		 * All but vertices are optional and can be NULL.
		 */
		const float3 *vertices;
		const float4 *colors;
		const float2 *texcoords;

		/* Vertex indices for the triangles. */
		const U16 *indices;

		/* Number of triangles. */
		U32 triangle_count;
	};

	/**
	 * Get required memory amount for the rasterization output.
	 * Width and height must be specified before calling this.
	 *
	 * This is helper util and completely optional.
	 */
	U32 GetRequiredMemoryAmount(const RasterizerOutput &self, bool color, bool depth);

	/**
	 * Initialize rasterizer output.
	 * Width and height must be specified before calling this.
	 *
	 * This is helper util and completely optional.
	 */
	void Initialize(RasterizerOutput &self, void *memory, bool color, bool depth);

	/**
	 * Transform number of triangles to rasterized buffers.
	 *
	 * You can split the process into multiple tiles. Each tile is completely
	 * independent from each other and can be processed in parallel.
	 */
	void Rasterize(RasterizerOutput &output, const RasterizerInput *input, U32 input_count, U32 tile_index, U32 tile_count);

	/**
	 * Clear color buffer
	 */
	void ClearColor(RasterizerOutput &output, float4 value, U32 tile_index, U32 tile_count);

	/**
	 * Clear depth buffer
	 */
	void ClearDepth(RasterizerOutput &output, float value, U32 tile_index, U32 tile_count);

	/**
	 * Blit output buffer to screen.
	 */
	void Blit(LockBufferInfo &output, RasterizerOutput &input, U32 tile_index, U32 tile_count);
}
