/**
 * Software rasterization API.
 */
#pragma once
#include "Vector.h"

namespace nmj
{
	struct LockBufferInfo;
	struct TriangleBin;

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
	 * Rasterizer output data.
	 */
	struct RasterizerOutput
	{
		/**
		 * 32bit RGB color buffer with 8bit channels.
		 *
		 * Buffer should be rounded up to size that matches the 2x2 block
		 * layout and 16 byte alignment is required.
		 *
		 * The memory is packed as 2x2 pixels block in following layout:
		 * 00: R1 G1 B1 X  R2 G2 B2 X
		 * 08: R3 G3 B3 X  R4 G4 B4 X
		 *
		 * From screen-space layout of:
		 *  ---------- X
		 * | RGB1 RGB2
		 * | RGB3 RGB4
		 * Y
		 */
		void *color_buffer;

		/**
		 * 24bit unsigned normalized depth buffer with 8bit stencil buffer.
		 *
		 * Buffer should be rounded up to size that matches the 2x2 block
		 * layout and 16 byte alignment is required.
		 *
		 * The memory is packed as 2x2 pixels block in following layout:
		 * 00: D1 S1 D2 S2
		 * 08: D3 S3 D4 S4
		 *
		 * From screen-space layout of:
		 *  -------- X
		 * | DS1 DS2 
		 * | DS3 DS4
		 * Y
		 */
		void *depth_buffer;

		/**
		 * Triangle bin
		 *
		 * Reserved for internal usage.
		 */
		TriangleBin *bin;

		/**
		 * Output resolution.
		 */
		U16 width, height;
	};

	/**
	 * Rasterizer input data.
	 */
	struct RasterizerInput
	{
		/* Vertex transform matrix, using row-vectors. */
		float4 transform[4];

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
	 * Rasterizer state.
	 */
	struct RasterizerState
	{
		RasterizerOutput *output;
		U32 flags;
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
	 * You can split the work into N amount of calls, which can be processed
	 * in parallel.
	 */
	void Rasterize(RasterizerState &state, const RasterizerInput *input, U32 input_count, U32 split_index = 0, U32 num_splits = 1);

	/**
	 * Clear color buffer
	 *
	 * You can split the work into N amount of calls, which can be processed
	 * in parallel.
	 */
	void ClearColor(RasterizerOutput &output, float4 value, U32 split_index = 0, U32 num_splits = 1);

	/**
	 * Clear depth buffer
	 *
	 * You can split the work into N amount of calls, which can be processed
	 * in parallel.
	 */
	void ClearDepth(RasterizerOutput &output, float value, U32 split_index = 0, U32 num_splits = 1);

	/**
	 * Blit output buffer to screen.
	 *
	 * You can split the work into N amount of calls, which can be processed
	 * in parallel.
	 */
	void Blit(LockBufferInfo &output, RasterizerOutput &input, U32 split_index = 0, U32 num_splits = 1);
}

