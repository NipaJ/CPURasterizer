/**
 * Software rasterization API.
 */
#pragma once

namespace nmj
{
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
	 * 
	 * Output buffers are constructed from 32x32 pixel tiles that are stored contiguously
	 * in memory. When the output resolution does not divide evenly with the tiles, only
	 * the remainder of the tile is used and rest of the tile is just padding with a
	 * undefined value.
	 * 
	 * As an example, 144x80 output buffer would look in screen-space as following:
	 *  -------------------- X
	 * | T1  T2  T3  T4  T5
	 * | T6  T7  T8  T9  T10
	 * | T11 T12 T13 T14 T15
	 * Y
	 * 
	 * Only half of the horizontal pixels of tiles T5, T10 and T15 are used, because
	 * 144 / 32 = 4.5 tiles. Same applies for tiles from T11 to T15, but vertically,
	 * since 80 / 32 = 2.5 tiles.
	 * 
	 * Tiles themselves are constructed from blocks of 2x2 pixels. Blocks are also stored
	 * contiguously in memory and have similar screen-space layout to tiles (y-axis pointing
	 * down).
	 */
	struct RasterizerOutput
	{
		/**
		 * 32bit RGB color buffer with 8bit channels.
		 * 16 byte alignment is required.
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
		 * 16 byte alignment is required.
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
		float transform[4][4];

		/**
		 * Per vertex information.
		 * All but vertices are optional and can be NULL.
		 */
		const float *vertices; // xyz per vertex
		const float *colors;   // rgba per vertex
		const float *texcoords; // xy per vertex

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
	void ClearColor(RasterizerOutput &output, float r, float g, float b, float a, U32 split_index = 0, U32 num_splits = 1);

	/**
	 * Clear depth buffer
	 *
	 * You can split the work into N amount of calls, which can be processed
	 * in parallel.
	 */
	void ClearDepth(RasterizerOutput &output, float depth, U8 stencil, U32 split_index = 0, U32 num_splits = 1);

	/**
	 * Blit output buffer to screen.
	 *
	 * You can split the work into N amount of calls, which can be processed
	 * in parallel.
	 */
	void Blit(void *output, U32 pitch, RasterizerOutput &input, U32 split_index = 0, U32 num_splits = 1);
}

