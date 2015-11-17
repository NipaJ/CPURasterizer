#define _CRT_SECURE_NO_WARNINGS 1

#include "General.h"
#include "Font.h"

#include <stdlib.h>
#include <stdio.h>

#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

namespace nmj
{
	struct CharacterInfo
	{
		unsigned char *bitmap;
		int cw, ch, xoff, yoff;
		int advance, lsb;
		float scale;
	};

	struct Font
	{
		CharacterInfo ci[256];
		float height;
	};

	Font *CreateFontFromFile(const char *filename, float height)
	{
		// Load the file into memory
		unsigned char *buffer;
		size_t buffer_size;
		{
			FILE *f = fopen(filename, "rb");
			if (f == NULL)
				return NULL;

			fseek(f, 0, SEEK_END);
			buffer_size = ftell(f);
			rewind(f);

			buffer = (unsigned char *)malloc(buffer_size);
			NMJ_ASSERT(buffer);
			fread(buffer, 1, buffer_size, f);

			fclose(f);
		}

		Font *self = (Font *)malloc(sizeof (Font));
		NMJ_ASSERT(self);
		self->height = height;

		stbtt_fontinfo font_info;
		stbtt_InitFont(&font_info, buffer, 0);

		for (unsigned i = 0; i < 256; ++i)
		{
			CharacterInfo &ci = self->ci[i];

			int cw, ch, xoff, yoff;
			int advance, lsb;

			stbtt_GetCodepointHMetrics(&font_info, i, &advance, &lsb);
			float scale = stbtt_ScaleForPixelHeight(&font_info, height);

			// Should probably handle the allocations bit smarter.. Oh well, it's just for debugging anyways :P
			ci.bitmap = stbtt_GetCodepointBitmap(&font_info, scale, scale, i, &cw, &ch, &xoff, &yoff);
			ci.xoff = xoff;
			ci.yoff = yoff;
			ci.cw = cw;
			ci.ch = ch;
			ci.advance = advance;
			ci.lsb = lsb;
			ci.scale = scale;
		}

		free(buffer);
		return self;
	}

	void Release(Font *self)
	{
		for (unsigned i = 0; i < 256; ++i)
			free(self->ci[i].bitmap);
		free(self);
	}

	// NOTE: Doesn't handle UTF-8 yet.
	void RenderText(Font *self, LockBufferInfo &frame_info, unsigned x, unsigned y, const char *str, const float4 &color)
	{
		U8 color_data[3] = { U8(color.x * 255.0f), U8(color.y * 255.0f), U8(color.z * 255.0f) };

		y += unsigned(self->height);

		for (const char *c = str; *c != '\0'; ++c)
		{
			CharacterInfo &ci = self->ci[*c];
			if (x >= frame_info.width)
				break;
			if (y >= frame_info.height)
				break;

			char *data = (char *)frame_info.data;
			for (unsigned iy = 0; iy < unsigned(ci.ch); ++iy)
			{
				int oy = y + iy + ci.yoff;
				if (oy < 0 || oy >= int(frame_info.height))
					continue;

				for (unsigned ix = 0; ix < unsigned(ci.cw); ++ix)
				{
					int ox = x + ix + ci.xoff;
					if (ox < 0 || ox >= int(frame_info.width))
						continue;

					U8 alpha = ci.bitmap[iy * ci.cw + ix];

					// NOTE: Should probably not use LockBufferInfo, since it could be a write-only video buffer.
					U32 *pixel = (U32 *)(data + ((oy * frame_info.pitch) + ox * 4));
					U8 bg_color[3] =
					{
						(NMJ_RED(*pixel) * (255 - alpha)) / 255,
						(NMJ_GREEN(*pixel) * (255 - alpha)) / 255,
						(NMJ_BLUE(*pixel) * (255 - alpha)) / 255,
					};

					*pixel = NMJ_RGBA(
						(alpha * color_data[0]) / 255 + bg_color[0],
						(alpha * color_data[1]) / 255 + bg_color[1],
						(alpha * color_data[2]) / 255 + bg_color[2],
						0
					);
				}
			}

			x += unsigned(ci.advance * ci.scale);
		}
	}
}
