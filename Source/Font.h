#pragma once
#include "PlatformAPI.h"
#include "Vector.h"

namespace nmj
{
	struct Font;

	Font *CreateFont(const char *filename, float height);
	void Release(Font *self);

	void RenderText(Font *self, LockBufferInfo &frame_info, unsigned x, unsigned y, const char *str, const float4 &color);
}

