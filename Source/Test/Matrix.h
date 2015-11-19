#pragma once
#include "MathUtils.h"
#include "Vector.h"

namespace nmj
{
	inline void Transpose(float4 (&out)[4])
	{
		float tmp;
		tmp = out[0][1], out[0][1] = out[1][0], out[1][0] = tmp;
		tmp = out[0][2], out[0][2] = out[2][0], out[2][0] = tmp;
		tmp = out[0][3], out[0][3] = out[3][0], out[3][0] = tmp;
		tmp = out[1][2], out[1][2] = out[2][1], out[2][1] = tmp;
		tmp = out[1][3], out[1][3] = out[3][1], out[3][1] = tmp;
		tmp = out[2][3], out[2][3] = out[3][2], out[3][2] = tmp;
	}

	inline void Mul(float4 (&out)[4], const float4 (&a)[4], const float4 (&b)[4])
	{
		for (unsigned y = 0; y < 4; ++y)
		{
			for (unsigned x = 0; x < 4; ++x)
			{
				out[y][x]  = a[y][0] * b[0][x];
				out[y][x] += a[y][1] * b[1][x];
				out[y][x] += a[y][2] * b[2][x];
				out[y][x] += a[y][3] * b[3][x];
			}
		}
	}

	inline void CreateIdentity(float4 (&out)[4])
	{
		out[0] = float4(1.0f, 0.0f, 0.0f, 0.0f);
		out[1] = float4(0.0f, 1.0f, 0.0f, 0.0f);
		out[2] = float4(0.0f, 0.0f, 1.0f, 0.0f);
		out[3] = float4(0.0f, 0.0f, 0.0f, 1.0f);
	}

	inline void CreatePerspectiveProjection(float4 (&out)[4], float fov_y, float aspect_ratio, float near_z, float far_z)
	{
		float yscale = 1.0f / tan(fov_y * 0.5f);
		float xscale = yscale / aspect_ratio;
		float c = far_z / (far_z - near_z);

		out[0] = float4(xscale, 0.0f, 0.0f, 0.0f);
		out[1] = float4(0.0f, yscale, 0.0f, 0.0f);
		out[2] = float4(0.0f, 0.0f, c, 1.0f);
		out[3] = float4(0.0f, 0.0f, -near_z * c, 0.0f);
	}

	inline void CreateCameraTransform(float4 (&out)[4], const float3 &pos, const float3 (&axis)[3])
	{
		out[0] = float4(axis[0].x, axis[1].x, -axis[2].x, 0.0f);
		out[1] = float4(axis[0].y, axis[1].y, -axis[2].y, 0.0f);
		out[2] = float4(axis[0].z, axis[1].z, -axis[2].z, 0.0f);
		out[3] = float4(Dot(axis[0], pos), Dot(axis[1], pos), Dot(axis[2] * -1.0f, pos), 1.0f);
	}

	inline void CreateTranslate(float4 (&out)[4], const float3 &pos)
	{
		out[0] = float4(1.0f, 0.0f, 0.0f, 0.0f);
		out[1] = float4(0.0f, 1.0f, 0.0f, 0.0f);
		out[2] = float4(0.0f, 0.0f, 1.0f, 0.0f);
		out[3] = float4(pos, 1.0f);
	}
}
