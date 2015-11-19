#pragma once
#include "MathUtils.h"

namespace nmj
{
	union float2 
	{
		struct { float x, y; };
		float v[2];

		float2() = default;

		NMJ_FORCEINLINE float2(float v)
			: x(v), y(v)
		{}

		NMJ_FORCEINLINE float2(float x, float y)
			: x(x), y(y)
		{}

		NMJ_FORCEINLINE float &operator [] (unsigned i) { return v[i]; }
		NMJ_FORCEINLINE float operator [] (unsigned i) const { return v[i]; }
	};

	union float3 
	{
		struct { float x, y, z; };
		float2 xy;
		float v[4];

		float3() = default;

		NMJ_FORCEINLINE float3(float v)
			: x(v), y(v), z(v)
		{}

		NMJ_FORCEINLINE float3(float x, float y, float z)
			: x(x), y(y), z(z)
		{}

		NMJ_FORCEINLINE float &operator [] (unsigned i) { return v[i]; }
		NMJ_FORCEINLINE float operator [] (unsigned i) const { return v[i]; }
	};

	union float4 
	{
		struct { float x, y, z, w; };
		struct { float2 xy, zw; };
		float3 xyz;
		float v[4];

		float4() = default;

		NMJ_FORCEINLINE float4(float v)
			: x(v), y(v), z(v), w(v)
		{}

		NMJ_FORCEINLINE float4(float x, float y, float z, float w)
			: x(x), y(y), z(z), w(w)
		{}

		NMJ_FORCEINLINE float4(float3 xyz, float w)
			: x(xyz.x), y(xyz.y), z(xyz.z), w(w)
		{}

		NMJ_FORCEINLINE float &operator [] (unsigned i) { return v[i]; }
		NMJ_FORCEINLINE float operator [] (unsigned i) const { return v[i]; }
	};

	// =====================================================================
	// Scalar multiplication.
	// =====================================================================

	NMJ_FORCEINLINE float3 operator * (float s, const float3 &v)
	{
		return float3(v.x * s, v.y * s, v.z * s);
	}

	NMJ_FORCEINLINE float4 operator * (float s, const float4 &v)
	{
		return float4(v.x * s, v.y * s, v.z * s, v.w * s);
	}

	NMJ_FORCEINLINE float3 operator * (const float3 &v, float s)
	{
		return float3(v.x * s, v.y * s, v.z * s);
	}

	NMJ_FORCEINLINE float4 operator * (const float4 &v, float s)
	{
		return float4(v.x * s, v.y * s, v.z * s, v.w * s);
	}

	NMJ_FORCEINLINE void operator *= (float3 &dest, float s)
	{
		dest.x *= s;
		dest.y *= s;
		dest.z *= s;
	}

	NMJ_FORCEINLINE void operator *= (float4 &dest, float s)
	{
		dest.x *= s;
		dest.y *= s;
		dest.z *= s;
		dest.w *= s;
	}

	// =====================================================================
	// Vector addition.
	// =====================================================================

	NMJ_FORCEINLINE float3 operator + (const float3 &a, const float3 &b)
	{
		return float3(a.x + b.x, a.y + b.y, a.z + b.z);
	}

	NMJ_FORCEINLINE float4 operator + (const float4 &a, const float4 &b)
	{
		return float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
	}

	NMJ_FORCEINLINE void operator += (float3 &dest, const float3 &src)
	{
		dest.x += src.x;
		dest.y += src.y;
		dest.z += src.z;
	}

	NMJ_FORCEINLINE void operator += (float4 &dest, const float4 &src)
	{
		dest.x += src.x;
		dest.y += src.y;
		dest.z += src.z;
		dest.w += src.w;
	}

	// =====================================================================
	// Vector subtraction.
	// =====================================================================

	NMJ_FORCEINLINE float3 operator - (const float3 &a, const float3 &b)
	{
		return float3(a.x - b.x, a.y - b.y, a.z - b.z);
	}

	NMJ_FORCEINLINE float4 operator - (const float4 &a, const float4 &b)
	{
		return float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
	}

	NMJ_FORCEINLINE void operator -= (float3 &dest, const float3 &src)
	{
		dest.x -= src.x;
		dest.y -= src.y;
		dest.z -= src.z;
	}

	NMJ_FORCEINLINE void operator -= (float4 &dest, const float4 &src)
	{
		dest.x -= src.x;
		dest.y -= src.y;
		dest.z -= src.z;
		dest.w -= src.w;
	}

	// =====================================================================
	// Dot product.
	// =====================================================================

	NMJ_FORCEINLINE float Dot(const float3 &a, const float3 &b)
	{
		float s = a.x * b.x;
		s += a.y * b.y;
		s += a.z * b.z;
		return s;
	}

	NMJ_FORCEINLINE float Dot(const float4 &a, const float4 &b)
	{
		float s = a.x * b.x;
		s += a.y * b.y;
		s += a.z * b.z;
		s += a.w * b.w;
		return s;
	}

	// =====================================================================
	// Length.
	// =====================================================================

	NMJ_FORCEINLINE float GetLength(const float3 &src)
	{
		float s = src.x * src.x;
		s += src.y * src.y;
		s += src.z * src.z;
		return Sqrt(s);
	}

	NMJ_FORCEINLINE float GetLength(const float4 &src)
	{
		float s = src.x * src.x;
		s += src.y * src.y;
		s += src.z * src.z;
		s += src.w * src.w;
		return Sqrt(s);
	}

	// =====================================================================
	// Normalize.
	// =====================================================================

	NMJ_FORCEINLINE void Normalize(float3 &dest)
	{
		float s = dest.x * dest.x;
		s += dest.y * dest.y;
		s += dest.z * dest.z;

		s = RSqrt(s);

		dest.x *= s;
		dest.y *= s;
		dest.z *= s;
	}

	NMJ_FORCEINLINE void Normalize(float4 &dest)
	{
		float s = dest.x * dest.x;
		s += dest.y * dest.y;
		s += dest.z * dest.z;
		s += dest.w * dest.w;

		s = RSqrt(s);

		dest.x *= s;
		dest.y *= s;
		dest.z *= s;
		dest.w *= s;
	}

	// =====================================================================
	// float3 specific.
	// =====================================================================

	NMJ_FORCEINLINE float3 Cross(const float3 &a, const float3 &b)
	{
		return float3(
			a.y * b.z - a.z * b.y,
			a.z * b.x - a.x * b.z,
			a.x * b.y - a.y * b.x
		);
	}

	NMJ_FORCEINLINE void Rotate(float3 &self, const float3 &axis, float angle)
	{
		float c = cos(angle);
		float d = (1 - c) * Dot(axis, self);
		float3 x = sin(angle) * Cross(axis, self);

		self *= c;
		self += x;
		self += d * axis;
	}
}
