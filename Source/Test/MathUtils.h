#pragma once
#include <math.h>
#include <emmintrin.h>

namespace nmj
{
	// Pi constant
	static const float Pi = 3.14159f;

	// Tau constant (Tau = 2Pi)
	static const float Tau = 6.28319f;

	// Absolute value
	NMJ_FORCEINLINE float Abs(float x)
	{
		__m128 v = _mm_set_ss(x);
		v = _mm_andnot_ps(_mm_set_ss(-0.0f), v);
		return _mm_cvtss_f32(v);
	}

	// Faster square root
	NMJ_FORCEINLINE float Sqrt(float x)
	{
		__m128 v = _mm_set_ss(x);
		return _mm_cvtss_f32(_mm_mul_ss(v, _mm_rsqrt_ss(v)));
	}

	// Faster reciprocal square root
	NMJ_FORCEINLINE float RSqrt(float x)
	{
		return _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(x)));
	}

	NMJ_FORCEINLINE S32 Max(S32 a, S32 b)
	{
		return a > b ? a : b;
	}

	NMJ_FORCEINLINE S32 Min(S32 a, S32 b)
	{
		return a < b ? a : b;
	}
}

