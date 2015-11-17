/**
 * Simple header providing general types and stuff.
 */
#pragma once
#include <assert.h>
#include <stdint.h>
#include <stddef.h>

// Disable some warning level 4 warnings, that are not helpful.
#pragma warning(disable : 4201)
#pragma warning(disable : 4324)
#pragma warning(disable : 4100)

/* Run-time assert. */
#if NMJ_DISABLE_ASSERT
	#define NMJ_ASSERT(p_expr) ((void)0)
#else
	#if NMJ_CUSTOM_ASSERT
		#define NMJ_ASSERT(p_expr) ((p_expr) ? ::nmj::CustomAssertImpl(#p_expr, __LINE__, __FILE__) : ((void)0))
	#else
		#define NMJ_ASSERT(p_expr) assert(p_expr)
	#endif
#endif

/* Static assert. */
#if NMJ_DISABLE_STATIC_ASSERT
	#define NMJ_STATIC_ASSERT(p_expr, p_message) enum {}
#else
	#define NMJ_STATIC_ASSERT(p_expr, p_message) static_assert(p_expr, p_message)
#endif

/* Force inline */
#if defined(_MSC_VER)
	#define NMJ_FORCEINLINE __forceinline
#else
	#error "NMJ_FORCEINLINE not defined for this platform."
#endif

namespace nmj
{
	/* Fixed-width integer types. */
	typedef int8_t  S8;
	typedef int16_t S16;
	typedef int32_t S32;
	typedef int64_t S64;
	typedef uint8_t  U8;
	typedef uint16_t U16;
	typedef uint32_t U32;
	typedef uint64_t U64;

	/* Integer types that can contain pointer values. */
	typedef intptr_t SPtr;
	typedef uintptr_t UPtr;

	/* Align size with power of two value. */
	template <typename T>
	NMJ_FORCEINLINE T GetAligned(T value, UPtr alignment)
	{
		const UPtr mask = alignment - 1;
		return (value + mask) & ~T(mask);
	}

	/* Align pointer with power of two value. */
	template <typename T>
	NMJ_FORCEINLINE T *GetAligned(T *value, UPtr alignment)
	{
		const UPtr mask = alignment - 1;

		UPtr addr = UPtr(value);
		addr += mask;
		addr &= ~mask;
		return (T *)addr;
	}

	/* Round up to specified unit. */
	template <typename T>
	NMJ_FORCEINLINE T RoundUpToUnit(const T value, const T unit)
	{
		return (value + (unit - 1)) / unit;
	}
}

