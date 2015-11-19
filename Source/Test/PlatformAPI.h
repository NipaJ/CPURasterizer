/**
 * Platform API.
 *
 * Simple platform API abstraction for quick tests and hacks.
 */
#pragma once

// Macros for translating color to/from software frame buffer format.
#if _WIN32 || _WIN64
	#define NMJ_RGBA(p_r, p_g, p_b, p_a) \
		(U32((((p_a) << 24) & 0xFF000000) | (((p_r) << 16) & 0x00FF0000) | (((p_g) << 8) & 0x0000FF00) | (((p_b) << 0) & 0x000000FF)))
	#define NMJ_ALPHA(p_color) \
		(U8(((p_color) >> 24) & 0xFF))
	#define NMJ_RED(p_color) \
		(U8(((p_color) >> 16) & 0xFF))
	#define NMJ_GREEN(p_color) \
		(U8(((p_color) >> 8) & 0xFF))
	#define NMJ_BLUE(p_color) \
		(U8(((p_color) >> 0) & 0xFF))
#else
	#error "NMJ_RGBA not defined for the platform."
#endif

namespace nmj
{
	// Minimal set of keycodes.
	enum KeyCode
	{
		KeyCodeTab,     KeyCodeEnter,   KeyCodeSpace,     KeyCodeLeft,     KeyCodeUp,       KeyCodeRight,
		KeyCodeDown,    KeyCode0,       KeyCode1,         KeyCode2,        KeyCode3,        KeyCode4,
		KeyCode5,       KeyCode6,       KeyCode7,         KeyCode8,        KeyCode9,        KeyCodeA,
		KeyCodeB,       KeyCodeC,       KeyCodeD,         KeyCodeE,        KeyCodeF,        KeyCodeG,
		KeyCodeH,       KeyCodeI,       KeyCodeJ,         KeyCodeK,        KeyCodeL,        KeyCodeM,
		KeyCodeN,       KeyCodeO,       KeyCodeP,         KeyCodeQ,        KeyCodeR,        KeyCodeS,
		KeyCodeT,       KeyCodeU,       KeyCodeV,         KeyCodeW,        KeyCodeX,        KeyCodeY,
		KeyCodeZ,       KeyCodeLWin,    KeyCodeRWin,      KeyCodeNumpad0,  KeyCodeNumpad1,  KeyCodeNumpad2,
		KeyCodeNumpad3, KeyCodeNumpad4, KeyCodeNumpad5,   KeyCodeNumpad6,  KeyCodeNumpad7,  KeyCodeNumpad8,
		KeyCodeNumpad9, KeyCodeF1,      KeyCodeF2,        KeyCodeF3,       KeyCodeF4,       KeyCodeF5,
		KeyCodeF6,      KeyCodeF7,      KeyCodeF8,        KeyCodeF9,       KeyCodeF10,      KeyCodeF11,
		KeyCodeF12,     KeyCodeLShift,  KeyCodeRShift,    KeyCodeLControl, KeyCodeRControl, KeyCodeLMenu,
		KeyCodeRMenu,   KeyCodeEsc,     KeyCodeBackspace,

		NumKeyCodes
	};

	enum MouseCaptureMode
	{
		// Shared mouse access with other applications. This is the default.
		MouseCaptureModeShared,
		// Hide the OS cursor and prevent mouse from interacting with other applications.
		MouseCaptureModeExclusive
	};

	typedef U8 MouseButtonFlags;
	enum
	{
		MouseButton1 = 0x01,
		MouseButton2 = 0x02,
		MouseButton3 = 0x04,
		MouseButton4 = 0x08,
		MouseButton5 = 0x10,
		MouseButton6 = 0x20,
		MouseButton7 = 0x40,
		MouseButton8 = 0x80
	};

	struct LockBufferInfo
	{
		// Data stored as (y * pitch + x) * bits per pixel
		void *data;
		U32 width;
		U32 height;
		U32 pitch;
	};

	// Event function types.
	typedef void KeyboardEvent(void *userdata, KeyCode code, bool down);
	typedef void MouseEvent(void *userdata, S16 delta_x, S16 delta_y, S16 delta_z, MouseButtonFlags down_flags);

	// PlatformAPI data is declared internally in the translation unit.
	struct PlatformAPI;
	struct SoftwareRenderer;

	// Platform main function
	void Main(PlatformAPI *);

	// Events setters.
	void SetKeyboardEvent(PlatformAPI *self, void *userdata, KeyboardEvent *event);
	void SetMouseEvent(PlatformAPI *self, void *userdata, MouseEvent *event);

	// Mouse capture mode
	void SetMouseCaptureMode(PlatformAPI *self, MouseCaptureMode mode);

	// Set application title
	void SetApplicationTitle(PlatformAPI *self, const char *title);

	// Call this once per frame. Returns false when quit is requested.
	bool Update(PlatformAPI *self);

	// Report fatal error.
	void FatalError(PlatformAPI *self, const char *msg);

	// Get time since the beginning of the application.
	// Returns seconds in fixed point 32.32
	U64 GetTime(PlatformAPI *self);

	// Create software renderer for the platform.
	// Software renderer is using R8G8B8X8 frame buffer.
	SoftwareRenderer *CreateSoftwareRenderer(PlatformAPI *self, unsigned width, unsigned height, bool fullscreen);
	void Release(SoftwareRenderer *self);

	// Lock software framebuffer
	bool LockBuffer(SoftwareRenderer *self, LockBufferInfo &info);
	void UnlockBuffer(SoftwareRenderer *self);
}

