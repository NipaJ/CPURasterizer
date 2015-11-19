#include "General.h"
#include "PlatformAPI.h"

#define SOFTWARE_RENDERER_GDI 1

#include <windows.h>
#include <stdlib.h>

namespace nmj
{
	enum RendererType
	{
		RendererTypeNone,
		RendererTypeSoftware
	};

	struct SoftwareRenderer
	{
		// Really there isn't any need for this and could use something like
		// CONTAINING_RECORD macro from FreeBSD kernel. Too bad C++ doesn't
		// have composite_cast thanks to virtual inheritance.
		PlatformAPI *api;

#if SOFTWARE_RENDERER_GDI
		// Software framebuffer info
		BITMAPINFO bitmapinfo;
		void *bitmap_buffer;
#endif
	};

	struct PlatformAPI
	{
		// Renderer
		RendererType renderer_type;
		union
		{
			SoftwareRenderer software;
		} renderer;

		// Event userdata
		void *keyboard_event_userdata;
		void *mouse_event_userdata;

		// Event procedures
		KeyboardEvent *keyboard_event;
		MouseEvent *mouse_event;

		// Win32 data
		HINSTANCE hinstance;
		HWND hwnd;

		// Times
		LARGE_INTEGER time_frequency, time_offset;

		// Raw input event buffer
		U32 rid_buffer_size;
		void *rid_buffer;

		// Mouse data
		MouseCaptureMode mouse_capture_mode;
		MouseButtonFlags mouse_button_states;
		S32 mouse_saved_x, mouse_saved_y;
		S32 mouse_cur_x, mouse_cur_y;

		// Keyboard state info.
		// Each key gets one bit reserved for it's state.
		U32 keyboard_state[(31 + NumKeyCodes) / 32];
	};

	static KeyCode TranslateVirtualKeyToKeyCode(USHORT vk)
	{
		// Check unique keys
		switch (vk)
		{
		case VK_TAB: return KeyCodeTab;
		case VK_RETURN: return KeyCodeEnter;
		case VK_SPACE: return KeyCodeSpace;
		case VK_LEFT: return KeyCodeLeft;
		case VK_UP: return KeyCodeUp;
		case VK_RIGHT: return KeyCodeRight;
		case VK_DOWN: return KeyCodeDown;
		case VK_LWIN: return KeyCodeLWin;
		case VK_RWIN: return KeyCodeRWin;
		case VK_NUMPAD0: return KeyCodeNumpad0;
		case VK_NUMPAD1: return KeyCodeNumpad1;
		case VK_NUMPAD2: return KeyCodeNumpad2;
		case VK_NUMPAD3: return KeyCodeNumpad3;
		case VK_NUMPAD4: return KeyCodeNumpad4;
		case VK_NUMPAD5: return KeyCodeNumpad5;
		case VK_NUMPAD6: return KeyCodeNumpad6;
		case VK_NUMPAD7: return KeyCodeNumpad7;
		case VK_NUMPAD8: return KeyCodeNumpad8;
		case VK_NUMPAD9: return KeyCodeNumpad9;
		case VK_F1: return KeyCodeF1;
		case VK_F2: return KeyCodeF2;
		case VK_F3: return KeyCodeF3;
		case VK_F4: return KeyCodeF4;
		case VK_F5: return KeyCodeF5;
		case VK_F6: return KeyCodeF6;
		case VK_F7: return KeyCodeF7;
		case VK_F8: return KeyCodeF8;
		case VK_F9: return KeyCodeF9;
		case VK_F10: return KeyCodeF10;
		case VK_F11: return KeyCodeF11;
		case VK_F12: return KeyCodeF12;
		case VK_LSHIFT: return KeyCodeLShift;
		case VK_RSHIFT: return KeyCodeRShift;
		case VK_LCONTROL: return KeyCodeLControl;
		case VK_RCONTROL: return KeyCodeRControl;
		case VK_LMENU: return KeyCodeLMenu;
		case VK_RMENU: return KeyCodeRMenu;
		case VK_ESCAPE: return KeyCodeEsc;
		case VK_BACK: return KeyCodeBackspace;
		default: break;
		}

		// Check numbers
		if (vk >= 0x30 && vk <= 0x39)
			return KeyCode(KeyCode0 + (vk - 0x30));

		// Check letters
		if (vk >= 0x41 && vk <= 0x5A)
			return KeyCode(KeyCodeA + (vk - 0x41));

		// NumKeyCodes means "not found"
		return NumKeyCodes;
	}

	static const TCHAR *window_class_name = TEXT("PlatformAPI_WindowClass");

	static LRESULT CALLBACK OnWindowCallback(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
	{
		// Get pointer to itself.
		PlatformAPI *self;
		if (uMsg == WM_CREATE)
		{
			CREATESTRUCT *cs = (CREATESTRUCT *)lParam;
			self = (PlatformAPI *)cs->lpCreateParams;

			SetWindowLongPtr(hwnd, GWLP_USERDATA, (LONG_PTR)self);
		}
		else
		{
			self = (PlatformAPI *)GetWindowLongPtr(hwnd, GWLP_USERDATA);
		}

		// Process different messages
		switch (uMsg)
		{
		case WM_INPUT:
			{
				UINT required_size;
				GetRawInputData(
					(HRAWINPUT)lParam,
					RID_INPUT,
					NULL,
					&required_size,
					sizeof (RAWINPUTHEADER)
				);

				if (self->rid_buffer_size < required_size)
				{
					self->rid_buffer = realloc(self->rid_buffer, required_size);
					self->rid_buffer_size = required_size;
				}

				GetRawInputData(
					(HRAWINPUT)lParam,
					RID_INPUT,
					self->rid_buffer,
					&required_size,
					sizeof (RAWINPUTHEADER)
				);

				RAWINPUT *raw = (RAWINPUT *)self->rid_buffer;
				if (raw->header.dwType == RIM_TYPEKEYBOARD)
				{
					KeyCode kc = TranslateVirtualKeyToKeyCode(raw->data.keyboard.VKey);

					bool new_state = ((raw->data.keyboard.Flags & RI_KEY_BREAK) == 0);
					bool old_state = (self->keyboard_state[kc / 32] >> (kc & 31)) & 1;

					if (new_state != old_state)
					{
						if (new_state)
							self->keyboard_state[kc / 31] |= 1 << (kc & 31);
						else
							self->keyboard_state[kc / 31] &= ~(1 << (kc & 31));

						self->keyboard_event(self->keyboard_event_userdata, kc, new_state);
					}
				}
				else if (raw->header.dwType == RIM_TYPEMOUSE)
				{
					S16 delta_x = 0, delta_y = 0, delta_z = 0;

					// Figure out x and y delta
					if (raw->data.mouse.usFlags & MOUSE_MOVE_ABSOLUTE)
					{
						 delta_x = S16(S32(raw->data.mouse.lLastX) - self->mouse_cur_x);
						 delta_y = S16(S32(raw->data.mouse.lLastY) - self->mouse_cur_y);

						 self->mouse_cur_x = S32(raw->data.mouse.lLastX);
						 self->mouse_cur_y = S32(raw->data.mouse.lLastY);
					}
					else
					{
						 delta_x = S16(raw->data.mouse.lLastX);
						 delta_y = S16(raw->data.mouse.lLastY);

						 self->mouse_cur_x += S32(raw->data.mouse.lLastX);
						 self->mouse_cur_y += S32(raw->data.mouse.lLastY);
					}

					USHORT button_flags = raw->data.mouse.usButtonFlags;

					// Figure out z delta
					if (button_flags & RI_MOUSE_WHEEL)
						delta_z = S16(raw->data.mouse.usButtonData);

					// Detect up and down button states
					U8 down_buttons = 0, up_buttons = 0;
					if (button_flags & RI_MOUSE_BUTTON_1_DOWN)
						down_buttons |= 0x01;
					if (button_flags & RI_MOUSE_BUTTON_2_DOWN)
						down_buttons |= 0x02;
					if (button_flags & RI_MOUSE_BUTTON_3_DOWN)
						down_buttons |= 0x04;
					if (button_flags & RI_MOUSE_BUTTON_4_DOWN)
						down_buttons |= 0x08;
					if (button_flags & RI_MOUSE_BUTTON_5_DOWN)
						down_buttons |= 0x10;
					if (button_flags & RI_MOUSE_BUTTON_1_UP)
						up_buttons |= 0x01;
					if (button_flags & RI_MOUSE_BUTTON_2_UP)
						up_buttons |= 0x02;
					if (button_flags & RI_MOUSE_BUTTON_3_UP)
						up_buttons |= 0x04;
					if (button_flags & RI_MOUSE_BUTTON_4_UP)
						up_buttons |= 0x08;
					if (button_flags & RI_MOUSE_BUTTON_5_UP)
						up_buttons |= 0x10;

					// Check if needs to be sent as two events.
					if (down_buttons & up_buttons)
					{
						MouseButtonFlags button_states = self->mouse_button_states;

						button_states |= down_buttons;
						self->mouse_event(self->mouse_event_userdata, delta_x, delta_y, delta_z, button_states);

						button_states &= ~up_buttons;
						self->mouse_event(self->mouse_event_userdata, delta_x, delta_y, delta_z, button_states);

						self->mouse_button_states = button_states;
					}
					else
					{
						MouseButtonFlags button_states = self->mouse_button_states;

						button_states |= down_buttons;
						button_states &= ~up_buttons;
						self->mouse_event(self->mouse_event_userdata, delta_x, delta_y, delta_z, button_states);

						self->mouse_button_states = button_states;
					}
				}
				return 0;
			}

		case WM_PAINT:
			{
#if SOFTWARE_RENDERER_GDI
				if (self->renderer_type == RendererTypeSoftware)
				{
					SoftwareRenderer *renderer = &self->renderer.software;

					PAINTSTRUCT ps;

					HDC hdc = BeginPaint(hwnd, &ps);

					RECT client_rect;
					GetClientRect(hwnd, &client_rect);

					unsigned dest_width = unsigned(client_rect.right - client_rect.left);
					unsigned dest_height = unsigned(client_rect.bottom - client_rect.top);
					unsigned src_width = unsigned(renderer->bitmapinfo.bmiHeader.biWidth);
					unsigned src_height = unsigned(-renderer->bitmapinfo.bmiHeader.biHeight);

					StretchDIBits(
						hdc,
						0, 0, dest_width, dest_height,
						0, 0, src_width, src_height,
						renderer->bitmap_buffer,
						&renderer->bitmapinfo,
						DIB_RGB_COLORS,
						SRCCOPY
					);

					EndPaint(hwnd, &ps);
					return 0;
				}
#endif
				break;
			}

		case WM_CLOSE:
			PostQuitMessage(0);
			break;
		}

		return DefWindowProc(hwnd, uMsg, wParam, lParam);
	}

	static void Initialize(PlatformAPI *self, HINSTANCE hinstance)
	{
		memset(self, 0, sizeof *self);
		self->hinstance = hinstance;

		// Query start time and stuff
		QueryPerformanceFrequency(&self->time_frequency);
		QueryPerformanceCounter(&self->time_offset);

		WNDCLASSEX wc;
		memset(&wc, 0, sizeof wc);
		wc.cbSize = sizeof wc;
		wc.lpfnWndProc = OnWindowCallback;
		wc.lpszClassName = window_class_name;
		wc.hInstance = hinstance;
		wc.hIcon = LoadIcon(NULL, IDI_APPLICATION);
		wc.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
		wc.hCursor = LoadCursor(NULL, IDC_ARROW);

		if (RegisterClassEx(&wc) == 0)
			FatalError(self, "Failed to register window class");

		self->hwnd = CreateWindowEx(
			0,
			window_class_name,
			TEXT("PlatformAPI Window"),
			WS_OVERLAPPEDWINDOW,
			CW_USEDEFAULT, CW_USEDEFAULT,
			CW_USEDEFAULT, CW_USEDEFAULT,
			NULL,
			NULL,
			hinstance,
			self
		);

		if (self->hwnd == NULL)
			FatalError(self, "Failed to create the default window.");

		// Register input devices and prepare for input reading
		{
			RAWINPUTDEVICE devices[2];

			// Preallocate raw-input buffer.
			self->rid_buffer_size = 512;
			self->rid_buffer = malloc(self->rid_buffer_size);

			// Keyboard
			devices[0].usUsagePage = 0x01;
			devices[0].usUsage = 0x06;
			devices[0].dwFlags = 0;
			devices[0].hwndTarget = self->hwnd;

			// Mouse
			devices[1].usUsagePage = 0x01;
			devices[1].usUsage = 0x02;
			devices[1].dwFlags = 0;
			devices[1].hwndTarget = self->hwnd;

			if (RegisterRawInputDevices(devices, 2, sizeof devices[0]) == FALSE)
				FatalError(self, "Failed to register the input devices.");
		}
	}

	void SetKeyboardEvent(PlatformAPI *self, void *userdata, KeyboardEvent *event)
	{
		self->keyboard_event_userdata = userdata;
		self->keyboard_event = event;
	}

	void SetMouseEvent(PlatformAPI *self, void *userdata, MouseEvent *event)
	{
		self->mouse_event_userdata = userdata;
		self->mouse_event = event;
	}

	void SetMouseCaptureMode(PlatformAPI *self, MouseCaptureMode mode)
	{
		if (self->mouse_capture_mode == mode)
			return;

		RAWINPUTDEVICE mouse;

		// Mouse
		mouse.usUsagePage = 0x01;
		mouse.usUsage = 0x02;
		mouse.dwFlags = 0;
		mouse.hwndTarget = self->hwnd;

		if (mode == MouseCaptureModeExclusive)
		{
			POINT cp;
			GetCursorPos(&cp);

			mouse.dwFlags |= RIDEV_NOLEGACY | RIDEV_CAPTUREMOUSE;
			self->mouse_saved_x = cp.x;
			self->mouse_saved_y = cp.y;

			ShowCursor(false);
		}

		if (RegisterRawInputDevices(&mouse, 1, sizeof mouse) == FALSE)
			FatalError(self, "Failed to set mouse capture mode.");

		if (mode == MouseCaptureModeShared)
		{
			SetCursorPos(self->mouse_saved_x, self->mouse_saved_y);
			ShowCursor(true);
		}

		self->mouse_capture_mode = mode;
	}

	void SetApplicationTitle(PlatformAPI *self, const char *title)
	{
		SetWindowTextA(self->hwnd, title);
	}

	bool Update(PlatformAPI *self)
	{
		MSG msg;

		while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			if (msg.message == WM_QUIT)
				return false;

			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
		
		return true;
	}

	void FatalError(PlatformAPI *self, const char *msg)
	{
		MessageBoxA(NULL, msg, "Fatal Error", MB_OK | MB_ICONERROR);
		ExitProcess((UINT)-1);
	}

	U64 GetTime(PlatformAPI *self)
	{
		LARGE_INTEGER time;
		QueryPerformanceCounter(&time);

		const U64 second = U64(1) << U64(32);
		U64 ret;
		if (self->time_frequency.QuadPart > second)
			ret = (time.QuadPart - self->time_offset.QuadPart) / (self->time_frequency.QuadPart / second);
		else
			ret = (time.QuadPart - self->time_offset.QuadPart) * (second / self->time_frequency.QuadPart);

		return ret;
	}

	SoftwareRenderer *CreateSoftwareRenderer(PlatformAPI *self, unsigned width, unsigned height, bool fullscreen)
	{
		NMJ_ASSERT(self->renderer_type == RendererTypeNone);
		self->renderer_type = RendererTypeSoftware;
		SoftwareRenderer *renderer = &self->renderer.software;

		renderer->api = self;

#if SOFTWARE_RENDERER_GDI
		// Setup bitmap info
		BITMAPINFO &bitmapinfo = renderer->bitmapinfo;
		memset(&bitmapinfo, 0, sizeof (BITMAPINFO));
		bitmapinfo.bmiHeader.biSize = sizeof (BITMAPINFO);
		bitmapinfo.bmiHeader.biWidth = width;
		bitmapinfo.bmiHeader.biHeight = -int(height);
		bitmapinfo.bmiHeader.biPlanes = 1;
		bitmapinfo.bmiHeader.biBitCount = 32;
		bitmapinfo.bmiHeader.biCompression = BI_RGB;

		// Allocate framebuffer.
		const size_t buffer_size = width * height * 4;
		renderer->bitmap_buffer = malloc(buffer_size);
		memset(renderer->bitmap_buffer, 0, buffer_size);

		// Figure out proper window size
		RECT window_rect;
		window_rect.left = 0;
		window_rect.top = 0;
		window_rect.right = width;
		window_rect.bottom = height;
		AdjustWindowRectEx(&window_rect, WS_OVERLAPPEDWINDOW, FALSE, 0);

		RECT old_window_rect;
		GetWindowRect(self->hwnd, &old_window_rect);
		window_rect.left += old_window_rect.left;
		window_rect.top += old_window_rect.top;
		window_rect.right += old_window_rect.left;
		window_rect.bottom += old_window_rect.top;
#endif

		// Resize window
		MoveWindow(
			self->hwnd,
			window_rect.left, window_rect.top,
			(window_rect.right - window_rect.left), (window_rect.bottom - window_rect.top),
			false
		);

		// Make the window visible
		ShowWindow(self->hwnd, SW_SHOW);
		UpdateWindow(self->hwnd);

		return renderer;
	}

	void Release(SoftwareRenderer *self)
	{
		PlatformAPI *api = self->api;
		NMJ_ASSERT(api->renderer_type == RendererTypeSoftware);

		free(self->bitmap_buffer);

		ShowWindow(api->hwnd, SW_HIDE);
		api->renderer_type = RendererTypeNone;
	}

	bool LockBuffer(SoftwareRenderer *self, LockBufferInfo &info)
	{
		info.width = U32(self->bitmapinfo.bmiHeader.biWidth);
		info.height = U32(-self->bitmapinfo.bmiHeader.biHeight);
		info.pitch = U32(self->bitmapinfo.bmiHeader.biWidth) * 4;
		info.data = self->bitmap_buffer;
		return true;
	}

	void UnlockBuffer(SoftwareRenderer *self)
	{
		RedrawWindow(self->api->hwnd, NULL, NULL, RDW_INTERNALPAINT | RDW_INVALIDATE);
	}
}

int CALLBACK WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
	using namespace nmj;

	static PlatformAPI api;
	Initialize(&api, hInstance);

	Main(&api);
	return 0;
}

