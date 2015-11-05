#include "General.h"

#include "PlatformAPI.h"
#include "Rasterizer.h"
#include "Font.h"
#include "Vector.h"
#include "Matrix.h"
#include "MathUtils.h"

#include <stdio.h>

namespace nmj
{
	struct Camera
	{
		float3 pos;
		float3 axis[3];
		float fov;
	};

	struct Scene
	{
	};

	struct Application
	{
		PlatformAPI *api;
		SoftwareRenderer *renderer;
		Font *font;
		bool mouse_exclusive;
		float mouse_sensitivity;

		float player_yaw, player_pitch;
		U32 player_flags;

		RasterizerOutput framebuffer;

		Camera camera;
		Scene scene;
	};

	enum
	{
		PlayerFlagMoveForward = 0x00000001,
		PlayerFlagMoveBackward = 0x00000002,
		PlayerFlagMoveRight = 0x00000004,
		PlayerFlagMoveLeft = 0x00000008
	};

	void RenderScene(const Scene &scene, RasterizerOutput &output, const Camera &camera)
	{
		static const float3 vertices[8] =
		{
			float3(-1.0f, +1.0f, +1.0f),
			float3(+1.0f, +1.0f, +1.0f),
			float3(+1.0f, -1.0f, +1.0f),
			float3(-1.0f, -1.0f, +1.0f),
			float3(-1.0f, +1.0f, -1.0f),
			float3(+1.0f, +1.0f, -1.0f),
			float3(+1.0f, -1.0f, -1.0f),
			float3(-1.0f, -1.0f, -1.0f)
		};
		static const float4 colors[8] =
		{
			float4(1.0f, 1.0f, 0.0f, 0.0f),
			float4(0.0f, 1.0f, 0.0f, 0.0f),
			float4(0.0f, 0.0f, 0.0f, 0.0f),
			float4(1.0f, 0.0f, 0.0f, 0.0f),
			float4(1.0f, 1.0f, 1.0f, 0.0f),
			float4(0.0f, 1.0f, 1.0f, 0.0f),
			float4(0.0f, 0.0f, 1.0f, 0.0f),
			float4(1.0f, 0.0f, 1.0f, 0.0f)
		};
		static const U16 indices[] =
		{
			/* Front  */ 0, 1, 2, 0, 2, 3,
			/* Back   */ 4, 6, 5, 4, 7, 6,
			/* Left   */ 4, 0, 3, 4, 3, 7,
			/* Right  */ 5, 2, 1, 5, 6, 2,
			/* Top    */ 0, 4, 5, 0, 5, 1,
			/* Bottom */ 3, 2, 6, 3, 6, 7
		};

		RasterizerState state[2];
		{
			float4 object_transform[4];
			float4 camera_transform[4], camera_projection[4], view_projection[4];
			CreateCameraTransform(camera_transform, camera.pos, camera.axis);
			CreatePerspectiveProjection(camera_projection, camera.fov, float(output.width) / float(output.height), 0.01f, 800.0f);
			Mul(view_projection, camera_transform, camera_projection);

			state[0].flags = RasterizerFlagColorWrite | RasterizerFlagDepthWrite | RasterizerFlagDepthTest;
			state[1].flags = RasterizerFlagColorWrite | RasterizerFlagDepthWrite | RasterizerFlagDepthTest;

			state[0].transform[0] = view_projection[0];
			state[0].transform[1] = view_projection[1];
			state[0].transform[2] = view_projection[2];
			state[0].transform[3] = view_projection[3];

			CreateTranslate(object_transform, float3(3.0f, 0.0f, 0.0f));
			Mul(state[1].transform, object_transform, view_projection);
		}

		RasterizerInput input[2];
		input[0].state = &state[0];
		input[0].vertices = vertices;
		input[0].colors = colors;
		input[0].texcoords = NULL;
		input[0].indices = indices;
		input[0].triangle_count = 12;
		input[1].state = &state[1];
		input[1].vertices = vertices;
		input[1].colors = colors;
		input[1].texcoords = NULL;
		input[1].indices = indices;
		input[1].triangle_count = 12;

		Rasterize(output, input, 2, 0, 1);
	}

	void OnKeyboardEvent(void *userdata, KeyCode code, bool down)
	{
		Application *app = (Application *)userdata;

		if (code == KeyCodeEsc && down)
		{
			if (app->mouse_exclusive)
			{
				SetMouseCaptureMode(app->api, MouseCaptureModeShared);
				app->mouse_exclusive = false;
			}
			else
			{
				SetMouseCaptureMode(app->api, MouseCaptureModeExclusive);
				app->mouse_exclusive = true;
			}
		}

		if (code == KeyCodeW)
		{
			if (down)
				app->player_flags |= PlayerFlagMoveForward;
			else
				app->player_flags &= ~PlayerFlagMoveForward;
		}
		if (code == KeyCodeS)
		{
			if (down)
				app->player_flags |= PlayerFlagMoveBackward;
			else
				app->player_flags &= ~PlayerFlagMoveBackward;
		}
		if (code == KeyCodeA)
		{
			if (down)
				app->player_flags |= PlayerFlagMoveLeft;
			else
				app->player_flags &= ~PlayerFlagMoveLeft;
		}
		if (code == KeyCodeD)
		{
			if (down)
				app->player_flags |= PlayerFlagMoveRight;
			else
				app->player_flags &= ~PlayerFlagMoveRight;
		}
	}

	void OnMouseEvent(void *userdata, S16 delta_x, S16 delta_y, S16 delta_z, MouseButtonFlags down_state)
	{
		Application *app = (Application *)userdata;

		if (app->mouse_exclusive)
		{
			app->player_yaw -= float(delta_x) * app->mouse_sensitivity;
			app->player_pitch += float(delta_y) * app->mouse_sensitivity;

			// Keep yaw between -180 and 180 degrees.
			app->player_yaw = fmod(app->player_yaw + Pi, Tau);
			if (app->player_yaw < 0.0f)
				app->player_yaw += Tau;
			app->player_yaw -= Pi;

			// Clamp pitch between -90 and 90 degrees.
			if (app->player_pitch > Pi * 0.5f)
				app->player_pitch = Pi * 0.5f;
			else if (app->player_pitch < -Pi * 0.5f)
				app->player_pitch = -Pi * 0.5f;
		}
	}

	void Main(PlatformAPI *api)
	{
		// Application settings
		Application app;
		app.api = api;
		SetApplicationTitle(api, "CPU Rasterizer");

		// Mouse setup
		app.mouse_exclusive = true;
		app.mouse_sensitivity = 0.8f * 0.0022f; // Sensitivity * source engine scale

		// Setup player properties
		app.player_yaw = 0.0f;
		app.player_pitch = 0.0f;
		app.player_flags = 0;

		// Setup camera
		app.camera.pos = float3(0.0f, 0.0f, -5.0f);
		app.camera.fov = Tau * 0.25f;
		
		// Set events
		SetKeyboardEvent(api, &app, OnKeyboardEvent);
		SetMouseEvent(api, &app, OnMouseEvent);
		SetMouseCaptureMode(api, MouseCaptureModeExclusive);

		// Create software renderer
		app.renderer = CreateSoftwareRenderer(api, 1280, 720, false);

		// Load default font.
		app.font = CreateFont("C:\\Windows\\Fonts\\calibrib.ttf", 18.0f);

		// Initialize the default framebuffer.
		{
			app.framebuffer.width = 1280;
			app.framebuffer.height = 720;

			U32 size = GetRequiredMemoryAmount(app.framebuffer, true, true);
			Initialize(app.framebuffer, malloc(size), true, true);
		}

		// Frame update loop
		U64 frame_start_time = GetTime(api);
		float frame_delta = 0.0001f;
		while (Update(api))
		{
			// Render the frame
			LockBufferInfo frame_info;
			LockBuffer(app.renderer, frame_info);
			{
				// Apply player rotation to camera.
				app.camera.axis[0] = float3(1.0f, 0.0f, 0.0f);
				app.camera.axis[1] = float3(0.0f, 1.0f, 0.0f);
				app.camera.axis[2] = float3(0.0f, 0.0f, 1.0f);
				Rotate(app.camera.axis[0], float3(1.0f, 0.0f, 0.0f), app.player_pitch);
				Rotate(app.camera.axis[1], float3(1.0f, 0.0f, 0.0f), app.player_pitch);
				Rotate(app.camera.axis[2], float3(1.0f, 0.0f, 0.0f), app.player_pitch);
				Rotate(app.camera.axis[0], float3(0.0f, 1.0f, 0.0f), app.player_yaw);
				Rotate(app.camera.axis[1], float3(0.0f, 1.0f, 0.0f), app.player_yaw);
				Rotate(app.camera.axis[2], float3(0.0f, 1.0f, 0.0f), app.player_yaw);

				// Apply player movement to camera
				float3 player_velocity = 0.0f;
				if (app.player_flags & PlayerFlagMoveForward)
					player_velocity += app.camera.axis[2];
				if (app.player_flags & PlayerFlagMoveBackward)
					player_velocity -= app.camera.axis[2];
				if (app.player_flags & PlayerFlagMoveRight)
					player_velocity -= app.camera.axis[0];
				if (app.player_flags & PlayerFlagMoveLeft)
					player_velocity += app.camera.axis[0];
				if (Dot(player_velocity, player_velocity) != 0.0f)
					Normalize(player_velocity);
				player_velocity *= 5.0f * frame_delta;
				app.camera.pos += player_velocity;

				// Clear frame buffers
				U64 clear_buffers_time = GetTime(api);
				ClearColor(app.framebuffer, float4(0.0f), 0, 1);
				ClearDepth(app.framebuffer, 0.0f, 0, 1);
				clear_buffers_time = GetTime(api) - clear_buffers_time;

				// Render scene
				U64 render_scene_time = GetTime(api);
				RenderScene(app.scene, app.framebuffer, app.camera);
				render_scene_time = GetTime(api) - render_scene_time;

				// Blit scene to screen.
				U64 blit_time = GetTime(api);
				Blit(frame_info, app.framebuffer, 0, 1);
				blit_time = GetTime(api) - blit_time;

				// Print debug stats
				{
					// unsafe
					static char buffer[1024];
					U32 line = 0;

					sprintf_s(buffer, sizeof buffer, "FPS: %.2f (%.2fms)", 1.0f / frame_delta, frame_delta * 1000.0f);
					RenderText(app.font, frame_info, 0, 18 * line++, buffer, float4(1.0f, 0.0f, 0.0f, 0.0f));

					sprintf_s(buffer, sizeof buffer, "ClearBuffers: %.3fms", (float(clear_buffers_time) / float(U64(1) << U64(32))) * 1000.0f);
					RenderText(app.font, frame_info, 0, 18 * line++, buffer, float4(1.0f, 0.0f, 0.0f, 0.0f));
					sprintf_s(buffer, sizeof buffer, "RenderScene: %.3fms", (float(render_scene_time) / float(U64(1) << U64(32))) * 1000.0f);
					RenderText(app.font, frame_info, 0, 18 * line++, buffer, float4(1.0f, 0.0f, 0.0f, 0.0f));
					sprintf_s(buffer, sizeof buffer, "Blit: %.3fms", (float(blit_time) / float(U64(1) << U64(32))) * 1000.0f);
					RenderText(app.font, frame_info, 0, 18 * line++, buffer, float4(1.0f, 0.0f, 0.0f, 0.0f));

					sprintf_s(buffer, sizeof buffer, "Position: [%.2f, %.2f, %.2f]", app.camera.pos.x, app.camera.pos.y, app.camera.pos.z);
					RenderText(app.font, frame_info, 0, 18 * line++, buffer, float4(1.0f, 0.0f, 0.0f, 0.0f));
					sprintf_s(buffer, sizeof buffer, "Yaw: %.2f", app.player_yaw / Pi * 180.0f);
					RenderText(app.font, frame_info, 0, 18 * line++, buffer, float4(1.0f, 0.0f, 0.0f, 0.0f));
					sprintf_s(buffer, sizeof buffer, "Pitch: %.2f", app.player_pitch / Pi * 180.0f);
					RenderText(app.font, frame_info, 0, 18 * line++, buffer, float4(1.0f, 0.0f, 0.0f, 0.0f));

					sprintf_s(buffer, sizeof buffer, "Axis X: [%.2f, %.2f, %.2f]", app.camera.axis[0].x, app.camera.axis[0].y, app.camera.axis[0].z);
					RenderText(app.font, frame_info, 0, 18 * line++, buffer, float4(1.0f, 0.0f, 0.0f, 0.0f));
					sprintf_s(buffer, sizeof buffer, "Axis Y: [%.2f, %.2f, %.2f]", app.camera.axis[1].x, app.camera.axis[1].y, app.camera.axis[1].z);
					RenderText(app.font, frame_info, 0, 18 * line++, buffer, float4(1.0f, 0.0f, 0.0f, 0.0f));
					sprintf_s(buffer, sizeof buffer, "Axis Z: [%.2f, %.2f, %.2f]", app.camera.axis[2].x, app.camera.axis[2].y, app.camera.axis[2].z);
					RenderText(app.font, frame_info, 0, 18 * line++, buffer, float4(1.0f, 0.0f, 0.0f, 0.0f));
				}

				// Calculate frame delta time
				{
					U64 time = GetTime(api);
					U64 delta = time - frame_start_time;
					frame_start_time = time;

					frame_delta = float(delta) / float(U64(1) << U64(32));
				}
			}
			UnlockBuffer(app.renderer);
		}
	}
}

