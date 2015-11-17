#include "General.h"

#include "PlatformAPI.h"
#include "Rasterizer.h"
#include "Font.h"
#include "Vector.h"
#include "Matrix.h"
#include "MathUtils.h"

#include <windows.h>
#include <process.h>
#include <stdio.h>
#include <vector>

namespace nmj
{
	// NOTE: We should probably query the CPU for it's cores and hyper-threading and decide this
	//       based on that information.
	enum { DefaultThreadAmount = 8 };

	struct Application;

	struct Camera
	{
		float3 pos;
		float3 axis[3];
		float fov;
	};

	struct Model
	{
		float3 *vertex_pos;
		float4 *vertex_color;
		U16 *indices;

		U32 triangle_count;
	};

	struct SceneObject
	{
		const Model *model;
		float4 transform[4];
	};

	struct Scene
	{
		std::vector<SceneObject> objects;
	};

	struct ThreadData
	{
		U32 index;
		Application *app;
	};

	struct Application
	{
		PlatformAPI *api;
		SoftwareRenderer *renderer;
		Font *font;
		bool mouse_exclusive;
		float mouse_sensitivity;
		float frame_delta;

		// Player info
		float player_yaw, player_pitch;
		U32 player_flags;

		// Rasterizer
		U32 rasterizer_event_id;
		HANDLE start_rasterization_event[2];
		HANDLE rasterizer_threads[DefaultThreadAmount];
		HANDLE rasterization_finished_event[DefaultThreadAmount];
		ThreadData rasterizer_data[DefaultThreadAmount];
		RasterizerOutput framebuffer;
		std::vector<RasterizerInput> rasterizer_input;

		// Profiler
		U64 clear_buffers_time;
		U64 render_scene_time;
		U64 blit_time;

		// Game world
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

	void CreateTestScene(Scene &scene)
	{
		static float3 vertices[8] =
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
		static float4 colors[8] =
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
		static U16 indices[] =
		{
			/* Front  */ 0, 1, 2, 0, 2, 3,
			/* Back   */ 4, 6, 5, 4, 7, 6,
			/* Left   */ 4, 0, 3, 4, 3, 7,
			/* Right  */ 5, 2, 1, 5, 6, 2,
			/* Top    */ 0, 4, 5, 0, 5, 1,
			/* Bottom */ 3, 2, 6, 3, 6, 7
		};
		static Model box_model =
		{
			vertices,
			colors,
			indices,
			12
		};

		scene.objects.resize(2);
		scene.objects[0].model = &box_model;
		scene.objects[1].model = &box_model;

		CreateIdentity(scene.objects[0].transform);
		CreateTranslate(scene.objects[1].transform, float3(3.0f, 0.0f, 0.0f));
	}

	unsigned int (__stdcall RasterizerThread)(void *userdata)
	{
		ThreadData *thread_data = (ThreadData *)userdata;
		U32 thread_index = thread_data->index;
		Application &app = *thread_data->app;

		U32 event_id = 0;
		for (;;)
		{
			WaitForSingleObject(app.start_rasterization_event[event_id], INFINITE);
			event_id = (event_id + 1) % 2;

			RasterizerState state;
			state.flags = RasterizerFlagColorWrite | RasterizerFlagDepthWrite | RasterizerFlagDepthTest;
			state.output = &app.framebuffer;
			Rasterize(state, app.rasterizer_input.data(), U32(app.rasterizer_input.size()), thread_index, DefaultThreadAmount);

			SetEvent(app.rasterization_finished_event[thread_index]);
		}
		// return 0;
	}

	void CreateRasterizerThreads(Application &app)
	{
		app.rasterizer_event_id = 0;
		app.start_rasterization_event[0] = CreateEvent(NULL, TRUE, FALSE, NULL);
		app.start_rasterization_event[1] = CreateEvent(NULL, TRUE, FALSE, NULL);

		for (U32 i = 0; i < DefaultThreadAmount; ++i)
		{
			app.rasterizer_data[i].index = i;
			app.rasterizer_data[i].app = &app;

			app.rasterization_finished_event[i] = CreateEvent(NULL, FALSE, FALSE, NULL);

			app.rasterizer_threads[i] = (HANDLE)_beginthreadex(NULL, 0, RasterizerThread, &app.rasterizer_data[i], 0, NULL);
		}
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

	void Build(std::vector<RasterizerInput> &self, const Scene &scene, float4 (&view_projection)[4])
	{
		U32 index = 0;
		self.resize(scene.objects.size());

		for (auto object : scene.objects)
		{
			const Model *model = object.model;

			RasterizerInput &ri = self[index++];
			ri.vertices = model->vertex_pos;
			ri.colors = model->vertex_color;
			ri.texcoords = NULL;
			ri.indices = model->indices;
			ri.triangle_count = model->triangle_count;

			Mul(ri.transform, object.transform, view_projection);
		}
	}

	void RenderFrame(Application &app, LockBufferInfo &frame_info)
	{
		// Clear frame buffers
		app.clear_buffers_time = GetTime(app.api);
		ClearColor(app.framebuffer, float4(0.0f), 0, 1);
		ClearDepth(app.framebuffer, 1.0f, 0, 1);
		app.clear_buffers_time = GetTime(app.api) - app.clear_buffers_time;

		// Render scene
		app.render_scene_time = GetTime(app.api);
		{
			// Calculate view_projection matrix.
			float4 camera_transform[4], camera_projection[4], view_projection[4];
			CreateCameraTransform(camera_transform, app.camera.pos, app.camera.axis);
			CreatePerspectiveProjection(camera_projection, app.camera.fov, float(app.framebuffer.width) / float(app.framebuffer.height), 0.1f, 100.0f);
			Mul(view_projection, camera_transform, camera_projection);

			// Build rasterizer input commands.
			Build(app.rasterizer_input, app.scene, view_projection);

			// Start the rasterizer threads
			U32 event_id = app.rasterizer_event_id;
			SetEvent(app.start_rasterization_event[event_id]);

			// Wait for the threads to finish.
			WaitForMultipleObjects(DefaultThreadAmount, app.rasterization_finished_event, TRUE, INFINITE);
			ResetEvent(app.start_rasterization_event[event_id]);
			app.rasterizer_event_id = (event_id + 1) % 2;
		}
		app.render_scene_time = GetTime(app.api) - app.render_scene_time;

		// Blit scene to screen.
		app.blit_time = GetTime(app.api);
		Blit(frame_info, app.framebuffer, 0, 1);
		app.blit_time = GetTime(app.api) - app.blit_time;
	}

	void PrintDebugStats(const Application &app, LockBufferInfo &frame_info)
	{
		// unsafe
		static char buffer[1024];
		U32 line = 0;

		sprintf_s(buffer, sizeof buffer, "FPS: %.2f (%.2fms)", 1.0f / app.frame_delta, app.frame_delta * 1000.0f);
		RenderText(app.font, frame_info, 0, 18 * line++, buffer, float4(1.0f, 0.0f, 0.0f, 0.0f));

		sprintf_s(buffer, sizeof buffer, "ClearBuffers: %.3fms", (float(app.clear_buffers_time) / float(U64(1) << U64(32))) * 1000.0f);
		RenderText(app.font, frame_info, 0, 18 * line++, buffer, float4(1.0f, 0.0f, 0.0f, 0.0f));
		sprintf_s(buffer, sizeof buffer, "RenderScene: %.3fms", (float(app.render_scene_time) / float(U64(1) << U64(32))) * 1000.0f);
		RenderText(app.font, frame_info, 0, 18 * line++, buffer, float4(1.0f, 0.0f, 0.0f, 0.0f));
		sprintf_s(buffer, sizeof buffer, "Blit: %.3fms", (float(app.blit_time) / float(U64(1) << U64(32))) * 1000.0f);
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

		// Setup scene
		CreateTestScene(app.scene);
		
		// Set events
		SetKeyboardEvent(api, &app, OnKeyboardEvent);
		SetMouseEvent(api, &app, OnMouseEvent);
		SetMouseCaptureMode(api, MouseCaptureModeExclusive);

		// Create software renderer
		app.renderer = CreateSoftwareRenderer(api, 1280, 720, false);

		// Load default font.
		app.font = CreateFontFromFile("C:\\Windows\\Fonts\\calibrib.ttf", 18.0f);

		// Initialize the rasterizer data
		{
			// Default framebuffer
			app.framebuffer.width = 1280;
			app.framebuffer.height = 720;
			U32 size = GetRequiredMemoryAmount(app.framebuffer, true, true);
			Initialize(app.framebuffer, malloc(size), true, true);

			// Threads
			CreateRasterizerThreads(app);
		}

		// Frame update loop
		U64 frame_start_time = GetTime(api);
		app.frame_delta = 0.0001f;
		while (Update(api))
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
			player_velocity *= 5.0f * app.frame_delta;
			app.camera.pos += player_velocity;

			// Render the frame
			LockBufferInfo frame_info;
			LockBuffer(app.renderer, frame_info);
			RenderFrame(app, frame_info);
			PrintDebugStats(app, frame_info);
			UnlockBuffer(app.renderer);

			// Calculate frame delta time
			U64 time = GetTime(api);
			U64 delta = time - frame_start_time;
			frame_start_time = time;
			app.frame_delta = float(delta) / float(U64(1) << U64(32));
		}
	}
}

