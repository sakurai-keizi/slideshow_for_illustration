// display_server.cpp - C++ OpenGL display server for slideshow.py
//
// Protocol (Python → stdin, binary):
//   Init:      [uint32 fps]
//   Per image: [uint32 w][uint32 h][uint32 ppf][uint32 dir][uint32 dur_ms][w*h*4 RGBA bytes]
//   dir: 0=left_to_right, 1=right_to_left, 2=top_to_bottom, 3=bottom_to_top
//   Pixels are already vertically flipped for OpenGL (row 0 = bottom).
//
// Protocol (C++ → stdout, text):
//   "READY\n"  - ready for next image (request it now)
//   "QUIT\n"   - user pressed ESC/Q

#include <GL/glew.h>
#include <SDL2/SDL.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

using namespace std::chrono;

static constexpr int BLUR_N = 4;

struct ImageData {
    uint32_t w = 0, h = 0, ppf = 0, dir = 0, dur_ms = 0;
    std::vector<uint8_t> pixels;
    bool valid = false;
};

static std::mutex              g_mtx;
static std::condition_variable g_cv_main;   // notify main: next image ready
static std::condition_variable g_cv_io;     // notify IO: main wants next image
static ImageData               g_next;
static bool                    g_next_ready = false;
static bool                    g_want_next  = false;
static std::atomic<bool>       g_quit{false};

static bool read_all(void* buf, size_t n) {
    char* p = static_cast<char*>(buf);
    while (n > 0) {
        size_t r = fread(p, 1, n, stdin);
        if (r == 0) return false;
        p += r; n -= r;
    }
    return true;
}

static void io_thread_func() {
    while (!g_quit) {
        {
            std::unique_lock<std::mutex> lk(g_mtx);
            g_cv_io.wait(lk, [] { return g_want_next || g_quit.load(); });
            if (g_quit) return;
            g_want_next = false;
        }

        fprintf(stdout, "READY\n");
        fflush(stdout);

        uint32_t hdr[5];
        if (!read_all(hdr, sizeof(hdr))) {
            g_quit = true; g_cv_main.notify_all(); return;
        }

        ImageData img;
        img.w = hdr[0]; img.h = hdr[1]; img.ppf = hdr[2];
        img.dir = hdr[3]; img.dur_ms = hdr[4];
        img.valid = true;
        img.pixels.resize((size_t)img.w * img.h * 4);

        if (!read_all(img.pixels.data(), img.pixels.size())) {
            g_quit = true; g_cv_main.notify_all(); return;
        }

        {
            std::lock_guard<std::mutex> lk(g_mtx);
            g_next = std::move(img);
            g_next_ready = true;
        }
        g_cv_main.notify_one();
    }
}

static int detect_fps(SDL_Window* win) {
    for (int i = 0; i < 3; i++) { glClear(GL_COLOR_BUFFER_BIT); SDL_GL_SwapWindow(win); }
    const int N = 10;
    auto t0 = steady_clock::now();
    for (int i = 0; i < N; i++) { glClear(GL_COLOR_BUFFER_BIT); SDL_GL_SwapWindow(win); }
    double m = N / duration<double>(steady_clock::now() - t0).count();
    for (int f : {24, 30, 48, 60, 75, 90, 120, 144, 165, 240})
        if (fabs(m - f) / f < 0.05) return f;
    return (int)round(m);
}

static void upload_texture(GLuint tex, const ImageData& img) {
    // PBO 経由でアップロード: CPU→PBO コピー後すぐにリターンし、
    // PBO→テクスチャ転送は GPU が非同期で行う。
    // これによりメインスレッドのブロック時間を最小化し、フレーム遅延を防ぐ。
    GLuint pbo;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, (GLsizeiptr)img.pixels.size(),
                 img.pixels.data(), GL_STREAM_DRAW);

    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    // nullptr: PBO バインド中は GPU が PBO からテクスチャへ非同期で転送
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.w, img.h, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glDeleteBuffers(1, &pbo);  // OpenGL の参照カウントにより GPU 転送完了まで実際には解放されない
}

int main() {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "SDL init: %s\n", SDL_GetError()); return 1;
    }
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 0);

    SDL_Window* win = SDL_CreateWindow("Slideshow",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        0, 0, SDL_WINDOW_FULLSCREEN_DESKTOP | SDL_WINDOW_OPENGL);
    if (!win) { fprintf(stderr, "Window: %s\n", SDL_GetError()); SDL_Quit(); return 1; }

    SDL_GLContext ctx = SDL_GL_CreateContext(win);
    glewExperimental = GL_TRUE;
    glewInit();
    SDL_GL_SetSwapInterval(1);  // 通常vsync（アダプティブvsyncは遅延時にティアリングを起こすため使わない）
    SDL_ShowCursor(SDL_DISABLE);

    GLint vp[4];
    glGetIntegerv(GL_VIEWPORT, vp);
    int sw = vp[2], sh = vp[3];

    glViewport(0, 0, sw, sh);
    glMatrixMode(GL_PROJECTION); glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
    glEnable(GL_TEXTURE_2D);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    glClearColor(0, 0, 0, 1);

    int fps = detect_fps(win);
    fprintf(stderr, "Display: %dx%d  FPS: %d\n", sw, sh, fps);

    // Report actual screen dimensions and fps to Python before any READY
    fprintf(stdout, "SCREEN %d %d %d\n", sw, sh, fps);
    fflush(stdout);

    // Kick off IO thread to request first image
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        g_want_next = true;
    }
    std::thread io_th(io_thread_func);

    // Wait for first image
    {
        std::unique_lock<std::mutex> lk(g_mtx);
        g_cv_main.wait(lk, [] { return g_next_ready || g_quit.load(); });
    }
    if (g_quit) {
        fprintf(stdout, "QUIT\n"); fflush(stdout);
        io_th.join();
        SDL_GL_DeleteContext(ctx); SDL_DestroyWindow(win); SDL_Quit();
        return 0;
    }

    ImageData cur;
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        cur = std::move(g_next);
        g_next_ready = false;
        // Immediately prefetch next image
        g_want_next = true;
    }
    g_cv_io.notify_one();

    GLuint tex = 0;
    glGenTextures(1, &tex);
    upload_texture(tex, cur);

    auto  t_start   = steady_clock::now();
    bool  slide_done = false;

    while (!g_quit) {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) g_quit = true;
            if (ev.type == SDL_KEYDOWN &&
                (ev.key.keysym.sym == SDLK_ESCAPE || ev.key.keysym.sym == SDLK_q))
                g_quit = true;
        }
        if (g_quit) break;

        double slide_dur = cur.dur_ms / 1000.0;
        double elapsed   = duration<double>(steady_clock::now() - t_start).count();
        double t         = std::min(elapsed / slide_dur, 1.0);
        double frame_dt  = 1.0 / (slide_dur * fps);
        double iw        = cur.w, ih = cur.h;
        double pan_dist  = (cur.dir < 2) ? iw - sw : ih - sh;
        if (pan_dist < 0) pan_dist = 0;
        float a = 1.0f / BLUR_N;

        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE);
        glColor4f(a, a, a, 1.f);
        glBindTexture(GL_TEXTURE_2D, tex);

        for (int i = 0; i < BLUR_N; i++) {
            double ts     = std::min(t + (i + 0.5) / BLUR_N * frame_dt, 1.0);
            double offset = ts * pan_dist;   // continuous sub-pixel for smooth blur

            double u0, u1, v_top, v_bot;
            if (cur.dir == 0 || cur.dir == 1) {
                // horizontal pan
                double x0 = (cur.dir == 0) ? offset : pan_dist - offset;
                double y0 = (ih - sh) / 2.0;
                u0    = x0 / iw; u1    = (x0 + sw) / iw;
                v_top = 1.0 - y0 / ih; v_bot = 1.0 - (y0 + sh) / ih;
            } else {
                // vertical pan
                double x0 = (iw - sw) / 2.0;
                double y0 = (cur.dir == 2) ? offset : pan_dist - offset;
                u0    = x0 / iw; u1    = (x0 + sw) / iw;
                v_top = 1.0 - y0 / ih; v_bot = 1.0 - (y0 + sh) / ih;
            }

            glBegin(GL_QUADS);
            glTexCoord2f((float)u0, (float)v_bot); glVertex2f(0, 0);
            glTexCoord2f((float)u1, (float)v_bot); glVertex2f(1, 0);
            glTexCoord2f((float)u1, (float)v_top); glVertex2f(1, 1);
            glTexCoord2f((float)u0, (float)v_top); glVertex2f(0, 1);
            glEnd();
        }

        glDisable(GL_BLEND);
        glColor4f(1, 1, 1, 1);

        SDL_GL_SwapWindow(win);   // vsync wait

        // Advance to next image when slide done and prefetch is ready
        if (!slide_done && elapsed >= slide_dur) slide_done = true;
        if (slide_done) {
            bool have_next;
            {
                std::lock_guard<std::mutex> lk(g_mtx);
                have_next = g_next_ready;
            }
            if (have_next) {
                {
                    std::lock_guard<std::mutex> lk(g_mtx);
                    cur = std::move(g_next);
                    g_next_ready = false;
                    g_want_next  = true;   // prefetch following image
                }
                g_cv_io.notify_one();
                upload_texture(tex, cur);
                t_start    = steady_clock::now();
                slide_done = false;
            }
            // else: keep rendering last frame until prefetch finishes
        }
    }

    fprintf(stdout, "QUIT\n"); fflush(stdout);
    g_quit = true;
    g_cv_io.notify_all();
    io_th.join();

    glDeleteTextures(1, &tex);
    SDL_GL_DeleteContext(ctx);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
