// display_test.cpp - C++ OpenGL vsync rendering test (Python オーバーヘッドなし)
// Build: bash build_test.sh
// Usage: ./display_test <image_path>
//
// slideshow.py と同じアルゴリズム（整数px/frameパン＋モーションブラー）を
// C++ で実装し、波打ちが Python 起因かどうかを確認する。

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <SDL2/SDL.h>
#include <GL/gl.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>
using namespace std;
using namespace std::chrono;

static constexpr double DURATION = 10.0;  // 1枚あたりの表示秒数
static constexpr int    BLUR_N   = 4;     // モーションブラーサンプル数

// vsync flip を計測してリフレッシュレートを返す（GL 初期化後に呼ぶこと）
static int detect_fps(SDL_Window* win) {
    for (int i = 0; i < 3; i++) { glClear(GL_COLOR_BUFFER_BIT); SDL_GL_SwapWindow(win); }
    const int N = 10;
    auto t0 = steady_clock::now();
    for (int i = 0; i < N; i++) { glClear(GL_COLOR_BUFFER_BIT); SDL_GL_SwapWindow(win); }
    double m = N / duration<double>(steady_clock::now() - t0).count();
    for (int f : {24,30,48,60,75,90,120,144,165,240})
        if (fabs(m - f) / f < 0.05) return f;
    return (int)round(m);
}

int main(int argc, char* argv[]) {
    if (argc < 2) { fprintf(stderr, "usage: %s <image>\n", argv[0]); return 1; }

    if (SDL_Init(SDL_INIT_VIDEO) < 0) { fprintf(stderr, "SDL: %s\n", SDL_GetError()); return 1; }
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 0);

    SDL_Window* win = SDL_CreateWindow("display_test",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        0, 0, SDL_WINDOW_FULLSCREEN_DESKTOP | SDL_WINDOW_OPENGL);
    if (!win) { fprintf(stderr, "Window: %s\n", SDL_GetError()); return 1; }

    SDL_GLContext ctx = SDL_GL_CreateContext(win);
    // vsync: 1=on, -1=adaptive(失敗したら1にフォールバック)
    if (SDL_GL_SetSwapInterval(-1) < 0) SDL_GL_SetSwapInterval(1);
    SDL_ShowCursor(SDL_DISABLE);

    // HiDPI 環境では SDL_GL_GetDrawableSize が論理サイズを返すことがある。
    // Python 版と同様に OpenGL のデフォルトビューポートから実フレームバッファサイズを取得する。
    GLint vp[4];
    glGetIntegerv(GL_VIEWPORT, vp);
    int sw = vp[2], sh = vp[3];

    // OpenGL 初期化（UV 空間: (0,0)=左下, (1,1)=右上）
    glViewport(0, 0, sw, sh);
    glMatrixMode(GL_PROJECTION); glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
    glEnable(GL_TEXTURE_2D);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    glClearColor(0, 0, 0, 1);

    int fps = detect_fps(win);
    printf("Display: %dx%d  FPS: %d\n", sw, sh, fps);

    // 画像ロード（Y 軸を反転して OpenGL に合わせる）
    int iw, ih;
    stbi_set_flip_vertically_on_load(1);
    unsigned char* pix = stbi_load(argv[1], &iw, &ih, nullptr, 4);
    if (!pix) { fprintf(stderr, "load failed: %s\n", argv[1]); return 1; }

    // cover スケール・パン計算
    double sc  = max((double)sw / iw, (double)sh / ih);
    int    tw  = (int)ceil(iw * sc);   // 仮想キャンバス幅
    int    th  = (int)ceil(ih * sc);   // 仮想キャンバス高
    double iw_sc = iw * sc;            // = tw (float 精度)
    double ih_sc = ih * sc;            // = th (float 精度)

    bool horiz    = (iw * sh > ih * sw);
    int  pan_dist = horiz ? tw - sw : th - sh;
    int  ppf      = max(1, (int)round((double)pan_dist / (DURATION * fps)));
    int  pan_frames = (int)ceil((double)pan_dist / ppf);
    double slide_dur = (double)pan_frames / fps;
    double frame_dt  = 1.0 / (slide_dur * fps);  // t 空間での 1 フレーム幅

    // 非パン軸のマージン（UV 計算用）
    double x_margin = (tw - sw) / 2.0;
    double y_margin = (th - sh) / 2.0;

    printf("Image: %dx%d  scale:%.3f  canvas:%dx%d  pan:%s  ppf:%d  dur:%.2fs\n",
        iw, ih, sc, tw, th, horiz ? "H" : "V", ppf, slide_dur);

    // テクスチャアップロード（オリジナルサイズ; UV 計算で cover スケールを表現）
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, iw, ih, 0, GL_RGBA, GL_UNSIGNED_BYTE, pix);
    stbi_image_free(pix);

    // レンダーループ
    bool   running = true;
    double elapsed = 0.0;
    auto   t_start = steady_clock::now();
    auto   t_last  = t_start;
    vector<double> ftimes;
    ftimes.reserve(fps + 1);

    while (running) {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) running = false;
            if (ev.type == SDL_KEYDOWN &&
                (ev.key.keysym.sym == SDLK_ESCAPE || ev.key.keysym.sym == SDLK_q))
                running = false;
        }

        double t = min(elapsed / slide_dur, 1.0);
        float  a = 1.0f / BLUR_N;

        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE);  // 加算ブレンド → 平均
        glColor4f(a, a, a, 1.f);
        glBindTexture(GL_TEXTURE_2D, tex);

        for (int i = 0; i < BLUR_N; i++) {
            // 現フレームの期間 [t, t+frame_dt] を N 等分してサンプル
            double ts  = min(t + (i + 0.5) / BLUR_N * frame_dt, 1.0);
            double off = min(ts * slide_dur * fps * ppf, (double)pan_dist);

            double u0, u1, v_top, v_bot;
            if (horiz) {
                // 水平パン: 左→右
                u0    =  off          / iw_sc;
                u1    = (off + sw)    / iw_sc;
                v_top = 1.0 - y_margin        / ih_sc;
                v_bot = 1.0 - (y_margin + sh) / ih_sc;
            } else {
                // 垂直パン: 上→下
                u0    =  x_margin         / iw_sc;
                u1    = (x_margin + sw)   / iw_sc;
                v_top = 1.0 - off         / ih_sc;
                v_bot = 1.0 - (off + sh)  / ih_sc;
            }

            glBegin(GL_QUADS);
            glTexCoord2f(u0, v_bot); glVertex2f(0, 0);
            glTexCoord2f(u1, v_bot); glVertex2f(1, 0);
            glTexCoord2f(u1, v_top); glVertex2f(1, 1);
            glTexCoord2f(u0, v_top); glVertex2f(0, 1);
            glEnd();
        }

        glDisable(GL_BLEND);
        glColor4f(1, 1, 1, 1);

        SDL_GL_SwapWindow(win);  // vsync まで待機

        auto   now = steady_clock::now();
        double ft  = duration<double>(now - t_last).count();
        t_last = now;
        ftimes.push_back(ft);
        if ((int)ftimes.size() > fps) ftimes.erase(ftimes.begin());

        elapsed = duration<double>(now - t_start).count();
        if (elapsed >= slide_dur) { t_start = now; elapsed = 0.0; }  // ループ

        if ((int)ftimes.size() >= 2) {
            double sum = 0, worst = 0; int drops = 0;
            double exp_ft = 1.0 / fps;
            for (double f : ftimes) {
                sum += f;
                if (f > worst) worst = f;
                if (f > exp_ft * 1.5) drops++;
            }
            printf("\rFPS:%.1f  worst:%.1fms  drops(>%.0fms):%d/%d     ",
                1.0 / (sum / ftimes.size()), worst * 1e3, exp_ft * 1.5e3,
                drops, (int)ftimes.size());
            fflush(stdout);
        }
    }

    puts("");
    glDeleteTextures(1, &tex);
    SDL_GL_DeleteContext(ctx);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
