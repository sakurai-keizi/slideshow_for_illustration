#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pygame",
#   "opencv-python",
#   "realesrgan",
#   "basicsr",
#   "torch",
#   "torchvision",
#   "PyOpenGL",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu121"
# url = "https://download.pytorch.org/whl/cu121"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu121" }
# torchvision = { index = "pytorch-cu121" }
# ///
"""
Slideshow - 指定フォルダ以下の画像を全画面でケンバーンズ風に表示する。
使い方: python slideshow.py <フォルダパス>
終了: ESC または Q キー
"""

import ctypes
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pygame
from OpenGL.GL import (
    GL_BLEND, GL_CLAMP_TO_EDGE, GL_COLOR_BUFFER_BIT, GL_LINEAR,
    GL_MODELVIEW, GL_MODULATE, GL_ONE, GL_PROJECTION, GL_QUADS,
    GL_RGB, GL_TEXTURE_2D, GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE,
    GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_UNPACK_ALIGNMENT, GL_UNSIGNED_BYTE, GL_VIEWPORT,
    glBegin, glBindTexture, glBlendFunc, glClear, glClearColor,
    glColor4f, glDeleteTextures, glDisable, glEnable, glEnd,
    glGenTextures, glGetIntegerv, glLoadIdentity, glMatrixMode,
    glPixelStorei, glTexCoord2f, glTexEnvf, glTexImage2D, glTexParameteri,
    glVertex2f, glViewport,
)
from OpenGL.GLU import gluOrtho2D

DURATION = 10.0          # 1枚あたりの表示秒数
MOTION_BLUR_SAMPLES = 4  # モーションブラーのサンプル数（多いほど滑らか、4で十分）
DEBUG_HORIZONTAL_ONLY = False  # デバッグ: True のとき横パン画像のみ表示（縦パン画像はスキップ）

MODEL_URL = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'
MODEL_PATH = Path.home() / '.cache' / 'realesrgan' / 'RealESRGAN_x4plus_anime_6B.pth'

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}


def get_display_fps(fallback: int = 60) -> int:
    """xrandr からディスプレイの現在のリフレッシュレートを取得する。取得できない場合は fallback を返す。"""
    try:
        out = subprocess.check_output(['xrandr'], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if '*' in line:
                m = re.search(r'(\d+\.\d+)\*', line)
                if m:
                    return round(float(m.group(1)))
    except Exception:
        pass
    print(f'警告: リフレッシュレートを取得できませんでした。{fallback}fps を使用します。', flush=True)
    return fallback


def collect_images(folder: str) -> list[str]:
    return [
        str(p) for p in Path(folder).rglob('*')
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def cover_scale(iw: int, ih: int, sw: int, sh: int) -> float:
    """画像がスクリーン全体を覆うスケール係数（CSS cover 相当）"""
    return max(sw / iw, sh / ih)


def _patch_torchvision() -> None:
    """torchvision 0.17+ で削除された functional_tensor を basicsr 向けに補完"""
    if 'torchvision.transforms.functional_tensor' not in sys.modules:
        import types
        import torchvision.transforms.functional as F
        mod = types.ModuleType('torchvision.transforms.functional_tensor')
        mod.rgb_to_grayscale = F.rgb_to_grayscale
        sys.modules['torchvision.transforms.functional_tensor'] = mod


def init_esrgan():
    _patch_torchvision()
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        print('Real-ESRGANモデルをダウンロード中...', flush=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print('ダウンロード完了', flush=True)

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=6, num_grow_ch=32, scale=4)
    return RealESRGANer(
        scale=4,
        model_path=str(MODEL_PATH),
        model=model,
        tile=512,
        tile_pad=10,
        pre_pad=0,
        half=True,   # FP16: RTX で高速化
        gpu_id=0,
    )


class SlideShow:
    def __init__(self, folder: str):
        print('Real-ESRGANを初期化中...', flush=True)
        self.upsampler = init_esrgan()
        print('初期化完了', flush=True)

        self._init_pygame()
        self._init_gl()
        self.fps = get_display_fps()
        print(f'ディスプレイ検出FPS: {self.fps}', flush=True)
        self._init_images(folder)

        self.last_pattern: str | None = None
        self.current_pattern: str = ''
        self.pan_tex: int | None = None        # OpenGL テクスチャ ID
        self.pan_tex_size: tuple[int, int] = (0, 0)  # (w, h)
        self.start_time: float = 0.0
        self.clock = pygame.time.Clock()

        self.pan_px_per_frame: int = 0
        self.slide_duration: float = DURATION

        self._prefetch_arr: np.ndarray | None = None
        self._prefetch_pattern: str = ''
        self._prefetch_pan_px_per_frame: int = 0
        self._prefetch_slide_duration: float = DURATION
        self._prefetch_ready = threading.Event()
        self._start_prefetch()

    def _init_pygame(self) -> None:
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_SWAP_CONTROL, 1)  # vsync
        self.screen = pygame.display.set_mode(
            (0, 0), pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.OPENGL
        )
        # HiDPI では get_size() が論理サイズを返すため、OpenGL のデフォルト
        # ビューポートから物理フレームバッファの実サイズを取得する
        vp = (ctypes.c_int * 4)()
        glGetIntegerv(GL_VIEWPORT, vp)
        self.sw, self.sh = int(vp[2]), int(vp[3])
        pygame.mouse.set_visible(False)
        pygame.display.set_caption('Slideshow')

    def _init_gl(self) -> None:
        """OpenGL の基本設定"""
        glViewport(0, 0, self.sw, self.sh)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0.0, 1.0, 0.0, 1.0)  # UV 空間: (0,0)=左下, (1,1)=右上
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glEnable(GL_TEXTURE_2D)
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)  # 頂点カラーでテクスチャをスケール
        glClearColor(0.0, 0.0, 0.0, 1.0)

    def _init_images(self, folder: str) -> None:
        self.all_images = collect_images(folder)
        if not self.all_images:
            print(f'画像が見つかりません: {folder}')
            pygame.quit()
            sys.exit(1)
        self.queue: list[str] = []
        self.shown: list[str] = []
        self._refill_queue()

    def _refill_queue(self) -> None:
        self.queue = self.all_images[:]
        random.shuffle(self.queue)
        self.shown = []

    def _next_image_path(self) -> str:
        if not self.queue:
            self._refill_queue()
        path = self.queue.pop()
        self.shown.append(path)
        return path

    def _next_pattern(self, candidates: list[str]) -> str:
        available = [p for p in candidates if p != self.last_pattern]
        if not available:
            available = candidates
        pattern = random.choice(available)
        self.last_pattern = pattern
        return pattern

    def _load_image_arr(self, path: str) -> np.ndarray | None:
        """BGR numpy 配列として画像を読み込む"""
        arr = cv2.imread(path, cv2.IMREAD_COLOR)
        if arr is None:
            print(f'読み込み失敗: {path}')
        return arr

    def _process_to_arr(self, arr_bgr: np.ndarray) -> tuple[np.ndarray, str, int]:
        """
        画像を処理してパン用 RGB 配列・パターン・パン速度（px/frame）を返す。
        横長は水平パン、縦長は垂直パン。いずれも cover サイズで画像全体を端から端まで表示。
        1フレームの移動量（整数）は合計パン時間が DURATION に最も近くなる値を選択。
        拡大が必要な場合は Real-ESRGAN → Lanczos、縮小のみの場合はバイラテラル → Lanczos。
        """
        ih, iw = arr_bgr.shape[:2]
        sc = cover_scale(iw, ih, self.sw, self.sh)

        if iw * self.sh > ih * self.sw:
            # 横長: 水平パン。cover サイズのみ（余白なし）で全体を表示。
            candidates = ['left_to_right', 'right_to_left']
            target_w = math.ceil(iw * sc)
            target_h = math.ceil(ih * sc)
            outscale = sc
            pan_dist = target_w - self.sw
        else:
            # 縦長: 垂直パン。cover サイズのみ（余白なし）で全体を表示。
            candidates = ['top_to_bottom', 'bottom_to_top']
            target_w = math.ceil(iw * sc)
            target_h = math.ceil(ih * sc)
            outscale = sc
            pan_dist = target_h - self.sh

        if pan_dist > 0:
            # 合計パン時間が DURATION に最も近くなる整数 px/frame を選択
            pan_px_per_frame = max(1, round(pan_dist / (DURATION * self.fps)))
            # 実際のパン完了時間（フレーム数から逆算）
            slide_duration = math.ceil(pan_dist / pan_px_per_frame) / self.fps
        else:
            # パン不要（画像がスクリーンとぴったり同サイズ）
            pan_px_per_frame = 0
            slide_duration = DURATION

        pattern = self._next_pattern(candidates)

        if sc > 1.0:
            # 拡大: Real-ESRGAN でアップスケール後に Lanczos で微調整
            arr_out, _ = self.upsampler.enhance(arr_bgr, outscale=outscale)
            arr_out = cv2.resize(arr_out, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        else:
            # 縮小: バイラテラルフィルタ + Lanczos
            arr_out = cv2.bilateralFilter(arr_bgr, d=5, sigmaColor=40, sigmaSpace=40)
            arr_out = cv2.resize(arr_out, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        return cv2.cvtColor(arr_out, cv2.COLOR_BGR2RGB), pattern, pan_px_per_frame, slide_duration

    def _upload_texture(self, arr_rgb: np.ndarray) -> tuple[int, int, int]:
        """
        RGB numpy (h, w, 3) を OpenGL テクスチャにアップロードし (tex_id, w, h) を返す。
        OpenGL はデータを下→上に読むため flipud して正立させる。
        """
        arr_flipped = np.ascontiguousarray(arr_rgb[::-1])
        h, w = arr_flipped.shape[:2]
        tex_id = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)  # RGB は3バイト/px なので1バイト境界に設定
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, arr_flipped)
        return tex_id, w, h

    def _is_horizontal(self, arr_bgr: np.ndarray) -> bool:
        ih, iw = arr_bgr.shape[:2]
        return iw * self.sh > ih * self.sw

    def _prefetch_worker(self) -> None:
        """バックグラウンドで次の画像を準備（numpy まで。GL アップロードはメインスレッドで行う）"""
        arr_bgr = None
        while arr_bgr is None:
            path = self._next_image_path()
            arr_bgr = self._load_image_arr(path)
            if arr_bgr is not None and DEBUG_HORIZONTAL_ONLY and not self._is_horizontal(arr_bgr):
                arr_bgr = None  # 縦パン画像はスキップ
        self._prefetch_arr, self._prefetch_pattern, self._prefetch_pan_px_per_frame, self._prefetch_slide_duration = self._process_to_arr(arr_bgr)
        self._prefetch_ready.set()

    def _start_prefetch(self) -> None:
        threading.Thread(target=self._prefetch_worker, daemon=True).start()

    def load_next_slide(self) -> None:
        self._prefetch_ready.wait()
        if self.pan_tex is not None:
            glDeleteTextures([self.pan_tex])
        self.pan_tex, pw, ph = self._upload_texture(self._prefetch_arr)
        self.pan_tex_size = (pw, ph)
        self.current_pattern = self._prefetch_pattern
        self.pan_px_per_frame = self._prefetch_pan_px_per_frame
        self.slide_duration = self._prefetch_slide_duration
        self._prefetch_ready.clear()
        self.start_time = time.perf_counter()
        self._start_prefetch()

    def _uv_for_t(self, t_sample: float) -> tuple[float, float, float, float]:
        """時刻 t_sample における UV 座標 (u0, u1, v_top, v_bot) を返す。
        pan_px_per_frame（整数）単位でパンし、画像全体を端から端まで表示する。"""
        pw, ph = self.pan_tex_size
        sw, sh = self.sw, self.sh
        pattern = self.current_pattern
        t_s = max(0.0, min(1.0, t_sample))

        if pattern in ('top_to_bottom', 'bottom_to_top'):
            # 画像全体を上端〜下端まで表示。
            # 基準位置は整数 px/frame だが、モーションブラー用に連続値で補間する。
            x0 = (pw - sw) / 2.0
            pan = ph - sh
            offset = min(t_s * DURATION * self.fps * self.pan_px_per_frame, float(pan))
            y0 = offset if pattern == 'top_to_bottom' else float(pan) - offset
        else:
            # 画像全体を左端〜右端まで表示。
            # 基準位置は整数 px/frame だが、モーションブラー用に連続値で補間する。
            y0 = (ph - sh) / 2.0
            pan = pw - sw
            offset = min(t_s * DURATION * self.fps * self.pan_px_per_frame, float(pan))
            x0 = offset if pattern == 'left_to_right' else float(pan) - offset

        return x0 / pw, (x0 + sw) / pw, 1.0 - y0 / ph, 1.0 - (y0 + sh) / ph

    def _render_gl(self, t: float) -> None:
        """
        モーションブラー付き OpenGL レンダリング。
        1フレーム期間を MOTION_BLUR_SAMPLES 等分した位置で描画し加算平均する。
        60Hz ディスプレイのサンプルアンドホールド特性による波打ちを解消する。
        """
        N = MOTION_BLUR_SAMPLES
        frame_dt = 1.0 / (DURATION * self.fps)  # 1フレーム分のアニメーション進行量  # 1フレーム分のアニメーション進行量
        a = 1.0 / N                         # 各サンプルの重み

        glClear(GL_COLOR_BUFFER_BIT)
        glEnable(GL_BLEND)
        glBlendFunc(GL_ONE, GL_ONE)   # 加算ブレンド → 平均色を実現
        glColor4f(a, a, a, 1.0)       # テクスチャを 1/N にスケール
        glBindTexture(GL_TEXTURE_2D, self.pan_tex)

        for i in range(N):
            # 現フレームが表示される期間 [t, t+frame_dt] を N 等分してサンプル
            t_sample = t + (i + 0.5) / N * frame_dt
            u0, u1, v_top, v_bot = self._uv_for_t(t_sample)
            glBegin(GL_QUADS)
            glTexCoord2f(u0, v_bot); glVertex2f(0.0, 0.0)
            glTexCoord2f(u1, v_bot); glVertex2f(1.0, 0.0)
            glTexCoord2f(u1, v_top); glVertex2f(1.0, 1.0)
            glTexCoord2f(u0, v_top); glVertex2f(0.0, 1.0)
            glEnd()

        glDisable(GL_BLEND)
        glColor4f(1.0, 1.0, 1.0, 1.0)  # 色をリセット

    def run(self) -> None:
        self.load_next_slide()
        elapsed = 0.0

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        pygame.quit()
                        sys.exit()

            t = min(elapsed / self.slide_duration, 1.0)
            self._render_gl(t)
            pygame.display.flip()          # vsync 待機
            self.clock.tick(self.fps)      # vsync が無効な環境でのフォールバック
            now = time.perf_counter()      # clock.tick 完了後に計測（正確な経過時間）

            elapsed = now - self.start_time
            if elapsed >= self.slide_duration:
                self.load_next_slide()
                elapsed = time.perf_counter() - self.start_time

            print(f'\rFPS: {self.clock.get_fps():.1f}', end='', flush=True)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('使い方: python slideshow.py <フォルダパス>')
        sys.exit(1)

    folder = sys.argv[1]
    if not os.path.isdir(folder):
        print(f"エラー: '{folder}' はディレクトリではありません")
        sys.exit(1)

    SlideShow(folder).run()
