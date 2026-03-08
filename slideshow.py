#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "opencv-python",
#   "realesrgan",
#   "basicsr",
#   "torch",
#   "torchvision",
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

import contextlib
import io
import math
import os
import random
import struct
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

DURATION = 10.0  # 1枚あたりの表示秒数

DEBUG_HORIZONTAL_ONLY = False  # デバッグ: True のとき横パン画像のみ表示
DEBUG_NO_ESRGAN       = False  # デバッグ: True のとき ESRGAN を使わずバイリニア拡大

MODEL_URL = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'
MODEL_PATH = Path.home() / '.cache' / 'realesrgan' / 'RealESRGAN_x4plus_anime_6B.pth'

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}

DIRECTION = {'left_to_right': 0, 'right_to_left': 1, 'top_to_bottom': 2, 'bottom_to_top': 3}


class _Frame(NamedTuple):
    """プリフェッチ済み画像データ（C++ サーバーに送る直前の状態）"""
    arr:            np.ndarray  # RGBA, OpenGL 用上下反転済み
    dir_int:        int         # DIRECTION の値
    ppf:            int         # pixels per frame
    slide_duration: float       # スライド表示秒数


def collect_images(folder: str) -> list[str]:
    return [
        str(p) for p in Path(folder).rglob('*')
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


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
        tile=256,
        tile_pad=10,
        pre_pad=0,
        half=True,
        gpu_id=0,
    )


def build_display_server() -> Path:
    """display_server.cpp をビルドして実行ファイルのパスを返す。"""
    src = Path(__file__).parent / 'display_server.cpp'
    out = Path(__file__).parent / 'display_server'

    if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        return out

    print('display_server をビルド中...', flush=True)
    try:
        cflags = subprocess.check_output(['pkg-config', '--cflags', 'sdl2'], text=True).strip()
        libs   = subprocess.check_output(['pkg-config', '--libs',   'sdl2'], text=True).strip()
    except subprocess.CalledProcessError:
        sys.exit('エラー: SDL2 開発ヘッダが見つかりません。\n'
                 '  sudo apt-get install -y libsdl2-dev  を実行してください。')

    ret = subprocess.run(f'g++ -O2 -o {out} {src} {cflags} {libs} -lGLEW -lGL -lm -pthread', shell=True)
    if ret.returncode != 0:
        sys.exit('エラー: display_server のビルドに失敗しました。')
    print('ビルド完了', flush=True)
    return out


class SlideShow:
    def __init__(self, folder: str):
        print('Real-ESRGANを初期化中...', flush=True)
        self.upsampler = init_esrgan()
        print('初期化完了', flush=True)

        # スクリーンサイズ・FPS は run() 内で C++ 実測値に設定される
        self.sw:  int = 0
        self.sh:  int = 0
        self.fps: int = 60

        self.all_images = collect_images(folder)
        if not self.all_images:
            sys.exit(f'画像が見つかりません: {folder}')
        self.queue: list[str] = []
        self._refill_queue()

        self.last_pattern: str | None = None

        # プリフェッチ（run() で C++ スクリーンサイズ確認後に開始）
        self._prefetch: _Frame | None = None
        self._prefetch_ready = threading.Event()

    # ── 画像キュー管理 ──────────────────────────────────────────────────────

    def _refill_queue(self) -> None:
        self.queue = self.all_images[:]
        random.shuffle(self.queue)

    def _next_image_path(self) -> str:
        if not self.queue:
            self._refill_queue()
        return self.queue.pop()

    def _next_pattern(self, candidates: list[str]) -> str:
        available = [p for p in candidates if p != self.last_pattern] or candidates
        pattern = random.choice(available)
        self.last_pattern = pattern
        return pattern

    # ── 画像処理 ────────────────────────────────────────────────────────────

    def _process(self, arr_bgr: np.ndarray) -> _Frame:
        """BGR 配列 → _Frame（OpenGL 用上下反転 RGBA、パン情報付き）"""
        ih, iw = arr_bgr.shape[:2]
        sw, sh = self.sw, self.sh

        if iw * sh > ih * sw:
            # 横長: 高さ固定、横スクロール
            target_h = sh
            target_w = -((-iw * sh) // ih)   # ceil(iw*sh/ih)、整数演算
            sc       = sh / ih
            pan_dist = target_w - sw
            pattern  = self._next_pattern(['left_to_right', 'right_to_left'])
        else:
            # 縦長: 横幅固定、縦スクロール
            target_w = sw
            target_h = -((-ih * sw) // iw)   # ceil(ih*sw/iw)、整数演算
            sc       = sw / iw
            pan_dist = target_h - sh
            pattern  = self._next_pattern(['top_to_bottom', 'bottom_to_top'])

        if pan_dist > 0:
            ppf            = max(1, round(pan_dist / (DURATION * self.fps)))
            slide_duration = math.ceil(pan_dist / ppf) / self.fps
        else:
            ppf            = 0
            slide_duration = DURATION

        if sc > 1.0 and not DEBUG_NO_ESRGAN:
            import torch
            with contextlib.redirect_stdout(io.StringIO()):
                arr_out, _ = self.upsampler.enhance(arr_bgr, outscale=4)
            torch.cuda.synchronize()
            arr_out = cv2.resize(arr_out, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        else:
            arr_out = cv2.resize(arr_bgr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        arr_rgba    = cv2.cvtColor(arr_out, cv2.COLOR_BGR2RGBA)
        arr_flipped = np.ascontiguousarray(arr_rgba[::-1])  # OpenGL 用上下反転
        return _Frame(arr_flipped, DIRECTION[pattern], ppf, slide_duration)

    # ── プリフェッチ ─────────────────────────────────────────────────────────

    def _prefetch_worker(self) -> None:
        arr_bgr = None
        while arr_bgr is None:
            path    = self._next_image_path()
            arr_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if arr_bgr is None:
                print(f'読み込み失敗: {path}', flush=True)
                continue
            if DEBUG_HORIZONTAL_ONLY:
                ih, iw = arr_bgr.shape[:2]
                if not (iw * self.sh > ih * self.sw):
                    arr_bgr = None  # 縦パン画像はスキップ
        self._prefetch = self._process(arr_bgr)
        self._prefetch_ready.set()

    def _start_prefetch(self) -> None:
        threading.Thread(target=self._prefetch_worker, daemon=True).start()

    # ── C++ サーバー通信 ─────────────────────────────────────────────────────

    def _send_image(self, proc: subprocess.Popen) -> None:
        """プリフェッチ完了を待ち、バイナリヘッダ＋ピクセルを C++ サーバーに送る。"""
        self._prefetch_ready.wait()
        frame = self._prefetch
        self._prefetch_ready.clear()
        self._start_prefetch()  # 次のプリフェッチを開始

        h, w = frame.arr.shape[:2]
        proc.stdin.write(struct.pack('<IIIII', w, h, frame.ppf, frame.dir_int,
                                     int(frame.slide_duration * 1000)))
        proc.stdin.write(frame.arr.tobytes())
        proc.stdin.flush()

    def run(self) -> None:
        proc = subprocess.Popen(
            [str(build_display_server())],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )

        # C++ が実際のスクリーンサイズと FPS を送ってくるのを待つ
        line = proc.stdout.readline().decode('utf-8', errors='replace').strip()
        if line.startswith('SCREEN '):
            _, w, h, fps = line.split()
            self.sw, self.sh, self.fps = int(w), int(h), int(fps)
            print(f'ディスプレイ: {self.sw}x{self.sh}  FPS: {self.fps}', flush=True)
        else:
            print(f'警告: SCREEN メッセージを受信できませんでした: {line!r}', flush=True)

        self._start_prefetch()  # 正しいスクリーンサイズが確定してから開始

        try:
            while True:
                line = proc.stdout.readline().decode('utf-8', errors='replace').strip()
                if line == 'QUIT' or not line:
                    break
                if line == 'READY':
                    self._send_image(proc)
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass
            proc.wait()
        print('\n終了', flush=True)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('使い方: python slideshow.py <フォルダパス>')
        sys.exit(1)

    folder = sys.argv[1]
    if not os.path.isdir(folder):
        print(f"エラー: '{folder}' はディレクトリではありません")
        sys.exit(1)

    SlideShow(folder).run()
