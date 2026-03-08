# Slideshow

指定フォルダ以下の画像をケンバーンズ風エフェクトで全画面スライドショー表示するツールです。

## 機能

- 指定フォルダ以下（サブフォルダ含む）の画像を全画面表示
- 以下のモーション（ケンバーンズ風）をランダムに適用（同じパターンが連続しない）
  - 上から下へのパン
  - 下から上へのパン
  - 左から右へのパン
  - 右から左へのパン
- 小さい画像は Real-ESRGAN で自動アップスケール（CLIP でアニメ/写真を自動判別し最適モデルを選択）
- 10秒ごとに画像を切り替え（全画像を一周するまで同じ画像は表示しない）
- C++ + OpenGL + モーションブラーによる滑らかな描画（リフレッシュレート自動検出 / vsync）

## 対応画像形式

`.jpg` `.jpeg` `.png` `.gif` `.bmp` `.webp` `.tiff` `.tif`

## 必要環境

- Python 3.12 以上
- CUDA 対応 GPU（Real-ESRGAN の高速処理に必要）
- [uv](https://docs.astral.sh/uv/) がインストールされていること
- SDL2 / GLEW 開発ライブラリ（C++ 表示サーバーのビルドに必要）

```bash
sudo apt-get install -y libsdl2-dev libglew-dev
```

## uv のインストール

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 使い方

### uv run で直接実行（推奨）

依存関係のインストールは不要です。初回実行時に自動でセットアップされます。

```bash
uv run slideshow.py <フォルダパス>
```

例:

```bash
uv run slideshow.py ~/Pictures
```

### シェバンで実行（Linux / macOS）

スクリプトに実行権限を付与すれば直接実行できます。

```bash
chmod +x slideshow.py
./slideshow.py <フォルダパス>
```

## 初回起動時の注意

Real-ESRGAN のモデルファイル（約 66MB）を自動でダウンロードします。
ダウンロード先: `~/.cache/realesrgan/RealESRGAN_x4plus_anime_6B.pth`

## 操作方法

| キー | 動作 |
|------|------|
| `ESC` または `Q` | 終了 |

## 設定（スクリプト先頭の定数）

| 定数 | デフォルト | 説明 |
|------|-----------|------|
| `DURATION` | `10.0` | 1枚あたりの表示秒数（パン距離から自動調整） |
| `DEBUG_HORIZONTAL_ONLY` | `False` | `True` のとき横パン画像のみ表示（デバッグ用） |
| `DEBUG_NO_ESRGAN` | `False` | `True` のとき ESRGAN を使わず Lanczos のみ（デバッグ用） |
