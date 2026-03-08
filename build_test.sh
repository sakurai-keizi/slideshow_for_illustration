#!/bin/bash
# 初回のみ: sudo apt-get install -y libsdl2-dev
set -e
cd "$(dirname "$0")"

if ! pkg-config --exists sdl2 2>/dev/null; then
    echo "SDL2 開発ヘッダが見つかりません。以下を実行してください:"
    echo "  sudo apt-get install -y libsdl2-dev"
    exit 1
fi

if [ ! -f stb_image.h ]; then
    echo "stb_image.h をダウンロード中..."
    curl -sL "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h" -o stb_image.h
fi

SDL_CFLAGS=$(pkg-config --cflags sdl2)
SDL_LIBS=$(pkg-config --libs sdl2)

g++ -O2 -o display_test display_test.cpp $SDL_CFLAGS $SDL_LIBS -lGL -lm
echo "ビルド完了: ./display_test <画像ファイル>"
