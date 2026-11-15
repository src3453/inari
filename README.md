# Inari 🦊

llama.cpp の `llama-server` を、黄色基調の豪華なモニタリング TUI 付きで扱うラッパー。

![overview](docs/overview.svg)

- **HuggingFace から GGUF をダウンロード** — TUI で検索 → 量子化を選んでダウンロード、または `repo:QUANT` 指定。分割 GGUF、mmproj に対応
- **起動引数のプロファイル化** — `~/.inari/profiles/*.toml` に保存
- **モニタリング TUI** — 7 ビュー（Overview / Serving / Requests / Engine / Hardware / Logs / Connect）
- **バックグラウンド起動** — TUI を閉じてもサーバーは動いたまま。`inari monitor` で再接続できる

## セットアップ

```bash
uv sync
uv run inari config set llama_server C:/llama.cpp/build/bin/llama-server.exe   # PATH に無い場合
```

NVIDIA GPU で負荷・温度・電力も見たいときは: `uv sync --extra nvidia`

## 使い方

```bash
# Hugging Face を TUI で検索してダウンロード（→ 完了後 p でプロファイル作成）
uv run inari hub qwen3 8b

# ダウンロード（候補表から選ぶ or :QUANT で直接指定）→ そのままプロファイル作成
uv run inari pull unsloth/Qwen3-8B-GGUF:Q4_K_M -p qwen3-8b

# プロファイル編集
uv run inari profile set qwen3-8b ctx-size=32768 flash-attn=on "-ot=exps=CPU" extra+=--no-webui
uv run inari profile show qwen3-8b
uv run inari profile ls

# 起動してモニタ
uv run inari serve qwen3-8b
uv run inari monitor            # 再接続
uv run inari monitor --url http://127.0.0.1:8080   # 外部で起動したサーバーを読み取り専用で監視
uv run inari ps / inari stop qwen3-8b

# モデル無しで TUI を試す
uv run inari demo
```

### プロファイルの書式

```toml
description = "Qwen3 8B"
model = "unsloth__Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"   # models_dir からの相対パス or 絶対パス
host = "127.0.0.1"
port = 8080
extra = ["--no-webui"]          # そのまま末尾に追加

[args]                          # --key value / true は単独フラグ / false は省略
ctx-size = 32768
n-gpu-layers = 99
jinja = true
flash-attn = "on"
"-ot" = "exps=CPU"              # '-' で始まるキーはそのまま使う

[env]
GGML_VK_VISIBLE_DEVICES = "0"
```

`--metrics` はモニタが使うので自動で付与されます。

### Hub（検索 TUI）のキー

| キー | 動作 |
|---|---|
| `/` | 検索語を入力（Enter で検索） |
| `o` | 並び順切替（downloads / trending / likes / updated） |
| `↑` `↓` `Enter` | リポジトリを選んでファイル一覧へ |
| `Enter` / `d` | 選んだ量子化をダウンロードキューに追加（`.part` から再開可能） |
| `m` | vision projector（mmproj）も一緒に落とす |
| `Tab` | ダウンロード一覧へ移動 → `p` でプロファイル作成、`x` でキャンセル |
| `Esc` | 戻る |
| `b` | Hugging Face のページをブラウザで開く |

### モニタのキー

| キー | 動作 |
|---|---|
| `1`–`7` / `←` `→` | ビュー切替 |
| `/` | フィルタ（Logs: 文字列, Requests: ステータス・タスク ID） |
| `Space` | 表示を一時停止 |
| `r` | 即時更新 |
| `t` | テーマ切替（inari / amber / lemon / kitsune） |
| `b` | llama.cpp の Web UI をブラウザで開く |
| `↑` `↓` `PgUp` `PgDn` `End` | ログ・リクエストのスクロール |
| `q` | 終了（管理中サーバーは「停止 / 動かしたまま」を選択） |

## データの取り方

- `/health` `/props` `/slots` `/metrics` を 0.5 秒ごとにポーリング
- サーバーログ（`~/.inari/logs/`）を tail して、デバイス・VRAM バッファ・オフロード層数・リクエスト履歴（prompt / キャッシュ再利用 / 出力 / tok/s / 所要時間）を抽出
- CPU / RAM / ディスク I/O / サーバープロセスは psutil、NVIDIA GPU は NVML（任意）

AMD / Intel GPU では負荷・温度は表示されませんが、VRAM 使用量は llama.cpp のバッファサイズから推定します。

保存先は `INARI_HOME`（既定 `~/.inari`）、モデル置き場は `INARI_MODELS` か `inari config set models_dir ...` で変更できます。
