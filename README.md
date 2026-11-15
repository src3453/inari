# Inari 🦊

llama.cpp の `llama-server` を、黄色基調の豪華なモニタリング TUI 付きで扱うラッパー。

![モニタリング TUI（Overview）](docs/monitor.png)

<table>
  <tr>
    <td><img src="docs/menu.png" alt="メインメニュー"></td>
    <td><img src="docs/chat.png" alt="ターミナルチャット"></td>
  </tr>
  <tr>
    <td align="center">メインメニュー</td>
    <td align="center">ターミナルチャット（<code>inari cli</code>）</td>
  </tr>
  <tr>
    <td colspan="2"><img src="docs/hub.png" alt="Hub（Hugging Face から GGUF をダウンロード）"></td>
  </tr>
  <tr>
    <td colspan="2" align="center">Hub — 量子化ごとの VRAM 適合を見ながら GGUF をダウンロード</td>
  </tr>
</table>

- **HuggingFace から GGUF をダウンロード** — TUI で検索 → 量子化を選んでダウンロード、または `repo:QUANT` 指定。分割 GGUF、mmproj に対応。保存先は llama.cpp の `-hf` と同じ Hugging Face キャッシュで、Xet / HTTP を切り替え可能
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

### メインメニュー

```bash
uv run inari                    # 引数なしで起動するとメインメニュー
```

保存済みプロファイルの一覧（モデル・ポート・起動状態）から選んで `Enter` を押すと、**Serve**（llama-server を起動してモニタ表示。起動中ならモニタに再接続）と **Chat (cli)**（その場でターミナルチャット。`inari cli` と同じ）のどちらで起動するかを選ぶ画面が出ます（`s` / `c` で直接選択、前回の選択が初期位置）。`c` でチャットを直接起動、`t` で選択中のプロファイルを Auto-tune（Balanced / Generation / Prompt / Thorough から選んで `inari tune` を実行。保存前に確認あり）、`o` で **Settings**（テーマ・ダウンロードバックエンドの変更、llama-server と設定ファイルの場所の確認）、`1`〜`9` で番号のプロファイルを選んで起動、`e` で選択中のプロファイルをセットアップウィザードで編集、`n` で「Setup & models」ウィザード（新規プロファイル作成・ローカルモデルの選択/ダウンロード/削除）、`i` で既存の llama-server コマンドを取り込み、`h` で Hub、`x` で起動中のサーバーを停止、`d` / `Del` で選択中のプロファイルを削除（確認モーダルあり。モデルファイルは残ります）、`q` で終了。モニタやウィザードを閉じるとメニューに戻ります。全コマンドの一覧は `inari --help`（パイプ等で TTY でないときは従来どおりコマンド一覧を表示）。

### Setup & models ウィザード（はじめての方はここから）

メインメニューの **+ Setup & models**（`n`）または `inari setup` で開きます。新しいプロファイルを作るウィザードですが、モデルステップはローカルモデルの管理画面も兼ねていて、プロファイルを保存せずにモデルのダウンロードや削除だけ行って `q` で抜けることもできます。

```bash
uv run inari setup              # = inari onboard。新しいプロファイルを対話的に作成
uv run inari setup qwen3-8b     # 既存プロファイルを編集
uv run inari setup qwen3-8b -a  # 全設定テーブル（Advanced）を直接開く
```

llama-server → モデル（`Enter` でこのプロファイルに使用、`h` で Hub からダウンロード、`d` / `Del` でローカルのモデルをディスクから削除。確認モーダルあり）→ 名前 → コンテキスト & KV キャッシュ → GPU オフロード → チャット / サンプリング → ネットワーク → **Advanced** → 確認・保存、の 9 ステップ。どのステップでも下部に「実行されるコマンド」が表示されます。

- **Advanced** は、設定済み llama-server の `--help` から読み取った全フラグを `Default`（llama.cpp 既定）/ `Tuned`（inari の推奨）/ `Yours`（あなたの値）で一覧表示。`/` 絞り込み、`x` experimental のみ、`Enter` 編集、`←→` 選択肢切替、`Del` リセット、`a` 名前で追加、`c` コマンドをコピー、`u` Tuned 値の適用 on/off
- **Tuned** は `llama-server --list-devices` の VRAM と GGUF メタデータ（学習時コンテキスト長、層数、MoE など）から計算
- `--help` の解析結果はビルドごとに `~/.inari/cache/` にキャッシュ
- **Auto-tune**（Context 以降のステップで `t`）は、Tuned の推測ではなく**実測**で決めます。llama-optimus / llama-dyno などのチューナーと同じく候補設定を試して速いものを残しますが、`llama-bench` ではなくこのプロファイルの `llama-server` コマンドそのもの（コンテキスト長・KV キャッシュ型・並列スロット・mmproj 込み）で起動して計測するので、本番のコンテキストで VRAM が足りない候補はここで落ち、数値も実際の serve と同じになります
  - 試す順番: 現在の設定（ベースライン）と llama.cpp の `--fit on` → GPU 配置（密モデルは `-ngl`、MoE は `--n-cpu-moe` を、ロード失敗時は CPU 側へ退避しつつ刻み幅を半分にしていく山登りで探索）→ `--ubatch-size` / `--batch-size` → `--threads`（CPU が計算に関わるときだけ）→（thorough のみ）配置を ±1 で再確認
  - Goal: `balanced`（2K プロンプト + 256 トークン応答の 1 リクエスト/分）/ `gen`（生成 tok/s）/ `prompt`（プロンプト処理 tok/s）。Depth: `quick`（最大 10 回起動）/ `thorough`（最大 24 回・各 2 回計測）
  - 結果は表で表示され、`Enter` で適用（`measured` として表示）。コンテキスト長と KV キャッシュ型は品質に関わるので変更しません。サーバー出力は `~/.inari/logs/autotune-<name>.log`
  - 既存プロファイルはコマンドラインからも: `uv run inari tune qwen3-8b [--goal gen] [--thorough] [-y]`

### 既存の llama-server コマンドから取り込む

```bash
uv run inari import             # コマンドを貼り付け（空行で確定）。メインメニューでは i
uv run inari import llama-server -hf unsloth/Qwen3-8B-GGUF:Q4_K_M -c 32768 -ngl 99 -fa on --jinja
uv run inari import -f run.sh   # スクリプト（.sh / .bat / .ps1）から読む
uv run inari import -y -p qwen3-8b "..."   # 確認なしでダウンロードし、ウィザードを開かずに保存
```

設定済み llama-server の `--help` を使って引数を読み取るので、`-c` / `--ctx-size` / `--ctx-size=…` のどの書き方でも同じキーになります。読み取った内容を表示したあと、セットアップウィザードの Review ステップで開くので、確認・調整してから保存できます。

- `-m PATH` → そのファイル（HF キャッシュ内なら `hf://` 参照に変換）。見つからなければローカルのモデルからファイル名で探す
- `-hf owner/repo[:QUANT]`（`-hff` も可）/ `-mu https://huggingface.co/…/resolve/…` → llama.cpp と同じ規則でファイルを選び（QUANT 省略時は Q4_K_M）、HF キャッシュにあればそれを使い、無ければサイズを確認してダウンロード。`-hf` と同様に mmproj も取得（`--no-mmproj` なら取得しない）
- `llama-server` / `llama server` / `llama serve` だけでなく `llama-cli` / `llama cli` のコマンドも取り込めます（llama-server に無いオプションや `-p`・`-cnv` などの CLI 専用オプションは除外）
- `--host` / `--port` はプロファイルの listen に、`FOO=1 llama-server …`・`$env:FOO="1"; …`・`set FOO=1 && …` は `[env]` に入る
- その llama-server のビルドが知らないフラグは `extra` にそのまま残し、`--hf-token` は保存しない（`inari config set hf_token` を使う）。`--metrics` は inari が自動で付けるので省く
- コマンドの llama-server が設定と違うビルドなら、プロファイルの `server_bin` に記録（未設定なら `llama_server` として登録）

```bash
# Hugging Face を TUI で検索してダウンロード（→ 完了後 p でプロファイル作成）
uv run inari hub qwen3 8b

# ダウンロード（候補表から選ぶ or :QUANT で直接指定）→ そのままプロファイル作成
uv run inari pull unsloth/Qwen3-8B-GGUF:Q4_K_M -p qwen3-8b
uv run inari pull unsloth/Qwen3-8B-GGUF:Q4_K_M --via http   # バックエンドを指定（auto / xet / http）

# ローカルのモデル一覧（HF キャッシュ + models_dir）
uv run inari models

# プロファイル編集
uv run inari profile set qwen3-8b ctx-size=32768 flash-attn=on "-ot=exps=CPU" extra+=--no-webui
uv run inari profile show qwen3-8b
uv run inari profile ls

# 起動してモニタ
uv run inari serve              # router モード：全プロファイルを 1 ポートで（モデルは要求に応じてロード）
uv run inari serve -l qwen3-8b  # router モード + 起動時に qwen3-8b をロード
uv run inari serve qwen3-8b     # 1 プロファイルだけを単体で起動
uv run inari monitor            # 再接続
uv run inari monitor --url http://127.0.0.1:8080   # 外部で起動したサーバーを読み取り専用で監視
uv run inari ps / inari stop qwen3-8b

# ターミナルでチャット（llama-cli 風）
uv run inari cli qwen3-8b       # = inari chat
uv run inari cli qwen3-8b -p "日本語で自己紹介して"   # 1 回だけ答えて終了（標準入力からも可）

# モデル無しで TUI を試す
uv run inari demo
```

### ターミナルでチャット（`inari cli`）

`llama-cli` のように、プロファイルのモデルとターミナルで直接会話します。中身は llama-server の `/v1/chat/completions`（ストリーミング）なので、プロファイルの設定（コンテキスト長・サンプリング・`--jinja` など）がそのまま使われます。

- そのプロファイルが既に単体で serve 中ならそのサーバーを、router が起動中なら router 経由（`"model": "<プロファイル名>"`）で使います。どちらも無ければチャット用に llama-server を起動し、終了時に停止します（`--keep` で起動したまま残す、`--no-router` で router を使わず単体起動）
- 応答は Rich で **Markdown としてレンダリング**（見出し・リスト・表・シンタックスハイライト付きコードブロック）。書き終わったブロックから確定表示し、書きかけのブロックはその下で逐次更新します。`--plain` / `/md` でプレーンテキスト表示に切り替え（パイプ出力時は常にプレーンテキスト）
- 応答はトークン単位で表示し、推論モデルの思考（`reasoning_content`）は薄い色で表示（`--no-think` / `/think` で非表示）。応答ごとに生成トークン数・tok/s・プロンプト処理速度を表示
- `Ctrl+C` で生成を中断、空のプロンプトで `Ctrl+C` か `Ctrl+D` / `/exit` で終了。行末 `\` または `"""` で囲むと複数行入力
- `/` を入力するとコマンド候補が表示されます（↑↓ で選択、Tab / Enter で確定）
- コマンド：`/clear`（会話をリセットして画面もクリア）、`/regen`（最後の応答を再生成）、`/undo`（最後のやり取りを削除）、`/system テキスト`（システムプロンプト設定）、`/think`、`/effort`（思考予算：`2048`・`8k`・`low`/`medium`/`high`/`max`・`off`・`auto`）、`/md`（Markdown 表示の切替）、`/info`、`/help`
- オプション：`-s/--system`、`--temp`、`-n/--n-predict`（1 応答の最大トークン）、`-e/--effort`（思考予算の初期値。`/effort` と同じ値）、`-p/--prompt`（1 回だけ答えて終了）、`--port`、`--url URL [-m MODEL]`（任意の起動中 llama-server と会話）
- プロファイル名を省略すると一覧から選択（単体で動いているサーバーが 1 つだけならそれを使用）


### router モード（`inari serve`）

プロファイル名を付けずに `inari serve` を実行すると、llama-server を [router モード](https://github.com/ggml-org/llama.cpp/tree/master/tools/server#using-multiple-models) で起動し、そのままモニタを開きます。

- すべてのプロファイルを、プロファイル名をセクション名にしたモデルプリセットとして `~/.inari/router/presets.ini` に書き出します（起動のたびに再生成）。プロファイルのない HF キャッシュ内のモデルは既定で候補から除外します（`--allow-no-profile-models` を付けると llama-server 自身が一覧に追加します）
- リクエストの `"model": "<プロファイル名>"` で使うモデルを選びます。未ロードのモデルは最初のリクエストで自動的にロードされます
- オプション：`--port`（既定 8080）、`--host`、`-l/--load PROFILE`（起動時にロード、複数指定可）、`--models-max N`（同時にロードするモデルの上限）、`--allow-no-profile-models`（プロファイルのないキャッシュ内モデルも候補に含める）
- モニタの **8 Models** タブ：`Enter` でロードしてそのモデルのモニタを表示、`u` でアンロード。表示中のモデルは、処理中または最後にロードされたモデルに自動で切り替わります（`m` で手動で次のモデルへ、`f` で自動追従に戻す）。Requests にはモデル名の列が付きます
- 再接続は `inari monitor router`、停止は `inari stop router`（各モデルのプロセスもまとめて停止）。メインメニューでは `r` で起動・再接続できます
- プロファイルの `host` / `port` は router では使われません。`LLAMA_ARG_*` 以外の `env` は router 全体に適用されます

### プロファイルの書式

```toml
description = "Qwen3 8B"
model = "hf://unsloth/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"   # HF キャッシュ参照 / models_dir からの相対パス / 絶対パス
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

`--metrics` はモニタが使うので自動で付与されます。`hf://owner/repo/file` という値（`model` や `mmproj` など）は、起動時に HF キャッシュ内のファイルパスに変換されます。

### モデルの保存先と llama.cpp との共有

ダウンロードしたモデルは **Hugging Face キャッシュ**（`~/.cache/huggingface/hub/models--owner--repo/`）に保存されます。llama.cpp の `-hf` や `huggingface-cli` と同じ場所・同じ形式なので、

- `inari pull` したモデルは `llama-server -hf owner/repo:QUANT` で再ダウンロードなしに使える
- `llama-server -hf` で落としたモデルも `inari models` に出てきて、プロファイルから使える
- 途中まで落としたファイル（`blobs/<sha256>.downloadInProgress`）は、HTTP なら inari と llama.cpp のどちらからでも再開できる
- 同じファイルを同時に落とそうとすると、片方が待つ（`.locks/` のロックを共有）

キャッシュの場所は llama.cpp と同じ順で決まります: `LLAMA_CACHE` → `HF_HUB_CACHE` → `HUGGINGFACE_HUB_CACHE` → `HF_HOME/hub` → `XDG_CACHE_HOME/huggingface/hub` → `~/.cache/huggingface/hub`。`inari config set hf_cache <dir>` で個別に指定した場合は、inari から起動する llama-server にも `LLAMA_CACHE` として渡します。

**ダウンロード方式**は `inari config set download_backend auto|xet|http`、`pull --via ...`、Hub TUI の `v` キーで切り替えられます。

| 方式 | 内容 |
|---|---|
| `auto`（既定） | `hf_xet` が入っていれば Xet。HTTP の途中ファイルがあるときは HTTP で再開 |
| `xet` | Xet（`hf_xet`）で取得。速いが、中止すると最初からやり直し |
| `http` | 普通の HTTP。Range で途中から再開できる |

以前の版で `~/.inari/models/owner__repo/` に落としたモデルは、`inari models migrate` で HF キャッシュへ移動できます。それを参照しているプロファイルも書き換えます。移行しなくても、これまで通り読み込めます。

### Hub（検索 TUI）のキー

| キー | 動作 |
|---|---|
| `/` | 検索語を入力（Enter で検索） |
| `o` | 並び順切替（downloads / trending / likes / updated） |
| `↑` `↓` `Enter` | リポジトリを選んでファイル一覧へ |
| `Enter` / `d` | 選んだ量子化をダウンロードキューに追加 |
| `m` | vision projector（mmproj）も一緒に落とす |
| `v` | ダウンロード方式を切替（auto / xet / http）。次に追加するダウンロードから有効 |
| `Tab` | ダウンロード一覧へ移動 → `p` でプロファイル作成、`x` でキャンセル |
| `Esc` | 戻る |
| `b` | Hugging Face のページをブラウザで開く |

ファイル一覧の **FIT** 列は、この PC の RAM + VRAM の総量から見た動作の目安です（重み + 控えめなコンテキスト分の余裕で概算。mmproj を付けるとその分も加算）。

| 表示 | 意味 |
|---|---|
| `◆ GPU` | VRAM に全部載る。最速 |
| `◐ GPU+RAM` | 一部を RAM にオフロード。動くが遅くなる |
| `○ CPU` | GPU なし。RAM から CPU で動かす |
| `△ tight` | ギリギリ。他のアプリを閉じ、コンテキストを小さめに |
| `✖ too big` | 足りない |

`★` はスペックからのおすすめで、ファイル一覧を開いたときのカーソル位置にもなります。VRAM に全部載る中で一番大きい量子化（ただし約 3.4 bit/weight 以上）を優先し、載らない場合は Q4_K_M 前後を RAM オフロード前提で選びます。F16 / BF16 は他に候補がないときだけ。GPU は `llama-server --list-devices` で検出するので CUDA / ROCm / Vulkan / Metal いずれでも VRAM が反映されます（検出完了まで数秒は RAM のみで判定）。

### モニタのキー

| キー | 動作 |
|---|---|
| `1`–`7` / `←` `→` | ビュー切替 |
| `/` | フィルタ（Logs: 文字列, Requests: ステータス・タスク ID） |
| `Space` | 表示を一時停止 |
| `r` | 即時更新 |
| `t` | テーマ選択（説明とカラープレビュー付き。↑↓ で即プレビュー、Enter で決定、Esc で戻す）: kitsune（既定）/ sora 水色 / matcha 緑 / sumi モノクロ / washi ライト端末用 |
| `b` | llama.cpp の Web UI をブラウザで開く |
| `↑` `↓` `PgUp` `PgDn` `End` | ログ・リクエストのスクロール |
| `q` | 終了（管理中サーバーは「停止 / 動かしたまま」を選択） |

## データの取り方

- `/health` `/props` `/slots` `/metrics` を 1 秒ごとにポーリング
- サーバーログ（`~/.inari/logs/`）を tail して、デバイス・VRAM バッファ・オフロード層数・リクエスト履歴（prompt / キャッシュ再利用 / 出力 / tok/s / 所要時間）を抽出
- CPU / RAM / ディスク I/O / サーバープロセスは psutil、NVIDIA GPU は NVML（任意）

AMD / Intel GPU では負荷・温度は表示されませんが、VRAM 使用量は llama.cpp のバッファサイズから推定します。

保存先は `INARI_HOME`（既定 `~/.inari`）。ダウンロードは HF キャッシュ（上記）に入ります。手で置いた GGUF 用の `models_dir` は `INARI_MODELS` か `inari config set models_dir ...` で変更できます。
