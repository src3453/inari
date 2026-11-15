"""A simulated llama-server so the TUI can be explored without a model.

Synthetic slots / props / metrics are produced in-process and realistic
llama.cpp log lines are fed through the real log parser.
"""

from __future__ import annotations

import random
import time

from inari.collector import Collector, Snapshot

BOOT_LOG = """\
$ llama-server --model Qwen3-8B-Q4_K_M.gguf --ctx-size 32768 --n-gpu-layers 99 --jinja --metrics
ggml_vulkan: Found 1 Vulkan devices:
llama_model_load_from_file_impl: using device Vulkan0 (AMD Radeon RX 7600) (unknown id) - 7920 MiB free
llama_model_loader: - kv   2:                               general.name str              = Qwen3 8B
print_info: file format = GGUF V3 (latest)
print_info: file type   = Q4_K - Medium
print_info: file size   = 4.68 GiB (4.91 BPW)
print_info: arch             = qwen3
print_info: n_ctx_train      = 40960
print_info: n_embd           = 4096
print_info: n_layer          = 36
print_info: n_head           = 32
print_info: n_head_kv        = 8
print_info: model type       = 8B
print_info: model params     = 8.19 B
print_info: vocab type       = BPE
print_info: n_vocab          = 151936
load_tensors: offloading 36 repeating layers to GPU
load_tensors: offloaded 37/37 layers to GPU
load_tensors:      Vulkan0 model buffer size =  4455.35 MiB
load_tensors:   CPU_Mapped model buffer size =   333.84 MiB
llama_context: n_ctx         = 32768
llama_kv_cache:    Vulkan0 KV buffer size =  1152.00 MiB
llama_context:    Vulkan0 compute buffer size =   304.75 MiB
llama_context: Vulkan_Host compute buffer size =    72.01 MiB
srv          init: initializing slots, n_slots = 1
main: model loaded
main: server is listening on http://127.0.0.1:8080 - starting the main loop
srv  update_slots: all slots are idle"""


class DemoCollector(Collector):
    def __init__(self) -> None:
        super().__init__("http://127.0.0.1:8080", name="demo", started=time.time() - 3725,
                         cmd=["llama-server", "--model", "Qwen3-8B-Q4_K_M.gguf", "--ctx-size", "32768",
                              "--n-gpu-layers", "99", "--jinja", "--metrics"])
        self.snap.has_logs = True
        self.parser.live = False
        for ln in BOOT_LOG.splitlines():
            self.parser.feed(ln)
        self.parser.live = True
        self.rng = random.Random(7)
        self.task = 0
        self.phase = "idle"
        self.phase_until = time.monotonic() + 1.5
        self.prompt = 0
        self.pp_done = 0
        self.decoded = 0
        self.target = 0
        self.history_tokens = 0
        self.totals = {"prompt": 0, "gen": 0}
        self._seed_history()

    def _seed_history(self) -> None:
        self.parser.live = False
        for _ in range(4):
            self._emit_full_request(self.rng.randint(200, 9000), self.rng.randint(40, 1500))
        self.parser.live = True
        for i, r in enumerate(self.parser.requests):
            r.started = time.time() - 2400 + i * 540

    def _emit_full_request(self, prompt: int, out: int) -> None:
        t = self.task
        reused = int(prompt * self.rng.choice([0, 0.9, 0.95]))
        feed = self.parser.feed
        feed(f"slot launch_slot_: id  0 | task {t} | processing task")
        feed(f"slot update_slots: id  0 | task {t} | new prompt, n_ctx_slot = 32768, n_keep = 0, n_prompt_tokens = {prompt}")
        feed(f"slot update_slots: id  0 | task {t} | kv cache rm [{reused}, end)")
        feed(f"slot update_slots: id  0 | task {t} | prompt done, n_past = {prompt}, n_tokens = {prompt - reused}")
        feed(f"slot      release: id  0 | task {t} | stop processing: n_past = {prompt + out}, truncated = 0")
        feed(f"slot print_timing: id  0 | task {t} |")
        pp_ms = (prompt - reused) / 820 * 1000 + 30
        g_ms = out / 41.5 * 1000
        feed(f"prompt eval time = {pp_ms:10.2f} ms / {prompt - reused:5d} tokens ({pp_ms / max(prompt - reused, 1):8.2f} ms per token, {(prompt - reused) / pp_ms * 1000:8.2f} tokens per second)")
        feed(f"       eval time = {g_ms:10.2f} ms / {out:5d} tokens ({g_ms / out:8.2f} ms per token, {out / g_ms * 1000:8.2f} tokens per second)")
        feed(f"      total time = {pp_ms + g_ms:10.2f} ms / {prompt - reused + out:5d} tokens")
        feed("srv  update_slots: all slots are idle")
        self.task += 1
        self.totals["prompt"] += prompt - reused
        self.totals["gen"] += out

    def _poll_server(self, s: Snapshot, dt: float) -> None:
        now = time.monotonic()
        feed = self.parser.feed
        t = self.task
        if self.phase == "idle" and now > self.phase_until:
            self.phase = "reading"
            self.prompt = self.rng.randint(800, 20000)
            self.reused = int(self.prompt * self.rng.choice([0, 0.92]))
            self.pp_done = self.reused
            self.decoded = 0
            self.target = self.rng.randint(150, 900)
            self.t0 = now
            feed(f"slot launch_slot_: id  0 | task {t} | processing task")
            feed(f"slot update_slots: id  0 | task {t} | new prompt, n_ctx_slot = 32768, n_keep = 0, n_prompt_tokens = {self.prompt}")
            feed(f"slot update_slots: id  0 | task {t} | kv cache rm [{self.reused}, end)")
        elif self.phase == "reading":
            self.pp_done = min(self.prompt, self.pp_done + int(820 * dt))
            prog = self.pp_done / self.prompt
            feed(f"slot update_slots: id  0 | task {t} | prompt processing progress, n_past = {self.pp_done}, n_tokens = 2048, progress = {prog:.6f}")
            if self.pp_done >= self.prompt:
                feed(f"slot update_slots: id  0 | task {t} | prompt done, n_past = {self.prompt}, n_tokens = {self.prompt - self.reused}")
                self.phase, self.t1 = "gen", now
        elif self.phase == "gen":
            self.decoded = min(self.target, self.decoded + int(self.rng.uniform(36, 46) * dt))
            if self.decoded >= self.target:
                pp_ms = (self.t1 - self.t0) * 1000 + 1
                g_ms = (now - self.t1) * 1000 + 1
                n_pp = self.prompt - self.reused
                feed(f"slot      release: id  0 | task {t} | stop processing: n_past = {self.prompt + self.decoded}, truncated = 0")
                feed(f"slot print_timing: id  0 | task {t} |")
                feed(f"prompt eval time = {pp_ms:10.2f} ms / {n_pp:5d} tokens ({pp_ms / max(n_pp, 1):8.2f} ms per token, {n_pp / pp_ms * 1000:8.2f} tokens per second)")
                feed(f"       eval time = {g_ms:10.2f} ms / {self.decoded:5d} tokens ({g_ms / self.decoded:8.2f} ms per token, {self.decoded / g_ms * 1000:8.2f} tokens per second)")
                feed(f"      total time = {pp_ms + g_ms:10.2f} ms / {n_pp + self.decoded:5d} tokens")
                feed("srv  update_slots: all slots are idle")
                self.totals["prompt"] += n_pp
                self.totals["gen"] += self.decoded
                self.task += 1
                self.phase, self.phase_until = "idle", now + self.rng.uniform(2, 6)

        busy = self.phase in ("reading", "gen")
        s.health = "ok"
        s.props = {"total_slots": 1, "model_path": "C:/models/Qwen3-8B-Q4_K_M.gguf", "build_info": "b6900-demo",
                   "default_generation_settings": {"n_ctx": 32768, "params": {"temperature": 0.7, "top_k": 20,
                                                                             "top_p": 0.8, "min_p": 0.0, "n_predict": -1}},
                   "modalities": {"vision": False, "audio": False},
                   "chat_template": "{%- for message in messages %}\n<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{%- endfor %}"}
        s.slots = [{"id": 0, "id_task": t if busy else -1, "n_ctx": 32768, "is_processing": busy,
                    "params": {"temperature": 0.7, "top_k": 20, "top_p": 0.8, "min_p": 0.0},
                    "next_token": [{"has_next_token": busy, "n_remain": -1,
                                    "n_decoded": self.decoded if self.phase == "gen" else 0}]}]
        s.metrics = {"prompt_tokens_total": self.totals["prompt"], "tokens_predicted_total": self.totals["gen"],
                     "prompt_tokens_seconds": 812.4, "predicted_tokens_seconds": 41.2,
                     "requests_processing": 1 if busy else 0, "requests_deferred": 0,
                     "n_decode_total": self.totals["gen"] + 300, "n_busy_slots_per_decode": 1.0}
        self._derive(s, dt)
        s.model = "Qwen3-8B-Q4_K_M"
