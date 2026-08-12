#!/usr/bin/env python3
"""
HTTP bridge so Unity can request MusicGen horror beds and wait for completion.

Uses the same CUDA path as iteration-2 gui_demo:
  MusicGenRealtimeGen -> dedicated GPU worker thread (fp16 + SDPA + TF32).

  GET  /health
  POST /generate  JSON { level, intensity, dissonance, seed, max_new_tokens? }
       -> audio/wav (waits until MusicGen finishes on GPU)
       -> if MusicGen unavailable/fails: procedural WAV fallback

Run (from this folder, with transformers + CUDA torch installed):
  python musicgen_unity_bridge.py --port 8765 --model models/musicgen-small --require-cuda
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import traceback
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.iteration2.musicgen_audio_runtime import MusicGenRealtimeGen  # noqa: E402
from src.iteration2.procedural_audio_gen import ProceduralAudioGen  # noqa: E402


def float_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _cuda_info() -> dict:
    info = {
        "cuda_available": False,
        "device": "cpu",
        "gpu_name": None,
        "vram_free_gb": None,
        "vram_total_gb": None,
    }
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["device"] = "cuda"
            info["gpu_name"] = torch.cuda.get_device_name(0)
            try:
                free_b, total_b = torch.cuda.mem_get_info()
                info["vram_free_gb"] = round(free_b / 1024**3, 2)
                info["vram_total_gb"] = round(total_b / 1024**3, 2)
            except Exception:
                pass
    except Exception as e:
        info["error"] = str(e)
    return info


class BridgeState:
    def __init__(
        self,
        model_path: Path,
        assets_dir: Path,
        max_new_tokens: int,
        require_cuda: bool,
        generate_timeout_s: float,
    ):
        self.model_path = model_path
        self.assets_dir = assets_dir
        self.max_new_tokens = max_new_tokens
        self.require_cuda = require_cuda
        self.generate_timeout_s = generate_timeout_s
        self.musicgen: MusicGenRealtimeGen | None = None
        self.procedural = ProceduralAudioGen(assets_dir)
        self.busy = False
        self.last_error: str | None = None
        self.last_source: str = "none"
        self.last_duration_s: float = 0.0
        self.last_gen_s: float = 0.0
        self.ready = False
        self.device: str = "cpu"
        self.gpu_name: str | None = None

    def init_models(self) -> None:
        cuda = _cuda_info()
        print(f"[bridge] CUDA available={cuda['cuda_available']} "
              f"gpu={cuda.get('gpu_name')} "
              f"vram_free={cuda.get('vram_free_gb')}GiB")
        if self.require_cuda and not cuda["cuda_available"]:
            raise SystemExit(
                "[bridge] --require-cuda set but torch.cuda.is_available() is False. "
                "Install CUDA torch in the same Python env iteration-2 uses "
                "(miniconda base has been verified)."
            )

        print(f"[bridge] loading MusicGen on GPU path from {self.model_path} ...")
        t0 = time.perf_counter()
        try:
            self.musicgen = MusicGenRealtimeGen(
                model_id=str(self.model_path),
                max_new_tokens=self.max_new_tokens,
                local_files_only=True,
                defer_init=False,
            )
            if not self.musicgen.available:
                self.last_error = self.musicgen.last_load_error or "MusicGen unavailable"
                print(f"[bridge] MusicGen not available: {self.last_error}")
            else:
                self.device = self.musicgen.device or "unknown"
                self.gpu_name = cuda.get("gpu_name") if self.device == "cuda" else None
                if self.require_cuda and self.device != "cuda":
                    raise SystemExit(
                        f"[bridge] MusicGen loaded on '{self.device}' but --require-cuda needs cuda"
                    )
                print(
                    f"[bridge] MusicGen ready in {time.perf_counter() - t0:.1f}s "
                    f"device={self.device} sr={self.musicgen.sample_rate} "
                    f"tokens={self.max_new_tokens}"
                )
        except SystemExit:
            raise
        except Exception as e:
            self.last_error = str(e)
            self.musicgen = None
            print(f"[bridge] MusicGen load failed: {e}")
            if self.require_cuda:
                raise SystemExit(f"[bridge] CUDA MusicGen required: {e}") from e
        print(f"[bridge] procedural available={self.procedural.available}")
        self.ready = True

    def generate(
        self,
        level: int,
        intensity: float,
        dissonance: float,
        seed: int,
        max_new_tokens: int | None = None,
    ) -> tuple[bytes, str, int]:
        """Return (wav_bytes, source, sample_rate). Blocks until GPU (or fallback) done."""
        self.busy = True
        self.last_error = None
        try:
            level = int(max(0, min(5, level)))
            intensity = float(np.clip(intensity, 0.0, 1.0))
            dissonance = float(np.clip(dissonance, 0.0, 1.0))
            seed = int(seed)

            wave = None
            source = "procedural"
            sr = 16000

            if self.musicgen is not None and self.musicgen.available:
                # Match gui_demo: optional per-request token override on the model,
                # then run on the dedicated CUDA worker thread.
                if max_new_tokens is not None:
                    self.musicgen.max_new_tokens = int(max(64, max_new_tokens))

                print(
                    f"[bridge] MusicGen GPU generate L{level} device={self.device} "
                    f"intensity={intensity:.2f} dissonance={dissonance:.2f} "
                    f"seed={seed} tokens={self.musicgen.max_new_tokens} ..."
                )
                t0 = time.perf_counter()
                try:
                    wave = self.musicgen.wait_for_waveform(
                        level,
                        intensity,
                        dissonance,
                        seed,
                        timeout_s=self.generate_timeout_s,
                    )
                    if wave is not None and wave.size > 0:
                        source = "musicgen"
                        sr = int(self.musicgen.sample_rate)
                        self.last_gen_s = time.perf_counter() - t0
                        print(
                            f"[bridge] MusicGen GPU done in {self.last_gen_s:.1f}s "
                            f"samples={wave.size} sr={sr} device={self.device}"
                        )
                except Exception as e:
                    self.last_error = str(e)
                    print(f"[bridge] MusicGen GPU failed, falling back to procedural: {e}")
                    traceback.print_exc()

            if wave is None or wave.size == 0:
                print(f"[bridge] procedural generate L{level} seed={seed}")
                wave = self.procedural.generate_clip(stress_level=level, seed=seed)
                if wave is not None and wave.size > 0:
                    reps = max(1, int(np.ceil(4.0 * 16000 / max(1, wave.size))))
                    wave = np.tile(wave, reps)[: int(4.0 * 16000)]
                    source = "procedural"
                    sr = 16000

            if wave is None or wave.size == 0:
                raise RuntimeError("Both MusicGen and procedural failed")

            peak = float(np.max(np.abs(wave))) + 1e-8
            if peak > 0.95:
                wave = wave * (0.95 / peak)

            wav_bytes = float_to_wav_bytes(wave, sr)
            self.last_source = source
            self.last_duration_s = float(wave.size / max(1, sr))
            return wav_bytes, source, sr
        finally:
            self.busy = False


STATE: BridgeState | None = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[http] " + (fmt % args) + "\n")

    def _send_json(self, code: int, payload: dict) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        assert STATE is not None
        if self.path.startswith("/health"):
            cuda = _cuda_info()
            mg = STATE.musicgen
            self._send_json(200, {
                "ok": True,
                "ready": STATE.ready,
                "busy": STATE.busy,
                "musicgen": bool(mg and mg.available),
                "procedural": STATE.procedural.available,
                "device": STATE.device if mg and mg.available else cuda.get("device"),
                "gpu_name": STATE.gpu_name or cuda.get("gpu_name"),
                "cuda_available": cuda.get("cuda_available"),
                "vram_free_gb": cuda.get("vram_free_gb"),
                "vram_total_gb": cuda.get("vram_total_gb"),
                "last_error": STATE.last_error,
                "last_source": STATE.last_source,
                "last_duration_s": STATE.last_duration_s,
                "last_gen_s": STATE.last_gen_s,
                "model_path": str(STATE.model_path),
                "max_new_tokens": STATE.max_new_tokens,
            })
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        assert STATE is not None
        if not self.path.startswith("/generate"):
            self._send_json(404, {"error": "not found"})
            return
        if STATE.busy:
            self._send_json(409, {"error": "busy generating"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length > 0 else b"{}"
            data = json.loads(body.decode("utf-8") or "{}")
            level = int(data.get("level", 0))
            intensity = float(data.get("intensity", level / 5.0))
            dissonance = float(data.get("dissonance", level / 5.0))
            seed = int(data.get("seed", int(time.time()) % 1_000_000))
            tokens = data.get("max_new_tokens")
            max_tokens = int(tokens) if tokens is not None else None

            wav_bytes, source, sr = STATE.generate(
                level, intensity, dissonance, seed, max_new_tokens=max_tokens
            )
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(wav_bytes)))
            self.send_header("X-Audio-Source", source)
            self.send_header("X-Sample-Rate", str(sr))
            self.send_header("X-Duration-S", f"{STATE.last_duration_s:.3f}")
            self.send_header("X-Device", STATE.device)
            self.send_header("X-Gen-Seconds", f"{STATE.last_gen_s:.3f}")
            if STATE.gpu_name:
                self.send_header("X-GPU-Name", STATE.gpu_name)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(wav_bytes)
        except Exception as e:
            traceback.print_exc()
            self._send_json(500, {"error": str(e)})


def main() -> None:
    global STATE
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="MusicGen Unity HTTP bridge (CUDA)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--model",
        default=str(BASE_DIR / "models" / "musicgen-small"),
        help="Local MusicGen snapshot folder",
    )
    parser.add_argument("--assets", default=str(BASE_DIR / "assets"))
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=192,
        help="Default MusicGen tokens (gui_demo default ~192; higher = longer GPU time)",
    )
    parser.add_argument(
        "--generate-timeout",
        type=float,
        default=180.0,
        help="Seconds to wait for GPU worker to finish one clip",
    )
    parser.add_argument(
        "--require-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail startup unless MusicGen loads on CUDA (default: true)",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[bridge] model path missing: {model_path}", file=sys.stderr)
        sys.exit(1)

    STATE = BridgeState(
        model_path,
        Path(args.assets),
        args.max_new_tokens,
        require_cuda=bool(args.require_cuda),
        generate_timeout_s=float(args.generate_timeout),
    )
    STATE.init_models()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[bridge] listening on http://{args.host}:{args.port} device={STATE.device}")
    print("[bridge] POST /generate  GET /health")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[bridge] shutting down")
        if STATE.musicgen is not None:
            STATE.musicgen.shutdown()
        server.shutdown()


if __name__ == "__main__":
    main()
