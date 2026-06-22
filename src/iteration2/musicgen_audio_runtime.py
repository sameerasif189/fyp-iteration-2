"""
Pretrained Meta MusicGen (Hugging Face) for runtime horror ambience — higher fidelity
than the small WaveGAN. Stress level + fusion scalars steer text prompts (0 ambient
→ 5 full horror), matching ``STRESS_PROFILES`` semantics.

Requires: ``pip install transformers accelerate``. Offline: snapshot weights to a folder
and load with ``local_files_only=True`` (``gui_demo.py`` defaults to local/offline MusicGen).
Use ``--audio-remote-ok`` in ``gui_demo`` to allow Hugging Face Hub/cache downloads.

Generation runs on a daemon thread so the pygame loop stays responsive.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any

import numpy as np

NUM_STRESS_LEVELS = 6

# Base prompts: horror *sound design* beds — avoid orchestral score / song language (model skews musical).
# L0--L3 are single prompts; L4 and L5 are pools of distinct setting variants
# so consecutive renders sound noticeably different instead of converging on a
# single "horror smear". A seed picks one variant per render in build_musicgen_prompt.
LEVEL_PROMPTS: dict[int, list[str]] = {
    0: [
        (
            "barely audible abandoned building air and duct rumble faint voltage hum dusty room tone "
            "very slow infra sub drift no rhythm no tonal center documentary horror ambience field recording"
        ),
    ],
    1: [
        (
            "empty corridor narrow wind cavity distant dripping water metal pings off concrete "
            "radio static washes irregular floor creaks creeping dread pure sound design no melody no chords"
        ),
    ],
    2: [
        (
            "pressure in the ears low mechanical whirr broken fluorescent buzz steam pipe knock "
            "granular hiss layers uncomfortable close-mic friction arrhythmic micro-scares no percussion groove"
        ),
    ],
    3: [
        (
            "tightening dread bandpass rising reversed metal shriek distant boiler rumble "
            "irregular sloppy pulse not a heartbeat distorted groan biomechanical wet resonance "
            "no melody no human voices"
        ),
    ],
    # Level 4 -- ~6 distinct horror "settings". Each is a complete sound design
    # bed without any musical scoring. No human voices, no screams, no choir.
    4: [
        (
            "abandoned industrial basement: metal stress groans concrete dust drifts heavy steel doors "
            "scraping across rebar pipe condensation drips into puddles dim machinery hum no music no vocals"
        ),
        (
            "derelict spaceship corridor: failing ventilation rattles cracked hull pings reactor coolant "
            "hissing through ruptured seams electrical arcing pops sub-bass hull creak no song no human voices"
        ),
        (
            "decaying carcass cavity: thick wet biology squelches gristle separation dry chitin clatter "
            "swarming insect chittering rib-cage resonance no melody no human screaming no choir"
        ),
        (
            "flooded sewer with rusted machinery: dripping echo through tile and brick distorted reverb "
            "metal turbine groan low frequency water sloshing rats scratching no orchestra no song"
        ),
        (
            "haunted forest at midnight: cracking branches under unseen weight wind through dead leaves "
            "owl calls warped through distance distant unidentified animal growl no melody no human cries"
        ),
        (
            "ruined power plant: transformer surges concrete dust falling crystallized metal stress "
            "intermittent klaxon decay copper whine fluorescent ballast crackle no drum kit no song"
        ),
    ],
    # Level 5 -- 6 distinct catastrophic horror beds. Maximum density without
    # tipping into musical score. Diversity is the point: explosion, collapse,
    # alien hive, hellish furnace, etc.
    5: [
        (
            "structural collapse: concrete shearing rebar snapping under enormous load dust avalanche "
            "secondary debris rain glass cascades subsonic pressure shockwave no music no vocals"
        ),
        (
            "hellish industrial furnace: roaring open flame steel slag pouring chains rattling on metal "
            "deafening boiler vent release distorted bass groan rhythmic only by pure chance not music"
        ),
        (
            "alien hive interior: layered wet chitin scrape distant mass of organisms breathing in unison "
            "fluid pulse through membrane walls dry crystalline crack no choir no song no melody"
        ),
        (
            "violent storm against ruined structure: torrential rain on corrugated metal sheet lightning "
            "thunderclap window glass blowing in concrete dust whipped by wind no orchestra no human voices"
        ),
        (
            "deep cavern with creature: enormous reverb tail crushing footsteps subterranean rock fall "
            "thick wet grunt distant reverberant snarl gravel and bone underfoot no song no music no vocals"
        ),
        (
            "warzone rubble at dusk: distant explosion concussions debris settling structural fires crackling "
            "drone of damaged engines metal twisting pure documentary sound effects no theme music no choir"
        ),
    ],
}


# Anti-music + anti-vocal anchor appended every render. The vocal clauses are
# important because MusicGen's training data is heavy on songs, and without
# explicit suppression it tends to emit human cries (which we explicitly do not
# want — the demo is environmental horror, not human suffering).
_HORROR_SOUND_DESIGN_ANCHOR = (
    "absolutely not a song: no catchy melody no chord progression no groovy drums no orchestral score. "
    "no human voices no female voices no male voices no screaming no crying no moaning no whispering "
    "no choir no singing no chanting no spoken word. "
    "harsh gritty documentary horror sound design bed, atonal, arrhythmic, mono-compatible, rumble-forward mix"
)


def _clip_level(level: int) -> int:
    return max(0, min(NUM_STRESS_LEVELS - 1, int(level)))


def guidance_for_stress_level(level: int, base: float) -> float:
    """Stronger prompt conditioning at high stress (esp. 4--5)."""
    lvl = _clip_level(level)
    bump = 0.12 * lvl
    if lvl >= 4:
        bump += 0.38 + 0.10 * float(lvl - 4)
    return float(min(9.8, base + bump))


def build_musicgen_prompt(
    level: int,
    audio_intensity: float,
    dissonance: float,
    seed: int = 0,
) -> str:
    """Merge a discrete tier setting with fusion audio heads.

    For L4/L5 (where multiple setting variants exist), ``seed`` picks one
    deterministically so consecutive renders explore different horror settings
    (industrial basement, derelict ship, hive, furnace, etc.) instead of
    converging on a single texture. Anti-music + anti-vocal anchors are
    always appended last so they dominate the conditioning.
    """
    lvl = _clip_level(level)
    ai = float(np.clip(audio_intensity, 0.0, 1.0))
    dis = float(np.clip(dissonance, 0.0, 1.0))
    tier = lvl / max(1, NUM_STRESS_LEVELS - 1)
    eff_ai = float(np.clip(ai * (0.78 + 0.22 * tier), 0.0, 1.0))
    eff_dis = float(np.clip(dis * (0.78 + 0.22 * tier), 0.0, 1.0))

    variants = LEVEL_PROMPTS[lvl]
    if len(variants) == 1:
        base = variants[0]
    else:
        base = variants[int(seed) % len(variants)]

    extra: list[str] = []

    mid = lvl + eff_ai * 0.38 + eff_dis * 0.38
    if mid < 1.28:
        extra.append(
            "very quiet restrained dynamics microscopic transients long empty silences bleak dead air"
        )
    if lvl <= 2:
        extra.append("keep level low uneasy space between events no tonal hook no melodic instrument")
    elif lvl == 3:
        extra.append(
            "building physical pressure midrange grime stinging highs no melodic solo no tonal hook"
        )
    elif lvl == 4:
        # Deliberately phrased to push MusicGen away from "intense music" and
        # toward sound-effects collage. The model otherwise loves orchestral pads.
        extra.append(
            "pure environmental horror sound effects: extreme transient violence "
            "smeared distorted bandwidth crushing lows ear-fatigue noise. "
            "this is not a music track, no theme, no rhythmic loop, no drum hits, no bassline"
        )
    else:
        extra.append(
            "unrelenting harsh noise-floor brutality visceral disgusting textures "
            "apocalyptic abrasion. this is not music, never resolves like a soundtrack theme, "
            "no rhythmic groove, no drone-melody, no chord pad"
        )

    if eff_ai > 0.58 + 0.06 * max(0, lvl - 3):
        extra.append("punishing loud bursts of destruction impacts and ripping distortion strips")
    if eff_dis > 0.52 + 0.06 * max(0, lvl - 3):
        extra.append(
            "sickening atonal scrape microtonal drift phasey ugly harmonics granular crackle "
            "radiator steam copper whine"
        )

    suffix = "; ".join(extra) if extra else "narrow claustrophobic stereo rust and damp concrete"
    return f"{base}. {suffix}. {_HORROR_SOUND_DESIGN_ANCHOR}."


def _to_mono_float32(wave: np.ndarray) -> np.ndarray:
    x = np.asarray(wave, dtype=np.float64)
    x = np.squeeze(x)
    if x.ndim == 2:
        x = np.mean(x, axis=0)
    x = np.clip(x.astype(np.float32), -1.0, 1.0)
    x -= float(np.mean(x))
    peak = float(np.max(np.abs(x))) + 1e-9
    if peak > 0.999:
        x = x / peak
    return x.astype(np.float32)


class MusicGenRealtimeGen:
    """
    Loads MusicGen on the foreground thread once, then renders clips on a daemon worker.
    Feed jobs with ``enqueue`` / ``enqueue_replace``; read completed numpy mono float buffers
    with ``poll_waveform``.
    """

    def __init__(
        self,
        model_id: str = "facebook/musicgen-small",
        max_new_tokens: int = 512,
        guidance_scale: float = 3.2,
        local_files_only: bool = False,
        *,
        defer_init: bool = False,
    ):
        self.model_id = model_id
        self.max_new_tokens = int(max(64, max_new_tokens))
        self.guidance_scale = float(guidance_scale)
        self._local_files_only = bool(local_files_only)

        self._processor: Any | None = None
        self._model: Any | None = None
        self._device: str | None = None
        self.sample_rate = 32000

        self._job_q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=2)
        self._out_q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._load_error: str | None = None
        self._init_thread: threading.Thread | None = None

        if defer_init:
            self._init_thread = threading.Thread(
                target=self._complete_initialization,
                name="musicgen-init",
                daemon=True,
            )
            self._init_thread.start()
        else:
            self._complete_initialization()

    def _complete_initialization(self) -> None:
        t0 = time.perf_counter()
        self._load_weights()
        if self.available:
            self._spawn_worker()
        load_s = time.perf_counter() - t0
        if self.available:
            print(f"[musicgen] ready on {self._device} (load_wall={load_s:.1f}s, sr={self.sample_rate})")
        elif self._load_error:
            print(f"[musicgen] init finished without model: {self._load_error[:200]}")

    def _load_weights(self) -> None:
        try:
            import torch  # noqa: WPS433
            from transformers import AutoProcessor  # noqa: WPS433
            from transformers import MusicgenForConditionalGeneration  # noqa: WPS433
        except ImportError as e:
            self._load_error = f"imports failed ({e}); pip install transformers accelerate"
            print(f"[musicgen] {self._load_error}")
            return

        try:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            if self._device == "cpu":
                print("[musicgen] CUDA not available — using CPU (expect very slow generation).")
            else:
                # 5060 / Blackwell loves TF32 + tuned cuDNN; SDPA also gives fused QKV
                # attention which is the main MusicGen-small bottleneck.
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cuda.matmul.allow_tf32 = True
                try:
                    torch.set_float32_matmul_precision("high")
                except Exception:
                    pass
                try:
                    free_b, total_b = torch.cuda.mem_get_info()
                    gpu_name = torch.cuda.get_device_name()
                    print(
                        f"[musicgen] GPU {gpu_name} "
                        f"vram_free={free_b/1024**3:.2f} GiB / {total_b/1024**3:.2f} GiB"
                    )
                except Exception:
                    pass

            dtype = torch.float16 if self._device == "cuda" else torch.float32
            lf = self._local_files_only
            self._processor = AutoProcessor.from_pretrained(self.model_id, local_files_only=lf)

            # Try SDPA attention first (fast fused kernels), fall back to default eager
            # if the installed transformers version doesn't support the kwarg.
            attn_impl_used = "default"
            try:
                self._model = MusicgenForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype,
                    local_files_only=lf,
                    attn_implementation="sdpa",
                )
                attn_impl_used = "sdpa"
            except (TypeError, ValueError):
                self._model = MusicgenForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype,
                    local_files_only=lf,
                )
            self._model.to(self._device)
            self._model.eval()
            # Reserve a TF32 cuda graph slot so the first inference doesn't pay
            # allocator + jit-cache cost at the same time.
            if self._device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            ae = getattr(self._model.config, "audio_encoder", None)
            if ae is not None and getattr(ae, "sampling_rate", None):
                self.sample_rate = int(ae.sampling_rate)
            print(f"[musicgen] attention={attn_impl_used} dtype={dtype}")
        except Exception as e:
            self._processor = None
            self._model = None
            self._device = None
            self._load_error = str(e)
            print(f"[musicgen] checkpoint load failed: {e}")

    def _spawn_worker(self) -> None:
        assert self._model is not None and self._processor is not None
        self._worker = threading.Thread(target=self._worker_loop, name="musicgen-worker", daemon=True)
        self._worker.start()

    @property
    def available(self) -> bool:
        return self._model is not None and self._processor is not None

    @property
    def last_load_error(self) -> str | None:
        return self._load_error

    @property
    def is_loading(self) -> bool:
        """Checkpoint / worker bootstrap still running on the init thread."""
        if self._init_thread is None:
            return False
        return self._init_thread.is_alive()

    def enqueue_replace(self, level: int, audio_intensity: float, dissonance: float, seed: int) -> None:
        if not self.available:
            return
        prompt = build_musicgen_prompt(level, audio_intensity, dissonance, seed=int(seed))
        lvl = _clip_level(level)
        tokens = self.max_new_tokens
        if lvl >= 4:
            tokens = int(min(1024, tokens + 32 + (lvl - 4) * 48))
        elif lvl >= 2:
            tokens = int(min(1024, tokens + int(12 * (lvl - 1))))
        payload = dict(
            level=lvl,
            prompt=prompt,
            seed=int(seed % (2**31)),
            max_new_tokens=int(max(64, tokens)),
            guidance_scale=guidance_for_stress_level(level, self.guidance_scale),
        )
        while True:
            try:
                self._job_q.put_nowait(payload)
                return
            except queue.Full:
                try:
                    _ = self._job_q.get_nowait()
                except queue.Empty:
                    pass

    def poll_waveform(self) -> np.ndarray | None:
        result = self.poll_result()
        if result is None:
            return None
        wave = result.get("wave")
        return wave if isinstance(wave, np.ndarray) else None

    def poll_result(self) -> dict[str, Any] | None:
        """Return the latest generated waveform with its stress-level metadata."""
        try:
            return self._out_q.get_nowait()
        except queue.Empty:
            return None

    def generate_now(
        self,
        level: int,
        audio_intensity: float,
        dissonance: float,
        seed: int,
        *,
        max_new_tokens: int | None = None,
    ) -> np.ndarray | None:
        """Synchronous one-shot generation on the *calling thread*.

        Used by the demo to pre-render one bed per stress level at startup so
        the runtime always has audio ready when a level is first entered, even
        before the async worker has had time to produce one. Returns a mono
        float32 waveform at ``self.sample_rate`` or ``None`` on failure.
        """
        if not self.available:
            return None
        try:
            import torch  # noqa: WPS433
        except ImportError:
            return None
        prompt = build_musicgen_prompt(level, audio_intensity, dissonance, seed=int(seed))
        lvl = _clip_level(level)
        tokens = self.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        if lvl >= 4:
            tokens = int(min(1024, tokens + 32 + (lvl - 4) * 48))
        elif lvl >= 2:
            tokens = int(min(1024, tokens + int(12 * (lvl - 1))))
        gs = float(np.clip(guidance_for_stress_level(level, self.guidance_scale), 1.0, 12.0))
        try:
            s32 = int(seed) % (2**31)
            torch.manual_seed(s32)
            if self._device == "cuda":
                torch.cuda.manual_seed_all(s32)
            batch = self._processor(text=[prompt], padding=True, return_tensors="pt")
            input_ids = batch["input_ids"].to(self._device)
            attn = batch.get("attention_mask")
            if attn is not None:
                attn = attn.to(self._device)
            with torch.inference_mode():
                wav = self._model.generate(
                    input_ids,
                    attention_mask=attn,
                    do_sample=True,
                    max_new_tokens=int(max(64, tokens)),
                    guidance_scale=gs,
                )
            wav_np = wav.float().detach().cpu().numpy()
            return _to_mono_float32(wav_np)
        except Exception as e:
            print(f"[musicgen] generate_now L{level} failed: {e}")
            return None

    def shutdown(self) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
        if self._init_thread is not None:
            self._init_thread.join(timeout=3.0)

    def _worker_loop(self) -> None:
        import torch

        model = self._model
        processor = self._processor
        assert model is not None and processor is not None

        while not self._stop.is_set():
            try:
                job = self._job_q.get(timeout=0.35)
            except queue.Empty:
                continue
            prompt = job["prompt"]
            level = int(job.get("level", 0))
            seed = int(job["seed"])
            mx = int(job.get("max_new_tokens", self.max_new_tokens))
            gs = float(np.clip(job.get("guidance_scale", self.guidance_scale), 1.0, 12.0))

            try:
                s32 = seed % (2**31)
                torch.manual_seed(s32)
                if self._device == "cuda":
                    torch.cuda.manual_seed_all(s32)

                batch = processor(text=[prompt], padding=True, return_tensors="pt")
                input_ids = batch["input_ids"].to(self._device)
                attn = batch.get("attention_mask")
                if attn is not None:
                    attn = attn.to(self._device)
                # MusicGen.generate does not accept ``generator=`` on recent transformers —
                # seeding globals above keeps runs reproducible enough.
                gen_kwargs = dict(
                    do_sample=True,
                    max_new_tokens=mx,
                    guidance_scale=gs,
                )
                with torch.inference_mode():
                    wav = model.generate(input_ids, attention_mask=attn, **gen_kwargs)
                wav_np = wav.float().detach().cpu().numpy()
                mono = _to_mono_float32(wav_np)
                while not self._out_q.empty():
                    try:
                        _ = self._out_q.get_nowait()
                    except queue.Empty:
                        break
                self._out_q.put_nowait(
                    {
                        "level": _clip_level(level),
                        "wave": mono,
                        "seed": seed,
                        "duration_s": float(mono.size / max(1, self.sample_rate)),
                    }
                )
            except Exception as e:
                print(f"[musicgen] synthesis error: {e}")
