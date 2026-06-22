import os
import sys
import time
import random
import math
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.table import Table
from rich.live import Live
from rich.align import Align

from src.iteration2.fusion_orchestrator import FusionOrchestrator
from src.iteration2.stress_schema import STRESS_PROFILES
from src.iteration2.performance import PerfMonitor

CATALOG = Path(__file__).resolve().parent / "assets" / "catalog.json"

STRESS_COLORS = {
    0: "green",
    1: "dark_green",
    2: "yellow",
    3: "dark_orange",
    4: "red",
    5: "bold red",
}

STRESS_BAR_CHARS = " ░▒▓█"

ENTITY_ART_4 = [
    "                          ",
    "           ░░░            ",
    "          ░▒▒░            ",
    "         ░▒▒▒░            ",
    "          ░▒░             ",
    "         ░▒▒▒░            ",
    "        ░▒   ▒░           ",
    "       ░▒     ▒░          ",
    "        ░       ░         ",
    "                          ",
]

ENTITY_ART_5 = [
    "                          ",
    "          ▓█▓█▓           ",
    "         █▓▒▒▓█           ",
    "        █▓▒██▒▓█          ",
    "         ▓▒▒▒▒▓           ",
    "        █▓▒▒▒▒▓█          ",
    "       █▓      ▓█         ",
    "      █▓   ██   ▓█        ",
    "       █          █       ",
    "      █▓          ▓█      ",
]

ENTITY_NONE = ["                          "] * 10

console = Console()


def make_audio_viz(intensity: float, dissonance: float, tick: int, width: int = 50) -> Text:
    txt = Text()
    num_bars = width
    for i in range(num_bars):
        phase = math.sin((i * 0.3) + (tick * 0.4)) * intensity
        noise = random.uniform(-0.3, 0.3) * dissonance
        val = max(0.0, min(1.0, 0.5 + phase + noise))
        bar_height = int(val * 7)
        bar_chars = "▁▂▃▄▅▆▇█"
        char = bar_chars[min(bar_height, len(bar_chars) - 1)]
        if intensity < 0.2:
            color = "green"
        elif intensity < 0.45:
            color = "yellow"
        elif intensity < 0.7:
            color = "dark_orange"
        else:
            color = "red"
        txt.append(char, style=color)
    return txt


def make_env_scene(corruption: float, fog: float, flicker_hz: float, tick: int, width: int = 50, height: int = 10) -> Text:
    rng = random.Random(tick * 7 + 42)
    txt = Text()
    fog_char_pool = " .·:;░"
    corruption_pool = "~^*#%&@"
    clean_pool = " .  .  . "

    flicker_on = math.sin(tick * flicker_hz * 0.5) > -0.2

    for row in range(height):
        for col in range(width):
            r = rng.random()
            if r < corruption:
                char = rng.choice(corruption_pool)
                if corruption > 0.5:
                    color = "red"
                elif corruption > 0.25:
                    color = "dark_orange"
                else:
                    color = "yellow"
            elif r < corruption + fog:
                idx = min(int(fog * (len(fog_char_pool) - 1)), len(fog_char_pool) - 1)
                char = fog_char_pool[idx]
                color = "bright_black"
            else:
                char = rng.choice(clean_pool)
                if flicker_on:
                    color = "white"
                else:
                    color = "bright_black"
            txt.append(char, style=color)
        if row < height - 1:
            txt.append("\n")
    return txt


def make_camera_frame(magnitude: float, tick: int, width: int = 54, height: int = 14) -> Text:
    txt = Text()
    rng = random.Random(tick * 13 + 99)

    border_thick = max(1, int(magnitude * 4))
    noise_density = magnitude * 0.6
    aberration = magnitude > 0.15
    warp = magnitude > 0.35

    for row in range(height):
        for col in range(width):
            in_border = (row < border_thick or row >= height - border_thick or
                         col < border_thick or col >= width - border_thick)

            if in_border:
                if magnitude > 0.5:
                    chars = "▓█░▒"
                    color = "red" if rng.random() > 0.4 else "dark_orange"
                elif magnitude > 0.2:
                    chars = "░▒▓ "
                    color = "yellow"
                else:
                    chars = "░ ░ "
                    color = "bright_black"
                txt.append(rng.choice(chars), style=color)
            elif rng.random() < noise_density:
                if aberration and rng.random() > 0.5:
                    txt.append(rng.choice("░▒"), style="magenta")
                else:
                    txt.append(rng.choice(".:·"), style="bright_black")
            elif warp and rng.random() < 0.08:
                txt.append(rng.choice("~≈"), style="cyan")
            else:
                txt.append(" ")
        if row < height - 1:
            txt.append("\n")
    return txt


def make_entity_panel(level: int, tick: int) -> Text:
    if level >= 5:
        art = ENTITY_ART_5
        color = "bold red"
        label = "ENTITY ACTIVE: silhouette_stare"
    elif level >= 4:
        art = ENTITY_ART_4
        flicker = (tick % 4) < 2
        color = "red" if flicker else "dark_red"
        label = "ENTITY GLIMPSE: shadow_pass"
    else:
        art = ENTITY_NONE
        color = "bright_black"
        label = "no entity"

    txt = Text()
    txt.append(f" {label}\n\n", style=color)
    for i, line in enumerate(art):
        txt.append(line, style=color)
        if i < len(art) - 1:
            txt.append("\n")
    return txt


def make_stress_bar(level: int) -> Text:
    txt = Text()
    for i in range(6):
        if i <= level:
            block = "████"
            color = STRESS_COLORS.get(i, "white")
        else:
            block = "░░░░"
            color = "bright_black"
        txt.append(f" {i}:{block}", style=color)
    return txt


def make_stats_table(frame_data: dict, perf_data: dict, tick: int) -> Table:
    t = Table(show_header=False, box=None, padding=(0, 1))
    t.add_column("key", style="bold cyan", width=18)
    t.add_column("value", width=30)

    se = frame_data["stress_event"]
    ac = frame_data["audio_command"]
    vc = frame_data["visual_command"]
    cc = frame_data["camera_command"]
    ec = frame_data["entity_command"]
    ga = frame_data["generated_assets"]

    t.add_row("Stress Level", f"[{STRESS_COLORS[se['level']]}]{se['level']} ({STRESS_PROFILES[se['level']].name})[/]")
    t.add_row("Audio Asset", f"[yellow]{ga['audio_asset']}[/]")
    t.add_row("Audio Intensity", f"{ac['intensity']:.3f}")
    t.add_row("Visual Asset", f"[dark_orange]{ga['visual_asset']}[/]")
    t.add_row("Corruption", f"{vc['corruption_alpha']:.3f}")
    t.add_row("Fog Density", f"{vc['fog_density']:.3f}")
    t.add_row("Camera Asset", f"[magenta]{ga['camera_asset']}[/]")
    t.add_row("Cam Magnitude", f"{cc['magnitude']:.3f}")
    t.add_row("Entity Asset", f"[red]{ga['entity_asset']}[/]")
    t.add_row("Entity Active", f"{'YES' if ec['enabled'] else 'no'}")
    t.add_row("Adaptive ms", f"{perf_data['adaptive_ms']:.3f}")
    t.add_row("Quality Tier", f"{int(perf_data['quality_tier'])}")
    t.add_row("Frame", f"{tick}")
    return t


def run_visual_demo():
    orchestrator = FusionOrchestrator(catalog_path=CATALOG)
    perf = PerfMonitor()
    start = time.perf_counter()
    tick = 0
    current_level = 0
    pending_input = None

    console.print(Panel(
        "[bold cyan]Iteration 2: Stress-Conditioned Asset Generation Demo[/]\n\n"
        "Type a stress level [bold]0-5[/] and press Enter at any time.\n"
        "The visuals will update in real time.\n"
        "Press [bold]q[/] + Enter to quit.",
        title="Horror Asset Generation",
        border_style="red",
    ))
    console.print()

    import threading
    input_queue = []
    running = True

    def input_thread():
        while running:
            try:
                line = input()
                input_queue.append(line.strip().lower())
            except EOFError:
                break

    t = threading.Thread(target=input_thread, daemon=True)
    t.start()

    with Live(console=console, refresh_per_second=4, screen=False) as live:
        while True:
            while input_queue:
                cmd = input_queue.pop(0)
                if cmd == "q":
                    return
                if cmd in {"0", "1", "2", "3", "4", "5"}:
                    current_level = int(cmd)

            elapsed = time.perf_counter() - start
            perf.begin()
            out = orchestrator.step(requested_level=current_level, now_s=elapsed, mode="manual")
            perf_metrics = perf.end()
            payload = out.to_dict()

            se = payload["stress_event"]
            ac = payload["audio_command"]
            vc = payload["visual_command"]
            cc = payload["camera_command"]
            level = se["level"]
            profile = STRESS_PROFILES[level]

            stress_bar = make_stress_bar(level)
            audio_viz = make_audio_viz(ac["intensity"], profile.dissonance[1], tick)
            env_scene = make_env_scene(vc["corruption_alpha"], vc["fog_density"], vc["light_flicker_hz"], tick)
            camera_frame = make_camera_frame(cc["magnitude"], tick)
            entity_panel = make_entity_panel(level, tick)
            stats = make_stats_table(payload, perf_metrics, tick)

            layout = Layout()
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="body"),
                Layout(name="footer", size=3),
            )

            header_text = Text()
            header_text.append("  STRESS: ", style="bold white")
            header_text.append_text(stress_bar)
            header_text.append(f"    t={elapsed:.1f}s", style="bright_black")
            layout["header"].update(Panel(header_text, border_style=STRESS_COLORS[level]))

            layout["body"].split_row(
                Layout(name="visuals", ratio=3),
                Layout(name="sidebar", ratio=1),
            )

            layout["visuals"].split_column(
                Layout(name="audio_panel", size=5),
                Layout(name="main_panels"),
            )

            layout["audio_panel"].update(Panel(audio_viz, title="Audio Waveform", border_style="yellow"))

            layout["main_panels"].split_row(
                Layout(name="env_panel"),
                Layout(name="right_stack"),
            )
            layout["env_panel"].update(Panel(env_scene, title="Environment", border_style="dark_orange"))

            layout["right_stack"].split_column(
                Layout(name="camera_panel"),
                Layout(name="entity_panel"),
            )
            layout["camera_panel"].update(Panel(camera_frame, title="Camera Distortion", border_style="magenta"))
            layout["entity_panel"].update(Panel(entity_panel, title="Entity", border_style="red"))

            layout["sidebar"].update(Panel(stats, title="Live Stats", border_style="cyan"))

            footer_text = Text()
            footer_text.append("  Input stress [0-5] + Enter  |  q = quit  |  ", style="bright_black")
            footer_text.append(f"Current input: {current_level}", style=f"bold {STRESS_COLORS[level]}")
            layout["footer"].update(Panel(footer_text, border_style="bright_black"))

            live.update(layout)
            tick += 1
            time.sleep(0.25)


if __name__ == "__main__":
    try:
        run_visual_demo()
    except KeyboardInterrupt:
        console.print("\n[bold red]Demo stopped.[/]")
