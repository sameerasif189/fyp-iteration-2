"""
Download free 3D meshes and create entity mesh catalog.

These assets are from public sample libraries and are usable for demo/prototyping.
"""

import argparse
import json
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent
MESH_DIR = BASE_DIR / "assets" / "entity_meshes"
CATALOG_PATH = MESH_DIR / "catalog.json"

MESH_SOURCES = [
    {
        "design": "wraith",
        "filename": "wraith_monster.glb",
        "url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/1.0/Monster/glTF-Binary/Monster.glb",
        "source": "Khronos glTF Sample Models",
        "license": "CC-BY 4.0 (verify attribution requirements)",
    },
    {
        "design": "brute",
        "filename": "brute_xbot.glb",
        "url": "https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/models/gltf/Xbot.glb",
        "source": "three.js example assets",
        "license": "Repository license (verify before redistribution)",
    },
    {
        "design": "crawler",
        "filename": "crawler_brainstem.glb",
        "url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/main/2.0/BrainStem/glTF-Binary/BrainStem.glb",
        "source": "Khronos glTF Sample Models",
        "license": "CC-BY 4.0 (verify attribution requirements)",
    },
]


def download_meshes(timeout_s: int = 60):
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    catalog = {}
    for item in MESH_SOURCES:
        out_path = MESH_DIR / item["filename"]
        print(f"[mesh] downloading {item['design']} -> {item['filename']}")
        resp = requests.get(item["url"], timeout=timeout_s)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        catalog[item["filename"]] = {
            "design": item["design"],
            "local_path": str(out_path.relative_to(BASE_DIR)).replace("/", "\\"),
            "url": item["url"],
            "source": item["source"],
            "license": item["license"],
        }
    CATALOG_PATH.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    print(f"[mesh] wrote catalog: {CATALOG_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    download_meshes(timeout_s=args.timeout)
