import json
from pathlib import Path
from typing import Dict, List


class PreloadedAssetBank:
    def __init__(self, catalog_path: Path):
        self.catalog_path = catalog_path
        with catalog_path.open("r", encoding="utf-8") as f:
            self.catalog: Dict[str, Dict[str, List[str]]] = json.load(f)

        self._active_cache: Dict[str, List[str]] = {}
        self._warm_cache: Dict[str, List[str]] = {}

    def get_assets(self, domain: str, level: int) -> List[str]:
        level_key = str(max(0, min(5, level)))
        return self.catalog.get(domain, {}).get(level_key, [])

    def warm_levels(self, level: int) -> None:
        levels = [max(0, min(5, x)) for x in (level - 1, level, level + 1)]
        self._active_cache.clear()
        self._warm_cache.clear()
        for domain in ("audio", "visual", "camera", "entity"):
            self._active_cache[domain] = self.get_assets(domain, level)
            warm = []
            for lv in levels:
                warm.extend(self.get_assets(domain, lv))
            self._warm_cache[domain] = sorted(set(warm))

    @property
    def active_cache_size(self) -> int:
        return sum(len(v) for v in self._active_cache.values())

    @property
    def warm_cache_size(self) -> int:
        return sum(len(v) for v in self._warm_cache.values())

