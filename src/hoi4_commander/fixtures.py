"""Явно синтетические кадры для проверки цикла без запущенной игры."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .models import Rect, WindowInfo
from .observer import Observer, TemplateMatcher, contained_file, read_bgr

ANCHORS = ("nation_germany", "date_1936_01_01", "game_paused")
SEQUENCE = (
    "focus_menu", "focus_available", "focus_start", "research_menu", "research_slot",
    "technology_available", "technology_start", "construction_menu", "civilian_factory_tool", "building_state",
)
MARKERS = ("focus_active", "research_active", "construction_queued")


def write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Не удалось создать тестовый PNG")
    path.write_bytes(encoded.tobytes())


def create_demo(directory: Path | str) -> Path:
    """Создаёт 11 PNG с переходами мирного сценария, без игровых ресурсов."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    # Повторный запуск использует только собственный, явно помеченный набор.
    if manifest_path.exists():
        matcher = TemplateMatcher.from_manifest(manifest_path)
        if not matcher.synthetic:
            raise ValueError("Нельзя перезаписать пользовательскую калибровку")
        return manifest_path
    rng = np.random.default_rng(1936)
    names = ANCHORS + SEQUENCE + MARKERS
    templates: dict[str, np.ndarray] = {}
    specs = []
    for index, name in enumerate(names):
        path = directory / f"template_{name}.png"
        if path.exists():
            raise ValueError("Каталог содержит посторонний файл; выберите пустой каталог")
        image = rng.integers(10, 245, size=(24, 36, 3), dtype=np.uint8)
        templates[name] = image
        write_png(path, image)
        specs.append({"id": name, "file": path.name, "threshold": 0.99})
    frames = []
    for state in range(len(SEQUENCE) + 1):
        image = np.full((600, 800, 3), (24, 30, 35), dtype=np.uint8)
        cv2.putText(image, "SYNTHETIC TEST - NOT HOI4", (70, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
        visible = list(ANCHORS)
        if state < len(SEQUENCE):
            visible.append(SEQUENCE[state])
        if state >= 3:
            visible.append("focus_active")
        if state >= 7:
            visible.append("research_active")
        if state >= 10:
            visible.append("construction_queued")
        for name in visible:
            index = names.index(name)
            x = 35 + (index % 5) * 145
            y = 120 + (index // 5) * 100
            image[y:y+24, x:x+36] = templates[name]
            cv2.putText(image, name[:20], (x, y+45), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1)
        path = directory / f"frame_{state:02}.png"
        if path.exists():
            raise ValueError("Каталог содержит посторонний файл; выберите пустой каталог")
        write_png(path, image)
        frames.append(path.name)
    manifest_path.write_text(json.dumps({"synthetic": True, "templates": specs, "frames": frames}, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


class DemoWindowBackend:
    def current_window(self) -> WindowInfo:
        return WindowInfo(handle=1, title="Синтетическая HOI4 — тест", process_name="hoi4.exe", client=Rect(x=100, y=100, width=800, height=600))


class DemoObserver(Observer):
    def __init__(self, directory: Path | str | None = None):
        self._temporary = tempfile.TemporaryDirectory(prefix="hoi4_demo_") if directory is None else None
        directory = Path(self._temporary.name) if self._temporary else Path(directory)
        manifest_path = create_demo(directory)
        matcher = TemplateMatcher.from_manifest(manifest_path)
        if len(matcher.manifest.frames) != len(SEQUENCE) + 1:
            raise ValueError("Неверное число синтетических состояний")
        self.frames = tuple(read_bgr(contained_file(directory, name)) for name in matcher.manifest.frames)
        self.state = 0
        super().__init__(DemoWindowBackend(), matcher, lambda _: self.frames[self.state].copy())

    def advance(self) -> None:
        """Только мок-ввод переводит среду в следующее тестовое состояние."""
        if self.state >= len(self.frames) - 1:
            raise RuntimeError("Синтетический сценарий уже завершён")
        self.state += 1
