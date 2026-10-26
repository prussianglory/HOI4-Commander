"""Захват клиентской области и однозначное распознавание сохранённых шаблонов."""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Callable, Protocol

import cv2
import numpy as np
from pydantic import Field

from .models import Observation, Rect, StrictModel, Target, WindowInfo
from .safety import SafetyError


class WindowBackend(Protocol):
    def current_window(self) -> WindowInfo: ...


class TemplateSpec(StrictModel):
    id: str = Field(min_length=1, max_length=128, pattern=r"^[a-z][a-z0-9_]*$")
    file: str = Field(min_length=1)
    threshold: float = Field(default=0.97, ge=0.9, le=1.0)


class TemplateManifest(StrictModel):
    synthetic: bool = False
    templates: tuple[TemplateSpec, ...]
    frames: tuple[str, ...] = ()


def contained_file(root: Path, filename: str) -> Path:
    """Не разрешает выход из каталога калибровки, в том числе через симлинк."""
    path = (root / filename).resolve()
    if not path.is_relative_to(root.resolve()) or Path(filename).is_absolute():
        raise ValueError("Файл шаблона должен находиться внутри каталога манифеста")
    if not path.is_file():
        raise ValueError("Файл шаблона или кадра отсутствует")
    return path


def read_bgr(path: Path) -> np.ndarray:
    """Читает PNG и пути с кириллицей одинаково на Linux и Windows."""
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Не удалось прочитать изображение")
    return image


class TemplateMatcher:
    def __init__(self, manifest: TemplateManifest, root: Path):
        self.manifest = manifest
        self.synthetic = manifest.synthetic
        self.root = root.resolve()
        self.templates: dict[str, tuple[TemplateSpec, np.ndarray]] = {}
        for spec in manifest.templates:
            if spec.id in self.templates:
                raise ValueError("Идентификаторы шаблонов должны быть уникальными")
            image = read_bgr(contained_file(self.root, spec.file))
            if image.shape[0] < 4 or image.shape[1] < 4 or float(image.std(axis=(0, 1)).mean()) < 3:
                raise ValueError("Шаблон слишком мал или не содержит различимых деталей")
            self.templates[spec.id] = (spec, image)
        if not self.templates:
            raise ValueError("Набор шаблонов пуст")

    @classmethod
    def from_manifest(cls, path: Path | str) -> TemplateMatcher:
        path = Path(path)
        return cls(TemplateManifest.model_validate_json(path.read_text(encoding="utf-8")), path.parent)

    def detect(self, image: np.ndarray, frame_id: str) -> tuple[dict[str, Target], frozenset[str]]:
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("Ожидается цветной кадр BGR uint8")
        found: dict[str, Target] = {}
        ambiguous: set[str] = set()
        height, width = image.shape[:2]
        for name, (spec, template) in self.templates.items():
            th, tw = template.shape[:2]
            if th > height or tw > width:
                continue
            result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
            # Допускается лишь пиксельный разброс одного пика, а не вся ширина значка.
            # Близкие самостоятельные совпадения тоже запрещают выбор цели.
            peaks: list[tuple[int, int, float]] = []
            work = result.copy()
            while len(peaks) < 2:
                _, score, _, point = cv2.minMaxLoc(work)
                if not np.isfinite(score) or score < spec.threshold:
                    break
                x, y = point
                peaks.append((x, y, score))
                x0, x1 = max(0, x - 1), min(work.shape[1], x + 2)
                y0, y1 = max(0, y - 1), min(work.shape[0], y + 2)
                work[y0:y1, x0:x1] = -1
            if len(peaks) > 1:
                ambiguous.add(name)
            elif peaks:
                x, y, score = peaks[0]
                found[name] = Target(id=name, rect=Rect(x=x, y=y, width=tw, height=th), confidence=min(1, max(0, score)), frame_id=frame_id)
        return found, frozenset(ambiguous)


def capture_client(window: WindowInfo) -> np.ndarray:
    """Снимает только клиентскую область выбранного окна, без записи на диск."""
    import mss

    rect = window.client
    with mss.mss() as screen:
        image = np.asarray(screen.grab({"left": rect.x, "top": rect.y, "width": rect.width, "height": rect.height}))
        return np.ascontiguousarray(image[:, :, :3])


class Observer:
    def __init__(self, window_backend: WindowBackend, matcher: TemplateMatcher, capture: Callable[[WindowInfo], np.ndarray] | None = None):
        self.window_backend = window_backend
        self.matcher = matcher
        self.capture = capture or capture_client

    def observe(self) -> Observation:
        window = self.window_backend.current_window()
        captured_at = time.monotonic()
        image = self.capture(window)
        if self.window_backend.current_window() != window:
            raise SafetyError("Активное окно изменилось во время захвата; кадр отброшен")
        if image.shape[:2] != (window.client.height, window.client.width):
            raise ValueError("Размер захвата не совпадает с клиентской областью")
        frame_id = uuid.uuid4().hex
        targets, ambiguous = self.matcher.detect(image, frame_id)
        return Observation(frame_id, captured_at, window, image, targets, ambiguous)


class SavedImageObserver(Observer):
    """Проверяет распознавание файла; не подтверждает работу захвата реальной игры."""
    def __init__(self, path: Path, window: WindowInfo, matcher: TemplateMatcher):
        class SavedWindow:
            def current_window(self) -> WindowInfo:
                return window

        super().__init__(SavedWindow(), matcher, lambda _: read_bgr(path))
