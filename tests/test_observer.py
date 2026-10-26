"""Проверки шаблонов на сохранённых синтетических изображениях."""
import json
from pathlib import Path

import numpy as np
import pytest

from hoi4_commander.fixtures import ANCHORS, SEQUENCE, DemoObserver, create_demo, write_png
from hoi4_commander.observer import Observer, SavedImageObserver, TemplateMatcher
from hoi4_commander.safety import SafetyError


def test_saved_images_complete_scenario(tmp_path):
    observer = DemoObserver(tmp_path)
    for state, expected in enumerate(SEQUENCE):
        observed = observer.observe()
        assert not observed.ambiguous
        assert set(ANCHORS).issubset(observed.targets)
        assert expected in observed.targets
        assert ("focus_active" in observed.targets) == (state >= 3)
        assert ("research_active" in observed.targets) == (state >= 7)
        for target in observed.targets.values():
            x, y = target.center
            assert 0 <= x < 800 and 0 <= y < 600
            assert target.frame_id == observed.frame_id
        observer.advance()
    assert "construction_queued" in observer.observe().targets
    with pytest.raises(RuntimeError):
        observer.advance()


def test_saved_image_observer(tmp_path):
    demo = DemoObserver(tmp_path)
    saved = SavedImageObserver(tmp_path / "frame_00.png", demo.window_backend.current_window(), demo.matcher)
    observed = saved.observe()
    assert "focus_menu" in observed.targets
    assert observed.window.client.x == 100


def test_duplicate_occurrences_are_ambiguous(tmp_path):
    demo = DemoObserver(tmp_path)
    image = demo.frames[0].copy()
    spec, template = demo.matcher.templates["focus_menu"]
    image[500:524, 700:736] = template
    targets, ambiguous = demo.matcher.detect(image, "test")
    assert "focus_menu" not in targets
    assert "focus_menu" in ambiguous


def test_missing_template_returns_no_target(tmp_path):
    demo = DemoObserver(tmp_path)
    targets, ambiguous = demo.matcher.detect(np.zeros((600, 800, 3), dtype=np.uint8), "empty")
    assert not targets
    assert not ambiguous


def test_template_larger_than_frame_is_not_found(tmp_path):
    demo = DemoObserver(tmp_path)
    targets, ambiguous = demo.matcher.detect(np.zeros((8, 8, 3), dtype=np.uint8), "tiny")
    assert not targets and not ambiguous


@pytest.mark.parametrize("filename", ["../outside.png", "/tmp/outside.png"])
def test_manifest_rejects_escaping_paths(tmp_path, filename):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"templates": [{"id": "target", "file": filename}]}))
    with pytest.raises(ValueError, match="внутри"):
        TemplateMatcher.from_manifest(path)


def test_constant_template_is_rejected(tmp_path):
    write_png(tmp_path / "empty.png", np.full((20, 20, 3), (17, 80, 230), dtype=np.uint8))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"templates": [{"id": "empty", "file": "empty.png"}]}))
    with pytest.raises(ValueError, match="деталей"):
        TemplateMatcher.from_manifest(manifest)


def test_threshold_cannot_be_disabled(tmp_path):
    manifest = create_demo(tmp_path)
    data = json.loads(manifest.read_text())
    data["templates"][0]["threshold"] = 0.1
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        TemplateMatcher.from_manifest(manifest)


def test_create_demo_does_not_overwrite_user_file(tmp_path):
    victim = tmp_path / "template_nation_germany.png"
    victim.write_bytes(b"existing")
    with pytest.raises(ValueError):
        create_demo(tmp_path)
    assert victim.read_bytes() == b"existing"


def test_focus_change_during_capture_discards_frame(tmp_path):
    demo = DemoObserver(tmp_path)

    class MovingWindow:
        def __init__(self):
            self.calls = 0

        def current_window(self):
            self.calls += 1
            window = demo.window_backend.current_window()
            return window if self.calls == 1 else window.model_copy(update={"handle": 2})

    observer = Observer(MovingWindow(), demo.matcher, lambda _: demo.frames[0].copy())
    with pytest.raises(SafetyError, match="изменилось"):
        observer.observe()


def test_close_duplicate_peaks_fail_closed(tmp_path):
    # Периодический рисунок допускает совпадения через два пикселя: выбор неоднозначен.
    pattern = np.indices((10, 10)).sum(axis=0) % 2 * 200
    template = np.repeat(pattern[:, :, None], 3, axis=2).astype(np.uint8)
    write_png(tmp_path / "periodic.png", template)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"templates": [{"id": "periodic", "file": "periodic.png", "threshold": 0.99}]}))
    matcher = TemplateMatcher.from_manifest(manifest)
    image = np.zeros((30, 30, 3), dtype=np.uint8)
    repeated = np.indices((10, 12)).sum(axis=0) % 2 * 200
    image[5:15, 5:17] = np.repeat(repeated[:, :, None], 3, axis=2)
    targets, ambiguous = matcher.detect(image, "close")
    assert "periodic" not in targets
    assert "periodic" in ambiguous
