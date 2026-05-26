from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


IMAGE_EXTENSIONS = {
    ".bmp",
    ".dib",
    ".jpeg",
    ".jpg",
    ".jpe",
    ".jp2",
    ".png",
    ".webp",
    ".pbm",
    ".pgm",
    ".ppm",
    ".pxm",
    ".pnm",
    ".pfm",
    ".sr",
    ".ras",
    ".tiff",
    ".tif",
    ".exr",
    ".hdr",
    ".pic",
}


@dataclass
class BufferedVideoReader:
    """
    Потоковый читатель видео, который всегда держит "самый свежий" кадр.

    - читает в отдельном daemon-thread
    - при окончании видео делает loop (перемотка на 0)
    - get_latest_frame(copy=False) возвращает ссылку на кадр без копии (double-buffer)
    - get_latest_frame(copy=True) — явная копия, если кадр нужно хранить дольше одного тика
    """

    path: str
    api_preference: int = cv2.CAP_ANY
    target_fps: float = 30.0

    def __post_init__(self) -> None:
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._frames: list[Optional[np.ndarray]] = [None, None]
        self._read_idx: int = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._is_static_image = False

        self._open()
        if self._cap is not None:
            self._thread = threading.Thread(target=self._run, name="buffered_video_reader", daemon=True)
            self._thread.start()

    def _open(self) -> None:
        image = self._read_static_image(self.path)
        if image is not None:
            self._is_static_image = True
            self._publish_frame(image)
            return

        cap = cv2.VideoCapture(self.path, self.api_preference)
        if not cap.isOpened():
            cap.release()
            raise FileNotFoundError(f"Не удалось открыть видео: {self.path}")
        self._cap = cap

    @staticmethod
    def _looks_like_image(path: str) -> bool:
        suffix = ""
        if "." in path:
            suffix = path.rsplit(".", 1)[-1].lower()
        return bool(suffix) and f".{suffix}" in IMAGE_EXTENSIONS

    @classmethod
    def _read_static_image(cls, path: str) -> Optional[np.ndarray]:
        if not cls._looks_like_image(path):
            return None

        try:
            data = np.fromfile(path, dtype=np.uint8)
        except OSError:
            return None
        if data.size == 0:
            return None

        image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if image is None:
            return None
        return cls._to_bgr_frame(image)

    @staticmethod
    def _to_bgr_frame(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.ndim == 3 and image.shape[2] == 3:
            return image
        raise ValueError(f"Unsupported image shape: {image.shape}")

    def _publish_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            write_idx = 1 - self._read_idx
            self._frames[write_idx] = frame
            self._read_idx = write_idx

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=1.0)
        with self._lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
            self._frames = [None, None]
            self._read_idx = 0

    def get_latest_frame(self, *, copy: bool = False) -> Optional[np.ndarray]:
        with self._lock:
            frame = self._frames[self._read_idx]
            if frame is None:
                return None
            return frame.copy() if copy else frame

    def _run(self) -> None:
        min_dt = 1.0 / max(1.0, float(self.target_fps))
        last_ts = 0.0

        while not self._stop.is_set():
            cap = self._cap
            if cap is None:
                time.sleep(0.05)
                continue

            ok, frame = cap.read()
            if not ok or frame is None:
                # loop: перемотка в начало и пробуем снова
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                except Exception:
                    pass
                time.sleep(0.01)
                continue

            self._publish_frame(frame)

            now = time.time()
            dt = now - last_ts
            if dt < min_dt:
                time.sleep(min_dt - dt)
            last_ts = time.time()

