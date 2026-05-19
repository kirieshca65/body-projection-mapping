from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


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

        self._open()
        self._thread = threading.Thread(target=self._run, name="buffered_video_reader", daemon=True)
        self._thread.start()

    def _open(self) -> None:
        cap = cv2.VideoCapture(self.path, self.api_preference)
        if not cap.isOpened():
            cap.release()
            raise FileNotFoundError(f"Не удалось открыть видео: {self.path}")
        self._cap = cap

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

            with self._lock:
                write_idx = 1 - self._read_idx
                self._frames[write_idx] = frame
                self._read_idx = write_idx

            now = time.time()
            dt = now - last_ts
            if dt < min_dt:
                time.sleep(min_dt - dt)
            last_ts = time.time()

