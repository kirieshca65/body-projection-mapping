from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import numpy as np
import cv2
import os
import tempfile
import shutil
import threading

from frame_perfome.video_stream import BufferedVideoReader


@dataclass
class FrameStorage:
    """
    Импортируйте в других модулях: from frame_contoller import frame_storage
    """
    webcam_frame: Optional[np.ndarray] = None      # Кадр с вебкамеры
    landmarks_frame: Optional[np.ndarray] = None   # Кадр с отображением обнаруженных точек
    tiles_frames: Optional[np.ndarray] = None      # Кадр с наложенным контентом
    preview_frames: Optional[np.ndarray] = None    # Кадр с наложенным контентом над вебкамерой
    mapping_frame: Optional[np.ndarray] = None     # Конечный кадр для вывода на проектор
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def set_webcam(self, frame: np.ndarray) -> None:
        with self._lock:
            self.webcam_frame = frame.copy() if frame is not None else None

    def set_landmarks(self, frame: np.ndarray) -> None:
        with self._lock:
            self.landmarks_frame = frame.copy() if frame is not None else None

    def set_tiles(self, frame: np.ndarray) -> None:
        with self._lock:
            self.tiles_frames = frame.copy() if frame is not None else None

    def set_preview(self, frame: np.ndarray) -> None:
        with self._lock:
            self.preview_frames = frame.copy() if frame is not None else None

    def set_mapping(self, frame: np.ndarray) -> None:
        with self._lock:
            self.mapping_frame = frame.copy() if frame is not None else None

    def get_webcam(self) -> Optional[np.ndarray]:
        with self._lock:
            return self.webcam_frame.copy() if self.webcam_frame is not None else None

    def get_landmarks(self) -> Optional[np.ndarray]:
        with self._lock:
            return self.landmarks_frame.copy() if self.landmarks_frame is not None else None

    def get_tiles(self) -> Optional[np.ndarray]:
        with self._lock:
            return self.tiles_frames.copy() if self.tiles_frames is not None else None

    def get_preview(self) -> Optional[np.ndarray]:
        with self._lock:
            return self.preview_frames.copy() if self.preview_frames is not None else None

    def get_mapping(self) -> Optional[np.ndarray]:
        with self._lock:
            return self.mapping_frame.copy() if self.mapping_frame is not None else None

    """Разрешение проектора и вебкамеры"""
    mapping_res : Optional[Tuple[int, int]] = None
    webcam_res : Optional[Tuple[int, int]] = None

    def set_mapping_res(self, width : int, height : int):
        with self._lock:
            self.mapping_res = (width, height)

    def get_mapping_res(self):
        with self._lock:
            return self.mapping_res
    
    def set_webcam_res(self, width : int, height : int):
        with self._lock:
            self.webcam_res = (width, height)

    def get_webcam_res(self):
        with self._lock:
            return self.webcam_res

@dataclass
class TilesStorage:
    torso: Optional[np.ndarray] = None
    l_arm: Optional[np.ndarray] = None
    r_arm: Optional[np.ndarray] = None
    l_leg: Optional[np.ndarray] = None
    r_leg: Optional[np.ndarray] = None

    texture: Optional[np.ndarray] = None
    # alpha маски (uint8 0..255) по имени файла
    masks_alpha: Optional[Dict[str, np.ndarray]] = None

    # Видео (лениво); те же начальные значения, что в __init__
    _video_path: Optional[str] = None
    _video_reader: Optional[BufferedVideoReader] = None
    _video_select_attempted: bool = False
    _overlay_cache: Dict[str, Tuple[Tuple[int, int], np.ndarray]] = field(
        default_factory=dict
    )

    @staticmethod
    def _imread_unicode(path: str) -> Optional[np.ndarray]:
        """
        Безопасное чтение изображений, в том числе по путям с кириллицей.
        """
        if not os.path.exists(path):
            return None
        # np.fromfile корректно работает с Unicode-путями на Windows
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        return img

    def change_texure(self, path: str):
        img = self._imread_unicode(path)
        if img is None:
            raise FileNotFoundError(f"Не удалось открыть изображение по пути: {path}")
        self.texture = img

    def __init__(self) -> None:
        """
        Инициализация текстуры:
        - исходный файл берётся из папки модуля (в том числе с кириллицей в пути),
        - копируется во временную директорию,
        - читается через _imread_unicode.
        """
        temp_dir = tempfile.gettempdir()
        module_dir = os.path.dirname(os.path.abspath(__file__))
        tiles_dir = os.path.join(module_dir, "frame_perfome", "tiles")
        source = os.path.join(tiles_dir, "test_body.png")

        name = os.path.basename(source)
        dest = os.path.join(temp_dir, f"body_tile_{name}")

        # Перекопировать, если файла нет или он старее исходного
        if not os.path.exists(dest) or os.path.getmtime(dest) < os.path.getmtime(source):
            shutil.copy2(source, dest)

        img = self._imread_unicode(dest)
        if img is None:
            # Последняя попытка — прочитать напрямую исходный файл
            img = self._imread_unicode(source)
        if img is None:
            raise FileNotFoundError(
                f"Не удалось открыть изображение ни по временном пути: {dest}, ни по исходному: {source}"
            )

        self.texture = img
        self.masks_alpha = {}
        self._overlay_cache = {}
        self._video_path = None
        self._video_reader = None
        self._video_select_attempted = False

        self._load_default_masks(tiles_dir)

    def _load_default_masks(self, tiles_dir: str) -> None:
        """
        Загружает 5 PNG-масок из app/frame_perfome/tiles/.
        Если какой-то файл отсутствует/битый — просто пропускаем его,
        чтобы не ломать весь пайплайн.
        """
        for name in (
            "mask_torso.png",
            "mask_forearm_r.png",
            "mask_forearm_l.png",
            "mask_thigh_r.png",
            "mask_thigh_l.png",
        ):
            path = os.path.join(tiles_dir, name)
            img = self._imread_unicode(path)
            if img is None:
                continue

            alpha: Optional[np.ndarray] = None
            if img.ndim == 3 and img.shape[2] == 4:
                alpha = img[:, :, 3]
            elif img.ndim == 3 and img.shape[2] >= 3:
                alpha = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
            elif img.ndim == 2:
                alpha = img

            if alpha is None:
                continue

            if alpha.dtype != np.uint8:
                alpha = np.clip(alpha, 0, 255).astype(np.uint8)

            self.masks_alpha[name] = alpha

    def get_torso(self):
        if self.texture is None:
            raise RuntimeError("Текстура торса не инициализирована в TilesStorage.")
        return self.texture

    def get_mask_alpha(self, name: str) -> Optional[np.ndarray]:
        return self.masks_alpha.get(name)

    def set_video_path(self, path: str) -> None:
        self._video_path = path
        # переинициализируем ридер при смене пути
        if self._video_reader is not None:
            try:
                self._video_reader.stop()
            except Exception:
                pass
        self._video_reader = BufferedVideoReader(path)

    def stop_videoreader(self) -> None:
        reader = self._video_reader
        if reader is None:
            return
        try:
            reader.stop()
        finally:
            self._video_reader = None

    def ensure_video_selected(self) -> None:
        """
        Запрашивает у пользователя видео-файл через диалог (один раз),
        если путь не задан и ридер не инициализирован.
        """
        if self._video_reader is not None:
            return
        if self._video_select_attempted:
            return
        self._video_select_attempted = True

        # tkinter для получения пути до видео
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception:
            return
        print("Открываем tinker")
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.askopenfilename(
                title="Выберите видео для масок",
                filetypes=[
                    ("Video files", "*.mp4;*.mov;*.mkv;*.avi;*.webm"),
                    ("All files", "*.*"),
                ],
            )
            print(path)
            try:
                root.destroy()
            except Exception:
                pass
        except Exception:
            return

        if path:
            try:
                self.set_video_path(path)
            except Exception:
                # не валим пайплайн, если видео не открылось
                return

    def get_video_frame(self) -> Optional[np.ndarray]:
        if self._video_reader is None:
            self.ensure_video_selected()
        if self._video_reader is None:
            return None
        return self._video_reader.get_latest_frame()

    def build_overlay_mask(self, mask_name: str) -> Optional[np.ndarray]:
        """
        Собирает BGRA: RGB берём из текущего кадра видео, alpha — из маски.
        Маску ресайзим к размеру видео-кадра.

        Кеширует последний результат (по mask_name и размеру).
        """
        alpha = self.get_mask_alpha(mask_name)
        if alpha is None:
            return None

        video_bgr = self.get_video_frame()
        if video_bgr is None:
            return None

        vh, vw = video_bgr.shape[:2]
        if alpha.shape[:2] != (vh, vw):
            # Маска — дискретная, поэтому интерполяция ближайшего соседа
            alpha = cv2.resize(alpha, (vw, vh), interpolation=cv2.INTER_AREA)

        h, w = vh, vw
        key = mask_name
        cached = self._overlay_cache.get(key)
        if cached is not None and cached[0] == (h, w):
            # Обновляем только RGB, alpha остаётся той же (фиксированная маска)
            overlay = cached[1]
        else:
            overlay = np.zeros((h, w, 4), dtype=np.uint8)
            overlay[:, :, 3] = alpha
            self._overlay_cache[key] = ((h, w), overlay)

        overlay[:, :, :3] = video_bgr
        return overlay

"""Единственный экземпляр — создаётся при первом импорте модуля"""
frames: FrameStorage = FrameStorage()
tiles: TilesStorage = TilesStorage()
#tiles.chenge_texure()
