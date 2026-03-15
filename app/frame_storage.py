from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import cv2
import os
import tempfile
import shutil


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

    def set_webcam(self, frame: np.ndarray) -> None:
        self.webcam_frame = frame.copy() if frame is not None else None

    def set_landmarks(self, frame: np.ndarray) -> None:
        self.landmarks_frame = frame.copy() if frame is not None else None

    def set_tiles(self, frame: np.ndarray) -> None:
        self.tiles_frames = frame.copy() if frame is not None else None

    def set_preview(self, frame: np.ndarray) -> None:
        self.preview_frames = frame.copy() if frame is not None else None

    def set_mapping(self, frame: np.ndarray) -> None:
        self.mapping_frame = frame.copy() if frame is not None else None

    def get_webcam(self) -> Optional[np.ndarray]:
        return self.webcam_frame

    def get_landmarks(self) -> Optional[np.ndarray]:
        return self.landmarks_frame

    def get_tiles(self) -> Optional[np.ndarray]:
        return self.tiles_frames

    def get_preview(self) -> Optional[np.ndarray]:
        return self.preview_frames

    def get_mapping(self) -> Optional[np.ndarray]:
        return self.mapping_frame

    """Разрешение проектора и вебкамеры"""
    mapping_res : Optional[List[2 : int]] = None
    webcam_res : Optional[List[2 : int]] = None

    def set_mapping_res(self, width : int, height : int):
        self.mapping_res = [width, height]

    def get_mapping_res(self):
        return self.mapping_res
    
    def set_webcam_res(self, width : int, height : int):
        self.webcam_res = [width, height]

    def get_webcam_res(self):
        return self.webcam_res

@dataclass
class TilesStorage:
    torso: Optional[np.ndarray] = None
    l_arm: Optional[np.ndarray] = None
    r_arm: Optional[np.ndarray] = None
    l_leg: Optional[np.ndarray] = None
    r_leg: Optional[np.ndarray] = None

    texture: Optional[np.ndarray] = None

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
        source = os.path.join(module_dir, "frame_perfome", "tiles", "test_body.png")

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

    def get_torso(self):
        if self.texture is None:
            raise RuntimeError("Текстура торса не инициализирована в TilesStorage.")
        return self.texture

"""Единственный экземпляр — создаётся при первом импорте модуля"""
frames: FrameStorage = FrameStorage()
tiles: TilesStorage = TilesStorage()
#tiles.chenge_texure()