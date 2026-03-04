from dataclasses import dataclass
from typing import Optional

import numpy as np


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


"""Единственный экземпляр — создаётся при первом импорте модуля"""
frames: FrameStorage = FrameStorage()
