from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import cv2
import os


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
    torso : Optional[np.ndarray] = None
    l_arm : Optional[np.ndarray] = None
    r_arm : Optional[np.ndarray] = None
    l_leg : Optional[np.ndarray] = None
    r_leg : Optional[np.ndarray] = None

    texture : Optional[np.ndarray] = None

    def change_texure(self, path : str):
        self.texture = cv2.imread(path)
    
    def change_texure(self):
        self.texture = cv2.imread("frame_perfome/tiles/test_body.png")
    
    def __init__(self) -> None:
        absolute_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frame_perfome','tiles', 'models')
        # Декодируем массив в изображение OpenCV
        img = cv2.imread("app/frame_perfome/tiles/test_body.png")
        self.texture = img
       
    def get_torso(self):
        return self.texture

"""Единственный экземпляр — создаётся при первом импорте модуля"""
frames: FrameStorage = FrameStorage()
tiles: TilesStorage = TilesStorage()
#tiles.chenge_texure()