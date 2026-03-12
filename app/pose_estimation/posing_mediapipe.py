import os
import shutil
import tempfile

import cv2
import numpy as np

from . import custom_pose_style

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import drawing_utils

from frame_storage import frames
from frame_perfome import overlay_torso
from frame_storage import tiles

latest_pose_frame = None
_landmarker = None

"""Модели mediapipe для отслеживания"""
_models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
_full_src = os.path.join(_models_dir, 'pose_landmarker_full.task')
_lite_src = os.path.join(_models_dir, 'pose_landmarker_lite.task')


def _model_path_for_mediapipe(source: str) -> str:
    """Возвращает путь к модели, доступный для MediaPipe"""
    temp_dir = tempfile.gettempdir()
    name = os.path.basename(source)
    dest = os.path.join(temp_dir, f"body_projection_{name}")
    if not os.path.exists(dest) or os.path.getmtime(dest) < os.path.getmtime(source):
        shutil.copy2(source, dest)
    return dest


model_path = _model_path_for_mediapipe(_full_src)

"""Импорт базовых параметров для модели PoseLandmarker"""
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
PoseLandmarkerResult = vision.PoseLandmarkerResult
VisionRunningMode = vision.RunningMode

def result_handler(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    """latest_pose_frame импортируется из capture_control для дальнейшего вывода"""
    global latest_pose_frame
    
    result = result.pose_landmarks
    frame_ = output_image.numpy_view().copy()
    #print('pose landmarker result: {}'.format(result))
    landmark_print(result, frame_, timestamp_ms)
    overlay_torso(result)
    
    

def landmark_print(landmarks, frame, timestamp: int):
    """Стили отрисовки: заполняем все индексы, чтобы не было KeyError"""
    pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)
    """Отрисовка каждой точки"""
    for landmark in landmarks:
        drawing_utils.draw_landmarks(
            image=frame,
            landmark_list=landmark,
            connections=custom_pose_style.PoseConnections.POSE_LANDMARKS,
            connection_drawing_spec = pose_connection_style,
            is_drawing_landmarks = False)

    """Конвертация из RGB в BGR для OpenCV"""
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frames.set_landmarks(frame_bgr)


"""Инициализация базовых параметров для модели"""
options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=result_handler)


def mp_track_pose(frame: np.ndarray, timestamp_ms: int) -> None:
    global _landmarker
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_rgb,
            )
    """Результат отслеживания переходит в через callback в print_result"""
    _landmarker.detect_async(mp_image, timestamp_ms)


def init_landmarker() -> None:
    global _landmarker
    if _landmarker is None:
        _landmarker = vision.PoseLandmarker.create_from_options(options)


def close_landmarker() -> None:
    global _landmarker
    if _landmarker is not None:
        _landmarker.close()
        _landmarker = None

