import cv2

import custom_pose
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles

latest_pose_frame = None
_landmarker = None

"""Модели mediapipe для отслеживания"""
lite_model = 'models/mediapipe/pose_landmarker_lite.task'
full_model = 'models/mediapipe/pose_landmarker_full.task'

model_path = lite_model

"""Импорт базовых параметров для модели PoseLandmarker"""
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
PoseLandmarkerResult = vision.PoseLandmarkerResult
VisionRunningMode = vision.RunningMode

def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    """latest_pose_frame импортируется из capture_control для дальнейшего вывода"""
    global latest_pose_frame
    
    #print('pose landmarker result: {}'.format(result))
    result = result.pose_landmarks
    frame_rgb = output_image.numpy_view().copy()

    """Стили отрисовки: заполняем все индексы, чтобы не было KeyError"""
    pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)
    base_landmark_style = drawing_utils.DrawingSpec(color=(224, 224, 224), thickness=2)
    index = 0
    """Отрисовка каждой точки"""
    for landmark in result:
        num_landmarks = len(landmark)
        pose_landmark_style = {i: base_landmark_style for i in range(num_landmarks)}
        for key, spec in custom_pose.get_default_pose_landmarks_style().items():
            idx = int(key)
            if 0 <= idx < num_landmarks:
                pose_landmark_style[idx] = spec
        drawing_utils.draw_landmarks(
            image=frame_rgb,
            landmark_list=landmark,
            connections=custom_pose.PoseConnections.POSE_LANDMARKS,
            landmark_drawing_spec=pose_landmark_style,
            connection_drawing_spec = pose_connection_style)

    """Конвертация из RGB в BGR для OpenCV"""
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    latest_pose_frame = frame_bgr
 


"""Инициализация базовых параметров для модели"""
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)


def mp_track_pose(frame: mp.Image, timestamp_ms: int) -> None:
    global _landmarker
    _landmarker.detect_async(frame, timestamp_ms)
"""Результат отслеживания переходит в через callback в print_result"""


def init_landmarker() -> None:
    global _landmarker
    if _landmarker is None:
        _landmarker = PoseLandmarker.create_from_options(options)


def close_landmarker() -> None:
    global _landmarker
    if _landmarker is not None:
        _landmarker.close()
        _landmarker = None
