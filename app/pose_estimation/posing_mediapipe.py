import os
import shutil
import tempfile

import cv2
import numpy as np
import threading
import queue
import copy

from . import custom_pose_style

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import drawing_utils

try:
    from ..frame_storage import frames, tiles
    from ..frame_perfome import draw_overlay
except ImportError:
    from frame_storage import frames, tiles
    from frame_perfome import draw_overlay


latest_pose_frame = None
_landmarker = None
_overlay_thread: threading.Thread | None = None
_overlay_stop: threading.Event | None = None
_overlay_queue: "queue.Queue[tuple[int, object, np.ndarray, object]]" = None


def _model_path_for_mediapipe(source: str) -> str:
    """Возвращает путь к модели, доступный для MediaPipe"""
    temp_dir = tempfile.gettempdir()
    name = os.path.basename(source)
    dest = os.path.join(temp_dir, f"body_projection_{name}")
    if not os.path.exists(dest) or os.path.getmtime(dest) < os.path.getmtime(source):
        shutil.copy2(source, dest)
    return dest


"""Модели mediapipe для отслеживания"""
_models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
_full_src = os.path.join(_models_dir, 'pose_landmarker_full.task')
_lite_src = os.path.join(_models_dir, 'pose_landmarker_lite.task')

""""""
model_path = _model_path_for_mediapipe(_full_src)


"""Импорт базовых параметров для модели PoseLandmarker"""
BaseOptions = mp.tasks.BaseOptions(model_asset_path=model_path)
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
PoseLandmarkerResult = vision.PoseLandmarkerResult
VisionRunningMode = vision.RunningMode


def result_handler(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    """latest_pose_frame импортируется из capture_control для дальнейшего вывода"""
    global latest_pose_frame
    
    segmentation_masks = result.segmentation_masks[0].numpy_view()
    result = result.pose_landmarks
    
    if segmentation_masks is not None:
        segmentation_masks = segmentation_masks.astype(np.uint8) * 255
        #segmentation_masks = cv2.cvtColor(segmentation_masks, cv2.COLOR_GRAY2BGR)

    if result is None:
        return
    frame_rgb = output_image.numpy_view().copy()
    #print('pose landmarker result: {}'.format(result))

    # Передаём landmarks+кадр в отдельный поток для overlay, чтобы:
    q = _overlay_queue
    if q is not None:
        #frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        payload = (timestamp_ms, result, frames.get_webcam(), segmentation_masks)
        try:
            q.put_nowait(payload)
        except queue.Full:
            try:
                _ = q.get_nowait()
                q.task_done()
            except queue.Empty:
                pass
            try:
                q.put_nowait(payload)
            except queue.Full:
                pass

    landmark_print(result, frame_rgb, timestamp_ms)
    
    
    
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
    base_options=BaseOptions,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=result_handler,
    output_segmentation_masks=True)


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

"""Запуск потока для отрисовки по Landmarks"""
def start_overlay_worker() -> None:
    """
    Запускает поток, который выполняет draw_overlay() по входящим landmarks.
    Держим maxsize=1, чтобы всегда обрабатывать только "самое свежее".
    """
    global _overlay_thread, _overlay_stop, _overlay_queue
    if _overlay_thread is not None and _overlay_thread.is_alive():
        return

    _overlay_queue = queue.Queue(maxsize=1)
    _overlay_stop = threading.Event()

    def _worker() -> None:
        assert _overlay_stop is not None
        assert _overlay_queue is not None
        while not _overlay_stop.is_set():
            try:
                _ts, lm, frame_bgr, seg = _overlay_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                draw_overlay(lm, frame_bgr, seg)
            finally:
                _overlay_queue.task_done()

    _overlay_thread = threading.Thread(target=_worker, name="overlay_worker", daemon=True)
    _overlay_thread.start()


def stop_overlay_worker() -> None:
    global _overlay_thread, _overlay_stop, _overlay_queue
    if _overlay_stop is not None:
        _overlay_stop.set()

    q = _overlay_queue
    if q is not None:
        while True:
            try:
                _ = q.get_nowait()
                q.task_done()
            except queue.Empty:
                break

    t = _overlay_thread
    if t is not None:
        try:
            t.join(timeout=1.0)
        except Exception:
            pass

    _overlay_thread = None
    _overlay_stop = None
    _overlay_queue = None

