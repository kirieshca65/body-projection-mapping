import time
import threading
import queue
import sys
import cv2
from cv2_enumerate_cameras import enumerate_cameras
from screeninfo import get_monitors

from frame_storage import frames, tiles
from pose_estimation import (
    mp_track_pose,
    init_landmarker,
    close_landmarker,
    start_overlay_worker,
    stop_overlay_worker,
)

def get_screens():
    screens = get_monitors()
    for monitor in screens:
        print(monitor)

    index = int(input("Enter the index of the monitor: "))
    monitor = screens[index]
    frames.set_mapping_res(monitor.width, monitor.height)



def _preferred_capture_backends() -> list[int]:
    # Подбираем бэкенды под ОС, чтобы работало на Windows/macOS.
    if sys.platform.startswith("win"):
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    if sys.platform == "darwin":
        return [cv2.CAP_AVFOUNDATION]
    # На Linux обычно V4L2, оставим как запасной вариант.
    return [cv2.CAP_V4L2]


def get_camera() -> cv2.VideoCapture:
    """Получение списка камер в системе (Windows/macOS/Linux)."""
    cameras: list = []
    for backend in _preferred_capture_backends():
        try:
            cams = enumerate_cameras(backend)
            if cams:
                cameras = cams
                break
        except Exception:
            continue

    if not cameras:
        # Фоллбэк: показываем хотя бы индексы 0..5
        cameras = [type("Cam", (), {"index": i, "backend": 0}) for i in range(6)]

    for camera in cameras:
        print(camera)

    while True:
        index = int(input("Enter the index of the camera: "))
        if index not in range(len(cameras)):
            continue
        backend = cameras[index].backend
        cap = cv2.VideoCapture(cameras[index].index, backend)
        if not cap.isOpened():
            print(f"Unable to open webcam with index {index}.")
            continue
        else:
            ok, frame = cap.read()
            if ok and frame is not None:
                height, width = frame.shape[:2]
                frames.set_webcam_res(width, height)
            #print(frames.get_webcam_res())
            return cap


def start() -> None:
    
    cap = get_camera()
    stop_event: threading.Event | None = None
    t: threading.Thread | None = None
    frame_queue: "queue.Queue[tuple[int, any]]" | None = None

    try:
        # Важно: выбор видео/диалог должен выполняться в main thread (Windows/tkinter).
        tiles.ensure_video_selected()
        init_landmarker()
        start_overlay_worker()

        cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Pose Estimation', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Preview', cv2.WINDOW_NORMAL)

        # Кадры читаем в главном потоке, обработку (mp_track_pose + downstream) — в рабочем.
        # maxsize=1: всегда обрабатываем "самый свежий" кадр, а не накапливаем задержку.
        frame_queue = queue.Queue(maxsize=1)
        stop_event = threading.Event()

        def worker() -> None:
            while not stop_event.is_set():
                try:
                    ts_ms, frm = frame_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                try:
                    # Важно: в callback'е overlay_* берут кадр из frames.get_webcam().
                    # Поэтому обновляем storage именно здесь, синхронно с отправкой в MediaPipe.
                    frames.set_webcam(frm)
                    mp_track_pose(frm, ts_ms)
                finally:
                    frame_queue.task_done()

        t = threading.Thread(target=worker, name="pose_worker", daemon=True)
        t.start()
        
        while True:
            success, frame = cap.read()
            #print(frame)

            if not success:
                continue

            frame_timestamp_ms = int(time.time() * 1000)
            # Положить в очередь свежий кадр (если очередь занята — выбросить старый).
            try:
                frame_queue.put_nowait((frame_timestamp_ms, frame))
            except queue.Full:
                try:
                    _ = frame_queue.get_nowait()
                    frame_queue.task_done()
                except queue.Empty:
                    pass
                try:
                    frame_queue.put_nowait((frame_timestamp_ms, frame))
                except queue.Full:
                    # Если прямо сейчас снова занято — просто пропускаем этот кадр.
                    pass

            pose_frame = frames.get_landmarks()
            if pose_frame is not None:
                cv2.imshow('Pose Estimation', pose_frame)
            
            preview_frame = frames.get_preview()
            if preview_frame is not None:
                cv2.imshow('Preview', preview_frame)
                
            cv2.imshow('Webcam', frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), ord('й'), ord('Й')):
                if stop_event is not None:
                    stop_event.set()
                break
    
    finally:
        if stop_event is not None:
            stop_event.set()
        if frame_queue is not None:
            while True:
                try:
                    _ = frame_queue.get_nowait()
                    frame_queue.task_done()
                except queue.Empty:
                    break
        if t is not None:
            try:
                t.join(timeout=1.0)
            except Exception:
                pass
        try:
            close_landmarker()
        except Exception:
            pass
        try:
            stop_overlay_worker()
        except Exception:
            pass
        try:
            tiles.stop_videoreader()
        except Exception:
            pass
        try:
            cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    #get_screens()
    start()
    
