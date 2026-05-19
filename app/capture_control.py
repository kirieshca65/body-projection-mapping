import time
import threading
import queue
import sys
import cv2
import numpy as np
from cv2_enumerate_cameras import enumerate_cameras

from screeninfo import get_monitors

try:
    from .frame_storage import frames, tiles
    from .projector_control import (
        build_calibration_image,
        estimate_homography_cam_to_proj,
    )
    from .pose_estimation import (
        mp_track_pose,
        init_landmarker,
        close_landmarker,
        start_overlay_worker,
        stop_overlay_worker,
    )
except ImportError:
    from frame_storage import frames, tiles
    from projector_control import (
        build_calibration_image,
        estimate_homography_cam_to_proj,
    )
    from pose_estimation import (
        mp_track_pose,
        init_landmarker,
        close_landmarker,
        start_overlay_worker,
        stop_overlay_worker,
    )

def _ui_monitor(screens, projector_monitor):
    """Монитор для окон управления (не проектор, если возможно)."""
    if not screens:
        return None
    for monitor in screens:
        if getattr(monitor, "is_primary", False):
            if projector_monitor is None or (
                monitor.x,
                monitor.y,
            ) != (projector_monitor.x, projector_monitor.y):
                return monitor
    if projector_monitor is not None:
        for monitor in screens:
            if (monitor.x, monitor.y) != (projector_monitor.x, projector_monitor.y):
                return monitor
    return screens[0]


def init_windows(projector_monitor=None) -> None:
    """Создаёт окна OpenCV и раскладывает их по экранам."""
    cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Pose Estimation", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mapping", cv2.WINDOW_NORMAL)

    placeholder = np.zeros((1, 1, 3), dtype=np.uint8)
    for name in ("Webcam", "Pose Estimation", "Preview", "Mapping"):
        cv2.imshow(name, placeholder)
    cv2.waitKey(1)

    screens = list(get_monitors())
    ui_monitor = _ui_monitor(screens, projector_monitor)
    if ui_monitor is not None:
        mx, my = ui_monitor.x, ui_monitor.y
        mw, mh = ui_monitor.width, ui_monitor.height
        half_w, half_h = mw // 2, mh // 2

        cv2.moveWindow("Webcam", mx + half_w, my)
        cv2.resizeWindow("Webcam", half_w, half_h)

        cv2.moveWindow("Preview", mx + half_w, my + half_h)
        cv2.resizeWindow("Preview", half_w, half_h)

        cv2.moveWindow("Pose Estimation", mx, my)
        cv2.resizeWindow("Pose Estimation", half_w, half_h)

    if projector_monitor is not None:
        cv2.moveWindow("Mapping", projector_monitor.x, projector_monitor.y)
        cv2.setWindowProperty(
            "Mapping",
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN,
        )


def select_projector_monitor():
    """Выбор монитора для проекции: задаёт разрешение маппинга и позицию окна Preview."""
    screens = list(get_monitors())
    if not screens:
        print("Мониторы не найдены (screeninfo); Preview останется обычным окном.")
        return None
    for i, monitor in enumerate(screens):
        print(f"{i}: {monitor}")
    while True:
        try:
            index = int(input("Индекс монитора-проектора: ").strip())
        except ValueError:
            continue
        if 0 <= index < len(screens):
            monitor = screens[index]
            frames.set_mapping_res(monitor.width, monitor.height)
            return monitor



def _preferred_capture_backends() -> list[int]:
    """Подбираем бэкенды под ОС, чтобы работало на Windows/macOS."""
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
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1980)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if not cap.isOpened():
            print(f"Unable to open webcam with index {index}.")
            continue
        else:
            ok, frame = cap.read()
            if ok and frame is not None:
                height, width = frame.shape[:2]
                frames.set_webcam_res(width, height)
            print(frames.get_webcam_res())
            return cap

   
def start() -> None:
    cap = get_camera()
    projector_monitor = select_projector_monitor()
    stop_event: threading.Event | None = None
    t: threading.Thread | None = None
    frame_queue: "queue.Queue[tuple[int, any]]" | None = None

    try:
        # Важно: выбор видео/диалог должен выполняться в main thread (Windows/tkinter).
        tiles.ensure_video_selected()
        init_landmarker()
        start_overlay_worker()

        init_windows(projector_monitor)

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

        homography_check = False
        
        print(
            "Калибровка: c — показать/скрыть ArUco на проекторе, "
            "h — оценить homography (камера→проектор) по кадру, q — выход."
            "d - сбросить homography"
        )

       
        while True:
            success, frame = cap.read()

            if not success:
                continue
            frames.set_webcam(frame)

            frame_timestamp_ms = int(time.time() * 1000)
            
            # Положить в очередь свежий кадр (если очередь занята — выбросить старый).
            try:
                if homography_check:
                    frame = frames.get_webcam_warped_to_projector()
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

            mapping_res = frames.get_mapping_res()
            if frames.get_calibration_active() and mapping_res is not None:
                pw, ph = mapping_res
                mapping_frame = build_calibration_image(pw, ph)
           
            else:
                mapping_frame = frames.get_mapping()
                """elif homography_check:
                mapping_frame = frames.get_webcam_warped_to_projector()"""

            if mapping_frame is not None:
                cv2.imshow('Mapping', mapping_frame)

            cv2.imshow('Webcam', frame)

            key = cv2.waitKey(1) & 0xFF
            if key:
                if key in (ord('q'), ord('Q')):
                    if stop_event is not None:
                        stop_event.set()
                    break
                
                if key in (ord("c"), ord("C")):
                    nxt = not frames.get_calibration_active()
                    frames.set_calibration_active(nxt)
                    print("Режим калибровки (ArUco на проекторе):", "вкл" if nxt else "выкл")

                if key in (ord("h"), ord("H")):
                    if mapping_res is None:
                        print("Сначала выберите монитор / задано mapping_res.")
                    else:
                        H, info = estimate_homography_cam_to_proj(frame, mapping_res)
                        if H is not None:
                            frames.set_homography_cam_to_proj(H)
                            print("Сохранена homography_cam_to_proj (камера→проектор).", info)
                            print("H (камера → проектор):")
                            for row in H:
                                print(
                                    "  ["
                                    + "  ".join(f"{v:12.6f}" for v in row)
                                    + "]"
                                )
                                homography_check = True
                        else:
                            print("Homography не оценена:", info)
                if key in (ord("d"), ord("D")):
                    frames.set_homography_cam_to_proj(None)
                    homography_check = False
                    print("Homography сброшено")           
        
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
    start()
    
