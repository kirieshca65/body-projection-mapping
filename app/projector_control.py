"""
Калибровка камера–проектор: паттерн ArUco в углах экрана и оценка гомографии.

Основная матрица H_cam_to_proj (камера → проектор): переводит точки из плоскости изображения
камеры в координаты экрана проектора — как «повернуть» и выровнять вид камеры под прямоугольник
проектора.

Применение H к кадру — frames.get_webcam_warped_to_projector() в frame_storage
(inv(H) для cv2.warpPerspective).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# Совместимо с OpenCV 4.13+ (aruco в основном пакете opencv-python).
ARUCO_DICTIONARY = cv2.aruco.DICT_4X4_50


def _marker_rects(width: int, height: int) -> Dict[int, Tuple[int, int, int]]:
    """Углы экрана: 0=TL, 1=TR, 2=BR, 3=BL. Возвращает id -> (ox, oy, side)."""
    margin = max(48, min(width, height) // 24)
    side = max(64, min(width, height) // 12)
    while 2 * margin + side > width or 2 * margin + side > height:
        margin = max(24, margin - 8)
        side = max(48, side - 8)
        if margin <= 24 and side <= 48:
            break
    w, h = width, height
    return {
        0: (margin, margin, side),
        1: (w - margin - side, margin, side),
        2: (w - margin - side, h - margin - side, side),
        3: (margin, h - margin - side, side),
    }


def _projector_corner_points(
    rects: Dict[int, Tuple[int, int, int]],
) -> Dict[int, np.ndarray]:
    """4 угла каждого маркера в пикселях проектора (совпадает с порядком OpenCV detectMarkers)."""
    out: Dict[int, np.ndarray] = {}
    for mid, (ox, oy, s) in rects.items():
        out[mid] = np.float32(
            [
                [ox, oy],
                [ox + s - 1, oy],
                [ox + s - 1, oy + s - 1],
                [ox, oy + s - 1],
            ]
        )
    return out


def get_aruco_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
    parameters = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def build_calibration_image(width: int, height: int) -> np.ndarray:
    """
    Белый фон и 4 маркера ArUco в углах (BGR, размер экрана проектора).

    Фон должен контрастировать с внешней рамкой маркера — на чисто чёрном экране
    OpenCV ArUco часто не находит контуры маркера.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
    canvas = np.full((height, width), 255, dtype=np.uint8)
    rects = _marker_rects(width, height)
    for mid, (ox, oy, side) in rects.items():
        marker = np.zeros((side, side), dtype=np.uint8)
        cv2.aruco.generateImageMarker(dictionary, mid, side, marker)
        y1, x1 = oy + side, ox + side
        if y1 <= height and x1 <= width and ox >= 0 and oy >= 0:
            canvas[oy:y1, ox:x1] = marker
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def estimate_homography_cam_to_proj(
    camera_bgr: np.ndarray,
    proj_wh: Tuple[int, int],
) -> Tuple[Optional[np.ndarray], str]:
    """
    По кадру камеры с видимым паттерном калибровки возвращает H: камера → проектор.

    Ожидается тот же шаблон маркеров, что в build_calibration_image(w,h).
    """
    pw, ph = proj_wh
    rects = _marker_rects(pw, ph)
    proj_corners = _projector_corner_points(rects)

    gray = cv2.cvtColor(camera_bgr, cv2.COLOR_BGR2GRAY)
    detector = get_aruco_detector()
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return None, "ArUco: маркеры не обнаружены"

    img_pts: list[list[float]] = []
    proj_pts: list[list[float]] = []
    ids_flat = ids.flatten().tolist()
    for i, marker_id in enumerate(ids_flat):
        if marker_id not in proj_corners:
            continue
        pc = proj_corners[int(marker_id)]
        ic = corners[i][0]  # (4, 2)
        for row in range(4):
            proj_pts.append(pc[row].tolist())
            img_pts.append(ic[row].tolist())

    if len(proj_pts) < 4:
        return (
            None,
            f"Нужны минимум 4 соответствия; найдено подходящих маркеров: {len(proj_pts)//4}",
        )

    pts_p = np.float32(proj_pts)
    pts_i = np.float32(img_pts)
    H, mask = cv2.findHomography(pts_i, pts_p, cv2.RANSAC, 5.0)
    if H is None:
        return None, "findHomography вернул None"

    # H: камера → проектор; проверка репроекции в координаты проектора.
    cam_h = np.hstack([pts_i, np.ones((len(pts_i), 1), dtype=np.float64)])
    pred = (H @ cam_h.T).T
    pred = pred[:, :2] / pred[:, 2:3]
    err_all = np.linalg.norm(pred - pts_p, axis=1)
    inliers = mask.ravel().astype(bool) if mask is not None else np.ones(len(pts_p), dtype=bool)
    if inliers.any():
        err_mean = float(err_all[inliers].mean())
    else:
        err_mean = float(err_all.mean())

    return H, f"точек {len(pts_p)}, inliers RANSAC {int(inliers.sum())}, ср. ошибка ~{err_mean:.2f} px"

