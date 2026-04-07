from typing import Optional
import cv2
import numpy as np

from frame_storage import frames, tiles


size_adjust : float =  1.25
"""
- width_scale: ширина полосы как доля длины отрезка
- extend_scale: насколько продлить отрезок за точки (доля длины)
"""
width_scale : float = 0.25
extend_scale : float = 0.08


class _ExpSmoother2D:
    """
    Экспоненциальное сглаживание (EMA) для 2D-точек.

    s_t = alpha * x_t + (1 - alpha) * s_{t-1}

    Хранит состояние по ключу (например, индекс лендмарки).
    """

    def __init__(self, alpha: float = 0.55):
        self.alpha = float(alpha)
        self._state: dict[int, np.ndarray] = {}

    def reset(self) -> None:
        self._state.clear()

    def update(self, key: int, value_xy: np.ndarray) -> np.ndarray:
        v = np.asarray(value_xy, dtype=np.float32).reshape(2)
        prev = self._state.get(int(key))
        if prev is None:
            out = v
        else:
            a = self.alpha
            out = (a * v) + ((1.0 - a) * prev)
        self._state[int(key)] = out
        return out


_POINT_SMOOTHER = _ExpSmoother2D(alpha=0.6)


def draw_overlay(landmarks, frame: Optional[np.ndarray] = None) -> None:
    """
    Рисует overlay на кадре, синхронном с landmarks.

    Если `frame` не передан — берём кадр из `frames.get_webcam()` (fallback).
    Для устранения рассинхрона предпочтительно всегда передавать кадр явно.
    """
    if frame is None or landmarks is None:
        return
    
    # Конечности: берём текущий кадр видео + фиксированную маску (BGRA)
    ov = tiles.build_overlay_mask("mask_forearm_r.png")
    if ov is not None:
        overlay_limbs(landmarks, (11, 13), frame, overlay_img=ov)

    ov = tiles.build_overlay_mask("mask_forearm_l.png")
    if ov is not None:
        overlay_limbs(landmarks, (12, 14), frame, overlay_img=ov)

    ov = tiles.build_overlay_mask("mask_thigh_r.png")
    if ov is not None:
        overlay_limbs(landmarks, (23, 25), frame, overlay_img=ov)

    ov = tiles.build_overlay_mask("mask_thigh_l.png")
    if ov is not None:
        overlay_limbs(landmarks, (24, 26), frame, overlay_img=ov)

    ov = tiles.build_overlay_mask("mask_torso.png")
    if ov is not None:
        overlay_torso(landmarks, frame, overlay_img=ov)
    frames.set_preview(frame)


def overlay_torso(
    landmarks,
    frame: np.ndarray,
    overlay_img: Optional[np.ndarray] = None,
):
    """
    Накладывает текстуру торса на текущий кадр вебкамеры.
    Размеры берутся непосредственно из кадра, чтобы избежать рассинхронизации.
    """
    if overlay_img is None:
        overlay_img = tiles.get_torso()

    if landmarks[0] is None or frame is None or overlay_img is None:
        return

    fh, fw = frame.shape[:2]

    # Извлекаем координаты 4 точек из MediaPipe (x, y в пикселях)
    # Порядок: [Левое плечо, Правое плечо, Правое бедро, Левое бедро]
    landmark = landmarks[0]
    torso_idx = (11, 12, 24, 23)
    dst_raw = [
        np.array([landmark[i].x * fw, landmark[i].y * fh], dtype=np.float32) for i in torso_idx
    ]
    dst_pts = np.array([_POINT_SMOOTHER.update(i, p) for i, p in zip(torso_idx, dst_raw)], dtype=np.float32)

    """Применение дополнительного увеличения"""
    global size_adjust
    center = dst_pts.mean(axis=0, keepdims=True)
    dst_pts = (dst_pts - center) * size_adjust + center


    # быстрый варп+альфа только по ROI
    _warp_and_blend_roi(frame, overlay_img, dst_pts)

    return

def overlay_limbs(
    landmarks,
    limbs: tuple[int, int],
    frame : np.ndarray,
    overlay_img: Optional[np.ndarray] = None,
):
    """
    Накладывает `overlay_img` по двум точкам (индексам лендмарок) как "полоску"
    вдоль отрезка между ними.
    """

    global width_scale, extend_scale
    if overlay_img is None:
        overlay_img = tiles.get_torso()

    if landmarks[0] is None or frame is None or overlay_img is None:
        return

    fh, fw = frame.shape[:2]
    landmark = landmarks[0]
    p1_raw = np.array([landmark[limbs[0]].x * fw, landmark[limbs[0]].y * fh], dtype=np.float32)
    p2_raw = np.array([landmark[limbs[1]].x * fw, landmark[limbs[1]].y * fh], dtype=np.float32)
    p1 = _POINT_SMOOTHER.update(int(limbs[0]), p1_raw)
    p2 = _POINT_SMOOTHER.update(int(limbs[1]), p2_raw)

    v = p2 - p1 

    length = float(np.linalg.norm(v))
    if length < 1.0:
        return

    u = v / length  # вдоль отрезка
    perp = np.array([-u[1], u[0]], dtype=np.float32)  # перпендикуляр

    half_width = max(1.0, length * float(width_scale) * 0.5)
    extend = length * float(extend_scale)

    p1e = p1 - u * extend
    p2e = p2 + u * extend

    # Четыре угла прямоугольника вокруг отрезка
    dst_pts = np.array(
        [
            p1e + perp * half_width,
            p2e + perp * half_width,
            p2e - perp * half_width,
            p1e - perp * half_width,
        ],
        dtype="float32",
    )


    # быстрый варп+альфа только по ROI
    _warp_and_blend_roi(frame, overlay_img, dst_pts)

    return


def _warp_and_blend_roi(
    frame: np.ndarray,
    overlay_img: np.ndarray,
    dst_pts: np.ndarray,
) -> None:
    """
    Warp overlay_img в ROI, заданный dst_pts (4x2), и смешать с frame in-place.
    """
    if frame is None or overlay_img is None:
        return

    fh, fw = frame.shape[:2]

    # ROI вокруг четырехугольника
    x0 = int(np.floor(np.min(dst_pts[:, 0])))
    y0 = int(np.floor(np.min(dst_pts[:, 1])))
    x1 = int(np.ceil(np.max(dst_pts[:, 0])))
    y1 = int(np.ceil(np.max(dst_pts[:, 1])))

    # небольшой отступ на случай округлений
    pad = 2
    x0 -= pad
    y0 -= pad
    x1 += pad
    y1 += pad

    # clip в границы кадра
    if x1 <= 0 or y1 <= 0 or x0 >= fw or y0 >= fh:
        return
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(fw, x1)
    y1 = min(fh, y1)

    roi_w = int(x1 - x0)
    roi_h = int(y1 - y0)
    if roi_w <= 1 or roi_h <= 1:
        return

    # src pts по размеру overlay
    h_img, w_img = overlay_img.shape[:2]
    src_pts = np.array(
        [[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]],
        dtype=np.float32,
    )

    # сдвигаем dst_pts в систему координат ROI
    dst_local = dst_pts.astype(np.float32, copy=False) - np.array([x0, y0], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_local)
    warped = cv2.warpPerspective(
        overlay_img,
        M,
        (roi_w, roi_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    roi = frame[y0:y1, x0:x1]

    if warped.ndim != 3 or warped.shape[2] < 3:
        return

    if warped.shape[2] == 4:
        alpha = warped[:, :, 3].astype(np.float32) * (1.0 / 255.0)
        if np.max(alpha) <= 0.0:
            return
        alpha3 = alpha[:, :, None]
        warped_rgb = warped[:, :, :3].astype(np.float32)
    else:
        alpha3 = 1.0
        warped_rgb = warped[:, :, :3].astype(np.float32)

    # Векторное смешивание (in-place запись обратно в roi)
    roi_f = roi.astype(np.float32)
    out = warped_rgb * alpha3 + roi_f * (1.0 - alpha3)
    np.clip(out, 0, 255, out=out)
    roi[:] = out.astype(np.uint8)
    