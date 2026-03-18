from typing import Optional
import cv2
import numpy as np

from frame_storage import frames, tiles


def draw_overlay(landmarks):
    frame = frames.get_webcam()
    if frame is None:
        return
    # Рисуем оверлеи на копии, и обновляем preview один раз
    preview = frame.copy()

    # Конечности: берём текущий кадр видео + фиксированную маску (BGRA)
    ov = tiles.build_overlay_bgra("mask_forearm_r.png")
    if ov is not None:
        overlay_limbs(landmarks, (11, 13), preview, overlay_img=ov)

    ov = tiles.build_overlay_bgra("mask_forearm_l.png")
    if ov is not None:
        overlay_limbs(landmarks, (12, 14), preview, overlay_img=ov)

    ov = tiles.build_overlay_bgra("mask_thigh_r.png")
    if ov is not None:
        overlay_limbs(landmarks, (23, 25), preview, overlay_img=ov)

    ov = tiles.build_overlay_bgra("mask_thigh_l.png")
    if ov is not None:
        overlay_limbs(landmarks, (24, 26), preview, overlay_img=ov)

    ov = tiles.build_overlay_bgra("mask_torso.png")
    if ov is not None:
        overlay_torso(landmarks, preview, overlay_img=ov)
    frames.set_preview(preview)


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

    if frame is None or overlay_img is None:
        return

    fh, fw = frame.shape[:2]

    # Извлекаем координаты 4 точек из MediaPipe (x, y в пикселях)
    # Порядок: [Левое плечо, Правое плечо, Правое бедро, Левое бедро]
    landmark = landmarks[0]
    dst_pts = np.array(
        [
            [landmark[11].x * fw, landmark[11].y * fh],
            [landmark[12].x * fw, landmark[12].y * fh],
            [landmark[24].x * fw, landmark[24].y * fh],
            [landmark[23].x * fw, landmark[23].y * fh],
        ],
        dtype="float32",
    )

    # расширяем четырёхугольник относительно центра, чтобы картинка выходила за пределы точек.
    scale = 1.2  # коэффициент "запаса" вокруг точек
    center = dst_pts.mean(axis=0, keepdims=True)
    dst_pts = (dst_pts - center) * scale + center


    # быстрый варп+альфа только по ROI
    _warp_and_blend_roi(frame, overlay_img, dst_pts)

    return

def overlay_limbs(
    landmarks,
    limbs: tuple[int, int],
    frame : np.ndarray,
    overlay_img: Optional[np.ndarray] = None,
    *,
    width_scale: float = 0.35,
    extend_scale: float = 0.10,
):
    """
    Накладывает `overlay_img` по двум точкам (индексам лендмарок) как "полоску"
    вдоль отрезка между ними.

    - width_scale: ширина полосы как доля длины отрезка
    - extend_scale: насколько продлить отрезок за точки (доля длины)
    """
    if overlay_img is None:
        overlay_img = tiles.get_torso()

    if frame is None or overlay_img is None:
        return

    fh, fw = frame.shape[:2]
    landmark = landmarks[0]
    p1 = np.array([landmark[limbs[0]].x * fw, landmark[limbs[0]].y * fh], dtype=np.float32)
    p2 = np.array([landmark[limbs[1]].x * fw, landmark[limbs[1]].y * fh], dtype=np.float32)

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

    # небольшая "подушка" на случай округлений
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