import cv2
import numpy as np

from frame_storage import frames, tiles

#torso_frame = tiles.torso.copy()

fr_width : int
fr_height : int



def overlay_torso(landmarks):
    global fr_width, fr_height
    fw = fr_width
    fh = fr_height
    overlay_img = tiles.get_torso()
    frame = frames.get_webcam()
    # 1. Извлекаем координаты 4 точек из MediaPipe (x, y в пикселях)
    # Порядок: [Левое плечо, Правое плечо, Правое бедро, Левое бедро]
    landmark = landmarks[0]
    dst_pts = np.array([
        [landmark[11].x * fw, landmark[11].y * fh],
        [landmark[12].x * fw, landmark[12].y * fh],
        [landmark[24].x * fw, landmark[24].y * fh],
        [landmark[23].x * fw, landmark[23].y * fh]
    ], dtype="float32")

    # 2. Координаты углов исходного изображения (прямоугольник)
    h_img, w_img = overlay_img.shape[:2]
    src_pts = np.array([
        [0, 0],              # Топ-лево (к левому плечу)
        [w_img, 0],          # Топ-право (к правому плечу)
        [w_img, h_img],      # Низ-право (к правому бедру)
        [0, h_img]           # Низ-лево (к левому бедру)
    ], dtype="float32")

    # 3. Вычисляем матрицу перспективы и трансформируем картинку
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_img = cv2.warpPerspective(overlay_img, M, (fw, fh))

    # 4. Наложение overlay поверх frame с учётом прозрачности
    if overlay_img.shape[2] == 4:
        alpha_mask = warped_img[:, :, 3] / 255.0
    else:
        alpha_mask = np.ones((fh, fw), dtype=np.float32)

    for c in range(3):
        frame[:, :, c] = (alpha_mask * warped_img[:, :, c] +
                          (1 - alpha_mask) * frame[:, :, c]).astype(np.uint8)

    frames.set_preview(frame)
    return frame

def init_frame():
    global fr_width, fr_height
    fr_width, fr_height = frames.get_webcam_res()