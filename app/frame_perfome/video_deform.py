import cv2
import numpy as np

torso_frame = None

def overlay_torso(frame, overlay_img, landmarks, frame_width, frame_height):
    # 1. Извлекаем координаты 4 точек из MediaPipe (x, y в пикселях)
    # Порядок: [Левое плечо, Правое плечо, Правое бедро, Левое бедро]
    landmark = landmarks[0]
    dst_pts = np.array([
        [landmark[11].x * frame_width, landmark[11].y * frame_height],
        [landmark[12].x * frame_width, landmark[12].y * frame_height],
        [landmark[24].x * frame_width, landmark[24].y * frame_height],
        [landmark[23].x * frame_width, landmark[23].y * frame_height]
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
    warped_img = cv2.warpPerspective(overlay_img, M, (frame_width, frame_height))

    # 4. Наложение с учетом прозрачности (Alpha-канал)
    # Создаем маску из альфа-канала трансформированного изображения
    if overlay_img.shape[2] == 4:
        alpha_mask = warped_img[:, :, 3] / 255.0
        for c in range(0, 3):
            frame[:, :, c] = (alpha_mask * warped_img[:, :, c] +
                              (1 - alpha_mask) * frame[:, :, c])
    
    torso_frame = frame

