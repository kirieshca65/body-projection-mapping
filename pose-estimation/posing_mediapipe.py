import time

import cv2
from cv2_enumerate_cameras import enumerate_cameras

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python.vision.hand_landmarker import landmark

# Модели mediapipe для отслеживания
lite_model = 'models/mediapipe/pose_landmarker_lite.task'
full_model = 'models/mediapipe/pose_landmarker_full.task'

model_path = lite_model

#Импорт базовых параметров для модели PoseLandmarker
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
PoseLandmarkerResult = vision.PoseLandmarkerResult
VisionRunningMode = vision.RunningMode

def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    drawing_utils.draw_landmarks(image=output_image.numpy_view(), landmark_list=result.pose_landmarks, connection_drawing_spec = drawing_styles.get_default_pose_landmarks_style())
    cv2.imshow('Pose Estimation', output_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()
    print('pose landmarker result: {}'.format(result))

# Инициализация базовых параметров для модели
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)


def run_webcam() -> None:
    
    cap = get_camera()
    
    with PoseLandmarker.create_from_options(options) as landmarker:
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=frame_rgb,
                )
                frame_timestamp_ms = int(time.time() * 1000)
                landmarker.detect_async(mp_image, frame_timestamp_ms)
        finally:
            cap.release()


def get_camera() -> cv2.VideoCapture:
    cameras = enumerate_cameras(cv2.CAP_DSHOW)
    for camera in cameras:
            print(camera)
    while True:
        index = int(input("Enter the index of the camera: "))
        if index not in range(len(cameras)):
            continue
        cap = cv2.VideoCapture(cameras[index].index, cameras[index].backend)
        if not cap.isOpened():
            print(f"Unable to open webcam with index {index}.")
            continue
        else:
            return cap

if __name__ == "__main__":
    run_webcam()

