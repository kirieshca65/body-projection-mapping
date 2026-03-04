import cv2
from cv2_enumerate_cameras import enumerate_cameras
import mediapipe as mp
import time

from frame_storage import frames
from pose_estimation import mp_track_pose, init_landmarker, close_landmarker


def get_camera() -> cv2.VideoCapture:
    """Получение списка камер в системе"""
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


def start() -> None:
    
    cap = get_camera()

    try:
        init_landmarker()

        cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Pose Estimation', cv2.WINDOW_NORMAL)
        #cv2.namedWindow('Tors Deform', cv2.WINDOW_NORMAL)
        
        while True:
            success, frame = cap.read()
            frames.set_webcam(frame.copy())
            if not success:
                continue

            pose_frame = frames.get_landmarks()
            if pose_frame is not None:
                cv2.imshow('Pose Estimation', pose_frame)
           
            cv2.imshow('Webcam', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_timestamp_ms = int(time.time() * 1000)
            mp_track_pose(frame, frame_timestamp_ms)
    
    finally:
        close_landmarker()
        cap.release()


if __name__ == "__main__":
    start()