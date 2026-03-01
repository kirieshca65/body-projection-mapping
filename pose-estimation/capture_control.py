import cv2
from cv2_enumerate_cameras import enumerate_cameras
import mediapipe as mp
import time

import posing_mediapipe as pose_mp

latest_camera_frame = None

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


def run_webcam() -> None:
    
    cap = get_camera()
    
    try:
        pose_mp.init_landmarker()
        cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Pose Estimation', cv2.WINDOW_NORMAL)
        while True:
            success, frame = cap.read()
            if not success:
                continue

            if pose_mp.latest_pose_frame is not None:
                cv2.imshow('Pose Estimation', pose_mp.latest_pose_frame)
            cv2.imshow('Webcam', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_rgb,
            )
            frame_timestamp_ms = int(time.time() * 1000)
            pose_mp.mp_track_pose(mp_image, frame_timestamp_ms)
    finally:
        pose_mp.close_landmarker()
        cap.release()




if __name__ == "__main__":
    run_webcam()