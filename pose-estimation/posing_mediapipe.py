import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

lite_model = "pose-estimation/models/mediapipe/pose_landmarker_lite.task"
full_model = "pose-estimation/models/mediapipe/pose_landmarker_full.task"

model_path = lite_model

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('pose landmarker result: {}'.format(result))

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

landmarker.detect_async(mp_image, frame_timestamp_ms)