import dataclasses
from typing import Mapping
import enum

from mediapipe.tasks.python.vision import drawing_utils
"""Изменение стандартных стилей отрисовки"""
_RADIUS = 5
_RED = (48, 48, 255)
_GREEN = (48, 255, 48)
_BLUE = (192, 101, 21)
_YELLOW = (0, 204, 255)
_GRAY = (128, 128, 128)
_PURPLE = (128, 64, 128)
_PEACH = (180, 229, 255)
_WHITE = (224, 224, 224)
_CYAN = (192, 255, 48)
_MAGENTA = (192, 48, 255)

_THICKNESS_POSE_LANDMARKS = 2


class PoseConnections:
  """The connections between pose landmarks."""

  @dataclasses.dataclass
  class Connection:
    """The connection class for pose landmarks."""
    start: int
    end: int

  POSE_LANDMARKS: list[Connection] = [
      Connection(11, 12),
      Connection(11, 13),
      Connection(13, 15),
      Connection(12, 14),
      Connection(14, 16),
      Connection(11, 23),
      Connection(12, 24),
      Connection(23, 24),
      Connection(23, 25),
      Connection(24, 26),
      Connection(25, 27),
      Connection(26, 28),
      Connection(27, 29),
      Connection(28, 30),
      Connection(29, 31),
      Connection(30, 32),
      Connection(27, 31),
      Connection(28, 32),
  ]


class PoseLandmark(enum.IntEnum):
  """The 12 pose landmarks."""

  LEFT_SHOULDER = 11
  RIGHT_SHOULDER = 12
  LEFT_ELBOW = 13
  RIGHT_ELBOW = 14
  LEFT_WRIST = 15
  RIGHT_WRIST = 16
  LEFT_HIP = 23
  RIGHT_HIP = 24
  LEFT_KNEE = 25
  RIGHT_KNEE = 26
  LEFT_ANKLE = 27
  RIGHT_ANKLE = 28

_POSE_LANDMARKS_LEFT = frozenset([
    PoseLandmark.LEFT_SHOULDER,
    PoseLandmark.LEFT_ELBOW,
    PoseLandmark.LEFT_WRIST,
    PoseLandmark.LEFT_HIP,
    PoseLandmark.LEFT_KNEE,
    PoseLandmark.LEFT_ANKLE,
])

_POSE_LANDMARKS_RIGHT = frozenset([
    PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.RIGHT_ELBOW,
    PoseLandmark.RIGHT_WRIST,
    PoseLandmark.RIGHT_HIP,
    PoseLandmark.RIGHT_KNEE,
    PoseLandmark.RIGHT_ANKLE,
])

_drawingSpec = drawing_utils.DrawingSpec

def get_default_pose_landmarks_style() -> Mapping[int, _drawingSpec]:
  """Returns the default pose landmarks drawing style.

  Returns:
      A mapping from each pose landmark to its default drawing spec.
  """
  pose_landmark_style = {}
  left_spec = _drawingSpec(
      color=(0, 138, 255), thickness=_THICKNESS_POSE_LANDMARKS
  )
  right_spec = _drawingSpec(
      color=(231, 217, 0), thickness=_THICKNESS_POSE_LANDMARKS
  )
  for landmark in _POSE_LANDMARKS_LEFT:
    pose_landmark_style[landmark] = left_spec
  for landmark in _POSE_LANDMARKS_RIGHT:
    pose_landmark_style[landmark] = right_spec
  return pose_landmark_style

