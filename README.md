# Базовые данные получаемые от Mediapipe
Внутри 2 массива: нормализованные координаы **Landmarks** и реальные координаты **WorldLandmarks**
* Landmarks
`x` и `y` от 0 до 1 относительно ширины изображения. `z` удаление от камеры относительно точки по середине между бедрами (0) - чем меньше значение, тем ближе к камере. 
* WorldLandmarks
Инетерпритирует данные под реальные размеры в метрах. Точкой отсчета является середины между бедрами
```
PoseLandmarkerResult:
  Landmarks:
    Landmark #0:
      x            : 0.638852
      y            : 0.671197
      z            : 0.129959
      visibility   : 0.9999997615814209
      presence     : 0.9999984502792358
    Landmark #1:
      x            : 0.634599
      y            : 0.536441
      z            : -0.06984
      visibility   : 0.999909
      presence     : 0.999958
    ... (33 landmarks per pose)
  WorldLandmarks:
    Landmark #0:
      x            : 0.067485
      y            : 0.031084
      z            : 0.055223
      visibility   : 0.9999997615814209
      presence     : 0.9999984502792358
    Landmark #1:
      x            : 0.063209
      y            : -0.00382
      z            : 0.020920
      visibility   : 0.999976
      presence     : 0.999998
    ... (33 world landmarks per pose)
  SegmentationMasks:
    ... (pictured below)
```

## Точки отслеживания Mediapipe
![Схема размещения точек](pose_landmarks_index.png)
Работаем только с частью точек:
```
11 - left shoulder
12 - right shoulder
13 - left elbow
14 - right elbow
15 - left wrist
16 - right wrist
23 - left hip
24 - right hip
25 - left knee
26 - right knee
27 - left ankle
28 - right ankle
```