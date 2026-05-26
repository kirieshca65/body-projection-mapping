from app.frame_storage import tiles
import cv2


print(tiles.get_torso())
print(type(tiles.get_torso()))
cv2.namedWindow('Tors Deform', cv2.WINDOW_NORMAL)
cv2.imshow('Tors Deform', tiles.get_torso())
input()
