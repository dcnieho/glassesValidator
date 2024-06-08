import cv2
from pathlib import Path


# Load the predefined dictionary
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)

board = cv2.aruco.CharucoBoard_create(7, 10, 4, 3, dictionary)
imboard = board.draw((1414, 2000))
cv2.imwrite(Path(__file__).resolve().parent / "chessboard1.png", imboard)