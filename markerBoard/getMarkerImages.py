import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, './..')
import utils

# get validation setup
validationSetup = utils.getValidationSetup(Path(__file__).resolve().parent.parent / 'config')

# Load the predefined dictionary
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)

# Generate the marker
sz = 1000
for i in range(250):
    markerImage = np.zeros((sz, sz), dtype=np.uint8)
    markerImage = cv2.aruco.drawMarker(dictionary, i, sz, markerImage, validationSetup['markerBorderBits'])

    cv2.imwrite(str(Path(__file__).resolve().parent / "all-markers" / "{}.png".format(i)), markerImage)