import cv2
import numpy as np
from pathlib import Path
import importlib.resources
import shutil


def deployMaker(outDir):
    outDir = Path(outDir)
    if not outDir.is_dir():
        raise RuntimeError('the requested directory "%s" does not exist' % outDir)

    # copy over all files
    for r in ['board.tex']:
        with importlib.resources.path(__package__, r) as p:
            shutil.copyfile(p, str(outDir/r))

    deployMarkerImages(outDir)

def deployMarkerImages(outDir):
    from .. import getValidationSetup

    outDir = Path(outDir) / "all-markers"
    if not outDir.is_dir():
        outDir.mkdir()

    # get validation setup
    validationSetup = getValidationSetup()

    # Load the predefined dictionary
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)

    # Generate the marker
    sz = 1000
    for i in range(250):
        markerImage = np.zeros((sz, sz), dtype=np.uint8)
        markerImage = cv2.aruco.drawMarker(dictionary, i, sz, markerImage, validationSetup['markerBorderBits'])

        cv2.imwrite(str(outDir / "{}.png".format(i)), markerImage)

def deployDefaultPdf(outFile):
    outFile = Path(outFile)
    if outFile.is_dir():
        outFile = outFile / 'board.pdf'

    with importlib.resources.path(__package__,'board.pdf') as p:
        shutil.copyfile(p, str(outFile))