import cv2
import numpy as np
import pathlib
import importlib.resources
import shutil


def deploy_maker(output_dir):
    output_dir = pathlib.Path(output_dir)
    if not output_dir.is_dir():
        raise RuntimeError('the requested directory "%s" does not exist' % output_dir)

    # copy over all files
    for r in ['poster.tex']:
        with importlib.resources.path(__package__, r) as p:
            shutil.copyfile(p, str(output_dir/r))

    deploy_marker_images(output_dir)

def deploy_marker_images(output_dir):
    from .. import get_validation_setup

    output_dir = pathlib.Path(output_dir) / "all-markers"
    if not output_dir.is_dir():
        output_dir.mkdir()

    # get validation setup
    validationSetup = get_validation_setup()

    # Load the predefined dictionary
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

    # Generate the marker
    sz = 1000
    for i in range(250):
        markerImage = np.zeros((sz, sz), dtype=np.uint8)
        markerImage = cv2.aruco.generateImageMarker(dictionary, i, sz, markerImage, validationSetup['markerBorderBits'])

        cv2.imwrite(str(output_dir / "{}.png".format(i)), markerImage)

def deploy_default_pdf(output_file_or_dir):
    output_file_or_dir = pathlib.Path(output_file_or_dir)
    if output_file_or_dir.is_dir():
        output_file_or_dir = output_file_or_dir / 'poster.pdf'

    with importlib.resources.path(__package__,'poster.pdf') as p:
        shutil.copyfile(p, str(output_file_or_dir))