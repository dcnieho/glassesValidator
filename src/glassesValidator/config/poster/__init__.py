import cv2
import numpy as np
import pandas as pd
import pathlib
import importlib.resources
import shutil
import math
from matplotlib import colors

from glassesTools import drawing, marker, plane, transforms

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

    # Generate the markers
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


class Poster(plane.Plane):
    posterImageFilename = 'referencePoster.png'
    aruco_dict          = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

    def __init__(self, configDir, validationSetup):
        from .. import get_markers

        if configDir is not None:
            configDir = pathlib.Path(configDir)

        # get marker width
        if validationSetup['mode'] == 'deg':
            self.cellSizeMm = 2.*math.tan(math.radians(.5))*validationSetup['distance']*10
        else:
            self.cellSizeMm = 10 # 1cm
        markerSize = self.cellSizeMm*validationSetup['markerSide']

        # get targets, set's center for the poster also based on centerTarget indicated in validationSetup
        self._get_targets(configDir, validationSetup)

        # call base class
        markers = get_markers(configDir, validationSetup['markerPosFile'])
        ref_image_store_path = None
        if configDir is not None:
            ref_image_store_path = configDir / self.posterImageFilename
        super(Poster, self).__init__(markers, markerSize, Poster.aruco_dict, validationSetup['markerBorderBits'], self.cellSizeMm, plane_center=self.center, ref_image_store_path=ref_image_store_path, ref_image_width=validationSetup['referencePosterWidth'])

    def _get_targets(self, config_dir, validationSetup):
        """ poster space: (0,0) is at center target, (-,-) bottom left """
        from .. import get_targets

        # read in target positions
        self.targets = {}
        targets = get_targets(config_dir, validationSetup['targetPosFile'])
        if targets is not None:
            targets.x = self.cellSizeMm * targets.x.astype('float32')
            targets.y = self.cellSizeMm * targets.y.astype('float32')
            center  = targets.loc[validationSetup['centerTarget'],['x','y']]    # need center in scaled space
            targets.x -= center.x
            targets.y -= center.y
            for idx, row in targets.iterrows():
                self.targets[idx] = marker.Marker(idx, row[['x','y']].values, color=row.color)
        else:
            center = pd.Series(data=[0.,0.],index=['x','y'])
        self.center = center.to_numpy().flatten()

    def _store_reference_image(self, path: pathlib.Path, width: int) -> np.ndarray:
        # first call superclass method to generate image without targets
        img = super(Poster, self)._store_reference_image(path, width)
        height = img.shape[0]

        # add targets
        subPixelFac = 8   # for sub-pixel positioning
        for key in self.targets:
            # 1. determine position on image
            circlePos = transforms.toImagePos(*self.targets[key].center, self.bbox,[width,height])

            # 2. draw
            clr = tuple([int(i*255) for i in colors.to_rgb(self.targets[key].color)[::-1]])  # need BGR color ordering
            drawing.openCVCircle(img, circlePos, 15, clr, -1, subPixelFac)

        if path:
            cv2.imwrite(str(path), img)

        return img

