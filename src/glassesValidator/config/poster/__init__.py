import cv2
import numpy as np
import pandas as pd
import pathlib
import importlib.resources
import shutil
import math
from matplotlib import colors

from glassesTools import aruco, drawing, marker, plane, transforms

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

    # generate and store the markers
    aruco.deploy_marker_images(output_dir, 1000, Poster.default_aruco_dict, validationSetup['markerBorderBits'])

def deploy_default_pdf(output_file_or_dir):
    output_file_or_dir = pathlib.Path(output_file_or_dir)
    if output_file_or_dir.is_dir():
        output_file_or_dir = output_file_or_dir / 'poster.pdf'

    with importlib.resources.path(__package__,'poster.pdf') as p:
        shutil.copyfile(p, str(output_file_or_dir))


class Poster(plane.Plane):
    posterImageFilename = 'referencePoster.png'
    default_aruco_dict  = cv2.aruco.DICT_4X4_250

    def __init__(self, configDir, validationSetup, **kwarg):
        from .. import get_markers

        if configDir is not None:
            configDir = pathlib.Path(configDir)

        # get marker width
        if validationSetup['mode'] == 'deg':
            self.cellSizeMm = 2.*math.tan(math.radians(.5))*validationSetup['distance']*10
        else:
            self.cellSizeMm = 10 # 1cm
        markerSize = self.cellSizeMm*validationSetup['markerSide']

        # get board size
        plane_size = plane.Coordinate(validationSetup['gridCols']*self.cellSizeMm, validationSetup['gridRows']*self.cellSizeMm)

        # get targets first, so that they can be drawn on the reference image
        self.targets: dict[int,marker.Marker] = {}
        origin = self._get_targets(configDir, validationSetup)

        # call base class
        markers = get_markers(configDir, validationSetup['markerPosFile'])
        ref_image_store_path = None
        if 'ref_image_store_path' in kwarg:
            ref_image_store_path = kwarg.pop('ref_image_store_path')
        elif configDir is not None:
            ref_image_store_path = configDir / self.posterImageFilename
        super(Poster, self).__init__(markers, markerSize, plane_size, Poster.default_aruco_dict, validationSetup['markerBorderBits'],self.cellSizeMm, "mm", ref_image_store_path=ref_image_store_path, ref_image_size=validationSetup['referencePosterSize'],**kwarg)

        # set center
        self.set_origin(origin)

    def set_origin(self, origin: plane.Coordinate):
        # set origin of plane. Origin location is on current (not original) plane
        # so set_origin([5., 0.]) three times in a row shifts the origin rightward by 15 units
        for i in self.targets:
            self.targets[i].shift(-np.array(origin))
        super(Poster, self).set_origin(origin)

    def _get_targets(self, config_dir, validationSetup) -> plane.Coordinate:
        """ poster space: (0,0) is origin (might be center target), (-,-) bottom left """
        from .. import get_targets

        # read in target positions
        targets = get_targets(config_dir, validationSetup['targetPosFile'])
        if targets is not None:
            targets['center'] = list(targets[['x','y']].values)
            targets['center'] *= self.cellSizeMm
            targets = targets.drop(['x','y'], axis=1)
            self.targets = {idx:marker.Marker(idx,**kwargs) for idx,kwargs in zip(targets.index.values,targets.to_dict(orient='records'))}
            origin = plane.Coordinate(*targets.loc[validationSetup['centerTarget']].center.copy())  # NB: need origin in scaled space
        else:
            origin = plane.Coordinate(0.,0.)
        return origin

    def _store_reference_image(self, path: pathlib.Path, width: int) -> np.ndarray:
        # first call superclass method to generate image without targets
        img = super(Poster, self)._store_reference_image(path, width)
        height = img.shape[0]

        # add targets
        subPixelFac = 8   # for sub-pixel positioning
        for key in self.targets:
            # 1. determine position on image
            circlePos = transforms.to_image_pos(*self.targets[key].center, self.bbox,[width,height])

            # 2. draw
            clr = tuple([int(i*255) for i in colors.to_rgb(self.targets[key].color)[::-1]])  # need BGR color ordering
            drawing.openCVCircle(img, circlePos, 15, clr, -1, subPixelFac)

        if path:
            cv2.imwrite(path, img)

        return img

