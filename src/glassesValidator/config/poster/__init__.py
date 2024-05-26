import cv2
import numpy as np
import pandas as pd
import pathlib
import importlib.resources
import shutil
import math
import tempfile
from matplotlib import colors

from glassesTools import aruco, drawing, marker, transforms

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


class Poster:
    posterImageFilename = 'referencePoster.png'
    aruco_dict          = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

    def __init__(self, configDir, validationSetup, imHeight = 400):
        if configDir is not None:
            configDir = pathlib.Path(configDir)

        # get marker width
        if validationSetup['mode'] == 'deg':
            self.cellSizeMm = 2.*math.tan(math.radians(.5))*validationSetup['distance']*10
        else:
            self.cellSizeMm = 10 # 1cm
        self.markerSize = self.cellSizeMm*validationSetup['markerSide']

        # get information about poster
        self._getTargetsAndKnownMarkers(configDir, validationSetup)

        # get image of poster
        useTempDir = configDir is None
        if useTempDir:
            tempDir = tempfile.TemporaryDirectory()
            configDir = pathlib.Path(tempDir.name)

        posterImage = configDir / self.posterImageFilename
        # 1 if doesn't exist, create
        if not posterImage.is_file():
            self._storeReferencePoster(posterImage, validationSetup)
        # 2. read image
        self.img = cv2.imread(str(posterImage), cv2.IMREAD_COLOR)

        if useTempDir:
            tempDir.cleanup()

        if imHeight==-1:
            self.scale = 1
        else:
            self.scale = float(imHeight)/self.img.shape[0]
            self.img = cv2.resize(self.img, None, fx=self.scale, fy=self.scale, interpolation = cv2.INTER_AREA)
        self.height, self.width, self.channels = self.img.shape

    def getImgCopy(self, asRGB=False):
        if asRGB:
            return self.img[:,:,[2,1,0]]    # indexing returns a copy
        else:
            return self.img.copy()

    def draw(self, img, x, y, subPixelFac=1, color=None, size=6):
        if not math.isnan(x):
            xy = transforms.toImagePos(x,y,self.bbox,[self.width, self.height])
            if color is None:
                drawing.openCVCircle(img, xy, 8, (0,255,0), -1, subPixelFac)
                color = (0,0,0)
            drawing.openCVCircle(img, xy, size, color, -1, subPixelFac)

    def _getTargetsAndKnownMarkers(self, config_dir, validationSetup):
        """ poster space: (0,0) is at center target, (-,-) bottom left """
        from .. import get_targets, get_markers

        # read in target positions
        self.targets = {}
        targets = get_targets(config_dir, validationSetup['targetPosFile'])
        if targets is not None:
            center  = targets.loc[validationSetup['centerTarget'],['x','y']]
            targets.x = self.cellSizeMm * (targets.x.astype('float32') - center.x)
            targets.y = self.cellSizeMm * (targets.y.astype('float32') - center.y)
            for idx, row in targets.iterrows():
                self.targets[idx] = marker.Marker(idx, row[['x','y']].values, color=row.color)
        else:
            center = pd.Series(data=[0.,0.],index=['x','y'])


        # read in aruco marker positions
        markerHalfSizeMm  = self.markerSize/2.
        self.knownMarkers = {}
        self.bbox         = []
        markerPos = get_markers(config_dir, validationSetup['markerPosFile'])
        if markerPos is not None:
            markerPos.x = self.cellSizeMm * (markerPos.x.astype('float32') - center.x)
            markerPos.y = self.cellSizeMm * (markerPos.y.astype('float32') - center.y)
            for idx, row in markerPos.iterrows():
                c   = row[['x','y']].values
                # rotate markers (negative because poster coordinate system)
                rot = row[['rotation_angle']].values[0]
                if rot%90 != 0:
                    raise ValueError("Rotation of a marker must be a multiple of 90 degrees")
                rotr= -math.radians(rot)
                R   = np.array([[math.cos(rotr), math.sin(rotr)], [-math.sin(rotr), math.cos(rotr)]])
                # top left first, and clockwise: same order as detected aruco marker corners
                tl = c + np.matmul(R,np.array( [ -markerHalfSizeMm , -markerHalfSizeMm ] ))
                tr = c + np.matmul(R,np.array( [  markerHalfSizeMm , -markerHalfSizeMm ] ))
                br = c + np.matmul(R,np.array( [  markerHalfSizeMm ,  markerHalfSizeMm ] ))
                bl = c + np.matmul(R,np.array( [ -markerHalfSizeMm ,  markerHalfSizeMm ] ))

                self.knownMarkers[idx] = marker.Marker(idx, c, corners=[ tl, tr, br, bl ], rot=rot)

            # determine bounding box of markers ([left, top, right, bottom])
            # NB: this assumes that poster has an outer edge of markers, i.e.,
            # that it does not have targets at its edges. Also assumes markers
            # are rotated by multiples of 90 degrees
            self.bbox.append(markerPos.x.min()-markerHalfSizeMm)
            self.bbox.append(markerPos.y.min()-markerHalfSizeMm)
            self.bbox.append(markerPos.x.max()+markerHalfSizeMm)
            self.bbox.append(markerPos.y.max()+markerHalfSizeMm)

    def getArucoBoard(self, unRotateMarkers=False):
        boardCornerPoints = []
        ids = []
        for key in self.knownMarkers:
            ids.append(key)
            cornerPoints = np.vstack(self.knownMarkers[key].corners).astype('float32')
            if unRotateMarkers:
                cornerPoints = marker.getUnrotated(cornerPoints,self.knownMarkers[key].rot)

            boardCornerPoints.append(cornerPoints)
        return aruco.create_board(boardCornerPoints, ids, self.aruco_dict)

    def _storeReferencePoster(self, posterImage, validationSetup):
        referenceBoard = self.getArucoBoard(unRotateMarkers = True)
        # get image with markers
        bboxExtents    = [self.bbox[2]-self.bbox[0], math.fabs(self.bbox[3]-self.bbox[1])]  # math.fabs to deal with bboxes where (-,-) is bottom left
        aspectRatio    = bboxExtents[0]/bboxExtents[1]
        refBoardWidth  = validationSetup['referencePosterWidth']
        refBoardHeight = math.ceil(refBoardWidth/aspectRatio)
        margin         = 1  # always 1 pixel, anything else behaves strangely (markers are drawn over margin as well)

        refBoardImage  = cv2.cvtColor(
            referenceBoard.generateImage(
                (refBoardWidth+2*margin,refBoardHeight+2*margin),margin,validationSetup['markerBorderBits']),
            cv2.COLOR_GRAY2RGB
        )
        # cut off this 1-pix margin
        assert refBoardImage.shape[0]==refBoardHeight+2*margin,"Output image height is not as expected"
        assert refBoardImage.shape[1]==refBoardWidth +2*margin,"Output image width is not as expected"
        refBoardImage  = refBoardImage[1:-1,1:-1,:]
        # walk through all markers, if any are supposed to be rotated, do so
        minX =  np.inf
        maxX = -np.inf
        minY =  np.inf
        maxY = -np.inf
        rots = []
        cornerPointsU = []
        for key in self.knownMarkers:
            cornerPoints = np.vstack(self.knownMarkers[key].corners).astype('float32')
            cornerPointsU.append(marker.getUnrotated(cornerPoints, self.knownMarkers[key].rot))
            rots.append(self.knownMarkers[key].rot)
            minX = np.min(np.hstack((minX,cornerPoints[:,0])))
            maxX = np.max(np.hstack((maxX,cornerPoints[:,0])))
            minY = np.min(np.hstack((minY,cornerPoints[:,1])))
            maxY = np.max(np.hstack((maxY,cornerPoints[:,1])))
        if np.any(np.array(rots)!=0):
            # determine where the markers are placed
            sizeX = maxX - minX
            sizeY = maxY - minY
            xReduction = sizeX / float(refBoardImage.shape[1])
            yReduction = sizeY / float(refBoardImage.shape[0])
            if xReduction > yReduction:
                nRows = int(sizeY / xReduction);
                yMargin = (refBoardImage.shape[0] - nRows) / 2;
                xMargin = 0
            else:
                nCols = int(sizeX / yReduction);
                xMargin = (refBoardImage.shape[1] - nCols) / 2;
                yMargin = 0

            for r,cpu in zip(rots,cornerPointsU):
                if r != 0:
                    # figure out where marker is
                    cpu -= np.array([[minX,minY]])
                    cpu[:,0] =       cpu[:,0] / sizeX  * float(refBoardImage.shape[1]) + xMargin
                    cpu[:,1] = (1. - cpu[:,1] / sizeY) * float(refBoardImage.shape[0]) + yMargin
                    sz = np.min(cpu[2,:]-cpu[0,:])
                    # get marker
                    cpu = np.floor(cpu)
                    idxs = np.floor([cpu[0,1], cpu[0,1]+sz, cpu[0,0], cpu[0,0]+sz]).astype('int')
                    mark = refBoardImage[idxs[0]:idxs[1], idxs[2]:idxs[3]]
                    # rotate (opposite because coordinate system) and put back
                    if r==-90:
                        mark = cv2.rotate(mark, cv2.ROTATE_90_CLOCKWISE)
                    elif r==90:
                        mark = cv2.rotate(mark, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif r==180:
                        mark = cv2.rotate(mark, cv2.ROTATE_180)

                    refBoardImage[idxs[0]:idxs[1], idxs[2]:idxs[3]] = mark

        # add targets
        subPixelFac = 8   # for sub-pixel positioning
        for key in self.targets:
            # 1. determine position on image
            circlePos = transforms.toImagePos(*self.targets[key].center, self.bbox,[refBoardWidth,refBoardHeight])

            # 2. draw
            clr = tuple([int(i*255) for i in colors.to_rgb(self.targets[key].color)[::-1]])  # need BGR color ordering
            drawing.openCVCircle(refBoardImage, circlePos, 15, clr, -1, subPixelFac)

        cv2.imwrite(str(posterImage), refBoardImage)

