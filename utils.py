import math
import cv2
import numpy as np
import pandas as pd
from shlex import shlex


class Marker:
    def __init__(self, key, center, corners=None, color=None):
        self.key = key
        self.center = center
        self.corners = corners
        self.color = color

    def __str__(self):
        ret = '[%s]: center @ (%.2f, %.2f)' % (self.key, self.center[0], self.center[1])
        return ret

def getValidationSetup(configDir):
    # read key=value pairs into dict
    with open(str(configDir / "validationSetup.txt")) as f:
        lexer = shlex(f)
        lexer.whitespace += '='
        lexer.wordchars += '.'  # don't split extensions of filenames in the input file
        validationSetup = dict(zip(lexer, lexer))

    # parse numerics into int or float
    for key,val in validationSetup.items():
        if np.all([c.isdigit() for c in val]):
            validationSetup[key] = int(val)
        else:
            try:
                validationSetup[key] = float(val)
            except:
                pass # just keep value as a string
    return validationSetup

def getKnownMarkers(configDir, validationSetup):
    """ (0,0) is at center target, (-,-) bottom left """
    cellSizeMm = 2.*math.tan(math.radians(.5))*validationSetup['distance']*10
    markerHalfSizeMm = cellSizeMm*validationSetup['markerSide']/2.
            
    # read in target positions
    markers = {}
    targets = pd.read_csv(str(configDir / validationSetup['targetPosFile']),index_col=0,names=['x','y','clr'])
    center  = targets.loc[validationSetup['centerTarget'],['x','y']]
    targets.x = cellSizeMm * (targets.x.astype('float32') - center.x)
    targets.y = cellSizeMm * (targets.y.astype('float32') - center.y)
    for idx, row in targets.iterrows():
        key = 't%d' % idx
        markers[key] = Marker(key, row[['x','y']].values, color=row.clr)
    
    # read in aruco marker positions
    markerPos = pd.read_csv(str(configDir / validationSetup['markerPosFile']),index_col=0,names=['x','y'])
    markerPos.x = cellSizeMm * (markerPos.x.astype('float32') - center.x)
    markerPos.y = cellSizeMm * (markerPos.y.astype('float32') - center.y)
    for idx, row in markerPos.iterrows():
        key = '%d' % idx
        c   = row[['x','y']].values
        # top left first, and clockwise: same order as detected aruco marker corners
        tl = c + np.array( [ -markerHalfSizeMm ,  markerHalfSizeMm ] )
        tr = c + np.array( [  markerHalfSizeMm ,  markerHalfSizeMm ] )
        br = c + np.array( [  markerHalfSizeMm , -markerHalfSizeMm ] )
        bl = c + np.array( [ -markerHalfSizeMm , -markerHalfSizeMm ] )
        markers[key] = Marker(key, c, corners=[ tl, tr, br, bl ])
    
    # determine bounding box of markers ([left, top, right, bottom])
    bbox = []
    bbox.append(markerPos.x.min()-markerHalfSizeMm)
    bbox.append(markerPos.y.max()+markerHalfSizeMm) # top is at positive
    bbox.append(markerPos.x.max()+markerHalfSizeMm)
    bbox.append(markerPos.y.min()-markerHalfSizeMm) # bottom is at negative

    return markers, bbox

def toNormPos(x,y,bbox):
    # transforms input (x,y) which is in image units (e.g. mm on an aruco board)
    # to a normalized position in the image, given the image's bounding box in
    # image units
    # (0,0) in bottom left

    extents = [bbox[2]-bbox[0], math.fabs(bbox[3]-bbox[1])]             # math.fabs to deal with bboxes where (-,-) is bottom left
    pos     = [(x-bbox[0])/extents[0], math.fabs(y-bbox[1])/extents[1]] # math.fabs to deal with bboxes where (-,-) is bottom left
    return pos

def toImagePos(x,y,bbox,imSize,margin=[0,0],subPixelFac=1):
    # transforms input (x,y) which is in image units (e.g. mm on an aruco board)
    # to a pixel position in the image, given the image's bounding box in
    # image units
    # imSize should be active image area in pixels, excluding margin
    # takes into account OpenCV's subPixelFac for subpixel positioning (see docs of
    # e.g. cv2.circle)

    # fractional position between bounding box edges, (0,0) in bottom left
    pos = toNormPos(x,y, bbox)
    # turn into int, add margin
    pos = tuple([int(round(p*s+m)*subPixelFac) for p,s,m in zip(pos,imSize,margin)])
    return pos

def transform(h, x, y):
    src = np.float32([[ [x,y] ]])
    dst = cv2.perspectiveTransform(src,h)
    return dst[0][0]


def distortPoint(p, cameraMatrix, distCoeff):
    fx = cameraMatrix[0][0]
    fy = cameraMatrix[1][1]
    cx = cameraMatrix[0][2]
    cy = cameraMatrix[1][2]

    k1 = distCoeff[0]
    k2 = distCoeff[1]
    k3 = distCoeff[4]
    p1 = distCoeff[2]
    p2 = distCoeff[3]

    x = (p[0] - cx) / fx
    y = (p[1] - cy) / fy

    r2 = x*x + y*y

    dx = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
    dy = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)

    dx = dx + (2 * p1 * x * y + p2 * (r2 + 2 * x * x))
    dy = dy + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)

    p[0] = dx * fx + cx;
    p[1] = dy * fy + cy;

    return p