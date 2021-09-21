import math
import cv2
import numpy as np
import pandas as pd
import itertools
from shlex import shlex


def getXYZLabels(stringList,N=3):
    if type(stringList) is not list:
        stringList = [stringList]
    return list(itertools.chain(*[[s+'_%s' % (chr(c)) for c in range(ord('x'), ord('x')+N)] for s in stringList]))

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
        lexer.commenters = '%'
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

def corners_intersection(corners):
    line1 = ( corners[0], corners[2] )
    line2 = ( corners[1], corners[3] )
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return np.array( [x,y] ).astype('float32')

def toNormPos(x,y,bbox):
    # transforms input (x,y) which is in image units (e.g. mm on an aruco board)
    # to a normalized position in the image, given the image's bounding box in
    # image units
    # (0,0) in bottom left

    extents = [bbox[2]-bbox[0], math.fabs(bbox[3]-bbox[1])]             # math.fabs to deal with bboxes where (-,-) is bottom left
    pos     = [(x-bbox[0])/extents[0], math.fabs(y-bbox[1])/extents[1]] # math.fabs to deal with bboxes where (-,-) is bottom left
    return pos

def toImagePos(x,y,bbox,imSize,margin=[0,0]):
    # transforms input (x,y) which is in image units (e.g. mm on an aruco board)
    # to a pixel position in the image, given the image's bounding box in
    # image units
    # imSize should be active image area in pixels, excluding margin

    # fractional position between bounding box edges, (0,0) in bottom left
    pos = toNormPos(x,y, bbox)
    # turn into int, add margin
    pos = [p*s+m for p,s,m in zip(pos,imSize,margin)]
    return pos

def transform(h, x, y):
    if math.isnan(x):
        return [math.nan, math.nan]

    src = np.float32([[ [x,y] ]])
    dst = cv2.perspectiveTransform(src,h)
    return dst[0][0]


def distortPoint(x, y, cameraMatrix, distCoeff):
    if math.isnan(x):
        return [math.nan, math.nan]

    fx = cameraMatrix[0][0]
    fy = cameraMatrix[1][1]
    cx = cameraMatrix[0][2]
    cy = cameraMatrix[1][2]

    k1 = distCoeff[0]
    k2 = distCoeff[1]
    k3 = distCoeff[4]
    p1 = distCoeff[2]
    p2 = distCoeff[3]

    x = (x - cx) / fx
    y = (y - cy) / fy

    r2 = x*x + y*y

    dx = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
    dy = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)

    dx = dx + (2 * p1 * x * y + p2 * (r2 + 2 * x * x))
    dy = dy + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)

    x = dx * fx + cx;
    y = dy * fy + cy;

    return np.float32([x, y]).flatten()

def undistortPoint(x, y, cameraMatrix, distCoeff):
    if math.isnan(x):
        return [math.nan, math.nan]

    p = np.float32([[[x, y]]])
    dst = cv2.undistortPoints(p, cameraMatrix, distCoeff, P=cameraMatrix)
    return dst[0][0]


def angle_between(v1, v2):
    #(180.0 / math.pi) * math.atan2(np.linalg.norm(np.cross(v1,v2)), np.dot(v1,v2))

    cx = v1[1] * v2[2] - v1[2] * v2[1]
    cy = v1[2] * v2[0] - v1[0] * v1[2]
    cz = v1[0] * v2[1] - v1[1] * v2[0]
    cross = math.sqrt( cx * cx + cy * cy + cz * cz )
    dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2] 
    return (180.0 / math.pi) * math.atan2( cross, dot )

def intersect_plane_ray(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
    # from https://rosettacode.org/wiki/Find_the_intersection_of_a_line_with_a_plane#Python

    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        raise RuntimeError("no intersection or line is within plane")
 
    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    return w + si * rayDirection + planePoint


def drawOpenCVCircle(img, center_coordinates, radius, color, thickness, subPixelFac):
    p = [np.round(x*subPixelFac) for x in center_coordinates]
    if np.all([not math.isnan(x) and abs(x)<np.iinfo(int).max for x in p]):
        p = tuple([int(x) for x in p])
        cv2.circle(img, p, radius*subPixelFac, color, thickness, lineType=cv2.LINE_AA, shift=int(math.log2(subPixelFac)))

def drawOpenCVLine(img, start_point, end_point, color, thickness, subPixelFac):
    sp = [np.round(x*subPixelFac) for x in start_point]
    ep = [np.round(x*subPixelFac) for x in   end_point]
    if np.all([not math.isnan(x) and abs(x)<np.iinfo(int).max for x in sp]) and np.all([not math.isnan(x) and abs(x)<np.iinfo(int).max for x in ep]):
        sp = tuple([int(x) for x in sp])
        ep = tuple([int(x) for x in ep])
        cv2.line(img, sp, ep, color, thickness, lineType=cv2.LINE_AA, shift=int(math.log2(subPixelFac)))
        
def drawOpenCVRectangle(img, p1, p2, color, thickness, subPixelFac):
    p1 = [np.round(x*subPixelFac) for x in p1]
    p2 = [np.round(x*subPixelFac) for x in p2]
    if np.all([not math.isnan(x) and abs(x)<np.iinfo(int).max for x in p1]) and np.all([not math.isnan(x) and abs(x)<np.iinfo(int).max for x in p2]):
        p1 = tuple([int(x) for x in p1])
        p2 = tuple([int(x) for x in p2])
        cv2.rectangle(img, p1, p2, color, thickness, lineType=cv2.LINE_AA, shift=int(math.log2(subPixelFac)))

def drawOpenCVFrameAxis(img, cameraMatrix, distCoeffs, rvec,  tvec,  armLength, thickness, subPixelFac):
    points = np.vstack((np.zeros((1,3)), armLength*np.eye(3)))
    points = cv2.projectPoints(points, rvec, tvec, cameraMatrix, distCoeffs)[0]
    drawOpenCVLine(img, points[0].flatten(), points[1].flatten(), (0, 0, 255), thickness, subPixelFac)
    drawOpenCVLine(img, points[0].flatten(), points[2].flatten(), (0, 255, 0), thickness, subPixelFac)
    drawOpenCVLine(img, points[0].flatten(), points[3].flatten(), (255, 0, 0), thickness, subPixelFac)

def drawArucoDetectedMarkers(img,corners,ids,borderColor=(0,255,0), drawIDs = True, subPixelFac=1):
    textColor   = [x for x in borderColor]
    cornerColor = [x for x in borderColor]
    textColor[0]  , textColor[1]   = textColor[1]  , textColor[0]       #   text color just sawp G and R
    cornerColor[1], cornerColor[2] = cornerColor[2], cornerColor[1]     # corner color just sawp G and B

    drawIDs = drawIDs and (ids is not None) and len(ids)>0

    for i in range(0, len(corners)):
        corner = corners[i][0]
        # draw marker sides
        for j in range(4):
            p0 = corner[j,:]
            p1 = corner[(j + 1) % 4,:]
            drawOpenCVLine(img, p0, p1, borderColor, 1, subPixelFac)
        
        # draw first corner mark
        p1 = corner[0]
        drawOpenCVRectangle(img, corner[0]-3, corner[0]+3, cornerColor, 1, subPixelFac)

        # draw IDs if wanted
        if drawIDs:
            c = corners_intersection(corner)
            cv2.putText(img, str(ids[i][0]), tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2, lineType=cv2.LINE_AA)