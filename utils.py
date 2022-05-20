import math
import cv2
import numpy as np
import pandas as pd
import csv
import itertools
import sys
import warnings
from shlex import shlex
import bisect
import mp4analyser.iso
from matplotlib import colors


def getXYZLabels(stringList,N=3):
    if type(stringList) is not list:
        stringList = [stringList]
    return list(itertools.chain(*[[s+'_%s' % (chr(c)) for c in range(ord('x'), ord('x')+N)] for s in stringList]))

def noneIfAnyNan(vals):
    if not np.any(np.isnan(vals)):
        return vals
    else:
        return None

def dataReaderHelper(entry,lbl,N=3,type='float32'):
    columns = getXYZLabels(lbl,N)
    if np.all([x in entry for x in columns]):
        return np.array([entry[x] for x in columns]).astype(type)
    else:
        return None

def allNanIfNone(vals,numel):
    if vals is None:
        return [math.nan for x in range(numel)]
    else:
        return vals

def cartesian_product(*arrays):
    ndim = len(arrays)
    return (np.stack(np.meshgrid(*arrays), axis=-1)
              .reshape(-1, ndim))

def getFrameTimestampsFromVideo(vid_file):
    """
    Parse the supplied video, return an array of frame timestamps
    """
    if vid_file.suffix in ['.mp4', '.mov']:
        # parse mp4 file
        boxes   = mp4analyser.iso.Mp4File(str(vid_file))
        # 1. find mdat box
        moov    = boxes.child_boxes[[i for i,x in enumerate(boxes.child_boxes) if x.type=='moov'][0]]
        # 2. find track boxes
        trakIdxs= [i for i,x in enumerate(moov.child_boxes) if x.type=='trak']
        # 3. check which track contains video
        trakIdx = [i for i,x in enumerate(boxes.get_summary()['track_list']) if x['media_type']=='video'][0]
        trak    = moov.child_boxes[trakIdxs[trakIdx]]
        # 4. get mdia
        mdia    = trak.child_boxes[[i for i,x in enumerate(trak.child_boxes) if x.type=='mdia'][0]]
        # 5. get time_scale field from mdhd
        time_base = mdia.child_boxes[[i for i,x in enumerate(mdia.child_boxes) if x.type=='mdhd'][0]].box_info['timescale']
        # 6. get minf
        minf    = mdia.child_boxes[[i for i,x in enumerate(mdia.child_boxes) if x.type=='minf'][0]]
        # 7. get stbl
        stbl    = minf.child_boxes[[i for i,x in enumerate(minf.child_boxes) if x.type=='stbl'][0]]
        # 8. get sample table from stts
        samp_table = stbl.child_boxes[[i for i,x in enumerate(stbl.child_boxes) if x.type=='stts'][0]].box_info['entry_list']
        # 9. now we have all the info to determine the timestamps of each frame
        df = pd.DataFrame(samp_table) # easier to use that way
        totalFrames = df['sample_count'].sum()
        frameTs = np.zeros(totalFrames)
        # first uncompress delta table
        idx = 0
        for count,dur in zip(df['sample_count'], df['sample_delta']):
            frameTs[idx:idx+count] = dur
            idx = idx+count
        # turn into timestamps, first in time_base units
        frameTs = np.roll(frameTs,1)
        frameTs[0] = 0.
        frameTs = np.cumsum(frameTs)
        # now into timestamps in ms
        frameTs = frameTs/time_base*1000
    else:
        # open file with opencv and get timestamps of each frame
        vid = cv2.VideoCapture(str(vid_file))
        frameTs = []
        while vid.isOpened():
            # get current time (we want start time of frame
            frameTs.append(vid.get(cv2.CAP_PROP_POS_MSEC))
            
            # Capture frame-by-frame
            ret, frame = vid.read()

            if not ret == True:
                break

        # release the video capture object
        vid.release()
        frameTs = np.array(frameTs)

    ### convert the frame_timestamps to dataframe
    frameIdx = np.arange(0, len(frameTs))
    frameTsDf = pd.DataFrame({'frame_idx': frameIdx, 'timestamp': frameTs})
    frameTsDf.set_index('frame_idx', inplace=True)
    
    return frameTsDf

def tssToFrameNumber(ts,frameTimestamps,mode='nearest'):
    df = pd.DataFrame(index=ts)
    df.insert(0,'frame_idx',np.int64(0))
    
    # get index where this ts would be inserted into the frame_timestamp array
    idxs = np.searchsorted(frameTimestamps, ts)
    if mode=='after':
        idxs = idxs.astype('float32')
        # out of range, set to nan
        idxs[idxs==0] = np.nan
        # -1: since idx points to frame timestamp for frame after the one during which the ts ocurred, correct
        idxs -= 1
    elif mode=='nearest':
        # implementation from https://stackoverflow.com/questions/8914491/finding-the-nearest-value-and-return-the-index-of-array-in-python/8929827#8929827
        # same logic as used by pupil labs
        idxs = np.clip(idxs, 1, len(frameTimestamps)-1)
        left = frameTimestamps[idxs-1]
        right = frameTimestamps[idxs]
        idxs -= ts - left < right - ts

    df=df.assign(frame_idx=idxs)
    if mode=='after':
        df=df.convert_dtypes() # turn into int64 again

    return df

class Marker:
    def __init__(self, key, center, corners=None, color=None, rot=0):
        self.key = key
        self.center = center
        self.corners = corners
        self.color = color
        self.rot = rot

    def __str__(self):
        ret = '[%s]: center @ (%.2f, %.2f), rot %.0f deg' % (self.key, self.center[0], self.center[1], self.rot)
        return ret

def getValidationSetup(configDir):
    # read key=value pairs into dict
    with open(configDir / "validationSetup.txt") as f:
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
    # transforms input (x,y) which is on a plane in world units
    # (e.g. mm on an aruco board) to a normalized position
    # in an image of the plane, given the image's bounding box in
    # world units
    # (0,0) is bottom left

    extents = [bbox[2]-bbox[0], math.fabs(bbox[3]-bbox[1])]             # math.fabs to deal with bboxes where (-,-) is bottom left
    pos     = [(x-bbox[0])/extents[0], math.fabs(y-bbox[1])/extents[1]] # math.fabs to deal with bboxes where (-,-) is bottom left
    return pos

def toImagePos(x,y,bbox,imSize,margin=[0,0]):
    # transforms input (x,y) which is on a plane in world units
    # (e.g. mm on an aruco board) to a pixel position in the
    # image, given the image's bounding box in world units
    # imSize should be active image area in pixels, excluding margin

    # fractional position between bounding box edges, (0,0) in bottom left
    pos = toNormPos(x,y, bbox)
    # turn into int, add margin
    pos = [p*s+m for p,s,m in zip(pos,imSize,margin)]
    return pos

def estimateHomography(known, detectedCorners, detectedIDs):
    # collect matching corners in image and in world
    pts_src = []
    pts_dst = []
    for i in range(0, len(detectedIDs)):
        key = '%d' % detectedIDs[i]
        if key in known:
            pts_src.extend( detectedCorners[i][0] )
            pts_dst.extend(    known[key].corners )

    if len(pts_src) < 4:
        return None, False

    # compute Homography
    pts_src = np.float32(pts_src)
    pts_dst = np.float32(pts_dst)
    h, _ = cv2.findHomography(pts_src, pts_dst)

    return h, True

def applyHomography(h, x, y):
    if math.isnan(x):
        return [math.nan, math.nan]

    src = np.asarray([x, y], dtype='float32').reshape((1, -1, 2))
    dst = cv2.perspectiveTransform(src,h)
    return dst.flatten()


def distortPoint(x, y, cameraMatrix, distCoeff):
    if math.isnan(x):
        return [math.nan, math.nan]
    
    # unproject, ignoring distortion as this is an undistored point
    points_2d = np.asarray([x, y], dtype='float32').reshape((1, -1, 2))
    points_2d = cv2.undistortPoints(points_2d, cameraMatrix, np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0]]))
    points_3d = cv2.convertPointsToHomogeneous(points_2d)
    points_3d.shape = -1, 3

    # reproject, applying distortion
    points_2d, _ = cv2.projectPoints(points_3d, np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), cameraMatrix, distCoeff)

    return points_2d.flatten()

def undistortPoint(x, y, cameraMatrix, distCoeff):
    if math.isnan(x):
        return [math.nan, math.nan]

    points_2d = np.asarray([x, y], dtype='float32').reshape((1, -1, 2))
    points_2d = cv2.undistortPoints(points_2d, cameraMatrix, distCoeff, P=cameraMatrix) # P=cameraMatrix to reproject to camera
    return points_2d.flatten()


def angle_between(v1, v2): 
    return (180.0 / math.pi) * math.atan2(np.linalg.norm(np.cross(v1,v2)), np.dot(v1,v2))

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
    # same as the openCV function, but with anti-aliasing for a nicer image if subPixelFac>1
    points = np.vstack((np.zeros((1,3)), armLength*np.eye(3)))
    points = cv2.projectPoints(points, rvec, tvec, cameraMatrix, distCoeffs)[0]
    drawOpenCVLine(img, points[0].flatten(), points[1].flatten(), (0, 0, 255), thickness, subPixelFac)
    drawOpenCVLine(img, points[0].flatten(), points[2].flatten(), (0, 255, 0), thickness, subPixelFac)
    drawOpenCVLine(img, points[0].flatten(), points[3].flatten(), (255, 0, 0), thickness, subPixelFac)

def drawArucoDetectedMarkers(img,corners,ids,borderColor=(0,255,0), drawIDs = True, subPixelFac=1):
    # same as the openCV function, but with anti-aliasing for a (much) nicer image if subPixelFac>1
    textColor   = [x for x in borderColor]
    cornerColor = [x for x in borderColor]
    textColor[0]  , textColor[1]   = textColor[1]  , textColor[0]       #   text color just swap G and R
    cornerColor[1], cornerColor[2] = cornerColor[2], cornerColor[1]     # corner color just swap G and B

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



class Gaze:
    def __init__(self, ts, vid2D, world3D=None, lGazeVec=None, lGazeOrigin=None, rGazeVec=None, rGazeOrigin=None):
        self.ts = ts
        self.vid2D = vid2D
        self.world3D = world3D
        self.lGazeVec= lGazeVec
        self.lGazeOrigin = lGazeOrigin
        self.rGazeVec= rGazeVec
        self.rGazeOrigin = rGazeOrigin
        
    @staticmethod
    def readDataFromFile(fileName):
        gazes = {}
        maxFrameIdx = 0
        with open(fileName, 'r' ) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for entry in reader:
                frame_idx   = float(entry['frame_idx'])
                ts          = float(entry['timestamp'])
                
                vid2D       = dataReaderHelper(entry,'vid_gaze_pos',2)
                world3D     = dataReaderHelper(entry,'3d_gaze_pos')
                lGazeVec    = dataReaderHelper(entry,'l_gaze_dir')
                lGazeOrigin = dataReaderHelper(entry,'l_gaze_ori')
                rGazeVec    = dataReaderHelper(entry,'r_gaze_dir')
                rGazeOrigin = dataReaderHelper(entry,'r_gaze_ori')
                gaze = Gaze(ts, vid2D, world3D, lGazeVec, lGazeOrigin, rGazeVec, rGazeOrigin)

                if frame_idx in gazes:
                    gazes[frame_idx].append(gaze)
                else:
                    gazes[frame_idx] = [gaze]

                maxFrameIdx = int(max(maxFrameIdx,frame_idx))

        return gazes,maxFrameIdx

    def draw(self, img, subPixelFac=1, camRot=None, camPos=None, cameraMatrix=None, distCoeff=None):
        drawOpenCVCircle(img, self.vid2D, 8, (0,255,0), 2, subPixelFac)
        # draw 3D gaze point as well, should coincide with 2D gaze point
        if self.world3D is not None and camRot is not None and camPos is not None and cameraMatrix is not None and distCoeff is not None:
            a = cv2.projectPoints(np.array(self.world3D).reshape(1,3),camRot,camPos,cameraMatrix,distCoeff)[0][0][0]
            drawOpenCVCircle(img, a, 5, (0,0,0), -1, subPixelFac)
            

def getMarkerUnrotated(cornerPoints, rot):
    # markers are rotated in multiples of 90 only, so can easily unrotate
    if rot == -90:
        # -90 deg
        cornerPoints = np.vstack((cornerPoints[-1,:], cornerPoints[0:3,:]))
    elif rot == 90:
        # 90 deg
        cornerPoints = np.vstack((cornerPoints[1:,:], cornerPoints[0,:]))
    elif rot == 180:
        # 180 deg
        cornerPoints = np.vstack((cornerPoints[2:,:], cornerPoints[0:2,:]))

    return cornerPoints

class Reference:
    boardImageFilename = 'referenceBoard.png'
    aruco_dict         = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)

    def __init__(self, configDir, validationSetup, imHeight = 400):
        # get marker width
        if validationSetup['mode'] == 'deg':
            self.cellSizeMm = 2.*math.tan(math.radians(.5))*validationSetup['distance']*10
        else:
            self.cellSizeMm = 10 # 1cm
        self.markerSize = self.cellSizeMm*validationSetup['markerSide']

        # get information about marker board
        self.knownMarkers, self.bbox = self._getKnownMarkers(configDir, validationSetup)

        # get image of markerboard
        boardImage = configDir / self.boardImageFilename
        # 1 if doesn't exist, create
        if not boardImage.is_file():
            self._storeReferenceBoard(configDir, validationSetup)
        # 2. read image
        self.img = cv2.imread(str(boardImage), cv2.IMREAD_COLOR)
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
            xy = toImagePos(x,y,self.bbox,[self.width, self.height])
            if color is None:
                drawOpenCVCircle(img, xy, 8, (0,255,0), -1, subPixelFac)
                color = (0,0,0)
            drawOpenCVCircle(img, xy, size, color, -1, subPixelFac)

    def _getKnownMarkers(self, configDir, validationSetup):
        """ board space: (0,0) is at center target, (-,-) bottom left """
        markerHalfSizeMm = self.markerSize/2.
            
        # read in target positions
        markers = {}
        targets = pd.read_csv(str(configDir / validationSetup['targetPosFile']),index_col=0,names=['x','y','clr'])
        center  = targets.loc[validationSetup['centerTarget'],['x','y']]
        targets.x = self.cellSizeMm * (targets.x.astype('float32') - center.x)
        targets.y = self.cellSizeMm * (targets.y.astype('float32') - center.y)
        for idx, row in targets.iterrows():
            key = 't%d' % idx
            markers[key] = Marker(key, row[['x','y']].values, color=row.clr)
    
        # read in aruco marker positions
        markerPos = pd.read_csv(str(configDir / validationSetup['markerPosFile']),index_col=0,names=['x','y','rot'])
        markerPos.x = self.cellSizeMm * (markerPos.x.astype('float32') - center.x)
        markerPos.y = self.cellSizeMm * (markerPos.y.astype('float32') - center.y)
        for idx, row in markerPos.iterrows():
            key = '%d' % idx
            c   = row[['x','y']].values
            # rotate markers (negative because poster coordinate system)
            rot = row[['rot']].values[0]
            if rot%90 != 0:
                raise ValueError("Rotation of a marker must be a multiple of 90 degrees")
            rotr= -math.radians(rot)
            R   = np.array([[math.cos(rotr), math.sin(rotr)], [-math.sin(rotr), math.cos(rotr)]])
            # top left first, and clockwise: same order as detected aruco marker corners
            tl = c + np.matmul(R,np.array( [ -markerHalfSizeMm ,  markerHalfSizeMm ] ))
            tr = c + np.matmul(R,np.array( [  markerHalfSizeMm ,  markerHalfSizeMm ] ))
            br = c + np.matmul(R,np.array( [  markerHalfSizeMm , -markerHalfSizeMm ] ))
            bl = c + np.matmul(R,np.array( [ -markerHalfSizeMm , -markerHalfSizeMm ] ))
        
            markers[key] = Marker(key, c, corners=[ tl, tr, br, bl ], rot=rot)
    
        # determine bounding box of markers ([left, top, right, bottom])
        # NB: this assumes that board has an outer edge of markers, i.e.,
        # that it does not have targets at its edges. Also assumes markers
        # are rotated by multiples of 90 degrees
        bbox = []
        bbox.append(markerPos.x.min()-markerHalfSizeMm)
        bbox.append(markerPos.y.max()+markerHalfSizeMm) # top is at positive
        bbox.append(markerPos.x.max()+markerHalfSizeMm)
        bbox.append(markerPos.y.min()-markerHalfSizeMm) # bottom is at negative

        return markers, bbox

    def getTargets(self):
        # Aruco markers have numeric keys, gaze targets have keys starting with 't'
        targets = {}
        for key in self.knownMarkers:
            if key.startswith('t'):
                targets[int(key[1:])] = self.knownMarkers[key]
        return targets
    
    def getArucoBoard(self, unRotateMarkers=False):
        boardCornerPoints = []
        ids = []
        for key in self.knownMarkers:
            if not key.startswith('t'):
                ids.append(int(key))
                cornerPoints = np.vstack(self.knownMarkers[key].corners).astype('float32')
                if unRotateMarkers:
                    cornerPoints = getMarkerUnrotated(cornerPoints,self.knownMarkers[key].rot)

                boardCornerPoints.append(cornerPoints)

        boardCornerPoints = np.dstack(boardCornerPoints)        # list of 2D arrays -> 3D array
        boardCornerPoints = np.rollaxis(boardCornerPoints,-1)   # 4x2xN -> Nx4x2
        boardCornerPoints = np.pad(boardCornerPoints,((0,0),(0,0),(0,1)),'constant', constant_values=(0.,0.)) # Nx4x2 -> Nx4x3
        return cv2.aruco.Board_create(boardCornerPoints, self.aruco_dict, np.array(ids))
            
    def _storeReferenceBoard(self, configDir, validationSetup):
        referenceBoard = self.getArucoBoard(unRotateMarkers = True)
        # get image with markers
        bboxExtents    = [self.bbox[2]-self.bbox[0], math.fabs(self.bbox[3]-self.bbox[1])]  # math.fabs to deal with bboxes where (-,-) is bottom left
        aspectRatio    = bboxExtents[0]/bboxExtents[1]
        refBoardWidth  = validationSetup['referenceBoardWidth']
        refBoardHeight = math.ceil(refBoardWidth/aspectRatio)
        margin         = 1  # always 1 pixel, anything else behaves strangely (markers are drawn over margin as well)
        refBoardImage  = cv2.cvtColor(
            cv2.aruco.drawPlanarBoard(
                referenceBoard,(refBoardWidth+2*margin,refBoardHeight+2*margin),margin,validationSetup['markerBorderBits']),
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
            if not key.startswith('t'):
                cornerPoints = np.vstack(self.knownMarkers[key].corners).astype('float32')
                cornerPointsU.append(getMarkerUnrotated(cornerPoints, self.knownMarkers[key].rot))
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
                    marker = refBoardImage[idxs[0]:idxs[1], idxs[2]:idxs[3]]
                    # rotate (opposite because coordinate system) and put back
                    if r==-90:
                        marker = cv2.rotate(marker, cv2.ROTATE_90_CLOCKWISE)
                    elif r==90:
                        marker = cv2.rotate(marker, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif r==180:
                        marker = cv2.rotate(marker, cv2.ROTATE_180)

                    refBoardImage[idxs[0]:idxs[1], idxs[2]:idxs[3]] = marker

        # add targets
        subPixelFac = 8   # for sub-pixel positioning
        for key in self.knownMarkers:
            if key.startswith('t'):
                # 1. determine position on image
                circlePos = toImagePos(*self.knownMarkers[key].center, self.bbox,[refBoardWidth,refBoardHeight])

                # 2. draw
                clr = tuple([int(i*255) for i in colors.to_rgb(self.knownMarkers[key].color)[::-1]])  # need BGR color ordering
                drawOpenCVCircle(refBoardImage, circlePos, 15, clr, -1, subPixelFac)

        cv2.imwrite(str(configDir / 'referenceBoard.png'), refBoardImage)

class GazeWorld:
    def __init__(self, ts, gaze3DRay=None, gaze3DHomography=None, lGazeOrigin=None, lGaze3D=None, rGazeOrigin=None, rGaze3D=None, gaze2DRay=None, gaze2DHomography=None, lGaze2D=None, rGaze2D=None):
        # 3D gaze (and plane info) is in world space, w.r.t. scene camera
        # 2D gaze is on the reference board
        self.ts = ts

        # in camera space
        self.gaze3DRay        = gaze3DRay           # 3D gaze point on plane (3D gaze point <-> camera ray intersected with plane)
        self.gaze3DHomography = gaze3DHomography    # gaze2DHomography in camera space
        self.lGazeOrigin      = lGazeOrigin
        self.lGaze3D          = lGaze3D             # 3D gaze point on plane ( left eye gaze vector intersected with plane)
        self.rGazeOrigin      = rGazeOrigin
        self.rGaze3D          = rGaze3D             # 3D gaze point on plane (right eye gaze vector intersected with plane)

        # in board space
        self.gaze2DRay        = gaze2DRay           # gaze3DRay in board space
        self.gaze2DHomography = gaze2DHomography    # Video gaze point directly mapped to board through homography transformation
        self.lGaze2D          = lGaze2D             # lGaze3D in board space
        self.rGaze2D          = rGaze2D             # rGaze3D in board space

    @staticmethod
    def getWriteHeader():
        header = ['gaze_timestamp']
        header.extend(getXYZLabels(['gazePosCam_vidPos_ray','gazePosCam_vidPos_homography']))
        header.extend(getXYZLabels(['gazeOriCamLeft','gazePosCamLeft']))
        header.extend(getXYZLabels(['gazeOriCamRight','gazePosCamRight']))
        header.extend(getXYZLabels('gazePosBoard2D_vidPos_ray',2))
        header.extend(getXYZLabels('gazePosBoard2D_vidPos_homography',2))
        header.extend(getXYZLabels('gazePosBoard2DLeft',2))
        header.extend(getXYZLabels('gazePosBoard2DRight',2))
        return header

    @staticmethod
    def getMissingWriteData():
        return [math.nan for x in range(24)]

    def getWriteData(self):
        writeData = [self.ts]
        # in camera space
        writeData.extend(allNanIfNone(self.gaze3DRay,3))
        writeData.extend(allNanIfNone(self.gaze3DHomography,3))
        writeData.extend(allNanIfNone(self.lGazeOrigin,3))
        writeData.extend(allNanIfNone(self.lGaze3D,3))
        writeData.extend(allNanIfNone(self.rGazeOrigin,3))
        writeData.extend(allNanIfNone(self.rGaze3D,3))
        # in board space
        writeData.extend(allNanIfNone(self.gaze2DRay,2))
        writeData.extend(allNanIfNone(self.gaze2DHomography,2))
        writeData.extend(allNanIfNone(self.lGaze2D,2))
        writeData.extend(allNanIfNone(self.rGaze2D,2))

        return writeData
    
    @staticmethod
    def readDataFromFile(fileName,start=None,end=None,stopOnceExceeded=False):
        gazes = {}
        readSubset = start is not None and end is not None
        with open(fileName, 'r' ) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for entry in reader:
                frame_idx = int(float(entry['frame_idx']))
                if readSubset and (frame_idx<start or frame_idx>end):
                    if stopOnceExceeded and frame_idx>end:
                        break
                    else:
                        continue
            
                ts = float(entry['gaze_timestamp'])
                gaze3DRay       = dataReaderHelper(entry,'gazePosCam_vidPos_ray')
                gaze3DHomography= dataReaderHelper(entry,'gazePosCam_vidPos_homography')
                lGazeOrigin     = dataReaderHelper(entry,'gazeOriCamLeft')
                lGaze3D         = dataReaderHelper(entry,'gazePosCamLeft')
                rGazeOrigin     = dataReaderHelper(entry,'gazeOriCamRight')
                rGaze3D         = dataReaderHelper(entry,'gazePosCamRight')
                gaze2DRay       = dataReaderHelper(entry,'gazePosBoard2D_vidPos_ray',2)
                gaze2DHomography= dataReaderHelper(entry,'gazePosBoard2D_vidPos_homography',2)
                lGaze2D         = dataReaderHelper(entry,'gazePosBoard2DLeft',2)
                rGaze2D         = dataReaderHelper(entry,'gazePosBoard2DRight',2)
                gaze = GazeWorld(ts, gaze3DRay, gaze3DHomography, lGazeOrigin, lGaze3D, rGazeOrigin, rGaze3D, gaze2DRay, gaze2DHomography, lGaze2D, rGaze2D)

                if frame_idx in gazes:
                    gazes[frame_idx].append(gaze)
                else:
                    gazes[frame_idx] = [gaze]

        return gazes

    def drawOnWorldVideo(self, img, cameraMatrix, distCoeff, subPixelFac=1):
        # project to camera, display
        # left eye
        if self.lGaze3D is not None:
            pPointCam = cv2.projectPoints(self.lGaze3D.reshape(1,3),np.zeros((1,3)),np.zeros((1,3)),cameraMatrix,distCoeff)[0][0][0]
            drawOpenCVCircle(img, pPointCam, 3, (0,0,255), -1, subPixelFac)
        # right eye
        if self.rGaze3D is not None:
            pPointCam = cv2.projectPoints(self.rGaze3D.reshape(1,3),np.zeros((1,3)),np.zeros((1,3)),cameraMatrix,distCoeff)[0][0][0]
            drawOpenCVCircle(img, pPointCam, 3, (255,0,0), -1, subPixelFac)
        # average
        if (self.lGaze3D is not None) and (self.rGaze3D is not None):
            pointCam  = np.array([(x+y)/2 for x,y in zip(self.lGaze3D,self.rGaze3D)]).reshape(1,3)
            pPointCam = cv2.projectPoints(pointCam,np.zeros((1,3)),np.zeros((1,3)),cameraMatrix,distCoeff)[0][0][0]
            if not math.isnan(pPointCam[0]):
                drawOpenCVCircle(img, pPointCam, 6, (255,0,255), -1, subPixelFac)

    def drawOnReferencePlane(self, img, reference, subPixelFac=1):
        # left eye
        if self.lGaze2D is not None:
            reference.draw(img, self.lGaze2D[0],self.lGaze2D[1], subPixelFac, (0,0,255), 3)
        # right eye
        if self.rGaze2D is not None:
            reference.draw(img, self.rGaze2D[0],self.rGaze2D[1], subPixelFac, (255,0,0), 3)
        # average
        if (self.lGaze2D is not None) and (self.rGaze2D is not None):
            average = np.array([(x+y)/2 for x,y in zip(self.lGaze2D,self.rGaze2D)])
            if not math.isnan(average[0]):
                reference.draw(img, average[0], average[1], subPixelFac, (255,0,255))
        # video gaze position
        if self.gaze2DHomography is not None:
            reference.draw(img, self.gaze2DHomography[0],self.gaze2DHomography[1], subPixelFac, (0,255,0), 5)
        if self.gaze2DRay is not None:
            reference.draw(img, self.gaze2DRay[0],self.gaze2DRay[1], subPixelFac, (0,0,0), 3)

class BoardPose:
    def __init__(self, frameIdx, nMarkers=0, rVec=None, tVec=None, hMat=None):
        self.frameIdx   = frameIdx
        
        # pose
        self.nMarkers   = nMarkers  # number of Aruco markers this pose estimate is based on
        self.rVec       = rVec
        self.tVec       = tVec
        self.determinePlanePointNormal()

        # homography
        self.hMat       = hMat.reshape(3,3) if hMat is not None else hMat

    @staticmethod
    def getWriteHeader():
        header = ['frame_idx','poseNMarker']
        header.extend(getXYZLabels(['poseRvec','poseTvec']))
        header.extend(['homography[%d,%d]' % (r,c) for r in range(3) for c in range(3)])
        return header

    @staticmethod
    def getMissingWriteData():
        dat = [0]
        return dat.extend([math.nan for x in range(15)])

    def getWriteData(self):
        writeData = [self.frameIdx, self.nMarkers]
        # in camera space
        writeData.extend(allNanIfNone(self.rVec.flatten(),3))
        writeData.extend(allNanIfNone(self.tVec.flatten(),3))
        writeData.extend(allNanIfNone(self.hMat.flatten(),9))

        return writeData
    
    @staticmethod
    def readDataFromFile(fileName,start=None,end=None,stopOnceExceeded=False):
        poses       = {}
        readSubset  = start is not None and end is not None
        data        = pd.read_csv(str(fileName), delimiter='\t',index_col=False)
        rCols       = [col for col in data.columns if 'poseRvec' in col]
        tCols       = [col for col in data.columns if 'poseTvec' in col]
        hCols       = [col for col in data.columns if 'homography' in col]
        # run through all columns
        for idx, row in data.iterrows():
            frame_idx = int(row['frame_idx'])
            if readSubset and (frame_idx<start or frame_idx>end):
                if stopOnceExceeded and frame_idx>end:
                    break
                else:
                    continue

            # get all values (None if all nan)
            args = tuple(noneIfAnyNan(row[c].to_numpy()) for c in (rCols,tCols,hCols))
            
            # insert if any non-None
            if not np.all([x is None for x in args]):   # check for not all isNone
                poses[frame_idx] = BoardPose(frame_idx,int(row['poseNMarker']),*args)

        return poses

    def setPose(self,rVec,tVec):
        self.rVec = rVec
        self.tVec = tVec
        self.determinePlanePointNormal()

    def determinePlanePointNormal(self):
        if (self.rVec is not None) and (self.tVec is not None):
            # get board normal
            RBoard           = cv2.Rodrigues(self.rVec)[0]
            self.planeNormal = np.matmul(RBoard, np.array([0,0,1.]))
            # get point on board (just use origin)
            RtBoard          = np.hstack((RBoard, self.tVec.reshape(3,1)))
            self.planePoint  = np.matmul(RtBoard, np.array([0, 0, 0., 1.]))
        else:
            self.planeNormal = None
            self.planePoint  = None

class Idx2Timestamp:
    def __init__(self, fileName):
        self.timestamps = {}
        with open(fileName, 'r' ) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for entry in reader:
                frame_idx = int(float(entry['frame_idx']))
                if frame_idx!=-1:
                    self.timestamps[frame_idx] = float(entry['timestamp'])

    def get(self, idx):
        if idx in self.timestamps:
            return self.timestamps[int(idx)]
        else:
            warnings.warn('frame_idx %d is not in set\n' % ( idx ), RuntimeWarning )
            return -1.

class Timestamp2Index:
    def __init__(self, fileName):
        self.indexes = []
        self.timestamps = []
        with open(fileName, 'r' ) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for entry in reader:
                self.indexes   .append(int(float(entry['frame_idx'])))
                self.timestamps.append(    float(entry['timestamp']))

    def find(self, ts):
        idx = min(bisect.bisect(self.timestamps, ts), len(self.indexes)-1)
        return self.indexes[idx]

def readCameraCalibrationFile(fileName):
    fs = cv2.FileStorage(str(fileName), cv2.FILE_STORAGE_READ)
    cameraMatrix    = fs.getNode("cameraMatrix").mat()
    distCoeff       = fs.getNode("distCoeff").mat()
    # camera extrinsics for 3D gaze
    cameraRotation  = fs.getNode("rotation").mat()
    if cameraRotation is not None:
        cameraRotation  = cv2.Rodrigues(cameraRotation)[0]  # need rotation vector, not rotation matrix
    cameraPosition  = fs.getNode("position").mat()
    fs.release()

    return (cameraMatrix,distCoeff,cameraRotation,cameraPosition)

def readMarkerIntervalsFile(fileName):
    analyzeFrames = []
    if fileName.is_file():
        with open(fileName, 'r' ) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for entry in reader:
                analyzeFrames.append(int(float(entry['start_frame'])))
                analyzeFrames.append(int(float(entry['end_frame'])))

    return None if len(analyzeFrames)==0 else analyzeFrames

def gazeToPlane(gaze,boardPose,cameraRotation,cameraPosition, cameraMatrix=None, distCoeffs=None):

    hasCameraPose = (boardPose.rVec is not None) and (boardPose.tVec is not None)
    gazeWorld   = GazeWorld(gaze.ts)
    if hasCameraPose:
        # set up camera to world and back matrices
        RBoard      = cv2.Rodrigues(boardPose.rVec)[0]
        RtBoard     = np.hstack((RBoard  ,                    boardPose.tVec.reshape(3,1)))
        RtBoardInv  = np.hstack((RBoard.T,np.matmul(-RBoard.T,boardPose.tVec.reshape(3,1))))

        # get transform from ET data's coordinate frame to camera's coordinate frame
        if cameraRotation is None:
            cameraRotation = np.zeros((3,1))
        RCam  = cv2.Rodrigues(cameraRotation)[0]
        if cameraPosition is None:
            cameraPosition = np.zeros((3,1))
        RtCam = np.hstack((RCam, cameraPosition))

        # project 3D gaze to reference board
        if gaze.world3D is not None:
            # turn 3D gaze point into ray from camera
            g3D = np.matmul(RCam,np.array(gaze.world3D).reshape(3,1))
            g3D /= np.sqrt((g3D**2).sum()) # normalize
            # find intersection of 3D gaze with board, draw
            g3Board  = intersect_plane_ray(boardPose.planeNormal, boardPose.planePoint, g3D.flatten(), np.array([0.,0.,0.]))  # vec origin (0,0,0) because we use g3D from camera's view point to be able to recreate Tobii 2D gaze pos data
            (x,y,z)  = np.matmul(RtBoardInv,np.append(g3Board,1.).reshape((4,1))).flatten() # z should be very close to zero
            gazeWorld.gaze3DRay = g3Board
            gazeWorld.gaze2DRay = [x, y]

    # unproject 2D gaze point on video to point on board (should yield values very close to
    # the above method of intersecting 3D gaze point ray with board)
    if boardPose.hMat is not None:
        ux, uy   = gaze.vid2D
        if (cameraMatrix is not None) and (distCoeffs is not None):
            ux, uy   = undistortPoint( ux, uy, cameraMatrix, distCoeffs)
        (xW, yW) = applyHomography(boardPose.hMat, ux, uy)
        gazeWorld.gaze2DHomography = [xW, yW]

        # get this point in board space
        if hasCameraPose:
            gazeWorld.gaze3DHomography = np.matmul(RtBoard,np.array([xW,yW,0.,1.]).reshape((4,1))).flatten()

    # project gaze vectors to reference board (and draw on video)
    if not hasCameraPose:
        # nothing to do anymore
        return gazeWorld

    gazeVecs    = [gaze.lGazeVec   , gaze.rGazeVec]
    gazeOrigins = [gaze.lGazeOrigin, gaze.rGazeOrigin]
    clrs        = [(0,0,255)       , (255,0,0)]
    eyes        = ['left'          , 'right']
    attrs       = [['lGazeOrigin','lGaze3D','lGaze2D'],['rGazeOrigin','rGaze3D','rGaze2D']]
    for gVec,gOri,clr,eye,attr in zip(gazeVecs,gazeOrigins,clrs,eyes,attrs):
        if gVec is None or gOri is None:
            continue
        # get gaze vector and point on vector (pupil center) ->
        # transform from ET data coordinate frame into camera coordinate frame
        gVec    = np.matmul(RCam ,          gVec    )
        gOri    = np.matmul(RtCam,np.append(gOri,1.))
        setattr(gazeWorld,attr[0],gOri)

        # intersect with board -> yield point on board in camera reference frame
        gBoard  = intersect_plane_ray(boardPose.planeNormal, boardPose.planePoint, gVec, gOri)
        setattr(gazeWorld,attr[1],gBoard)
                        
        # transform intersection with board from camera space to board space
        if not math.isnan(gBoard[0]):
            (x,y,z) = np.matmul(RtBoardInv,np.append(gBoard,1.).reshape((4,1))).flatten() # z should be very close to zero
            pgBoard = [x,y]
        else:
            pgBoard = [np.nan,np.nan]
        setattr(gazeWorld,attr[2],pgBoard)

    return gazeWorld

def selectDictRange(theDict,start,end):
    return {k: theDict[k] for k in theDict if k>=start and k<=end}