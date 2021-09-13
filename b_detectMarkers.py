#!/usr/bin/python

import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
import csv
import math
import pandas as pd


class Marker:
    def __init__(self, key, center, corners=None):
        self.key = key
        self.center = center
        self.corners = corners

    def __str__(self):
        ret = '[%s]: center @ (%.2f, %.2f)' % (self.key, self.center[0], self.center[1])
        return ret

def getKnownMarkers(markerDir, validationSetup):
    """ (0,0) is at center target, (-,-) bottom left """
    cellSizeCm = 2.*math.tan(math.radians(.5))*validationSetup['distance']
    markerHalfSizeCm = cellSizeCm*validationSetup['markerSide']/2.
            
    # read in target positions
    markers = {}
    targets = pd.read_csv(str(markerDir / validationSetup['targetPosFile']),names=['id','x','y','clr'])
    targets = targets[['id','x','y']].values.astype('float32')
    center  = targets[np.where(targets[:,0]==validationSetup['centerTarget']),1:3].flatten()
    targets[:,1:3] = targets[:,1:3]-center
    for i in range(targets.shape[0]):
        key = 't%d' % targets[i,0]
        c   = cellSizeCm * targets[i,1:3]
        markers[key] = Marker(key, c)
    
    # read in aruco marker positions
    markerPos = pd.read_csv(str(markerDir / validationSetup['markerPosFile']),names=['id','x','y'])
    markerPos = markerPos.values.astype('float32')
    markerPos[:,1:3] = markerPos[:,1:3]-center
    for i in range(markerPos.shape[0]):
        key = '%d' % markerPos[i,0]
        c   = cellSizeCm * markerPos[i,1:3]
        # top left first, and clockwise: same order as detected aruco marker corners
        tl = c + np.array( [ -markerHalfSizeCm ,  markerHalfSizeCm ] )
        tr = c + np.array( [  markerHalfSizeCm ,  markerHalfSizeCm ] )
        br = c + np.array( [  markerHalfSizeCm , -markerHalfSizeCm ] )
        bl = c + np.array( [ -markerHalfSizeCm , -markerHalfSizeCm ] )
        markers[key] = Marker(key, c, [ tl, tr, br, bl ])
        
    return markers


def estimateTransform(known, detectedCorners, detectedIDs):
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


def process(inputDir,basePath):
    markerDir = basePath / "markerLayout"
    # open file with information about Aruco marker and Gaze target locations
    validationSetup = {}
    with open(str(markerDir / "validationSetup.txt")) as setupFile:
        for line in setupFile:
            name, var = line.partition("=")[::2]
            try:
                validationSetup[name.strip()] = float(var)
            except ValueError:
                validationSetup[name.strip()] = var.strip()
    
    # open video file, query it for size
    inVideo = str(inputDir / 'worldCamera.mp4')
    cap    = cv2.VideoCapture( inVideo )
    if not cap.isOpened():
        raise RuntimeError('the file "{}" could not be opened'.format(inVideo))
    width  = float( cap.get(cv2.CAP_PROP_FRAME_WIDTH ) )
    height = float( cap.get(cv2.CAP_PROP_FRAME_HEIGHT ) )
    
    # get info about markers on our board
    # Aruco markers have numeric keys, gaze targets have keys starting with 't'
    aruco_dict   = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
    knownMarkers = getKnownMarkers(markerDir, validationSetup)
    centerTarget = knownMarkers['t%d'%validationSetup['centerTarget']].center
    
    # turn into aruco board object to be used for pose estimation
    boardCornerPoints = []
    ids = []
    for key in knownMarkers:
        if not key.startswith('t'):
            ids.append(int(key))
            boardCornerPoints.append(np.vstack(knownMarkers[key].corners).astype('float32'))
    boardCornerPoints = np.dstack(boardCornerPoints)        # list of 2D arrays -> 3D array
    boardCornerPoints = np.rollaxis(boardCornerPoints,-1)   # 4x2xN -> Nx4x2
    boardCornerPoints = np.pad(boardCornerPoints,((0,0),(0,0),(0,1)),'constant', constant_values=(0.,0.)) # Nx4x2 -> Nx4x3
    referenceBoard    = cv2.aruco.Board_create(boardCornerPoints, aruco_dict, np.array(ids))
    
    # setup aruco marker detection
    parameters = cv2.aruco.DetectorParameters_create()
    parameters.markerBorderBits       = 1
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX;

    # get camera calibration info
    fs = cv2.FileStorage(str(inputDir / "calibration.xml"), cv2.FILE_STORAGE_READ)
    cameraMatrix = fs.getNode("cameraMatrix").mat()
    distCoeff    = fs.getNode("distCoeff").mat()
    fs.release()

    # prep output file
    csv_file = open(str(inputDir / 'transformations.tsv'), 'w', newline='')
    csv_writer = csv.writer(csv_file, delimiter='\t')
    header = ['frame_idx']
    header.extend(['transformation[%d,%d]' % (r,c) for r in range(3) for c in range(3)])
    header.append('poseNMarker')
    header.extend(['poseRvec[%d]' % (v) for v in range(3)])
    header.extend(['poseTvec[%d]' % (v) for v in range(3)])
    header.extend(['centerTarget[%d]' % (v) for v in range(2)])
    csv_writer.writerow( header )

    frame_idx = 0
    stopAllProcessing = False
    while True:
        # process frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # detect markers, undistort
        corners , ids, rejectedImgPoints = \
            cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
        
        if np.all(ids != None):
            if len(ids) >= validationSetup['minNumMarkers']:
                # undistort markers, get homography (image to world transform)
                cornersU = [cv2.undistortPoints(x, cameraMatrix, distCoeff, P=cameraMatrix) for x in corners]
                H, status = estimateTransform(knownMarkers, cornersU, ids)
                if status:
                    # get camera pose
                    nMarkersUsed, Rvec, Tvec = cv2.aruco.estimatePoseBoard(corners, ids, referenceBoard, cameraMatrix, distCoeff)
                    
                    # find where target is expected to be in the image
                    iH = np.linalg.inv(H)
                    target = transform(iH, centerTarget[0], centerTarget[1])
                    target = distortPoint( target, cameraMatrix, distCoeff)
                    
                    # draw target location on image
                    if target[0] >= 0 and target[0] < width and target[1] >= 0 and target[1] < height:
                        x = int(round(target[0]))
                        y = int(round(target[1]))
                        cv2.line(frame, (x,0), (x,int(height)),(0,255,0),1)
                        cv2.line(frame, (0,y), (int(width),y) ,(0,255,0),1)
                        cv2.circle(frame, (x,y), 5, (0,255,0), -1)
                        
                    # store homography, pose and target location to file
                    writeDat = [frame_idx]
                    writeDat.extend( H.flatten() )
                    writeDat.append( nMarkersUsed )
                    writeDat.extend( Rvec.flatten() )
                    writeDat.extend( Tvec.flatten() )
                    writeDat.extend( target )
                    csv_writer.writerow( writeDat )

            # if any markers were detected, draw where on the frame
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # for debug, can draw rejected markers on frame
        # cv2.aruco.drawDetectedMarkers(frame, rejectedImgPoints, None, borderColor=(148,0,211))
                

        cv2.imshow(inputDir.name,frame)
        
        # quit fully
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            stopAllProcessing = True
            break
        # goto next
        if key == ord('n'):
            break
        # screenshot
        if key == ord('s'):
            cv2.imwrite(str(inputDir / ('frame_%d.png' % frame_idx)), frame)
        
        frame_idx += 1

    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()

    return stopAllProcessing


if __name__ == '__main__':
    basePath = Path(__file__).resolve().parent
    for d in (basePath / 'data' / 'preprocced').iterdir():
        if d.is_dir():
            if process(d,basePath):
                break
