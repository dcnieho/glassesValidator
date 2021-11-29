#!/usr/bin/python3
# NB: this is a combination of the b_ and c_ steps, not actively maintained.
# Not guaranteed to provide the same output as b_ and c_ steps, or to work at all

import sys
from pathlib import Path
import math
import bisect
import csv

import cv2
import numpy as np
import pandas as pd
from matplotlib import colors

import utils

gShowVisualization  = False      # if true, draw each frame and overlay info about detected markers and board


def process(inputDir,basePath):
    global gShowVisualization

    print('processing: {}'.format(inputDir.name))
    
    configDir = basePath / "config"
    # open file with information about Aruco marker and Gaze target locations
    validationSetup = utils.getValidationSetup(configDir)
    
    # open input video file, query it for size
    inVideo = str(inputDir / 'worldCamera.mp4')
    vidIn  = cv2.VideoCapture( inVideo )
    if not vidIn.isOpened():
        raise RuntimeError('the file "{}" could not be opened'.format(inVideo))
    width  = vidIn.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vidIn.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps    = vidIn.get(cv2.CAP_PROP_FPS)

    # open output scene video file
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    vidOutScene = cv2.VideoWriter(str(inputDir / 'detectOutput_scene.mp4'), fourcc, fps, (int(width), int(height)))

    # open output reference board video file
    reference = Reference(str(inputDir / 'referenceBoard.png'), configDir, validationSetup)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    vidOutBoard = cv2.VideoWriter(str(inputDir / 'detectOutput_board.mp4'), fourcc, fps, (reference.width, reference.height))
    
    # get info about markers on our board
    # Aruco markers have numeric keys, gaze targets have keys starting with 't'
    aruco_dict   = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
    knownMarkers, markerBBox = utils.getKnownMarkers(configDir, validationSetup)
    
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
    parameters.markerBorderBits       = validationSetup['markerBorderBits']
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX;

    # get frame timestamps lookup file
    i2t = Idx2Timestamp(str(inputDir / 'frameTimestamps.tsv'))

    fs = cv2.FileStorage(str(inputDir / "calibration.xml"), cv2.FILE_STORAGE_READ)
    cameraMatrix    = fs.getNode("cameraMatrix").mat()
    distCoeff       = fs.getNode("distCoeff").mat()
    # camera extrinsics for 3D gaze
    cameraRotation  = cv2.Rodrigues(fs.getNode("rotation").mat())[0]    # need rotation vector, not rotation matrix
    cameraPosition  = fs.getNode("position").mat()
    fs.release()

    # Read gaze data
    gazes,maxFrameIdx = utils.getGazeData(inputDir / 'gazeData.tsv')
    
    frame_idx = 0
    armLength = 2.*math.tan(math.radians(.5))*validationSetup['distance']*10*validationSetup['markerSide']/2 # arms of axis are half a marker long
    subPixelFac = 8   # for sub-pixel positioning
    stopAllProcessing = False
    while True:
        # process frame-by-frame
        ret, frame = vidIn.read()
        if not ret:
            break
        refImg = reference.getImgCopy()

        # detect markers, undistort
        corners, ids, rejectedImgPoints = \
            cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

        # get board pose, draw marker and board pose
        gotPose = False
        if np.all(ids != None):
            if len(ids) >= validationSetup['minNumMarkers']:
                # get camera pose
                nMarkersUsed, rVec, tVec = cv2.aruco.estimatePoseBoard(corners, ids, referenceBoard, cameraMatrix, distCoeff)
                       
                # draw axis indicating board pose (origin and orientation)
                if nMarkersUsed>0:
                    utils.drawOpenCVFrameAxis(frame, cameraMatrix, distCoeff, rVec, tVec, armLength, 3, subPixelFac)
                    gotPose = True

            # if any markers were detected, draw where on the frame
            utils.drawArucoDetectedMarkers(frame, corners, ids, subPixelFac=subPixelFac)
        
        # process gaze
        frame_ts  = i2t.get(frame_idx)
        if frame_idx in gazes:
            for gaze in gazes[frame_idx]:

                # draw 2D gaze point
                gaze.draw(frame, subPixelFac)

                # draw 3D gaze point as well, should coincide with 2D gaze point
                a = cv2.projectPoints(np.array(gaze.world3D).reshape(1,3),cameraRotation,cameraPosition,cameraMatrix,distCoeff)[0][0][0]
                utils.drawOpenCVCircle(frame, a, 6, (0,0,0), -1, subPixelFac)

                
                # if we have pose information, figure out where gaze vectors
                # intersect with reference board. Do same for 3D gaze point
                # (the projection of which coincides with 2D gaze provided by
                # the eye tracker)
                # store positions on marker board plane in camera coordinate frame to
                # file, along with gaze vector origins in same coordinate frame
                offsets = {}
                if gotPose:
                    # get board normal
                    RBoard      = cv2.Rodrigues(rVec)[0]
                    boardNormal = np.matmul(RBoard, np.array([0,0,1.]))
                    # get point on board (just use origin)
                    RtBoard     = np.hstack((RBoard  ,                    tVec.reshape(3,1)))
                    RtBoardInv  = np.hstack((RBoard.T,np.matmul(-RBoard.T,tVec.reshape(3,1))))
                    boardPoint  = np.matmul(RtBoard,np.array([0, 0, 0., 1.]))

                    # get transform from ET data's coordinate frame to camera's coordinate frame
                    RCam        = cv2.Rodrigues(cameraRotation)[0]
                    RtCam       = np.hstack((RCam, cameraPosition))

                    # project 3D gaze to reference board
                    # turn 3D gaze point into ray from camera
                    g3D = np.matmul(RCam,np.array(gaze.world3D).reshape(3,1))
                    g3D /= np.sqrt((g3D**2).sum()) # normalize
                    # find intersection of 3D gaze with board, draw
                    g3Board  = utils.intersect_plane_ray(boardNormal, boardPoint, g3D.flatten(), np.array([0.,0.,0.]))  # vec origin (0,0,0) because we use g3D from camera's view point to be able to recreate Tobii 2D gaze pos data
                    (x,y,z)  = np.matmul(RtBoardInv,np.append(g3Board,1.).reshape((4,1))).flatten() # z should be very close to zero
                    reference.draw(refImg, x, y, subPixelFac)
                    offsets['3D_gaze_point'] = [x,y]

                    # project gaze vectors to reference board (and draw on video)
                    gazeVecs    = [gaze.lGazeVec   , gaze.rGazeVec]
                    gazeOrigins = [gaze.lGazeOrigin, gaze.rGazeOrigin]
                    clrs        = [(0,0,255)       , (255,0,0)]
                    boardPosCam = []
                    for gVec,gOri,clr,eye in zip(gazeVecs,gazeOrigins,clrs,['left','right']):
                        # get gaze vector and point on vector (pupil center) ->
                        # transform from ET data coordinate frame into camera coordinate frame
                        gVec    = np.matmul(RtCam,np.append(gVec,1.))
                        gOri    = np.matmul(RtCam,np.append(gOri,1.))
                        # intersect with board -> yield point on board in camera reference frame
                        gBoard  = utils.intersect_plane_ray(boardNormal, boardPoint, gVec, gOri)
                        boardPosCam.append(gBoard)
                        # project and draw on video
                        pgBoard = cv2.projectPoints(gBoard.reshape(1,3),np.zeros((1,3)),np.zeros((1,3)),cameraMatrix,distCoeff)[0][0][0]
                        utils.drawOpenCVCircle(frame, pgBoard, 6, clr, -1, subPixelFac)

                        # transform intersection with board from camera space to board space, draw on reference board
                        if not math.isnan(gBoard[0]):
                            (x,y,z) = np.matmul(RtBoardInv,np.append(gBoard,1.).reshape((4,1))).flatten() # z should be very close to zero
                            reference.draw(refImg, x, y, subPixelFac, clr)
                            offsets[eye] = [x,y]

                    # make average gaze point
                    # on reference
                    if 'left' in offsets and 'right' in offsets:
                        # on reference
                        offsets['average'] = [(x+y)/2 for x,y in zip(offsets['left'],offsets['right'])]
                        reference.draw(refImg, offsets['average'][0], offsets['average'][1], subPixelFac, (255,0,255), 3)
                    # on video
                    if len(boardPosCam)==2:
                        gBoard = np.array([(x+y)/2 for x,y in zip(*boardPosCam)]).reshape(1,3)
                        pgBoard = cv2.projectPoints(gBoard,np.zeros((1,3)),np.zeros((1,3)),cameraMatrix,distCoeff)[0][0][0]
                        utils.drawOpenCVCircle(frame, pgBoard, 3, (255,0,255), -1, subPixelFac)
        
        # annotate frame
        cv2.rectangle(frame,(0,int(height)),(int(0.25*width),int(height)-30),(0,0,0),-1)
        cv2.putText(frame, '%8.2f [%6d]' % (frame_ts,frame_idx), (0, int(height)-5), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255))

        # store to file
        vidOutScene.write(frame)
        vidOutBoard.write(refImg)


        if gShowVisualization:
            cv2.imshow('frame',frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                # quit fully
                stopAllProcessing = True
                break
            if key == ord('n'):
                # goto next
                break
        elif (frame_idx+1)%100==0:
            print('  frame {}'.format(frame_idx+1))

        frame_idx += 1
        
    vidIn.release()
    vidOutScene.release()
    cv2.destroyAllWindows()

    return stopAllProcessing



if __name__ == '__main__':
    basePath = Path(__file__).resolve().parent
    for d in (basePath / 'data' / 'preprocced').iterdir():
        if d.is_dir():
            if process(d,basePath):
                break
