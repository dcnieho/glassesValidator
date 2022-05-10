#!/usr/bin/python3
# NB: this is a combination of the b_ and c_ steps, not actively maintained.
# Not guaranteed to provide the same output as b_ and c_ steps, or to work at all

import sys
from pathlib import Path
import math
import csv

import cv2
import numpy as np

import utils

gShowVisualization  = False     # if true, draw each frame and overlay info about detected markers and board


def process(inputDir,basePath):
    global gShowVisualization

    print('processing: {}'.format(inputDir.name))
    
    configDir = basePath / "config"
    # open file with information about Aruco marker and Gaze target locations
    validationSetup = utils.getValidationSetup(configDir)

    if gShowVisualization:
        cv2.namedWindow("frame")
        cv2.namedWindow("reference")
    
    # open input video file, query it for size
    inVideo = inputDir / 'worldCamera.mp4'
    if not inVideo.is_file():
        inVideo = inputDir / 'worldCamera.avi'
    vidIn  = cv2.VideoCapture( str(inVideo) )
    if not vidIn.isOpened():
        raise RuntimeError('the file "{}" could not be opened'.format(str(inVideo)))
    width  = vidIn.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vidIn.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps    = vidIn.get(cv2.CAP_PROP_FPS)

    # open output scene video file
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    vidOutScene = cv2.VideoWriter(str(inputDir / 'detectOutput_scene.mp4'), fourcc, fps, (int(width), int(height)))

    # open output reference board video file
    reference = utils.Reference(str(inputDir / 'referenceBoard.png'), configDir, validationSetup)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    vidOutBoard = cv2.VideoWriter(str(inputDir / 'detectOutput_board.mp4'), fourcc, fps, (reference.width, reference.height))
    
    # get info about markers on our board
    # Aruco markers have numeric keys, gaze targets have keys starting with 't'
    aruco_dict   = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
    knownMarkers, markerBBox = utils.getKnownMarkers(configDir, validationSetup)
    centerTarget = knownMarkers['t%d'%validationSetup['centerTarget']].center
    
    # turn into aruco board object to be used for pose estimation
    referenceBoard = utils.getReferenceBoard(knownMarkers, aruco_dict)

    # setup aruco marker detection
    parameters = cv2.aruco.DetectorParameters_create()
    parameters.markerBorderBits       = validationSetup['markerBorderBits']
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX;

    # get frame timestamps lookup file
    i2t = utils.Idx2Timestamp(str(inputDir / 'frameTimestamps.tsv'))

    # get camera calibration info
    cameraMatrix,distCoeff,cameraRotation,cameraPosition = utils.readCameraCalibrationFile(inputDir / "calibration.xml")

    # Read gaze data
    gazes,maxFrameIdx = utils.Gaze.readDataFromFile(inputDir / 'gazeData.tsv')
    
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
                if (cameraMatrix is not None) and (distCoeff is not None):
                    nMarkersUsed, rVec, tVec = cv2.aruco.estimatePoseBoard(corners, ids, referenceBoard, cameraMatrix, distCoeff)
                else:
                    nMarkersUsed = 0
                       
                # draw axis indicating board pose (origin and orientation)
                if nMarkersUsed>0:
                    utils.drawOpenCVFrameAxis(frame, cameraMatrix, distCoeff, rVec, tVec, armLength, 3, subPixelFac)
                    gotPose = True

                # also get homography (direct image plane to plane in world transform). Use undistorted marker corners
                if (cameraMatrix is not None) and (distCoeff is not None):
                    cornersU = [cv2.undistortPoints(x, cameraMatrix, distCoeff, P=cameraMatrix) for x in corners]
                else:
                    cornersU = corners
                H, status = utils.estimateHomography(knownMarkers, cornersU, ids)

                if status:
                    # find where target is expected to be in the image
                    iH = np.linalg.inv(H)
                    target = utils.applyHomography(iH, centerTarget[0], centerTarget[1])
                    if (cameraMatrix is not None) and (distCoeff is not None):
                        target = utils.distortPoint(*target, cameraMatrix, distCoeff)
                    # draw target location on image
                    if target[0] >= 0 and target[0] < width and target[1] >= 0 and target[1] < height:
                        utils.drawOpenCVCircle(frame, target, 3, (0,0,0), -1, subPixelFac)

            # if any markers were detected, draw where on the frame
            utils.drawArucoDetectedMarkers(frame, corners, ids, subPixelFac=subPixelFac)
        
        # process gaze
        if frame_idx in gazes:
            for gaze in gazes[frame_idx]:

                # draw 2D gaze point
                gaze.draw(frame, subPixelFac)

                # draw 3D gaze point as well, should coincide with 2D gaze point
                if gaze.world3D is not None:
                    a = cv2.projectPoints(np.array(gaze.world3D).reshape(1,3),cameraRotation,cameraPosition,cameraMatrix,distCoeff)[0][0][0]
                    utils.drawOpenCVCircle(frame, a, 6, (0,0,0), -1, subPixelFac)
                
                # if we have pose information, figure out where gaze vectors
                # intersect with reference board. Do same for 3D gaze point
                # (the projection of which coincides with 2D gaze provided by
                # the eye tracker)
                if gotPose:
                    gazeWorld = utils.gazeToPlane(gaze,rVec,tVec,cameraRotation,cameraPosition)

                    # draw gazes on video and reference image
                    gazeWorld.drawOnWorldVideo(frame, cameraMatrix, distCoeff, subPixelFac)
                    gazeWorld.drawOnReferencePlane(refImg, reference, subPixelFac)
        
        # annotate frame
        frame_ts  = i2t.get(frame_idx)
        cv2.rectangle(frame,(0,int(height)),(int(0.25*width),int(height)-30),(0,0,0),-1)
        cv2.putText(frame, '%8.2f [%6d]' % (frame_ts,frame_idx), (0, int(height)-5), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255))

        # store to file
        vidOutScene.write(frame)
        vidOutBoard.write(refImg)


        if gShowVisualization:
            cv2.imshow('frame',frame)
            cv2.imshow('reference',refImg)
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
