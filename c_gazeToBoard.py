#!/usr/bin/python3

import sys
from pathlib import Path
import math
import bisect

import cv2
import numpy as np
import pandas as pd
import csv
import time

import utils

gShowVisualization  = True      # if true, draw each frame and overlay info about detected markers and board
gShowReference      = True
gFPSFac             = 1


def process(inputDir,basePath):
    global gShowVisualization
    global gShowReference
    global gFPSFac

    print('processing: {}'.format(inputDir.name))
    
    configDir = basePath / "config"
    # open file with information about Aruco marker and Gaze target locations
    validationSetup = utils.getValidationSetup(configDir)

    if gShowVisualization:
        cv2.namedWindow("frame")
        if gShowReference:
            cv2.namedWindow("reference")

        reference = utils.Reference(str(inputDir / 'referenceBoard.png'), configDir, validationSetup)
        i2t = utils.Idx2Timestamp(str(inputDir / 'frameTimestamps.tsv'))
    
    # get camera calibration info
    cameraMatrix,distCoeff,cameraRotation,cameraPosition = utils.getCameraCalibrationInfo(inputDir / "calibration.xml")

    # open video, if wanted
    if gShowVisualization:
        cap         = cv2.VideoCapture(str(inputDir / 'worldCamera.mp4'))
        width       = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height      = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        ifi         = 1000./cap.get(cv2.CAP_PROP_FPS)/gFPSFac

    # Read gaze data
    gazes,maxFrameIdx = utils.getGazeData(inputDir / 'gazeData.tsv')

    # Read pose of marker board
    rVec,tVec = utils.getMarkerBoardPose(inputDir / 'boardPose.tsv')

    csv_file = open(str(inputDir / 'gazeWorldPos.tsv'), 'w', newline='')
    csv_writer = csv.writer(csv_file, delimiter='\t')
    header = ['frame_idx', 'frame_timestamp', 'gaze_timestamp']
    header.extend(utils.getXYZLabels(['planePoint','planeNormal']))
    header.extend(utils.getXYZLabels('gazeCam3D_vidPos'))
    header.extend(utils.getXYZLabels('gazeBoard2D_vidPos',2))
    header.extend(utils.getXYZLabels(['gazeOriLeft','gazeCam3DLeft']))
    header.extend(utils.getXYZLabels('gazeBoard2DLeft',2))
    header.extend(utils.getXYZLabels(['gazeOriRight','gazeCam3DRight']))
    header.extend(utils.getXYZLabels('gazeBoard2DRight',2))
    csv_writer.writerow(header)
    
    subPixelFac = 8   # for sub-pixel positioning
    stopAllProcessing = False
    for frame_idx in range(maxFrameIdx+1):
        startTime = time.perf_counter()
        if gShowVisualization:
            ret, frame = cap.read()
            if not ret: # we reached the end; done
                break

            refImg = reference.getImgCopy()
            

        if frame_idx in gazes:
            for gaze in gazes[frame_idx]:

                # draw 2D gaze point
                if gShowVisualization:
                    gaze.draw(frame, subPixelFac)

                # draw 3D gaze point as well, should coincide with 2D gaze point
                a = cv2.projectPoints(np.array(gaze.world3D).reshape(1,3),cameraRotation,cameraPosition,cameraMatrix,distCoeff)[0][0][0]
                if gShowVisualization:
                    utils.drawOpenCVCircle(frame, a, 6, (0,0,0), -1, subPixelFac)

                
                # if we have pose information, figure out where gaze vectors
                # intersect with reference board. Do same for 3D gaze point
                # (the projection of which coincides with 2D gaze provided by
                # the eye tracker)
                # store positions on marker board plane in camera coordinate frame to
                # file, along with gaze vector origins in same coordinate frame
                writeData = [frame_idx]
                if frame_idx in rVec:
                    gazeWorld = utils.gazeToPlane(gaze,rVec[frame_idx],tVec[frame_idx],cameraRotation,cameraPosition)
                    
                    # draw gazes on video and reference image
                    if gShowVisualization:
                        gazeWorld.drawOnWorldVideo(frame, cameraMatrix, distCoeff, subPixelFac)
                        if gShowReference:
                            gazeWorld.drawOnReferencePlane(refImg, reference, subPixelFac)

                    # store gaze-on-plane to csv
                    writeData.extend(gazeWorld.getWriteData())
                else:
                    writeData.extend(utils.GazeWorld.getMissingWriteData())
                csv_writer.writerow( writeData )

        if gShowVisualization:
            if gShowReference:
                cv2.imshow("reference", refImg)

            # if we have board pose, draw board origin on video
            if frame_idx in rVec:
                a = cv2.projectPoints(np.zeros((1,3)),rVec[frame_idx],tVec[frame_idx],cameraMatrix,distCoeff)[0][0][0]
                utils.drawOpenCVCircle(frame, a, 3, (0,255,0), -1, subPixelFac)
                utils.drawOpenCVLine(frame, (a[0],0), (a[0],height), (0,255,0), 1, subPixelFac)
                utils.drawOpenCVLine(frame, (0,a[1]), (width,a[1]) , (0,255,0), 1, subPixelFac)
                
            frame_ts  = i2t.get(frame_idx)
            cv2.rectangle(frame,(0,int(height)),(int(0.25*width),int(height)-30),(0,0,0),-1)
            cv2.putText(frame, '%8.2f [%6d]' % (frame_ts,frame_idx), (0, int(height)-5), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255))
            cv2.imshow('frame',frame)
            key = cv2.waitKey(max(1,int(round(ifi-(time.perf_counter()-startTime)*1000)))) & 0xFF
            if key == ord('q'):
                # quit fully
                stopAllProcessing = True
                break
            if key == ord('n'):
                # goto next
                break
            if key == ord('s'):
                # screenshot
                cv2.imwrite(str(inputDir / ('calc_frame_%d.png' % frame_idx)), frame)
        elif (frame_idx+1)%100==0:
            print('  frame {}'.format(frame_idx+1))

    csv_file.close()
    if gShowVisualization:
        cap.release()
        cv2.destroyAllWindows()

    return stopAllProcessing



if __name__ == '__main__':
    basePath = Path(__file__).resolve().parent
    for d in (basePath / 'data' / 'preprocced').iterdir():
        if d.is_dir():
            if process(d,basePath):
                break
