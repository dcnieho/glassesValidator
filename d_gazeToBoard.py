#!/usr/bin/python3

import sys
from pathlib import Path
import math

import cv2
import numpy as np
import csv
import time

import utils

gShowVisualization  = False     # if true, draw each frame and overlay info about detected markers and board
gShowReference      = True
qShowOnlyIntervals  = True      # if true, shows only frames in the marker intervals (if available)
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

        # get info about markers on our board
        knownMarkers, markerBBox = utils.getKnownMarkers(configDir, validationSetup)
        centerTarget = knownMarkers['t%d'%validationSetup['centerTarget']].center
    
    # get camera calibration info
    cameraMatrix,distCoeff,cameraRotation,cameraPosition = utils.readCameraCalibrationFile(inputDir / "calibration.xml")
    hasCameraMatrix = cameraMatrix is not None
    hasDistCoeff    = distCoeff is not None

    # get interval coded to be analyzed, if any
    analyzeFrames   = utils.readMarkerIntervalsFile(inputDir / "markerInterval.tsv")
    hasAnalyzeFrames= qShowOnlyIntervals and analyzeFrames is not None

    # open video, if wanted
    if gShowVisualization:
        cap         = cv2.VideoCapture(str(inputDir / 'worldCamera.mp4'))
        width       = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height      = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        ifi         = 1000./cap.get(cv2.CAP_PROP_FPS)/gFPSFac

    # Read gaze data
    gazes,maxFrameIdx = utils.Gaze.readDataFromFile(inputDir / 'gazeData.tsv')

    # Read pose of marker board
    rVec,tVec,homography = utils.readBoardPoseFile(inputDir / 'boardPose.tsv')

    csv_file = open(inputDir / 'gazeWorldPos.tsv', 'w', newline='')
    csv_writer = csv.writer(csv_file, delimiter='\t')
    header = ['frame_idx']
    header.extend(utils.GazeWorld.getWriteHeader())
    csv_writer.writerow(header)
    
    subPixelFac = 8   # for sub-pixel positioning
    stopAllProcessing = False
    for frame_idx in range(maxFrameIdx+1):
        startTime = time.perf_counter()
        if gShowVisualization:
            ret, frame = cap.read()
            if (not ret) or (hasAnalyzeFrames and frame_idx > analyzeFrames[-1]):
                # done
                break

            if hasAnalyzeFrames:
                # check we're in a current interval, else skip processing
                # NB: have to spool through like this, setting specific frame to read
                # with cap.get(cv2.CAP_PROP_POS_FRAMES) doesn't seem to work reliably
                # for VFR video files
                inIval = False
                for f in range(0,len(analyzeFrames),2):
                    if frame_idx>=analyzeFrames[f] and frame_idx<=analyzeFrames[f+1]:
                        inIval = True
                        break
                if not inIval:
                    # no need to show this frame
                    continue

            refImg = reference.getImgCopy()
            

        if frame_idx in gazes:
            for gaze in gazes[frame_idx]:

                # draw 2D gaze point
                if gShowVisualization:
                    gaze.draw(frame, subPixelFac=subPixelFac, camRot=cameraRotation, camPos=cameraPosition, cameraMatrix=cameraMatrix, distCoeff=distCoeff)
                
                # if we have pose information, figure out where gaze vectors
                # intersect with reference board. Do same for 3D gaze point
                # (the projection of which coincides with 2D gaze provided by
                # the eye tracker)
                # store positions on marker board plane in camera coordinate frame to
                # file, along with gaze vector origins in same coordinate frame
                writeData = [frame_idx]
                if frame_idx in rVec or frame_idx in homography:
                    R = rVec[frame_idx] if frame_idx in rVec else None
                    T = tVec[frame_idx] if frame_idx in tVec else None
                    H = homography[frame_idx] if frame_idx in homography else None
                    gazeWorld = utils.gazeToPlane(gaze,R,T,cameraRotation,cameraPosition, cameraMatrix, distCoeff, H)
                    
                    # draw gazes on video and reference image
                    if gShowVisualization:
                        gazeWorld.drawOnWorldVideo(frame, cameraMatrix, distCoeff, subPixelFac)
                        if gShowReference:
                            gazeWorld.drawOnReferencePlane(refImg, reference, subPixelFac)

                    # store gaze-on-plane to csv
                    writeData.extend(gazeWorld.getWriteData())
                    csv_writer.writerow( writeData )

        if gShowVisualization:
            if gShowReference:
                cv2.imshow("reference", refImg)

            # if we have board pose, draw board origin on video
            if frame_idx in rVec or frame_idx in homography:
                if frame_idx in rVec and hasCameraMatrix and hasDistCoeff:
                    a = cv2.projectPoints(np.zeros((1,3)),rVec[frame_idx],tVec[frame_idx],cameraMatrix,distCoeff)[0].flatten()
                else:
                    iH = np.linalg.inv(homography[frame_idx])
                    a = utils.applyHomography(iH, centerTarget[0], centerTarget[1])
                    if hasCameraMatrix and hasDistCoeff:
                        a = utils.distortPoint(*a, cameraMatrix, distCoeff)
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
        elif (frame_idx)%100==0:
            print('  frame {}'.format(frame_idx))

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
