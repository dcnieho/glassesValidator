#!/usr/bin/python

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

from ffpyplayer.player import MediaPlayer

gShowReference      = True      # if true, also draw reference board with gaze overlaid on it


def process(inputDir,basePath):
    global gShowReference

    print('processing: {}'.format(inputDir.name))
    
    configDir = basePath / "config"
    # open file with information about Aruco marker and Gaze target locations
    validationSetup = utils.getValidationSetup(configDir)

    # Read gaze data
    gazes,maxFrameIdx = utils.getGazeData(inputDir / 'gazeData.tsv')

    # Read pose of marker board, if available
    hasBoardPose = False
    if (inputDir / 'boardPose.tsv').is_file():
        rVec,tVec = utils.getMarkerBoardPose(inputDir / 'boardPose.tsv')
        hasBoardPose = True

    # Read gaze on board data, if available
    hasWorldGaze = False
    if (inputDir / 'gazeWorldPos.tsv').is_file():
        gazesWorld = utils.getGazeWorldData(inputDir / 'gazeWorldPos.tsv')
        hasWorldGaze = True

    # get camera calibration info
    cameraMatrix,distCoeff = utils.getCameraCalibrationInfo(inputDir / "calibration.xml")[0:2]

    # set up video playback
    # 1. OpenCV for showing the frame
    cv2.namedWindow("frame",cv2.WINDOW_NORMAL)
    cap     = cv2.VideoCapture( str(inputDir / 'worldCamera.mp4') )
    width   = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 2. if wanted, second OpenCV window for reference board
    if gShowReference and hasWorldGaze:
        cv2.namedWindow("reference")
        reference = utils.Reference(str(inputDir / 'referenceBoard.png'), configDir, validationSetup)
    # 3. timestamp info for relating audio to video frames
    t2i = utils.Timestamp2Index( str(inputDir / 'frameTimestamps.tsv') )
    # 4. mediaplayer for sound playback, brief wait or get_pts() below crashes
    ff_opts = {'vn' : False, 'volume': 1. }#{'sync':'video', 'framedrop':True}
    player = MediaPlayer(str(inputDir / 'worldCamera.mp4'), ff_opts=ff_opts)
    time.sleep(0.1)

    # show
    lastIdx = None
    subPixelFac = 8   # for sub-pixel positioning
    armLength = 2.*math.tan(math.radians(.5))*validationSetup['distance']*10*validationSetup['markerSide']/2 # arms of axis are half a marker long
    stopAllProcessing = False
    while True:
        frame, val = player.get_frame(force_refresh=True,show=False)
        if val == 'eof':
            break

        audio_pts = player.get_pts()    # this is audio_pts because we're in default audio sync mode

        # the audio is my shepherd and nothing shall I lack :-)
        # From experience, PROP_POS_MSEC is utterly broken; let's use indexes instead
        frame_idx = t2i.find(audio_pts*1000)  # audio_pts is in seconds, our frame timestamps are in ms
        idxOffset = cap.get(cv2.CAP_PROP_POS_FRAMES) - frame_idx
        if abs(idxOffset) > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        if lastIdx is None or lastIdx!=frame_idx:
            ret, frame = cap.read()
            if gShowReference and hasWorldGaze:
                refImg = reference.getImgCopy()

            # if we have board pose, draw board origin on video
            if hasBoardPose and frame_idx in rVec:
                utils.drawOpenCVFrameAxis(frame, cameraMatrix, distCoeff, rVec[frame_idx], tVec[frame_idx], armLength, 3, subPixelFac)

            # if have gaze for this frame, draw it
            # NB: usually have multiple gaze samples for a video frame, draw one
            if frame_idx in gazes:
                gazes[frame_idx][0].draw(frame, subPixelFac)
               
            # if have gaze in world info, draw it too (also only first)
            if hasWorldGaze and frame_idx in gazesWorld:
                gazesWorld[frame_idx][0].drawOnWorldVideo(frame, cameraMatrix, distCoeff, subPixelFac)
                if gShowReference:
                    gazesWorld[frame_idx][0].drawOnReferencePlane(refImg, reference, subPixelFac)

            cv2.rectangle(frame,(0,int(height)),(int(0.25*width),int(height)-30), (0,0,0), -1)
            cv2.putText(frame, ("%8.2f [%6d]" % (audio_pts, frame_idx) ), (0, int(height)-5), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255),2)

            
            cv2.imshow("frame", frame)
            if width>1280:
                cv2.resizeWindow('frame', 1280,720)
            if gShowReference and hasWorldGaze:
                cv2.imshow("reference", refImg)
            lastIdx = frame_idx

        key = cv2.waitKey(1) & 0xFF
        if key == ord('k'):
            player.seek(audio_pts+10, relative=False)
        if key == ord('j'):
            player.seek(max(0,audio_pts-10), relative=False)
        if key == ord('l'):
            player.seek(audio_pts+5, relative=False)
        if key == ord('h'):
            player.seek(max(0,audio_pts-5), relative=False)
        if key == ord('p'):
            player.toggle_pause()
        if key == ord('q'):
            # quit fully
            stopAllProcessing = True
            break
        if key == ord('n'):
            # goto next
            break

    cap.release()
    cv2.destroyAllWindows()
    return stopAllProcessing



if __name__ == '__main__':
    basePath = Path(__file__).resolve().parent
    for d in (basePath / 'data' / 'preprocced').iterdir():
        if d.is_dir():
            if process(d,basePath):
                break
