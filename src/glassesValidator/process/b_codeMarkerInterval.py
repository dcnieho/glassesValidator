#!/usr/bin/python

from pathlib import Path
import numpy as np

import cv2
import csv

from ffpyplayer.player import MediaPlayer

from .. import config
from .. import utils

# This script shows a video player that is used to indicate the interval(s)
# during which the marker board should be found in the video and in later
# steps data quality computed. So this interval/these intervals would for
# instance be the exact interval during which the subject performs the 
# validation task.
# This script can be run directly on recordings converted to the common format
# with the a_* scripts, but output from steps c_detectMarkers and d_gazeToBoard
# (which can be run before this script, they will just process the whole video)
# will also be shown if available.


def process(inputDir, configDir=None, showReference=False):
    # if showReference, also draw reference board with gaze overlaid on it (if available)
    inputDir  = Path(inputDir)
    if configDir is not None:
        configDir = pathlib.Path(configDir)

    print('processing: {}'.format(inputDir.name))
    utils.update_recording_status(inputDir, utils.Task.Coded, utils.Status.Running)
    
    # open file with information about Aruco marker and Gaze target locations
    validationSetup = config.getValidationSetup(configDir)
    reference = utils.Reference(configDir, validationSetup)

    # Read gaze data
    gazes,maxFrameIdx = utils.Gaze.readDataFromFile(inputDir / 'gazeData.tsv')

    # Read pose of marker board, if available
    hasBoardPose = False
    if (inputDir / 'boardPose.tsv').is_file():
        poses = utils.BoardPose.readDataFromFile(inputDir / 'boardPose.tsv')
        hasBoardPose = True

    # Read gaze on board data, if available
    hasWorldGaze = False
    if (inputDir / 'gazeWorldPos.tsv').is_file():
        gazesWorld = utils.GazeWorld.readDataFromFile(inputDir / 'gazeWorldPos.tsv')
        hasWorldGaze = True

    # get camera calibration info
    cameraMatrix,distCoeff = utils.readCameraCalibrationFile(inputDir / "calibration.xml")[0:2]
    hasCamCal = (cameraMatrix is not None) and (distCoeff is not None)

    # get interval coded to be analyzed, if available
    analyzeFrames = utils.readMarkerIntervalsFile(inputDir / "markerInterval.tsv")
    if analyzeFrames is None:
        analyzeFrames = []

    # set up video playback
    # 1. OpenCV window for scene video
    cv2.namedWindow("frame",cv2.WINDOW_NORMAL)
    # 2. if wanted and available, second OpenCV window for reference board with gaze on that plane
    showReference &= hasWorldGaze  # no reference board if we don't have world gaze, it'd be empty and pointless
    if showReference:
        cv2.namedWindow("reference")
    # 3. timestamp info for relating audio to video frames
    t2i = utils.Timestamp2Index( inputDir / 'frameTimestamps.tsv' )
    i2t = utils.Idx2Timestamp( inputDir / 'frameTimestamps.tsv' )
    # 4. mediaplayer for the actual video playback, with sound if available
    inVideo = inputDir / 'worldCamera.mp4'
    if not inVideo.is_file():
        inVideo = inputDir / 'worldCamera.avi'
    ff_opts = {'volume': 1., 'sync':'audio', 'framedrop':True}
    player = MediaPlayer(str(inVideo), ff_opts=ff_opts)

    # show
    subPixelFac = 8   # for sub-pixel positioning
    armLength = reference.markerSize/2 # arms of axis are half a marker long
    stopAllProcessing = False
    hasResized = False
    while True:
        frame, val = player.get_frame(force_refresh=True)
        if val == 'eof':
            player.toggle_pause()
        if frame is not None:
            image, pts = frame
            width, height = image.get_size()
            frame = cv2.cvtColor(np.asarray(image.to_memoryview()[0]).reshape((height,width,3)), cv2.COLOR_RGB2BGR)
            del image

        if frame is not None:
            # the audio is my shepherd and nothing shall I lack :-)
            frame_idx = t2i.find(pts*1000)  # pts is in seconds, our frame timestamps are in ms
            if showReference:
                refImg = reference.getImgCopy()

            # if we have board pose, draw board origin on video
            if hasBoardPose and frame_idx in poses and hasCamCal:
                utils.drawOpenCVFrameAxis(frame, cameraMatrix, distCoeff, poses[frame_idx].rVec, poses[frame_idx].tVec, armLength, 3, subPixelFac)

            # if have gaze for this frame, draw it
            # NB: usually have multiple gaze samples for a video frame, draw one
            if frame_idx in gazes:
                gazes[frame_idx][0].draw(frame, subPixelFac)
               
            # if have gaze in world info, draw it too (also only first)
            if hasWorldGaze and frame_idx in gazesWorld:
                if hasCamCal:
                    gazesWorld[frame_idx][0].drawOnWorldVideo(frame, cameraMatrix, distCoeff, subPixelFac)
                if showReference:
                    gazesWorld[frame_idx][0].drawOnReferencePlane(refImg, reference, subPixelFac)

            analysisIntervalIdx = None
            analysisLbl = ''
            for f in range(0,len(analyzeFrames)-1,2):   # -1 to make sure we don't try incomplete intervals
                if frame_idx>=analyzeFrames[f] and frame_idx<=analyzeFrames[f+1]:
                    analysisIntervalIdx = f
                if len(analysisLbl)>0:
                    analysisLbl += ', '
                analysisLbl += '{} -- {}'.format(*analyzeFrames[f:f+2])
            if len(analyzeFrames)%2:
                if len(analyzeFrames)==1:
                    analysisLbl +=   '{} -- xx'.format(analyzeFrames[-1])
                else:
                    analysisLbl += ', {} -- xx'.format(analyzeFrames[-1])
            
            # annotate what frame we're on
            frameClr = (0,0,255) if analysisIntervalIdx is not None else (0,0,0)
            cv2.rectangle(frame,(0,int(height)),(int(0.35*width),int(height)-30), frameClr, -1)
            cv2.putText(frame, ("%8.3f [%6d]" % (pts, frame_idx) ), (0, int(height)-5), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255),2)
            # annotate analysis intervals
            cv2.rectangle(frame,(0,30),(int(width),0), frameClr, -1)
            cv2.putText(frame, (analysisLbl), (0, 25), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255),2)

            
            if frame is not None:
                cv2.imshow("frame", frame)
                if not hasResized:
                    if width>1280 or height>800:
                        sFac = max([width/1280., height/800.])
                        cv2.resizeWindow('frame', round(width/sFac), round(height/sFac))
                    else:
                        cv2.resizeWindow('frame', width, height)
                    hasResized = True
                        
            if showReference:
                cv2.imshow("reference", refImg)

        key = cv2.waitKey(1) & 0xFF
        # seek: don't ask me why, but relative seeking works best for backward,
        # and seeking to absolute pts best for forward seeking.
        if key == ord('j'):
            step = (i2t.get(frame_idx)-i2t.get(max(0,frame_idx-1)))/1000
            player.seek(-step)                              # back one frame
        elif key == ord('k'):
            nextTs = i2t.get(frame_idx+1)
            if nextTs != -1.:
                step = (nextTs-i2t.get(max(0,frame_idx)))/1000
                player.seek(pts+step, relative=False)       # forward one frame
        elif key in [ord('h'), ord('H')]:
            step = 1 if key==ord('h') else 10
            player.seek(-step)                              # back one or ten seconds
        elif key in [ord('l'), ord('L')]:
            step = 1 if key==ord('l') else 10
            player.seek(pts+step, relative=False)           # forward one or ten seconds

        elif key == ord('p'):
            player.toggle_pause()
            if not player.get_pause():
                player.seek(0)  # needed to get frames rolling in again, apparently, after seeking occurred while paused

        elif key == ord('f'):
            if not frame_idx in analyzeFrames:
                analyzeFrames.append(frame_idx)
                analyzeFrames.sort()
        elif key == ord('d'):
            if frame_idx in analyzeFrames:
                # delete this one marker from analysis frames
                analyzeFrames.remove(frame_idx)
            elif analysisIntervalIdx is not None:
                # delete current interval from analysis frames
                del analyzeFrames[analysisIntervalIdx:analysisIntervalIdx+2]

        elif key in [ord('s'), ord('S')]:
            if (analysisIntervalIdx is not None) and (frame_idx!=analyzeFrames[analysisIntervalIdx]):
                # seek to start of current interval
                ts = i2t.get(analyzeFrames[analysisIntervalIdx])
                player.seek(ts/1000, relative=False)
            else:
                # seek to start of next or previous analysis interval, if any
                forward = key==ord('s')
                if forward:
                    idx = next((x for x in analyzeFrames[ 0:(len(analyzeFrames)//2)*2:2 ] if x>frame_idx), None) # slice gets starts of all whole intervals
                else:
                    idx = next((x for x in analyzeFrames[(len(analyzeFrames)//2)*2-2::-2] if x<frame_idx), None) # slice gets starts of all whole intervals in reverse order
                if idx is not None:
                    ts = i2t.get(idx)
                    player.seek(ts/1000, relative=False)
        elif key in [ord('e'), ord('E')]:
            if (analysisIntervalIdx is not None) and (frame_idx!=analyzeFrames[analysisIntervalIdx+1]):
                # seek to end of current interval
                ts = i2t.get(analyzeFrames[analysisIntervalIdx+1])
                player.seek(ts/1000, relative=False)
            else:
                # seek to end of next or previous analysis interval, if any
                forward = key==ord('e')
                if forward:
                    idx = next((x for x in analyzeFrames[1:(len(analyzeFrames)//2)*2:2] if x>frame_idx), None) # slice gets ends of all whole intervals
                else:
                    idx = next((x for x in analyzeFrames[(len(analyzeFrames)//2)*2::-2] if x<frame_idx), None) # slice gets ends of all whole intervals in reverse order
                if idx is not None:
                    ts = i2t.get(idx)
                    player.seek(ts/1000, relative=False)

        elif key == ord('q'):
            # quit fully
            stopAllProcessing = True
            break
        elif key == ord('n'):
            # goto next
            break
        
    player.close_player()
    cv2.destroyAllWindows()

    # store coded interval to file, if available
    with open(inputDir / 'markerInterval.tsv', 'w', newline='') as file:
        csv_writer = csv.writer(file, delimiter='\t')
        csv_writer.writerow(['start_frame', 'end_frame'])
        for f in range(0,len(analyzeFrames)-1,2):   # -1 to make sure we don't write out incomplete intervals
            csv_writer.writerow(analyzeFrames[f:f+2])

    utils.update_recording_status(inputDir, utils.Task.Coded, utils.Status.Finished)

    return stopAllProcessing
