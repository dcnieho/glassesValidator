#!/usr/bin/python

from pathlib import Path
import numpy as np

import cv2
import csv
import time

import utils

from ffpyplayer.player import MediaPlayer

# This script shows a video player that is used to indicate the interval(s)
# during which the marker board should be found in the video and in later
# steps data quality computed. So this interval/these intervals would for
# instance be the exact interval during which the subject performs the 
# validation task.
# This script can be run directly on recordings converted to the common format
# with the a_* scripts, but output from steps c_detectMarkers and d_gazeToBoard
# (which can be run before this script, they will just process the whole video)
# will also be shown if available.

gShowReference      = True      # if true, also draw reference board with gaze overlaid on it (if available)


def process(inputDir,basePath):
    global gShowReference

    print('processing: {}'.format(inputDir.name))
    
    configDir = basePath / "config"
    # open file with information about Aruco marker and Gaze target locations
    validationSetup = utils.getValidationSetup(configDir)
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
    # 1. OpenCV for showing the frame
    cv2.namedWindow("frame",cv2.WINDOW_NORMAL)
    inVideo = inputDir / 'worldCamera.mp4'
    if not inVideo.is_file():
        inVideo = inputDir / 'worldCamera.avi'
    cap    = cv2.VideoCapture(str(inVideo))
    width   = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frate   = cap.get(cv2.CAP_PROP_FPS)
    # 2. if wanted and available, second OpenCV window for reference board with gaze on that plane
    gShowReference &= hasWorldGaze  # no reference board if we don't have world gaze, it'd be empty and pointless
    if gShowReference:
        cv2.namedWindow("reference")
    # 3. timestamp info for relating audio to video frames
    t2i = utils.Timestamp2Index( str(inputDir / 'frameTimestamps.tsv') )
    i2t = utils.Idx2Timestamp( str(inputDir / 'frameTimestamps.tsv') )
    # 4. mediaplayer for sound playback, brief wait or get_pts() below crashes
    ff_opts = {'vn' : False, 'volume': 1. }#{'sync':'video', 'framedrop':True}
    inAudio = inputDir / 'worldCamera.wav'
    if not inAudio.is_file():
        inAudio = inputDir / 'worldCamera.mp4'
    if not inAudio.is_file():
        inAudio = inputDir / 'worldCamera.avi'
    player = MediaPlayer(str(inAudio), ff_opts=ff_opts)
    time.sleep(0.1)

    # show
    lastIdx = None
    subPixelFac = 8   # for sub-pixel positioning
    armLength = reference.markerSize/2 # arms of axis are half a marker long
    stopAllProcessing = False
    shouldRedraw = False
    while True:
        frame, val = player.get_frame(force_refresh=True,show=False)
        if val == 'eof':
            player.toggle_pause()

        audio_pts = player.get_pts()    # this is audio_pts because we're in default audio sync mode

        # the audio is my shepherd and nothing shall I lack :-)
        frame_idx = t2i.find(audio_pts*1000)  # audio_pts is in seconds, our frame timestamps are in ms
        idxOffset = cap.get(cv2.CAP_PROP_POS_FRAMES) - frame_idx
        if abs(idxOffset) > 0:
            if 'pupil' in inputDir.stem:
                # this seems to work better for the pupil core and invisible's world videos
                cap.set(cv2.CAP_PROP_POS_MSEC, audio_pts*1000)
            else:
                # for SMI and Tobii G2/G3, this appears more accurate
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        if lastIdx is None or lastIdx!=frame_idx or shouldRedraw:
            shouldRedraw = False
            ret, frame = cap.read()
            if gShowReference:
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
                if gShowReference:
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
                analysisLbl += ', {} -- xx'.format(analyzeFrames[-1])
            
            # annotate what frame we're on
            frameClr = (0,0,255) if analysisIntervalIdx is not None else (0,0,0)
            cv2.rectangle(frame,(0,int(height)),(int(0.25*width),int(height)-30), frameClr, -1)
            cv2.putText(frame, ("%8.2f [%6d]" % (audio_pts, frame_idx) ), (0, int(height)-5), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255),2)
            # annotate analysis intervals
            cv2.rectangle(frame,(0,30),(int(width),0), frameClr, -1)
            cv2.putText(frame, (analysisLbl), (0, 25), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255),2)

            
            cv2.imshow("frame", frame)
            if width>1280:
                cv2.resizeWindow('frame', 1280,720)
            if gShowReference:
                cv2.imshow("reference", refImg)
            lastIdx = frame_idx

        key = cv2.waitKey(1) & 0xFF
        if key == ord('j'):
            player.seek(max(0,audio_pts-1/frate), relative=False)   # back one frame
        elif key == ord('k'):
            player.seek(max(0,audio_pts+1/frate), relative=False)   # forward one frame
        elif key == ord('h'):
            player.seek(max(0,audio_pts-1), relative=False)         # back one second
        elif key == ord('l'):
            player.seek(audio_pts+1, relative=False)                # forward one second

        elif key == ord('p'):
            player.toggle_pause()

        elif key == ord('f'):
            if not frame_idx in analyzeFrames:
                analyzeFrames.append(frame_idx)
                analyzeFrames.sort()
                shouldRedraw = True
        elif key == ord('d'):
            if frame_idx in analyzeFrames:
                # delete this one marker from analysis frames
                analyzeFrames.remove(frame_idx)
                shouldRedraw = True
            elif analysisIntervalIdx is not None:
                # delete current interval from analysis frames
                del analyzeFrames[analysisIntervalIdx:analysisIntervalIdx+2]
                shouldRedraw = True

        elif key in [ord('s'), ord('S')]:
            if (analysisIntervalIdx is not None) and (frame_idx!=analyzeFrames[analysisIntervalIdx]):
                # seek to start of current interval
                ts = i2t.get(analyzeFrames[analysisIntervalIdx])
                player.seek(ts/1000, relative=False)
            else:
                # seek to start of next or previous analysis interval, if any
                forward = key==ord('s')
                if forward:
                    idx = next((x for x in analyzeFrames[0:(len(analyzeFrames)//2)*2:2] if x>frame_idx), None) # slice gets starts of all whole intervals
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

    cap.release()
    cv2.destroyAllWindows()

    # store coded interval to file, if available
    with open(inputDir / 'markerInterval.tsv', 'w', newline='') as file:
        csv_writer = csv.writer(file, delimiter='\t')
        csv_writer.writerow(['start_frame', 'end_frame'])
        for f in range(0,len(analyzeFrames)-1,2):   # -1 to make sure we don't write out incomplete intervals
            csv_writer.writerow(analyzeFrames[f:f+2])

    return stopAllProcessing



if __name__ == '__main__':
    basePath = Path(__file__).resolve().parent
    for d in (basePath / 'data' / 'preprocced').iterdir():
        if d.is_dir():
            if process(d,basePath):
                break
