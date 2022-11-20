#!/usr/bin/python

import pathlib
import numpy as np

import cv2
import csv

from ffpyplayer.player import MediaPlayer

import sys
isMacOS = sys.platform.startswith("darwin")
if isMacOS:
    import AppKit

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


def process(input_dir, config_dir=None, show_reference=False):
    # if showReference, also draw reference board with gaze overlaid on it (if available)
    input_dir  = pathlib.Path(input_dir)
    if config_dir is not None:
        config_dir = pathlib.Path(config_dir)

    print('processing: {}'.format(input_dir.name))
    utils.update_recording_status(input_dir, utils.Task.Coded, utils.Status.Running)
    
    # open file with information about Aruco marker and Gaze target locations
    validationSetup = config.get_validation_setup(config_dir)
    reference = utils.Reference(config_dir, validationSetup)

    # Read gaze data
    gazes,maxFrameIdx = utils.Gaze.readDataFromFile(input_dir / 'gazeData.tsv')

    # Read pose of marker board, if available
    hasBoardPose = False
    if (input_dir / 'boardPose.tsv').is_file():
        try:
            poses = utils.BoardPose.readDataFromFile(input_dir / 'boardPose.tsv')
            hasBoardPose = True
        except:
            # ignore when file can't be read or is empty
            pass

    # Read gaze on board data, if available
    hasWorldGaze = False
    if (input_dir / 'gazeWorldPos.tsv').is_file():
        try:
            gazesWorld = utils.GazeWorld.readDataFromFile(input_dir / 'gazeWorldPos.tsv')
            hasWorldGaze = True
        except:
            # ignore when file can't be read or is empty
            pass

    # get camera calibration info
    cameraMatrix,distCoeff = utils.readCameraCalibrationFile(input_dir / "calibration.xml")[0:2]
    hasCamCal = (cameraMatrix is not None) and (distCoeff is not None)

    # get interval coded to be analyzed, if available
    analyzeFrames = utils.readMarkerIntervalsFile(input_dir / "markerInterval.tsv")
    if analyzeFrames is None:
        analyzeFrames = []

    # set up video playback
    # 1. OpenCV window for scene video
    cv2.namedWindow("code validation intervals",cv2.WINDOW_NORMAL)
    # 2. if wanted and available, second OpenCV window for reference board with gaze on that plane
    show_reference &= hasWorldGaze  # no reference board if we don't have world gaze, it'd be empty and pointless
    if show_reference:
        cv2.namedWindow("reference")
    # 3. timestamp info for relating audio to video frames
    t2i = utils.Timestamp2Index( input_dir / 'frameTimestamps.tsv' )
    i2t = utils.Idx2Timestamp( input_dir / 'frameTimestamps.tsv' )
    # 4. mediaplayer for the actual video playback, with sound if available
    inVideo = input_dir / 'worldCamera.mp4'
    if not inVideo.is_file():
        inVideo = input_dir / 'worldCamera.avi'
    ff_opts = {'volume': 1., 'sync': 'audio', 'framedrop': True}
    player = MediaPlayer(str(inVideo), ff_opts=ff_opts)

    # show
    subPixelFac = 8   # for sub-pixel positioning
    armLength = reference.markerSize/2 # arms of axis are half a marker long
    stopAllProcessing = False
    hasResized = False
    hasRequestedFocus = not isMacOS # False only if on Mac OS, else True since its a no-op
    showHelp = False
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
            if show_reference:
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
                if show_reference:
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
            text = "%8.3f [%6d]" % (pts, frame_idx) 
            textSize,baseline = cv2.getTextSize(text,cv2.FONT_HERSHEY_PLAIN,2,2)
            cv2.rectangle(frame,(0,int(height)),(textSize[0]+2,int(height)-textSize[1]-baseline-5), frameClr, -1)
            cv2.putText(frame, (text), (2, int(height)-5), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255),2)
            # annotate analysis intervals
            textSize,baseline = cv2.getTextSize(analysisLbl,cv2.FONT_HERSHEY_PLAIN,2,2)
            cv2.rectangle(frame,(0,textSize[1]+baseline+5),(textSize[0]+5,0), frameClr, -1)
            cv2.putText(frame, (analysisLbl), (0, textSize[1]+baseline), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255),2)
            # show help
            if not showHelp:
                cv2.putText(frame,("Press I for help"), (0,55), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (0,255,255),2)
            else:
                cv2.putText(frame,("H: back 1 s, shift+H: back 10 s"), (0,55), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (0,255,255),2)
                cv2.putText(frame,("L: forward 1 s, shift+L: forward 10 s"), (0,85), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (0,255,255),2)
                cv2.putText(frame,("J: back 1 frame, K: forward 1 frame"), (0,115), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (0,255,255),2)
                cv2.putText(frame,("P: pause or resume playback"), (0,145), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (0,255,255),2)
                cv2.putText(frame,("F: mark frame, D: delete frame or current interval"), (0,175), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (0,255,255),2)
                cv2.putText(frame,("S: seek to start of next interval, shift+S seek to start of previous interval"), (0,205), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (0,255,255),2)
                cv2.putText(frame,("E: seek to end of next interval, shift+E seek to end of previous interval"), (0,235), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (0,255,255),2)
                cv2.putText(frame,("N/Q: quit"), (0,265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (0,255,255),2)
                cv2.putText(frame,("I: toggle help"), (0,295), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (0,255,255),2)
            
            if frame is not None:
                cv2.imshow("code validation intervals", frame)
                if not hasResized:
                    if width>1280 or height>800:
                        sFac = max([width/1280., height/800.])
                        cv2.resizeWindow('code validation intervals', round(width/sFac), round(height/sFac))
                    else:
                        cv2.resizeWindow('code validation intervals', width, height)
                    hasResized = True
                        
            if show_reference:
                cv2.imshow("reference", refImg)

        if not hasRequestedFocus:
            AppKit.NSApplication.sharedApplication().activateIgnoringOtherApps_(1)
            hasRequestedFocus = True

        key = cv2.waitKey(1) & 0xFF
        if key == ord('i'):
            showHelp = not showHelp
        # seek: don't ask me why, but relative seeking works best for backward,
        # and seeking to absolute pts best for forward seeking.
        elif key == ord('j'):
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
    cv2.waitKey(1)

    # store coded interval to file, if available
    with open(input_dir / 'markerInterval.tsv', 'w', newline='') as file:
        csv_writer = csv.writer(file, delimiter='\t')
        csv_writer.writerow(['start_frame', 'end_frame'])
        for f in range(0,len(analyzeFrames)-1,2):   # -1 to make sure we don't write out incomplete intervals
            csv_writer.writerow(analyzeFrames[f:f+2])

    utils.update_recording_status(input_dir, utils.Task.Coded, utils.Status.Finished)

    return stopAllProcessing
