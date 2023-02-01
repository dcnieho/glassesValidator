#!/usr/bin/python3

import pathlib

import cv2
import numpy as np
import csv
import threading

from .. import config
from .. import utils
from ._image_gui import GUI, generic_tooltip, qns_tooltip


stopAllProcessing = False
def process(working_dir, config_dir=None, show_visualization=False, show_poster=True, show_only_intervals=True):
    # if show_visualization, each frame is shown in a viewer, overlaid with info about detected markers and poster
    # if show_poster, gaze in poster space is also drawn in a separate window
    # if show_only_intervals, only the coded validation episodes (if available) are shown in the viewer while the rest of the scene video is skipped past
    working_dir  = pathlib.Path(working_dir)
    if config_dir is not None:
        config_dir = pathlib.Path(config_dir)

    print('processing: {}'.format(working_dir.name))

    # if we need gui, we run processing in a separate thread (GUI needs to be on the main thread for OSX, see https://github.com/pthom/hello_imgui/issues/33)
    if show_visualization:
        gui = GUI(use_thread = False)
        gui.set_interesting_keys('qns')
        gui.register_draw_callback('status',lambda: generic_tooltip(qns_tooltip()))
        frame_win_id = gui.add_window(working_dir.name)
        poster_win_id= None
        if show_poster:
            poster_win_id = gui.add_window("poster")

        proc_thread = threading.Thread(target=do_the_work, args=(working_dir, config_dir, gui, frame_win_id, show_poster, poster_win_id, show_only_intervals))
        proc_thread.start()
        gui.start()
        proc_thread.join()
        return stopAllProcessing
    else:
        return do_the_work(working_dir, config_dir, None, None, False, None, False)


def do_the_work(working_dir, config_dir, gui, frame_win_id, show_poster, poster_win_id, show_only_intervals):
    global stopAllProcessing
    show_visualization = gui is not None

    utils.update_recording_status(working_dir, utils.Task.Gaze_Tranformed_To_Poster, utils.Status.Running)

    # open file with information about ArUco marker and Gaze target locations
    validationSetup = config.get_validation_setup(config_dir)

    # prep visualizations, if any
    if show_visualization:
        poster      = utils.Poster(config_dir, validationSetup)
        centerTarget= poster.targets[validationSetup['centerTarget']].center
        i2t         = utils.Idx2Timestamp(working_dir / 'frameTimestamps.tsv')

        cap         = cv2.VideoCapture(str(working_dir / 'worldCamera.mp4'))
        width       = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height      = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # get camera calibration info
    cameraMatrix,distCoeff,cameraRotation,cameraPosition = utils.readCameraCalibrationFile(working_dir / "calibration.xml")
    hasCameraMatrix = cameraMatrix is not None
    hasDistCoeff    = distCoeff is not None

    # get interval coded to be analyzed, if any
    analyzeFrames   = utils.readMarkerIntervalsFile(working_dir / "markerInterval.tsv")
    hasAnalyzeFrames= show_only_intervals and analyzeFrames is not None

    # Read gaze data
    print('  gazeData')
    gazes,maxFrameIdx = utils.Gaze.readDataFromFile(working_dir / 'gazeData.tsv')

    # Read camera pose w.r.t. poster
    print('  posterPose')
    poses = utils.PosterPose.readDataFromFile(working_dir / 'posterPose.tsv')

    csv_file = open(working_dir / 'gazePosterPos.tsv', 'w', newline='')
    csv_writer = csv.writer(csv_file, delimiter='\t')
    header = ['frame_idx']
    header.extend(utils.GazePoster.getWriteHeader())
    csv_writer.writerow(header)

    subPixelFac = 8   # for sub-pixel positioning
    stopAllProcessing = False
    for frame_idx in range(maxFrameIdx+1):
        if show_visualization:
            ret, frame = cap.read()
            if (not ret) or (hasAnalyzeFrames and frame_idx > analyzeFrames[-1]):
                # done
                break

            keys = gui.get_key_presses()
            if 'q' in keys:
                # quit fully
                stopAllProcessing = True
                break
            if 'n' in keys:
                # goto next
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
            if show_poster:
                refImg = poster.getImgCopy()


        if frame_idx in gazes:
            for gaze in gazes[frame_idx]:

                # draw gaze point on scene video
                if show_visualization:
                    gaze.draw(frame, subPixelFac=subPixelFac, camRot=cameraRotation, camPos=cameraPosition, cameraMatrix=cameraMatrix, distCoeff=distCoeff)

                # if we have pose information, figure out where gaze vectors
                # intersect with poster. Do same for 3D gaze point
                # (the projection of which coincides with 2D gaze provided by
                # the eye tracker)
                # store positions on poster plane in camera coordinate frame to
                # file, along with gaze vector origins in same coordinate frame
                writeData = [frame_idx]
                if frame_idx in poses:
                    gazePoster = utils.gazeToPlane(gaze,poses[frame_idx],cameraRotation,cameraPosition, cameraMatrix, distCoeff)

                    # draw gazes on video and poster
                    if show_visualization:
                        gazePoster.drawOnWorldVideo(frame, cameraMatrix, distCoeff, subPixelFac)
                        if show_poster:
                            gazePoster.drawOnPoster(refImg, poster, subPixelFac)

                    # store gaze-on-poster to csv
                    writeData.extend(gazePoster.getWriteData())
                    csv_writer.writerow( writeData )

        if show_visualization:
            frame_ts  = i2t.get(frame_idx)
            if show_poster:
                gui.update_image(refImg, frame_ts/1000., frame_idx, window_id = poster_win_id)

            # if we have poster pose, draw poster origin on video
            if frame_idx in poses:
                if poses[frame_idx] is not None and hasCameraMatrix and hasDistCoeff:
                    a = cv2.projectPoints(np.zeros((1,3)),poses[frame_idx].rVec,poses[frame_idx].tVec,cameraMatrix,distCoeff)[0].flatten()
                else:
                    iH = np.linalg.inv(poses[frame_idx].hMat)
                    a = utils.applyHomography(iH, centerTarget[0], centerTarget[1])
                    if hasCameraMatrix and hasDistCoeff:
                        a = utils.distortPoint(*a, cameraMatrix, distCoeff)
                utils.drawOpenCVCircle(frame, a, 3, (0,255,0), -1, subPixelFac)
                utils.drawOpenCVLine(frame, (a[0],0), (a[0],height), (0,255,0), 1, subPixelFac)
                utils.drawOpenCVLine(frame, (0,a[1]), (width,a[1]) , (0,255,0), 1, subPixelFac)

            # keys is populated above
            if 's' in keys:
                # screenshot
                cv2.imwrite(str(working_dir / ('calc_frame_%d.png' % frame_idx)), frame)

            gui.update_image(frame, frame_ts/1000., frame_idx, window_id = frame_win_id)
            closed, = gui.get_state()
            if closed:
                stopAllProcessing = True
                break

        if (frame_idx)%100==0:
            print('  frame {}'.format(frame_idx))

    csv_file.close()
    if show_visualization:
        cap.release()
        gui.stop()

    utils.update_recording_status(working_dir, utils.Task.Gaze_Tranformed_To_Poster, utils.Status.Finished)

    return stopAllProcessing