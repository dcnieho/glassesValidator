#!/usr/bin/python3

import pathlib

import cv2
import numpy as np
import csv
import time
from .. import config
from .. import utils


def process(working_dir, config_dir=None, show_visualization=False, show_poster=True, show_only_intervals=True, fps_fac=1):
    # if show_visualization, each frame is shown in a viewer, overlaid with info about detected markers and poster
    # if show_poster, gaze in poster space is also drawn in a separate window
    # if show_only_intervals, only the coded validation episodes (if available) are shown in the viewer while the rest of the scene video is skipped past
    working_dir  = pathlib.Path(working_dir)
    if config_dir is not None:
        config_dir = pathlib.Path(config_dir)

    print('processing: {}'.format(working_dir.name))
    utils.update_recording_status(working_dir, utils.Task.Gaze_Tranformed_To_Poster, utils.Status.Running)

    # open file with information about ArUco marker and Gaze target locations
    validationSetup = config.get_validation_setup(config_dir)

    if show_visualization:
        cv2.namedWindow("frame")
        if show_poster:
            cv2.namedWindow("poster")

        poster      = utils.Poster(config_dir, validationSetup)
        centerTarget= poster.targets[validationSetup['centerTarget']].center
        i2t         = utils.Idx2Timestamp(working_dir / 'frameTimestamps.tsv')

    # get camera calibration info
    cameraMatrix,distCoeff,cameraRotation,cameraPosition = utils.readCameraCalibrationFile(working_dir / "calibration.xml")
    hasCameraMatrix = cameraMatrix is not None
    hasDistCoeff    = distCoeff is not None

    # get interval coded to be analyzed, if any
    analyzeFrames   = utils.readMarkerIntervalsFile(working_dir / "markerInterval.tsv")
    hasAnalyzeFrames= show_only_intervals and analyzeFrames is not None

    # open video, if wanted
    if show_visualization:
        cap         = cv2.VideoCapture(str(working_dir / 'worldCamera.mp4'))
        width       = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height      = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ifi         = 1000./cap.get(cv2.CAP_PROP_FPS)/fps_fac

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
        startTime = time.perf_counter()
        if show_visualization:
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
            if show_poster:
                cv2.imshow("poster", refImg)

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
                cv2.imwrite(str(working_dir / ('calc_frame_%d.png' % frame_idx)), frame)
        elif (frame_idx)%100==0:
            print('  frame {}'.format(frame_idx))

    csv_file.close()
    if show_visualization:
        cap.release()
        cv2.destroyAllWindows()

    utils.update_recording_status(working_dir, utils.Task.Gaze_Tranformed_To_Poster, utils.Status.Finished)

    return stopAllProcessing