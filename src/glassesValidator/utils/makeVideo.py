#!/usr/bin/python3
# NB: this is a combination of the c_ and d_ steps. Since it is maintained separately and honestly a bit of an
# after-thought, it may not be in sync with the c_ and d_ steps.

import shutil
import os
import pathlib

import cv2
import numpy as np

from .. import config
from .. import utils

from ffpyplayer.writer import MediaWriter
from ffpyplayer.pic import Image
import ffpyplayer.tools
from fractions import Fraction


def process(working_dir, config_dir=None, show_rejected_markers=False, add_audio_to_poster_video=False, show_visualization=False):
    # if show_rejected_markers, rejected ArUco marker candidates are also drawn on the video. Possibly useful for debug
    # if add_audio_to_poster_video, audio is added to poster video, not only to the scene video
    # if show_visualization, the generated video is shown as it is created in a viewer
    working_dir  = pathlib.Path(working_dir)
    if config_dir is not None:
        config_dir = pathlib.Path(config_dir)

    print('processing: {}'.format(working_dir.name))

    # open file with information about Aruco marker and Gaze target locations
    validationSetup = config.get_validation_setup(config_dir)

    if show_visualization:
        cv2.namedWindow("scene camera")
        cv2.namedWindow("poster")

    # open input video file, query it for size
    inVideo = working_dir / 'worldCamera.mp4'
    if not inVideo.is_file():
        inVideo = working_dir / 'worldCamera.avi'
    vidIn  = cv2.VideoCapture( str(inVideo) )
    if not vidIn.isOpened():
        raise RuntimeError('the file "{}" could not be opened'.format(str(inVideo)))
    width  = vidIn.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vidIn.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps    = vidIn.get(cv2.CAP_PROP_FPS)

    # get info about markers on our poster
    poster      = utils.Poster(config_dir, validationSetup)
    centerTarget= poster.targets[validationSetup['centerTarget']].center
    # turn into aruco board object to be used for pose estimation
    arucoBoard  = poster.getArucoBoard()

    # prep output video files
    # get which pixel format
    codec    = ffpyplayer.tools.get_format_codec(fmt='mp4')
    pix_fmt  = ffpyplayer.tools.get_best_pix_fmt('bgr24',ffpyplayer.tools.get_supported_pixfmts(codec))
    fpsFrac  = Fraction(fps).limit_denominator(10000).as_integer_ratio()
    # scene video
    out_opts = {'pix_fmt_in':'bgr24', 'pix_fmt_out':pix_fmt, 'width_in':int(      width    ), 'height_in':int(      height    ),'frame_rate':fpsFrac}
    vidOutScene  = MediaWriter(str(working_dir / 'detectOutput_scene.mp4') , [out_opts], overwrite=True)
    # poster video
    out_opts = {'pix_fmt_in':'bgr24', 'pix_fmt_out':pix_fmt, 'width_in':int(poster.width), 'height_in':int(poster.height),'frame_rate':fpsFrac}
    vidOutPoster = MediaWriter(str(working_dir / 'detectOutput_poster.mp4'), [out_opts], overwrite=True)

    # setup aruco marker detection
    parameters = cv2.aruco.DetectorParameters_create()
    parameters.markerBorderBits       = validationSetup['markerBorderBits']
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    # get frame timestamps lookup file
    i2t = utils.Idx2Timestamp(working_dir / 'frameTimestamps.tsv')

    # get camera calibration info
    cameraMatrix,distCoeff,cameraRotation,cameraPosition = utils.readCameraCalibrationFile(working_dir / "calibration.xml")
    hasCameraMatrix = cameraMatrix is not None
    hasDistCoeff    = distCoeff is not None

    # Read gaze data
    gazes,maxFrameIdx = utils.Gaze.readDataFromFile(working_dir / 'gazeData.tsv')

    # get interval coded to be analyzed, if available
    analyzeFrames = utils.readMarkerIntervalsFile(working_dir / "markerInterval.tsv")
    if analyzeFrames is None:
        analyzeFrames = []

    frame_idx = 0
    armLength = poster.markerSize/2 # arms of axis are half a marker long
    subPixelFac = 8   # for sub-pixel positioning
    stopAllProcessing = False
    while True:
        # process frame-by-frame
        ret, frame = vidIn.read()
        if not ret:
            break
        refImg = poster.getImgCopy()

        # detect markers, undistort
        corners, ids, rejectedImgPoints = \
            cv2.aruco.detectMarkers(frame, poster.aruco_dict, parameters=parameters)
        recoveredIds = None

        # get camera pose w.r.t. poster, draw marker and poster pose
        gotPose = False
        if np.all(ids != None):
            if len(ids) >= validationSetup['minNumMarkers']:
                pose = utils.PosterPose(frame_idx)
                # get camera pose
                if hasCameraMatrix and hasDistCoeff:
                    # Refine detected markers (eliminates markers not part of our poster, adds missing markers to the poster)
                    corners, ids, rejectedImgPoints, recoveredIds = utils.arucoRefineDetectedMarkers(
                            image = frame, arucoBoard = arucoBoard,
                            detectedCorners = corners, detectedIds = ids, rejectedCorners = rejectedImgPoints,
                            cameraMatrix = cameraMatrix, distCoeffs = distCoeff)

                    pose.nMarkers, rVec, tVec = cv2.aruco.estimatePoseBoard(corners, ids, arucoBoard, cameraMatrix, distCoeff, np.empty(1), np.empty(1))

                    # draw axis indicating poster pose (origin and orientation)
                    if pose.nMarkers>0:
                        # set pose
                        pose.setPose(rVec,tVec)
                        # and draw
                        utils.drawOpenCVFrameAxis(frame, cameraMatrix, distCoeff, pose.rVec, pose.tVec, armLength, 3, subPixelFac)
                        gotPose = True

                # also get homography (direct image plane to plane in world transform). Use undistorted marker corners
                if hasCameraMatrix and hasDistCoeff:
                    cornersU = [cv2.undistortPoints(x, cameraMatrix, distCoeff, P=cameraMatrix) for x in corners]
                else:
                    cornersU = corners
                H, status = utils.estimateHomography(poster.knownMarkers, cornersU, ids)

                if status:
                    pose.hMat = H
                    # find where target is expected to be in the image
                    iH = np.linalg.inv(pose.hMat)
                    target = utils.applyHomography(iH, centerTarget[0], centerTarget[1])
                    if hasCameraMatrix and hasDistCoeff:
                        target = utils.distortPoint(*target, cameraMatrix, distCoeff)
                    # draw target location on image
                    if target[0] >= 0 and target[0] < width and target[1] >= 0 and target[1] < height:
                        utils.drawOpenCVCircle(frame, target, 3, (0,0,0), -1, subPixelFac)

            # if any markers were detected, draw where on the frame
            utils.drawArucoDetectedMarkers(frame, corners, ids, subPixelFac=subPixelFac, specialHighlight=[recoveredIds,(255,255,0)])

        # for debug, can draw rejected markers on frame
        if show_rejected_markers:
            cv2.aruco.drawDetectedMarkers(frame, rejectedImgPoints, None, borderColor=(211,0,148))

        # process gaze
        if frame_idx in gazes:
            for gaze in gazes[frame_idx]:
                # draw gaze point on scene video
                gaze.draw(frame, subPixelFac=subPixelFac, camRot=cameraRotation, camPos=cameraPosition, cameraMatrix=cameraMatrix, distCoeff=distCoeff)

                # if we have pose information, figure out where gaze vectors
                # intersect with poster. Do same for 3D gaze point
                # (the projection of which coincides with 2D gaze provided by
                # the eye tracker)
                if gotPose:
                    gazePoster = utils.gazeToPlane(gaze,pose,cameraRotation,cameraPosition, cameraMatrix, distCoeff)

                    # draw gazes on video and poster
                    gazePoster.drawOnWorldVideo(frame, cameraMatrix, distCoeff, subPixelFac)
                    gazePoster.drawOnPoster(refImg, poster, subPixelFac)

        # annotate frame
        analysisIntervalIdx = None
        for f in range(0,len(analyzeFrames)-1,2):   # -1 to make sure we don't try incomplete intervals
            if frame_idx>=analyzeFrames[f] and frame_idx<=analyzeFrames[f+1]:
                analysisIntervalIdx = f
        frameClr = (0,0,255) if analysisIntervalIdx is not None else (0,0,0)

        frame_ts  = i2t.get(frame_idx)
        text = '%6.3f [%6d]' % (frame_ts/1000.,frame_idx)
        textSize,baseline = cv2.getTextSize(text,cv2.FONT_HERSHEY_PLAIN,2,2)
        cv2.rectangle(frame,(0,int(height)),(textSize[0]+2,int(height)-textSize[1]-baseline-5), frameClr, -1)
        cv2.putText(frame, (text), (2, int(height)-5), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255),2)

        # store to file
        img = Image(plane_buffers=[frame.flatten().tobytes()], pix_fmt='bgr24', size=(int(width), int(height)))
        vidOutScene.write_frame(img=img, pts=frame_idx/fps)
        img = Image(plane_buffers=[refImg.flatten().tobytes()], pix_fmt='bgr24', size=(poster.width, poster.height))
        vidOutPoster.write_frame(img=img, pts=frame_idx/fps)


        if show_visualization:
            cv2.imshow('scene camera',frame)
            cv2.imshow('poster',refImg)
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
    vidOutScene.close()
    vidOutPoster.close()
    cv2.destroyAllWindows()

    # if ffmpeg is on path, add audio to scene and optionally poster video
    if shutil.which('ffmpeg') is not None:
        todo = [working_dir / 'detectOutput_scene.mp4']
        if add_audio_to_poster_video:
            todo.append(working_dir / 'detectOutput_poster.mp4')

        for f in todo:
            # move file to temp name
            tempName = f.parent / (f.stem + '_temp' + f.suffix)
            shutil.move(str(f),str(tempName))

            # add audio
            cmd_str = ' '.join(['ffmpeg', '-y', '-i', '"'+str(tempName)+'"', '-i', '"'+str(inVideo)+'"', '-vcodec', 'copy', '-acodec', 'copy', '-map', '0:v:0', '-map', '1:a:0?', '"'+str(f)+'"'])
            os.system(cmd_str)
            # clean up
            tempName.unlink(missing_ok=True)

    return stopAllProcessing