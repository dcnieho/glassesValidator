#!/usr/bin/python3
# NB: this is a combination of the c_ and d_ steps, not actively maintained.
# Not guaranteed to provide the same output as c_ and d_ steps, or to work at all

import shutil
import os
from pathlib import Path

import cv2
import numpy as np

import utils

from ffpyplayer.writer import MediaWriter
from ffpyplayer.pic import Image
import ffpyplayer.tools
from fractions import Fraction

gShowVisualization      = False     # if true, draw each frame and overlay info about detected markers and board
gAddAudioToBoardVideo   = False     # if true, audio will be added to reference board video, not only to the scene video


def process(inputDir,basePath):
    global gShowVisualization
    global gAddAudioToBoardVideo

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

    # get info about markers on our board
    reference       = utils.Reference(configDir, validationSetup)
    centerTarget    = reference.getTargets()[validationSetup['centerTarget']].center
    # turn into aruco board object to be used for pose estimation
    referenceBoard  = reference.getArucoBoard()
    
    # prep output video files
    # get which pixel format
    codec    = ffpyplayer.tools.get_format_codec(fmt='mp4')
    pix_fmt  = ffpyplayer.tools.get_best_pix_fmt('bgr24',ffpyplayer.tools.get_supported_pixfmts(codec))
    fpsFrac  = Fraction(fps).limit_denominator(10000).as_integer_ratio()
    # scene video
    out_opts = {'pix_fmt_in':'bgr24', 'pix_fmt_out':pix_fmt, 'width_in':int(      width    ), 'height_in':int(      height    ),'frame_rate':fpsFrac}
    vidOutScene = MediaWriter(str(inputDir / 'detectOutput_scene.mp4'), [out_opts], overwrite=True)
    # reference board video
    out_opts = {'pix_fmt_in':'bgr24', 'pix_fmt_out':pix_fmt, 'width_in':int(reference.width), 'height_in':int(reference.height),'frame_rate':fpsFrac}
    vidOutBoard = MediaWriter(str(inputDir / 'detectOutput_board.mp4'), [out_opts], overwrite=True)

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
    armLength = reference.markerSize/2 # arms of axis are half a marker long
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
            cv2.aruco.detectMarkers(frame, reference.aruco_dict, parameters=parameters)

        # get board pose, draw marker and board pose
        gotPose = False
        if np.all(ids != None):
            if len(ids) >= validationSetup['minNumMarkers']:
                pose = utils.BoardPose(frame_idx)
                # get camera pose
                if (cameraMatrix is not None) and (distCoeff is not None):
                    pose.nMarkers, rVec, tVec = cv2.aruco.estimatePoseBoard(corners, ids, referenceBoard, cameraMatrix, distCoeff)
                    
                    # draw axis indicating board pose (origin and orientation)
                    if pose.nMarkers>0:
                        # set pose
                        pose.setPose(rVec,tVec)
                        # and draw
                        utils.drawOpenCVFrameAxis(frame, cameraMatrix, distCoeff, pose.rVec, pose.tVec, armLength, 3, subPixelFac)
                        gotPose = True

                # also get homography (direct image plane to plane in world transform). Use undistorted marker corners
                if (cameraMatrix is not None) and (distCoeff is not None):
                    cornersU = [cv2.undistortPoints(x, cameraMatrix, distCoeff, P=cameraMatrix) for x in corners]
                else:
                    cornersU = corners
                H, status = utils.estimateHomography(reference.knownMarkers, cornersU, ids)

                if status:
                    pose.hMat = H
                    # find where target is expected to be in the image
                    iH = np.linalg.inv(pose.hMat)
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

                # draw gaze point on scene video
                gaze.draw(frame, subPixelFac=subPixelFac, camRot=cameraRotation, camPos=cameraPosition, cameraMatrix=cameraMatrix, distCoeff=distCoeff)
                
                # if we have pose information, figure out where gaze vectors
                # intersect with reference board. Do same for 3D gaze point
                # (the projection of which coincides with 2D gaze provided by
                # the eye tracker)
                if gotPose:
                    gazeWorld = utils.gazeToPlane(gaze,pose,cameraRotation,cameraPosition)

                    # draw gazes on video and reference image
                    gazeWorld.drawOnWorldVideo(frame, cameraMatrix, distCoeff, subPixelFac)
                    gazeWorld.drawOnReferencePlane(refImg, reference, subPixelFac)
        
        # annotate frame
        frame_ts  = i2t.get(frame_idx)
        cv2.rectangle(frame,(0,int(height)),(int(0.25*width),int(height)-30),(0,0,0),-1)
        cv2.putText(frame, '%6.3f [%6d]' % (frame_ts/1000.,frame_idx), (0, int(height)-5), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255))

        # store to file
        img = Image(plane_buffers=[frame.flatten().tobytes()], pix_fmt='bgr24', size=(int(width), int(height)))
        vidOutScene.write_frame(img=img, pts=frame_idx/fps)
        img = Image(plane_buffers=[refImg.flatten().tobytes()], pix_fmt='bgr24', size=(reference.width, reference.height))
        vidOutBoard.write_frame(img=img, pts=frame_idx/fps)


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
    vidOutScene.close()
    vidOutBoard.close()
    cv2.destroyAllWindows()

    # if ffmpeg is on path, add audio to scene and optionally board video
    if shutil.which('ffmpeg') is not None:
        todo = [inputDir / 'detectOutput_scene.mp4']
        if gAddAudioToBoardVideo:
            todo.append(inputDir / 'detectOutput_board.mp4')

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



if __name__ == '__main__':
    basePath = Path(__file__).resolve().parent
    for d in (basePath / 'data' / 'preprocced').iterdir():
        if d.is_dir():
            if process(d,basePath):
                break