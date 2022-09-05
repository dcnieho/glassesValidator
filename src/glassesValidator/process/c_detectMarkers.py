from pathlib import Path

import cv2
import numpy as np
import csv
import time

from .. import config
from .. import utils


def process(inputDir,configDir=None,visualizeDetection=False,showRejectedMarkers=False,FPSFac=1):
    # if visualizeDetection, draw each frame and overlay info about detected markers and board
    # if showRejectedMarkers, rejected marker candidates are also drawn on frame. Possibly useful for debug
    inputDir  = Path(inputDir)
    if configDir is not None:
        configDir = pathlib.Path(configDir)

    print('processing: {}'.format(inputDir.name))

    # open file with information about Aruco marker and Gaze target locations
    validationSetup = config.getValidationSetup(configDir)
    
    # open video file, query it for size
    inVideo = inputDir / 'worldCamera.mp4'
    if not inVideo.is_file():
        inVideo = inputDir / 'worldCamera.avi'
    cap    = cv2.VideoCapture(str(inVideo))
    if not cap.isOpened():
        raise RuntimeError('the file "{}" could not be opened'.format(str(inVideo)))
    width  = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ifi    = 1000./cap.get(cv2.CAP_PROP_FPS)/FPSFac
    
    # get info about markers on our board
    reference       = utils.Reference(configDir, validationSetup)
    centerTarget    = reference.targets[validationSetup['centerTarget']].center
    # turn into aruco board object to be used for pose estimation
    referenceBoard  = reference.getArucoBoard()
    
    # setup aruco marker detection
    parameters = cv2.aruco.DetectorParameters_create()
    parameters.markerBorderBits       = validationSetup['markerBorderBits']
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    # get camera calibration info
    cameraMatrix,distCoeff = utils.readCameraCalibrationFile(inputDir / "calibration.xml")[0:2]
    hasCameraMatrix = cameraMatrix is not None
    hasDistCoeff    = distCoeff is not None

    # get interval coded to be analyzed, if any
    analyzeFrames   = utils.readMarkerIntervalsFile(inputDir / "markerInterval.tsv")
    hasAnalyzeFrames= analyzeFrames is not None

    # prep output file
    csv_file = open(inputDir / 'boardPose.tsv', 'w', newline='')
    csv_writer = csv.writer(csv_file, delimiter='\t')
    header = utils.BoardPose.getWriteHeader()
    csv_writer.writerow(header)

    frame_idx = -1
    stopAllProcessing = False
    armLength = reference.markerSize/2 # arms of axis are half a marker long
    subPixelFac = 8   # for sub-pixel positioning
    while True:
        startTime = time.perf_counter()

        # process frame-by-frame
        ret, frame = cap.read()
        frame_idx += 1
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
                # no need to process this frame
                continue

        # detect markers, undistort
        corners, ids, rejectedImgPoints = \
            cv2.aruco.detectMarkers(frame, reference.aruco_dict, parameters=parameters)
        recoveredIds = None
        
        if np.all(ids != None):
            if len(ids) >= validationSetup['minNumMarkers']:
                pose = utils.BoardPose(frame_idx)
                
                # get camera pose
                if hasCameraMatrix and hasDistCoeff:
                    # Refine detected markers (eliminates markers not part of our board, adds missing markers to the board)
                    corners, ids, rejectedImgPoints, recoveredIds = utils.arucoRefineDetectedMarkers(
                            image = frame, board = referenceBoard,
                            detectedCorners = corners, detectedIds = ids, rejectedCorners = rejectedImgPoints,
                            cameraMatrix = cameraMatrix, distCoeffs = distCoeff)

                    pose.nMarkers, rVec, tVec = cv2.aruco.estimatePoseBoard(corners, ids, referenceBoard, cameraMatrix, distCoeff, np.empty(1), np.empty(1))
                
                    if pose.nMarkers>0:
                        # set pose
                        pose.setPose(rVec,tVec)
                        # and draw if wanted
                        if visualizeDetection:
                            # draw axis indicating board pose (origin and orientation)
                            utils.drawOpenCVFrameAxis(frame, cameraMatrix, distCoeff, pose.rVec, pose.tVec, armLength, 3, subPixelFac)

                # also get homography (direct image plane to plane in world transform). Use undistorted marker corners
                if hasCameraMatrix and hasDistCoeff:
                    cornersU = [cv2.undistortPoints(x, cameraMatrix, distCoeff, P=cameraMatrix) for x in corners]
                else:
                    cornersU = corners
                H, status = utils.estimateHomography(reference.knownMarkers, cornersU, ids)

                if status:
                    pose.hMat = H
                    if visualizeDetection:
                        # find where target is expected to be in the image
                        iH = np.linalg.inv(pose.hMat)
                        target = utils.applyHomography(iH, centerTarget[0], centerTarget[1])
                        if hasCameraMatrix and hasDistCoeff:
                            target = utils.distortPoint(*target, cameraMatrix, distCoeff)
                        # draw target location on image
                        if target[0] >= 0 and target[0] < width and target[1] >= 0 and target[1] < height:
                            utils.drawOpenCVCircle(frame, target, 3, (0,0,0), -1, subPixelFac)

                if pose.nMarkers>0 or status:
                    csv_writer.writerow( pose.getWriteData() )

            # if any markers were detected, draw where on the frame
            if visualizeDetection:
                utils.drawArucoDetectedMarkers(frame, corners, ids, subPixelFac=subPixelFac, specialHighlight=[recoveredIds,(255,255,0)])

        # for debug, can draw rejected markers on frame
        if visualizeDetection and showRejectedMarkers:
            cv2.aruco.drawDetectedMarkers(frame, rejectedImgPoints, None, borderColor=(211,0,148))
                
        if visualizeDetection:
            cv2.imshow(inputDir.name,frame)
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
                cv2.imwrite(str(inputDir / ('detect_frame_%d.png' % frame_idx)), frame)
        elif (frame_idx)%100==0:
            print('  frame {}'.format(frame_idx))

    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()

    return stopAllProcessing


if __name__ == '__main__':
    basePath = Path(__file__).resolve().parent
    for d in (basePath / 'data' / 'preprocced').iterdir():
        if d.is_dir():
            if process(d,basePath):
                break
