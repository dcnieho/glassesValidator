from pathlib import Path

import cv2
import numpy as np
import csv
import math
from matplotlib import colors
import time
import utils

gVisualizeDetection = False     # if true, draw each frame and overlay info about detected markers and board
gShowRejectedMarkers= False     # if true, rejected marker candidates are also drawn on frame. Possibly useful for debug
gFPSFac             = 1


def storeReferenceBoard(inputDir,validationSetup,knownMarkers, aruco_dict,markerBBox):
    referenceBoard = utils.getReferenceBoard(knownMarkers, aruco_dict, unRotateMarkers = True)
    # get image with markers
    bboxExtents    = [markerBBox[2]-markerBBox[0], math.fabs(markerBBox[3]-markerBBox[1])]  # math.fabs to deal with bboxes where (-,-) is bottom left
    aspectRatio    = bboxExtents[0]/bboxExtents[1]
    refBoardWidth  = validationSetup['referenceBoardWidth']
    refBoardHeight = math.ceil(refBoardWidth/aspectRatio)
    margin         = 1  # always 1 pixel, anything else behaves strangely (markers are drawn over margin as well)
    refBoardImage  = cv2.cvtColor(
        cv2.aruco.drawPlanarBoard(
            referenceBoard,(refBoardWidth+2*margin,refBoardHeight+2*margin),margin,validationSetup['markerBorderBits']),
        cv2.COLOR_GRAY2RGB
    )
    # cut off this 1-pix margin
    assert refBoardImage.shape[0]==refBoardHeight+2*margin,"Output image height is not as expected"
    assert refBoardImage.shape[1]==refBoardWidth +2*margin,"Output image width is not as expected"
    refBoardImage  = refBoardImage[1:-1,1:-1,:]
    # walk through all markers, if any are supposed to be rotated, do so
    minX =  np.inf
    maxX = -np.inf
    minY =  np.inf
    maxY = -np.inf
    rots = []
    cornerPointsU = []
    for key in knownMarkers:
        if not key.startswith('t'):
            cornerPoints = np.vstack(knownMarkers[key].corners).astype('float32')
            cornerPointsU.append(utils.getMarkerUnrotated(cornerPoints, knownMarkers[key].rot))
            rots.append(knownMarkers[key].rot)
            minX = np.min(np.hstack((minX,cornerPoints[:,0])))
            maxX = np.max(np.hstack((maxX,cornerPoints[:,0])))
            minY = np.min(np.hstack((minY,cornerPoints[:,1])))
            maxY = np.max(np.hstack((maxY,cornerPoints[:,1])))
    if np.any(np.array(rots)!=0):
        # determine where the markers are placed
        sizeX = maxX - minX
        sizeY = maxY - minY
        xReduction = sizeX / float(refBoardImage.shape[1])
        yReduction = sizeY / float(refBoardImage.shape[0])
        if xReduction > yReduction:
            nRows = int(sizeY / xReduction);
            yMargin = (refBoardImage.shape[0] - nRows) / 2;
            xMargin = 0
        else:
            nCols = int(sizeX / yReduction);
            xMargin = (refBoardImage.shape[1] - nCols) / 2;
            yMargin = 0
        
        for r,cpu in zip(rots,cornerPointsU):
            if r != 0:
                # figure out where marker is
                cpu -= np.array([[minX,minY]])
                cpu[:,0] =       cpu[:,0] / sizeX  * float(refBoardImage.shape[1]) + xMargin
                cpu[:,1] = (1. - cpu[:,1] / sizeY) * float(refBoardImage.shape[0]) + yMargin
                sz = np.min(cpu[2,:]-cpu[0,:])
                # get marker
                cpu = np.floor(cpu)
                idxs = np.floor([cpu[0,1], cpu[0,1]+sz, cpu[0,0], cpu[0,0]+sz]).astype('int')
                marker = refBoardImage[idxs[0]:idxs[1], idxs[2]:idxs[3]]
                # rotate (opposite because coordinate system) and put back
                if r==-90:
                    marker = cv2.rotate(marker, cv2.ROTATE_90_CLOCKWISE)
                elif r==90:
                    marker = cv2.rotate(marker, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif r==180:
                    marker = cv2.rotate(marker, cv2.ROTATE_180)

                refBoardImage[idxs[0]:idxs[1], idxs[2]:idxs[3]] = marker

    # add targets
    subPixelFac = 8   # for sub-pixel positioning
    for key in knownMarkers:
        if key.startswith('t'):
            # 1. determine position on image
            circlePos = utils.toImagePos(*knownMarkers[key].center, markerBBox,[refBoardWidth,refBoardHeight])

            # 2. draw
            clr = tuple([int(i*255) for i in colors.to_rgb(knownMarkers[key].color)[::-1]])  # need BGR color ordering
            utils.drawOpenCVCircle(refBoardImage, circlePos, 15, clr, -1, subPixelFac)

    cv2.imwrite(str(inputDir / 'referenceBoard.png'), refBoardImage)


def process(inputDir,basePath):
    global gVisualizeDetection
    global gShowRejectedMarkers
    global gFPSFac

    print('processing: {}'.format(inputDir.name))

    configDir = basePath / "config"
    # open file with information about Aruco marker and Gaze target locations
    validationSetup = utils.getValidationSetup(configDir)
    
    # open video file, query it for size
    inVideo = inputDir / 'worldCamera.mp4'
    if not inVideo.is_file():
        inVideo = inputDir / 'worldCamera.avi'
    cap    = cv2.VideoCapture(str(inVideo))
    if not cap.isOpened():
        raise RuntimeError('the file "{}" could not be opened'.format(str(inVideo)))
    width  = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ifi    = 1000./cap.get(cv2.CAP_PROP_FPS)/gFPSFac
    
    # get info about markers on our board
    # Aruco markers have numeric keys, gaze targets have keys starting with 't'
    aruco_dict   = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
    knownMarkers, markerBBox = utils.getKnownMarkers(configDir, validationSetup)
    centerTarget = knownMarkers['t%d'%validationSetup['centerTarget']].center
    
    # turn into aruco board object to be used for pose estimation
    referenceBoard = utils.getReferenceBoard(knownMarkers, aruco_dict)
    # store image of reference board to file
    storeReferenceBoard(inputDir,validationSetup,knownMarkers,aruco_dict,markerBBox)
    
    # setup aruco marker detection
    parameters = cv2.aruco.DetectorParameters_create()
    parameters.markerBorderBits       = validationSetup['markerBorderBits']
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX;

    # get camera calibration info
    cameraMatrix,distCoeff = utils.getCameraCalibrationInfo(inputDir / "calibration.xml")[0:2]
    hasCameraMatrix = cameraMatrix is not None
    hasDistCoeff    = distCoeff is not None

    # prep output file
    csv_file = open(inputDir / 'boardPose.tsv', 'w', newline='')
    csv_writer = csv.writer(csv_file, delimiter='\t')
    header = ['frame_idx']
    header.append('poseNMarker')
    header.extend(utils.getXYZLabels(['poseRvec','poseTvec']))
    header.extend(['transformation[%d,%d]' % (r,c) for r in range(3) for c in range(3)])
    csv_writer.writerow( header )

    frame_idx = 0
    stopAllProcessing = False
    armLength = 2.*math.tan(math.radians(.5))*validationSetup['distance']*10*validationSetup['markerSide']/2 # arms of axis are half a marker long
    subPixelFac = 8   # for sub-pixel positioning
    while True:
        startTime = time.perf_counter()
        # process frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # detect markers, undistort
        corners, ids, rejectedImgPoints = \
            cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
        
        if np.all(ids != None):
            if len(ids) >= validationSetup['minNumMarkers']:
                # get camera pose
                if hasCameraMatrix and hasDistCoeff:
                    nMarkersUsed, rVec, tVec = cv2.aruco.estimatePoseBoard(corners, ids, referenceBoard, cameraMatrix, distCoeff)
                else:
                    nMarkersUsed = 0
                
                writeDat = [frame_idx]
                if nMarkersUsed>0:
                    # draw axis indicating board pose (origin and orientation)
                    if gVisualizeDetection and nMarkersUsed>0:
                        utils.drawOpenCVFrameAxis(frame, cameraMatrix, distCoeff, rVec, tVec, armLength, 3, subPixelFac)

                    # store pose to file
                    writeDat.append( nMarkersUsed )
                    writeDat.extend( rVec.flatten() )
                    writeDat.extend( tVec.flatten() )
                else:
                    writeDat.extend([math.nan for x in range(7)])

                # also get homography (direct image plane to plane in world transform). Use undistorted marker corners
                if hasCameraMatrix and hasDistCoeff:
                    cornersU = [cv2.undistortPoints(x, cameraMatrix, distCoeff, P=cameraMatrix) for x in corners]
                else:
                    cornersU = corners
                H, status = utils.estimateHomography(knownMarkers, cornersU, ids)

                if status:
                    # find where target is expected to be in the image
                    iH = np.linalg.inv(H)
                    target = utils.applyHomography(iH, centerTarget[0], centerTarget[1])
                    if hasCameraMatrix and hasDistCoeff:
                        target = utils.distortPoint(*target, cameraMatrix, distCoeff)
                    # draw target location on image
                    if target[0] >= 0 and target[0] < width and target[1] >= 0 and target[1] < height:
                        utils.drawOpenCVCircle(frame, target, 3, (0,0,0), -1, subPixelFac)

                    # write transform to file
                    writeDat.extend( H.flatten() )
                else:
                    writeDat.extend([math.nan for x in range(9)])

                if nMarkersUsed>0 or status:
                    csv_writer.writerow( writeDat )

            # if any markers were detected, draw where on the frame
            if gVisualizeDetection:
                utils.drawArucoDetectedMarkers(frame, corners, ids, subPixelFac=subPixelFac)

        # for debug, can draw rejected markers on frame
        if gVisualizeDetection and gShowRejectedMarkers:
            cv2.aruco.drawDetectedMarkers(frame, rejectedImgPoints, None, borderColor=(211,0,148))
                
        if gVisualizeDetection:
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
        
        frame_idx += 1

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
