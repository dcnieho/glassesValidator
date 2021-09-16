from pathlib import Path

import cv2
import numpy as np
import csv
import math
from matplotlib import colors
import time
import utils

gFPSFac         = 1


def storeReferenceBoard(referenceBoard,inputDir,validationSetup,knownMarkers,markerBBox):
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
    # add targets
    subPixelFac = 8   # for sub-pixel positioning
    for key in knownMarkers:
        if key.startswith('t'):
            # 1. determine position on image
            circlePos = utils.toImagePos(*knownMarkers[key].center, markerBBox,[refBoardWidth,refBoardHeight], subPixelFac=subPixelFac)

            # 2. draw
            clr = tuple([int(i*255) for i in colors.to_rgb(knownMarkers[key].color)[::-1]])  # need BGR color ordering
            refBoardImage = cv2.circle(refBoardImage, tuple(circlePos), 15*subPixelFac, clr, -1, lineType=cv2.LINE_AA, shift=int(math.log2(subPixelFac)))

    cv2.imwrite(str(inputDir / 'referenceBoard.png'), refBoardImage)


def estimateTransform(known, detectedCorners, detectedIDs):
    # collect matching corners in image and in world
    pts_src = []
    pts_dst = []
    for i in range(0, len(detectedIDs)):
        key = '%d' % detectedIDs[i]
        if key in known:
            pts_src.extend( detectedCorners[i][0] )
            pts_dst.extend(    known[key].corners )

    if len(pts_src) < 4:
        return None, False

    # compute Homography
    pts_src = np.float32(pts_src)
    pts_dst = np.float32(pts_dst)
    h, _ = cv2.findHomography(pts_src, pts_dst)

    return h, True


def process(inputDir,basePath):
    global gFPSFac

    configDir = basePath / "config"
    # open file with information about Aruco marker and Gaze target locations
    validationSetup = utils.getValidationSetup(configDir)
    
    # open video file, query it for size
    inVideo = str(inputDir / 'worldCamera.mp4')
    cap    = cv2.VideoCapture( inVideo )
    if not cap.isOpened():
        raise RuntimeError('the file "{}" could not be opened'.format(inVideo))
    width  = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ifi    = 1000./cap.get(cv2.CAP_PROP_FPS)/gFPSFac
    
    # get info about markers on our board
    # Aruco markers have numeric keys, gaze targets have keys starting with 't'
    aruco_dict   = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
    knownMarkers, markerBBox = utils.getKnownMarkers(configDir, validationSetup)
    centerTarget = knownMarkers['t%d'%validationSetup['centerTarget']].center
    
    # turn into aruco board object to be used for pose estimation
    boardCornerPoints = []
    ids = []
    for key in knownMarkers:
        if not key.startswith('t'):
            ids.append(int(key))
            boardCornerPoints.append(np.vstack(knownMarkers[key].corners).astype('float32'))
    boardCornerPoints = np.dstack(boardCornerPoints)        # list of 2D arrays -> 3D array
    boardCornerPoints = np.rollaxis(boardCornerPoints,-1)   # 4x2xN -> Nx4x2
    boardCornerPoints = np.pad(boardCornerPoints,((0,0),(0,0),(0,1)),'constant', constant_values=(0.,0.)) # Nx4x2 -> Nx4x3
    referenceBoard    = cv2.aruco.Board_create(boardCornerPoints, aruco_dict, np.array(ids))
    # store image of reference board to file
    storeReferenceBoard(referenceBoard,inputDir,validationSetup,knownMarkers,markerBBox)
    
    # setup aruco marker detection
    parameters = cv2.aruco.DetectorParameters_create()
    parameters.markerBorderBits       = validationSetup['markerBorderBits']
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX;

    # get camera calibration info
    fs = cv2.FileStorage(str(inputDir / "calibration.xml"), cv2.FILE_STORAGE_READ)
    cameraMatrix = fs.getNode("cameraMatrix").mat()
    distCoeff    = fs.getNode("distCoeff").mat()
    fs.release()

    # prep output file
    csv_file = open(str(inputDir / 'transformations.tsv'), 'w', newline='')
    csv_writer = csv.writer(csv_file, delimiter='\t')
    header = ['frame_idx']
    header.extend(['transformation[%d,%d]' % (r,c) for r in range(3) for c in range(3)])
    header.append('poseNMarker')
    header.extend(['poseRvec[%d]' % (v) for v in range(3)])
    header.extend(['poseTvec[%d]' % (v) for v in range(3)])
    header.extend(['centerTarget[%d]' % (v) for v in range(2)])
    csv_writer.writerow( header )

    frame_idx = 0
    stopAllProcessing = False
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
                # undistort markers, get homography (image to world transform)
                cornersU = [cv2.undistortPoints(x, cameraMatrix, distCoeff, P=cameraMatrix) for x in corners]
                H, status = estimateTransform(knownMarkers, cornersU, ids)
                if status:
                    # get camera pose
                    nMarkersUsed, Rvec, Tvec = cv2.aruco.estimatePoseBoard(corners, ids, referenceBoard, cameraMatrix, distCoeff)
                    
                    # find where target is expected to be in the image
                    iH = np.linalg.inv(H)
                    target = utils.transform(iH, centerTarget[0], centerTarget[1])
                    target = utils.distortPoint( *target, cameraMatrix, distCoeff)
                    
                    # draw target location on image
                    subPixelFac = 8   # for sub-pixel positioning
                    if target[0] >= 0 and target[0] < width and target[1] >= 0 and target[1] < height:
                        x = int(round(target[0]*subPixelFac))
                        y = int(round(target[1]*subPixelFac))
                        cv2.line(frame, (x,0), (x,int(height*subPixelFac)),(0,255,0),1, lineType=cv2.LINE_AA, shift=int(math.log2(subPixelFac)))
                        cv2.line(frame, (0,y), (int(width*subPixelFac),y) ,(0,255,0),1, lineType=cv2.LINE_AA, shift=int(math.log2(subPixelFac)))
                        cv2.circle(frame, (x,y), 3*subPixelFac, (0,255,0), -1, lineType=cv2.LINE_AA, shift=int(math.log2(subPixelFac)))
                       
                    # draw axis indicating board pose
                    if nMarkersUsed>0:
                        cv2.aruco.drawAxis(frame,cameraMatrix, distCoeff,Rvec, Tvec, 2*2.*math.tan(math.radians(.5))*validationSetup['distance']*10)    # arms of grid are 2 poster grid cells

                    # store homography, pose and target location to file
                    writeDat = [frame_idx]
                    writeDat.extend( H.flatten() )
                    writeDat.append( nMarkersUsed )
                    writeDat.extend( Rvec.flatten() )
                    writeDat.extend( Tvec.flatten() )
                    writeDat.extend( target )
                    csv_writer.writerow( writeDat )

            # if any markers were detected, draw where on the frame
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # for debug, can draw rejected markers on frame
        # cv2.aruco.drawDetectedMarkers(frame, rejectedImgPoints, None, borderColor=(211,0,148))
                

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
