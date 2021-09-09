#!/usr/bin/python3

import sys
import argparse
import os
import math
import bisect

import cv2
import numpy as np
import csv
from csv import DictReader

sys.path.insert(0, './markers')
from detect import transform
from detect import getKnownMarkers

gShowReference = True
gWaitTime = 1
gFrameSkipCount=120

class Gaze:
    def __init__(self, ts, x, y, confidence):
        self.ts = ts
        self.x = x
        self.y = y
        self.confidence = confidence
        self.ux = x
        self.uy = y
        self.xCm = -1
        self.yCm = -1


    def draw(self, img):
        x = int(self.x)
        y = int(self.y)
        g = 255 * self.confidence
        r = 255 * (1 - g)
        b = 0
        cv2.circle(img, (x, y), 15, (b, g, r), 5)



def angle_between(v1, v2):
    #(180.0 / math.pi) * math.atan2(np.linalg.norm(np.cross(v1,v2)), np.dot(v1,v2))

    cx = v1[1] * v2[2] - v1[2] * v2[1]
    cy = v1[2] * v2[0] - v1[0] * v1[2]
    cz = v1[0] * v2[1] - v1[1] * v2[0]
    cross = math.sqrt( cx * cx + cy * cy + cz * cz )
    dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2] 
    return (180.0 / math.pi) * math.atan2( cross, dot )


class Reference:
    def __init__(self, fileName):
        self.img = cv2.imread(fileName, cv2.IMREAD_COLOR)
        self.img = cv2.resize(self.img, (400,280))
        referencePoints = getKnownMarkers()
        self.height, self.width, self.channels = self.img.shape
        self.xScale = self.width / float(referencePoints['tr'].center[0])
        self.yScale = self.height / float(referencePoints['tr'].center[1])
        self.gtCm = [ float(referencePoints['gt'].center[0]), float(referencePoints['gt'].center[1]) ]

    def getImgCopy(self):
        return self.img.copy()

    def error(self, x, y):
        distCm = 150
        vgaze = np.array( [ x - self.gtCm[0], y - self.gtCm[1], distCm ] )
        vgt = np.array( [ 0, 0, distCm ] )
        return angle_between(vgaze, vgt), vgaze[0], vgaze[1]

    def draw(self, img, x, y):
        x = int(round(x * self.xScale))
        y = self.height - int(round(y * self.yScale))
        cv2.circle( img, (x,y), 8, (0,0,0), -1)
        cv2.circle( img, (x,y), 4, (0,255,0), -1)

class Idx2Timestamp:
    def __init__(self, fileName):
        self.timestamps = []
        with open(fileName, 'r' ) as f:
            reader = DictReader(f, delimiter='\t')
            for entry in reader:
                self.timestamps.append(1e-3*float(entry['timestamp']))

    def get(self, idx):
        if idx < len(self.timestamps):
            return self.timestamps[int(idx)]
        else:
            sys.stderr.write("[WARNING] %d requested (from %d)\n" % ( idx, len(self.timestamps) ) )
            return self.timestamps[-1]


def undistortPoints(x, y, cameraMatrix, distCoeffs):
    p = np.float32([[[x, y]]])
    dst = cv2.undistortPoints(p, cameraMatrix, distCoeffs, None, cameraMatrix)
    return dst[0][0][0], dst[0][0][1]

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def process(inputDir):
    global gShowReference
    global gWaitTime
    global gFrameSkipCount

    save = True

    cv2.namedWindow("frame")
    if gShowReference:
        cv2.namedWindow("reference")

    reference = Reference('reference.png')

    i2t = Idx2Timestamp( os.path.join(inputDir, 'frame_timestamps.tsv') )

    doUndistort = '/eyerec/' in inputDir or '/pupil-labs/' in inputDir or '/grip/' in inputDir

    fs = cv2.FileStorage("markers/calibration.xml", cv2.FILE_STORAGE_READ)
    cameraMatrix = fs.getNode("cameraMatrix").mat()
    distCoeffs = fs.getNode("distCoeffs").mat()

    cap = cv2.VideoCapture( os.path.join(inputDir, 'worldCamera.mp4') )
    width = float( cap.get(cv2.CAP_PROP_FRAME_WIDTH ) )
    height = float( cap.get(cv2.CAP_PROP_FRAME_HEIGHT ) )

    # Read gaze data
    gazes = {}
    with open( os.path.join(inputDir, 'gazeData_world.tsv'), 'r' ) as f:
        reader = DictReader(f, delimiter='\t')
        for entry in reader:
            frame_idx = int(float(entry['frame_idx']))
            confidence = float(entry['confidence'])
            try:
                ts = float(entry['timestamp'])
                gx = float(entry['norm_pos_x']) * width
                gy = float(entry['norm_pos_y']) * height
                gaze = Gaze(ts, gx, gy, confidence)
                if doUndistort:
                    gaze.ux, gaze.uy = undistortPoints(gaze.x, gaze.y, cameraMatrix, distCoeffs)

                if frame_idx in gazes:
                    gazes[frame_idx].append(gaze)
                else:
                    gazes[frame_idx] = [ gaze ]
            except Exception as e:
                sys.stderr.write(str(e) + '\n')
                sys.stderr.write('[WARNING] Problematic entry: %s\n' % (entry) )


    # Read ground truth and transformation
    gt = {}
    transformation = {}
    with open( os.path.join(inputDir, 'transformations.tsv'), 'r' ) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for entry in reader:
            frame_idx = int(entry['frame_idx'])

            # ground truth pixel position in undistorted image
            tmp = entry['gt'].split(',')
            gt[frame_idx] = ( float(tmp[0]), float(tmp[1]) )

            # transformation from undistorted image to reference (in cm!)
            tmp = entry['transformation'].split(',')
            values = []
            for i in range(0,9):
                values.append( float(tmp[i]) )
            transformation[frame_idx] = np.asarray(values).reshape(3,3)

    if save:
        csv_file = open(os.path.join(inputDir, 'report.tsv'), 'w')
        csv_writer = csv.writer(csv_file, delimiter='\t', lineterminator='\n')
        csv_writer.writerow( ['frame_idx', 'timestamp', 'confidence', 'errorDeg', 'dxCm', 'dyCm', 'gaze_ts', 'gaze_x', 'gaze_y'] ) 

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    while(True):
        ret, frame = cap.read()
        if not ret: # we reached the end; fix it
            break

        # CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
        next_frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES) 
        frame_idx = next_frame_idx - 1
        frame_ts  = i2t.get(frame_idx)

        refImg = reference.getImgCopy()
        if frame_idx in gazes:
            for gaze in gazes[frame_idx]:
                gaze.draw(frame)
                # if we can, already transform the gaze
                hasTransformation = frame_idx in transformation
                if hasTransformation:
                    ux = gaze.x
                    uy = gaze.y
                    if doUndistort:
                        ux, uy = undistortPoints( gaze.x, gaze.y, cameraMatrix, distCoeffs)
                    (gaze.xCm, gaze.yCm) = transform(transformation[frame_idx],
                            ux, uy)
                    reference.draw(refImg, gaze.xCm, gaze.yCm)
                    angleDeviation, dxCm, dyCm = reference.error(gaze.xCm, gaze.yCm)
                    #print('%10d\t%10.3f\t%10.3f\t%10.3f' % ( frame_idx, frame_ts, gaze.confidence, angleDeviation ) )
                    if save:
                        csv_writer.writerow([ frame_idx, frame_ts, gaze.confidence, angleDeviation, dxCm, dyCm, gaze.ts, gaze.x, gaze.y] )

        if gShowReference:
            cv2.imshow("reference", refImg)

        if frame_idx in gt:
            x = int(round(gt[frame_idx][0]))
            y = int(round(gt[frame_idx][1]))
            cv2.line(frame, (x,0), (x,int(height)),(0,255,0),1)
            cv2.line(frame, (0,y), (int(width),y),(0,255,0),1)

        cv2.rectangle(frame,(0,int(height)),(int(0.25*width),int(height)-30),(0,0,0),-1)
        cv2.putText(frame, '%8.2f [%6d]' % (frame_ts,frame_idx), (0, int(height)-5), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255))
        #cv2.imshow('frame',frame)
        cv2.waitKey(gWaitTime)

    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inputDir', help='path to the input dir')
    args = parser.parse_args()

    if not os.path.isdir(args.inputDir):
        print('Invalid input dir: {}'.format(args.inputDir))
        sys.exit()
    else:
        # run preprocessing on this data
        process(args.inputDir)
