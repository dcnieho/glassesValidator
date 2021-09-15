#!/usr/bin/python3

import sys
from pathlib import Path
import math
import bisect

import cv2
import numpy as np
import pandas as pd
import csv
import time

import utils

gShowReference  = True
gFPSFac         = 1

class Gaze:
    def __init__(self, ts, x, y):
        self.ts = ts
        self.x = x
        self.y = y
        self.xCm = -1
        self.yCm = -1


    def draw(self, img):
        if not math.isnan(self.x):
            x = int(self.x)
            y = int(self.y)
            g = 255
            r = 255 * (1 - g)
            b = 0
            cv2.circle(img, (x, y), 15, (b, g, r), 5)


class Reference:
    def __init__(self, fileName, markerDir, validationSetup):
        self.img = cv2.imread(fileName, cv2.IMREAD_COLOR)
        self.scale = 400./self.img.shape[0]
        self.img = cv2.resize(self.img, None, fx=self.scale, fy=self.scale, interpolation = cv2.INTER_AREA)
        self.height, self.width, self.channels = self.img.shape
        # get marker info
        _, self.bbox = utils.getKnownMarkers(markerDir, validationSetup)

    def getImgCopy(self):
        return self.img.copy()

    def error(self, x, y):
        distCm = 150
        vgaze = np.array( [ x - self.gtCm[0], y - self.gtCm[1], distCm ] )
        vgt = np.array( [ 0, 0, distCm ] )
        return angle_between(vgaze, vgt), vgaze[0], vgaze[1]

    def draw(self, img, x, y, subPixelFac):
        xy = tuple(utils.toImagePos(x,y,self.bbox,[self.width, self.height],subPixelFac=subPixelFac))
        cv2.circle(img, xy, 8*subPixelFac, (0, 0 ,0), -1, lineType=cv2.LINE_AA, shift=int(math.log2(subPixelFac)))
        cv2.circle(img, xy, 4*subPixelFac, (0,255,0), -1, lineType=cv2.LINE_AA, shift=int(math.log2(subPixelFac)))

class Idx2Timestamp:
    def __init__(self, fileName):
        self.timestamps = []
        with open(fileName, 'r' ) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for entry in reader:
                self.timestamps.append(float(entry['timestamp']))

    def get(self, idx):
        if idx < len(self.timestamps):
            return self.timestamps[int(idx)]
        else:
            sys.stderr.write("[WARNING] %d requested (from %d)\n" % ( idx, len(self.timestamps) ) )
            return self.timestamps[-1]


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def process(inputDir,basePath):
    global gShowReference
    global gFPSFac
    
    configDir = basePath / "config"
    # open file with information about Aruco marker and Gaze target locations
    validationSetup = utils.getValidationSetup(configDir)

    cv2.namedWindow("frame")
    if gShowReference:
        cv2.namedWindow("reference")

    reference = Reference(str(inputDir / 'referenceBoard.png'), configDir, validationSetup)

    i2t = Idx2Timestamp(str(inputDir / 'frame_timestamps.tsv'))

    fs = cv2.FileStorage(str(inputDir / "calibration.xml"), cv2.FILE_STORAGE_READ)
    cameraMatrix = fs.getNode("cameraMatrix").mat()
    distCoeff    = fs.getNode("distCoeff").mat()
    fs.release()

    cap         = cv2.VideoCapture(str(inputDir / 'worldCamera.mp4'))
    width       = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    ifi         = 1000./cap.get(cv2.CAP_PROP_FPS)/gFPSFac

    # Read gaze data
    gazes = {}
    with open( str(inputDir / 'gazeData.tsv'), 'r' ) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for entry in reader:
            frame_idx = int(float(entry['frame_idx']))
            try:
                ts = float(entry['timestamp'])
                gx = float(entry['gaze_pos_x']) * width
                gy = float(entry['gaze_pos_y']) * height
                gaze = Gaze(ts, gx, gy)

                if frame_idx in gazes:
                    gazes[frame_idx].append(gaze)
                else:
                    gazes[frame_idx] = [gaze]
            except Exception as e:
                sys.stderr.write(str(e) + '\n')
                sys.stderr.write('[WARNING] Problematic entry: %s\n' % (entry) )


    # Read transformation and location of center target
    centerTarget = {}
    transformation = {}
    temp = pd.read_csv(str(inputDir / 'transformations.tsv'), delimiter='\t')
    ctCols    = [col for col in temp.columns if 'centerTarget' in col]
    transCols = [col for col in temp.columns if 'transformation' in col]
    for idx, row in temp.iterrows():
        frame_idx = int(row['frame_idx'])

        # get center target pixel position in undistorted image
        centerTarget[frame_idx] = row[ctCols].values

        # transformation from undistorted image to reference (in mm!)
        transformation[frame_idx] = row[transCols].values.reshape(3,3)

    csv_file = open(str(inputDir / 'report.tsv'), 'w')
    csv_writer = csv.writer(csv_file, delimiter='\t', lineterminator='\n')
    csv_writer.writerow( ['frame_idx', 'timestamp', 'errorDeg', 'dxCm', 'dyCm', 'gaze_ts', 'gaze_x', 'gaze_y'] ) 

    subPixelFac = 8   # for sub-pixel positioning
    stopAllProcessing = False
    while(True):
        startTime = time.perf_counter()
        ret, frame = cap.read()
        if not ret: # we reached the end; done
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
                if frame_idx in transformation:
                    ux = gaze.x
                    uy = gaze.y
                    ux, uy = utils.undistortPoint( gaze.x, gaze.y, cameraMatrix, distCoeff)
                    (gaze.xCm, gaze.yCm) = utils.transform(transformation[frame_idx], ux, uy)
                    reference.draw(refImg, gaze.xCm, gaze.yCm, subPixelFac)
                    #angleDeviation, dxCm, dyCm = reference.error(gaze.xCm, gaze.yCm)
                    #print('%10d\t%10.3f\t%10.3f\t%10.3f' % ( frame_idx, frame_ts, gaze.confidence, angleDeviation ) )
                    #csv_writer.writerow([ frame_idx, frame_ts, angleDeviation, dxCm, dyCm, gaze.ts, gaze.x, gaze.y] )

        if gShowReference:
            cv2.imshow("reference", refImg)

        if frame_idx in centerTarget:
            x = int(round(centerTarget[frame_idx][0]*subPixelFac))
            y = int(round(centerTarget[frame_idx][1]*subPixelFac))
            cv2.line(frame, (x,0), (x,int(height*subPixelFac)),(0,255,0),1, lineType=cv2.LINE_AA, shift=3)
            cv2.line(frame, (0,y), (int(width*subPixelFac),y) ,(0,255,0),1, lineType=cv2.LINE_AA, shift=3)
            cv2.circle(frame, (x,y), 3*subPixelFac, (0,255,0), -1, lineType=cv2.LINE_AA, shift=int(math.log2(subPixelFac)))

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
            cv2.imwrite(str(inputDir / ('calc_frame_%d.png' % frame_idx)), frame)
            
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
