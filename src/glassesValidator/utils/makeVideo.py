import shutil
import os
import pathlib

import cv2
import numpy as np
import threading

import sys
isMacOS = sys.platform.startswith("darwin")
if isMacOS:
    import AppKit

from glassesTools import aruco, gaze_headref, ocv, timestamps, transforms
from glassesTools.video_gui import GUI, generic_tooltip_drawer

from .. import config
from .. import utils

from ffpyplayer.writer import MediaWriter
from ffpyplayer.pic import Image
import ffpyplayer.tools
from fractions import Fraction


stopAllProcessing = False
def process(working_dir, config_dir=None, show_rejected_markers=False, add_audio_to_poster_video=False, show_visualization=False):
    # if show_rejected_markers, rejected ArUco marker candidates are also drawn on the video. Possibly useful for debug
    # if add_audio_to_poster_video, audio is added to poster video, not only to the scene video
    # if show_visualization, the generated video is shown as it is created in a viewer
    working_dir  = pathlib.Path(working_dir)
    if config_dir is not None:
        config_dir = pathlib.Path(config_dir)

    print('processing: {}'.format(working_dir.name))

    if show_visualization:
        # We run processing in a separate thread (GUI needs to be on the main thread for OSX, see https://github.com/pthom/hello_imgui/issues/33)
        gui = GUI(use_thread = False)
        key_tooltip = {
            "q": "Quit",
            'n': 'Next'
        }
        gui.set_interesting_keys(list(key_tooltip.keys()))
        gui.register_draw_callback('status',lambda: generic_tooltip_drawer(key_tooltip))
        main_win_id = gui.add_window(working_dir.name)

        proc_thread = threading.Thread(target=do_the_work, args=(working_dir, config_dir, gui, main_win_id, show_rejected_markers, add_audio_to_poster_video, show_visualization))
        proc_thread.start()
        gui.start()
        proc_thread.join()
    else:
        do_the_work(working_dir, config_dir, None, None, show_rejected_markers, add_audio_to_poster_video, show_visualization)
    return stopAllProcessing

def do_the_work(working_dir, config_dir, gui, main_win_id, show_rejected_markers, add_audio_to_poster_video, show_visualization):
    global stopAllProcessing

    # open file with information about Aruco marker and Gaze target locations
    validationSetup = config.get_validation_setup(config_dir)

    # open input video file, query it for size
    inVideo = working_dir / 'worldCamera.mp4'
    if not inVideo.is_file():
        inVideo = working_dir / 'worldCamera.avi'
    vidIn   = ocv.CV2VideoReader(inVideo, timestamps.from_file(working_dir / 'frameTimestamps.tsv'))
    width   = vidIn.get_prop(cv2.CAP_PROP_FRAME_WIDTH)
    height  = vidIn.get_prop(cv2.CAP_PROP_FRAME_HEIGHT)
    fps     = vidIn.get_prop(cv2.CAP_PROP_FPS)

    # get info about markers on our poster
    poster      = config.poster.Poster(config_dir, validationSetup)
    # turn into aruco board object to be used for pose estimation
    arucoBoard  = poster.get_aruco_board()
    # get poster image width, height
    ref_img     = poster.get_ref_image(400)
    ref_height, ref_width, _ = ref_img.shape

    # prep output video files
    # get which pixel format
    codec    = ffpyplayer.tools.get_format_codec(fmt='mp4')
    pix_fmt  = ffpyplayer.tools.get_best_pix_fmt('bgr24',ffpyplayer.tools.get_supported_pixfmts(codec))
    fpsFrac  = Fraction(fps).limit_denominator(10000).as_integer_ratio()
    # scene video
    out_opts = {'pix_fmt_in':'bgr24', 'pix_fmt_out':pix_fmt, 'width_in':int(  width  ), 'height_in':int(  height  ),'frame_rate':fpsFrac}
    vidOutScene  = MediaWriter(str(working_dir / 'detectOutput_scene.mp4') , [out_opts], overwrite=True)
    # poster video
    out_opts = {'pix_fmt_in':'bgr24', 'pix_fmt_out':pix_fmt, 'width_in':int(ref_width), 'height_in':int(ref_height),'frame_rate':fpsFrac}
    vidOutPoster = MediaWriter(str(working_dir / 'detectOutput_poster.mp4'), [out_opts], overwrite=True)

    # get camera calibration info
    cameraParams = ocv.CameraParams.readFromFile(working_dir / "calibration.xml")

    # setup aruco marker detection
    detector = aruco.ArUcoDetector(arucoBoard.getDictionary(), {'markerBorderBits': validationSetup['markerBorderBits']})
    detector.set_board(arucoBoard)
    detector.set_intrinsics(cameraParams)

    # Read gaze data
    gazes = gaze_headref.read_dict_from_file(working_dir / 'gazeData.tsv')[0]

    # get interval coded to be analyzed, if available
    analyzeFrames = utils.readMarkerIntervalsFile(working_dir / "markerInterval.tsv")
    if analyzeFrames is None:
        analyzeFrames = []

    frame_idx = -1
    armLength = poster.marker_size/2 # arms of axis are half a marker long
    subPixelFac = 8   # for sub-pixel positioning
    stopAllProcessing = False
    hasRequestedFocus = not isMacOS # False only if on Mac OS, else True since its a no-op
    while True:
        # process frame-by-frame
        done, frame, frame_idx, frame_ts = vidIn.read_frame(report_gap=True)
        # TODO: if there is a discontinuity, fill in the missing frames so audio stays in sync
        # check if we're done
        if done:
            break
        vidIn.report_frame()

        if frame is None:
            # we don't have a valid frame, use a fully black frame
            frame = np.zeros((int(height),int(width),3), np.uint8)   # black image
        refImg = poster.get_ref_image(ref_width)

        # detect markers
        pose, detect_dict = detector.detect_and_estimate(frame, frame_idx, min_num_markers=validationSetup['minNumMarkers'])
        # draw detection and pose
        detector.visualize(frame, pose, detect_dict, armLength, subPixelFac, show_rejected_markers)

        # process gaze
        if frame_idx in gazes:
            for gaze in gazes[frame_idx]:
                # draw gaze point on scene video
                gaze.draw(frame, cameraParams, subPixelFac)

                # if we have pose information, figure out where gaze vectors
                # intersect with poster. Do same for 3D gaze point
                # (the projection of which coincides with 2D gaze provided by
                # the eye tracker)
                if pose.pose_N_markers>0 or pose.homography_N_markers>0:
                    gazePoster = transforms.gazeToPlane(gaze, pose, cameraParams)

                    # draw gazes on video and poster
                    gazePoster.drawOnWorldVideo(frame, cameraParams, subPixelFac)
                    gazePoster.drawOnPlane(refImg, poster, subPixelFac)

        # annotate frame
        analysisIntervalIdx = None
        for f in range(0,len(analyzeFrames)-1,2):   # -1 to make sure we don't try incomplete intervals
            if frame_idx>=analyzeFrames[f] and frame_idx<=analyzeFrames[f+1]:
                analysisIntervalIdx = f
        frameClr = (0,0,255) if analysisIntervalIdx is not None else (0,0,0)

        text = '%6.3f [%6d] (%s markers)' % (frame_ts/1000.,frame_idx, pose.pose_N_markers)
        textSize,baseline = cv2.getTextSize(text,cv2.FONT_HERSHEY_PLAIN,2,2)
        cv2.rectangle(frame,(0,int(height)),(textSize[0]+2,int(height)-textSize[1]-baseline-5), frameClr, -1)
        cv2.putText(frame, (text), (2, int(height)-5), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255),2)

        # store to file
        img = Image(plane_buffers=[frame.flatten().tobytes()], pix_fmt='bgr24', size=(int(width), int(height)))
        vidOutScene.write_frame(img=img, pts=frame_idx/fps)
        img = Image(plane_buffers=[refImg.flatten().tobytes()], pix_fmt='bgr24', size=(ref_width, ref_height))
        vidOutPoster.write_frame(img=img, pts=frame_idx/fps)


        if show_visualization:
            gui.update_image(frame, frame_ts/1000., frame_idx, window_id = main_win_id)

            if not hasRequestedFocus:
                AppKit.NSApplication.sharedApplication().activateIgnoringOtherApps_(1)
                hasRequestedFocus = True

            keys = gui.get_key_presses()
            if 'q' in keys:
                # quit fully
                stopAllProcessing = True
                break
            if 'n' in keys:
                # goto next
                break

            closed, = gui.get_state()
            if closed:
                stopAllProcessing = True
                break

    vidOutScene.close()
    vidOutPoster.close()
    if show_visualization:
        gui.stop()

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
            cmd_str = ' '.join(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', '"'+str(tempName)+'"', '-i', '"'+str(inVideo)+'"', '-vcodec', 'copy', '-acodec', 'copy', '-map', '0:v:0', '-map', '1:a:0?', '"'+str(f)+'"'])
            os.system(cmd_str)
            # clean up
            if f.exists():
                tempName.unlink(missing_ok=True)

    return stopAllProcessing