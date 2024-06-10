import pathlib

import cv2
import threading

from glassesTools import drawing, gaze_headref, gaze_worldref, intervals, ocv, plane, recording, timestamps
from glassesTools.video_gui import GUI, generic_tooltip_drawer, qns_tooltip

from .. import config
from .. import utils


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
        gui.register_draw_callback('status',lambda: generic_tooltip_drawer(qns_tooltip()))
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

    utils.update_recording_status(working_dir, utils.Task.Gaze_Tranformed_To_Poster, utils.Status.Running)

    # get camera calibration info
    cameraParams      = ocv.CameraParams.readFromFile(working_dir / "calibration.xml")

    # Read gaze data
    gazes,maxFrameIdx = gaze_headref.read_dict_from_file(working_dir / 'gazeData.tsv')

    # Read camera pose w.r.t. poster
    poses       = plane.read_dict_from_file(working_dir / 'posterPose.tsv')

    # transform
    plane_gazes = gaze_worldref.gazes_head_to_world(poses, gazes, cameraParams)

    # store to file
    gaze_worldref.write_dict_to_file(plane_gazes, working_dir / 'gazePosterPos.tsv', skip_missing=True)

    utils.update_recording_status(working_dir, utils.Task.Gaze_Tranformed_To_Poster, utils.Status.Finished)


    # done if no visualization wanted
    if not gui is not None:
        return False

    # prep visualizations
    # get info about recording
    recInfo         = recording.Recording.load_from_json(working_dir)
    # open file with information about ArUco marker and Gaze target locations
    validationSetup = config.get_validation_setup(config_dir)
    poster          = config.poster.Poster(config_dir, validationSetup)

    cap             = ocv.CV2VideoReader(recInfo.get_scene_video_path(), timestamps.from_file(working_dir / 'frameTimestamps.tsv'))
    width           = cap.get_prop(cv2.CAP_PROP_FRAME_WIDTH)
    height          = cap.get_prop(cv2.CAP_PROP_FRAME_HEIGHT)

    # get interval coded to be analyzed, if any
    analyzeFrames   = utils.readMarkerIntervalsFile(working_dir / "markerInterval.tsv")

    subPixelFac = 8   # for sub-pixel positioning
    stopAllProcessing = False
    for frame_idx in range(maxFrameIdx+1):
        done, frame, frame_idx, frame_ts = cap.read_frame(report_gap=True)
        if done or (show_only_intervals and intervals.beyond_last_interval(frame_idx, analyzeFrames)):
            break

        keys = gui.get_key_presses()
        if 'q' in keys:
            # quit fully
            stopAllProcessing = True
            break
        if 'n' in keys:
            # goto next
            break

        # check we're in a current interval, else skip processing
        # NB: have to spool through like this, setting specific frame to read
        # with cap.get(cv2.CAP_PROP_POS_FRAMES) doesn't seem to work reliably
        # for VFR video files
        if show_only_intervals and not intervals.is_in_interval(frame_idx, analyzeFrames):
            # no need to show this frame
            continue

        if show_poster:
            refImg = poster.get_ref_image(400)


        if frame_idx in gazes:
            for gaze_head, gaze_world in zip(gazes[frame_idx],plane_gazes[frame_idx]):
                # draw gaze point on scene video
                gaze_head.draw(frame, cameraParams, subPixelFac)

                # draw plane gazes on video and poster
                if frame_idx in poses:
                    gaze_world.drawOnWorldVideo(frame, cameraParams, subPixelFac)
                    if show_poster:
                        gaze_world.drawOnPlane(refImg, poster, subPixelFac)

        if show_poster:
            gui.update_image(refImg, frame_ts/1000., frame_idx, window_id = poster_win_id)

        # if we have poster pose, draw poster origin on video
        if frame_idx in poses:
            a = poses[frame_idx].getOriginOnImage(cameraParams)
            drawing.openCVCircle(frame, a, 3, (0,255,0), -1, subPixelFac)
            drawing.openCVLine(frame, (a[0],0), (a[0],height), (0,255,0), 1, subPixelFac)
            drawing.openCVLine(frame, (0,a[1]), (width,a[1]) , (0,255,0), 1, subPixelFac)

        # keys is populated above
        if 's' in keys:
            # screenshot
            cv2.imwrite(working_dir / f'calc_frame_{frame_idx}.png', frame)

        gui.update_image(frame, frame_ts/1000., frame_idx, window_id = frame_win_id)
        closed, = gui.get_state()
        if closed:
            stopAllProcessing = True
            break


    gui.stop()

    return stopAllProcessing