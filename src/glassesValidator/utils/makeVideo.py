import shutil
import os
import pathlib

import cv2
import numpy as np

from glassesTools import annotation, aruco, gaze_headref, gaze_worldref, naming, ocv, propagating_thread, recording, timestamps, transforms
from glassesTools.gui import video_player

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

    if show_visualization:
        # We run processing in a separate thread (GUI needs to be on the main thread for OSX, see https://github.com/pthom/hello_imgui/issues/33)
        gui = video_player.GUI(use_thread = False)
        gui.add_window(working_dir.name)
        gui.set_show_controls(True)
        gui.set_show_play_percentage(True)
        gui.set_show_action_tooltip(True)

        proc_thread = propagating_thread.PropagatingThread(target=do_the_work, args=(working_dir, config_dir, gui, show_rejected_markers, add_audio_to_poster_video), cleanup_fun=gui.stop)
        proc_thread.start()
        gui.start()
        proc_thread.join()
    else:
        do_the_work(working_dir, config_dir, None, show_rejected_markers, add_audio_to_poster_video)

def do_the_work(working_dir, config_dir, gui: video_player.GUI, show_rejected_markers, add_audio_to_poster_video):
    has_gui = gui is not None
    sub_pixel_fac = 8   # for anti-aliased drawing

    # get info about recording
    recInfo = recording.Recording.load_from_json(working_dir)

    # open file with information about Aruco marker and Gaze target locations
    validationSetup = config.get_validation_setup(config_dir)
    # get info about markers on our poster
    poster          = config.poster.Poster(config_dir, validationSetup)
    # get poster image width, height
    ref_img         = poster.get_ref_image(400)
    ref_height, ref_width, _ = ref_img.shape

    # get interval(s) coded to be analyzed, if any
    analyzeFrames   = utils.readMarkerIntervalsFile(working_dir / "markerInterval.tsv")
    if analyzeFrames is None:
        analyzeFrames = []
    else:
        # flatten
        analyzeFrames = [i for iv in analyzeFrames for i in iv]
    episodes = {annotation.Event.Validate: analyzeFrames}

    # Read gaze data
    gazes_head  = gaze_headref.read_dict_from_file(working_dir / naming.gaze_data_fname)[0]

    # get camera calibration info
    cameraParams = ocv.CameraParams.read_from_file(working_dir / "calibration.xml")

    # build pose estimator
    in_video = recInfo.get_scene_video_path()   # get video file to process
    video_ts = timestamps.VideoTimestamps(working_dir / naming.frame_timestamps_fname)
    pose_estimator = aruco.PoseEstimator(in_video, video_ts, cameraParams)
    pose_estimator.add_plane('validate',
                             {'plane': poster, 'aruco_params': {'markerBorderBits': validationSetup['markerBorderBits']}, 'min_num_markers': validationSetup['minNumMarkers']})
    pose_estimator.set_visualize_on_frame(True)
    pose_estimator.show_rejected_markers = show_rejected_markers

    # prep output video files
    width, height, fps = pose_estimator.get_video_info()
    # get which pixel format
    codec    = ffpyplayer.tools.get_format_codec(fmt='mp4')
    pix_fmt  = ffpyplayer.tools.get_best_pix_fmt('bgr24',ffpyplayer.tools.get_supported_pixfmts(codec))
    fpsFrac  = Fraction(fps).limit_denominator(10000).as_integer_ratio()
    # scene video
    out_opts = {'pix_fmt_in':'bgr24', 'pix_fmt_out':pix_fmt, 'width_in':  width  , 'height_in':  height  ,'frame_rate':fpsFrac}
    vidOutScene  = MediaWriter(str(working_dir / 'detectOutput_scene.mp4') , [out_opts], overwrite=True)
    # poster video
    out_opts = {'pix_fmt_in':'bgr24', 'pix_fmt_out':pix_fmt, 'width_in':ref_width, 'height_in':ref_height,'frame_rate':fpsFrac}
    vidOutPoster = MediaWriter(str(working_dir / 'detectOutput_poster.mp4'), [out_opts], overwrite=True)

    # if we have a gui, set it up
    if has_gui:
        gui.set_frame_size((width, height), gui.main_window_id)
        gui.set_show_timeline(True, video_ts, episodes, gui.main_window_id)
        gui.set_show_annotation_label(False, gui.main_window_id)
        gui.set_timecode_position('r', gui.main_window_id)
        gui.set_show_action_tooltip(True, gui.main_window_id)
        # add window for poster
        poster_win_id = gui.add_window('poster')
        gui.set_frame_size((ref_width, ref_height), poster_win_id)

    should_exit = False
    while True:
        status, pose, _, _, (frame, frame_idx, frame_ts) = pose_estimator.process_one_frame()
        # TODO: if there is a discontinuity, fill in the missing frames so audio stays in sync
        # check if we're done
        if status==aruco.Status.Finished:
            break
        # NB: no need to handle aruco.Status.Skip, since we didn't provide the pose estimator with any analysis intervals (we want to process the whole video)
        pose = pose['validate']

        if frame is None:
            # we don't have a valid frame, use a fully black frame
            frame = np.zeros((height,width,3), np.uint8)   # black image
        refImg = poster.get_ref_image(ref_width)

        # process gaze
        if frame_idx in gazes_head:
            for gaze in gazes_head[frame_idx]:
                # draw gaze point on scene video
                gaze.draw(frame, cameraParams, sub_pixel_fac)

                # figure out where gaze vectors intersect with poster
                gazePoster = gaze_worldref.from_head(pose, gaze, cameraParams)
                # and draw gazes on video and poster
                gazePoster.draw_on_world_video(frame, cameraParams, sub_pixel_fac)
                gazePoster.draw_on_plane(refImg, poster, sub_pixel_fac)

        # annotate frame
        analysisIntervalIdx = None
        for f in range(0,len(analyzeFrames)-1,2):   # -1 to make sure we don't announce incomplete intervals
            if frame_idx>=analyzeFrames[f] and frame_idx<=analyzeFrames[f+1]:
                analysisIntervalIdx = f
        frameClr = (0,0,255) if analysisIntervalIdx is not None else (0,0,0)

        text = '%6.3f [%6d] (%s markers)' % (frame_ts/1000.,frame_idx, pose.pose_N_markers)
        textSize,baseline = cv2.getTextSize(text,cv2.FONT_HERSHEY_PLAIN,2,2)
        cv2.rectangle(frame,(0,height),(textSize[0]+2,height-textSize[1]-baseline-5), frameClr, -1)
        cv2.putText(frame, (text), (2, height-5), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255),2)

        # store to file
        img = Image(plane_buffers=[frame.flatten().tobytes()], pix_fmt='bgr24', size=(width, height))
        vidOutScene.write_frame(img=img, pts=frame_idx/fps)
        img = Image(plane_buffers=[refImg.flatten().tobytes()], pix_fmt='bgr24', size=(ref_width, ref_height))
        vidOutPoster.write_frame(img=img, pts=frame_idx/fps)

        if has_gui:
            gui.update_image(frame , frame_ts/1000., frame_idx, window_id=gui.main_window_id)
            gui.update_image(refImg, frame_ts/1000., frame_idx, window_id=poster_win_id)

            requests = gui.get_requests()
            for r,_ in requests:
                if r=='exit':   # only request we need to handle
                    should_exit = True
                    break
            if should_exit:
                break

    vidOutScene.close()
    vidOutPoster.close()
    if has_gui:
        gui.stop()

    # if ffmpeg is on path, add audio to scene and optionally poster video
    if shutil.which('ffmpeg') is not None:
        todo = [working_dir / 'detectOutput_scene.mp4']
        if add_audio_to_poster_video:
            todo.append(working_dir / 'detectOutput_poster.mp4')

        for f in todo:
            # move file to temp name
            tempName = f.parent / (f.stem + '_temp' + f.suffix)
            shutil.move(f, tempName)

            # add audio
            cmd_str = ' '.join(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', '"'+str(tempName)+'"', '-i', '"'+str(in_video)+'"', '-vcodec', 'copy', '-acodec', 'copy', '-map', '0:v:0', '-map', '1:a:0?', '"'+str(f)+'"'])
            os.system(cmd_str)
            # clean up
            if f.exists():
                tempName.unlink(missing_ok=True)
            else:
                shutil.move(tempName, f)