import pathlib
import numpy as np

from glassesTools.validation import config, assign_fixations

from .. import utils


def process(working_dir, config_dir=None, do_global_shift=True, max_dist_fac=.5):
    working_dir  = pathlib.Path(working_dir)
    if config_dir is not None:
        config_dir = pathlib.Path(config_dir)

    print('processing: {}'.format(working_dir.name))
    utils.update_recording_status(working_dir, utils.Task.Fixation_Assigned, utils.Status.Running, skip_if_missing=True)

    # open file with information about ArUco marker and gaze target locations
    validationSetup = config.get_validation_setup(config_dir)

    # get interval coded to be analyzed
    analyzeFrames = utils.readMarkerIntervalsFile(working_dir / 'markerInterval.tsv')
    if analyzeFrames is None:
        print('  no marker intervals defined for this recording, skipping')
        return

    validation_plane = config.plane.ValidationPlane(config_dir, validationSetup)

    targets = {t_id: np.append(validation_plane.targets[t_id].center, 0.) for t_id in validation_plane.targets}   # get centers of targets
    plot_limits = [[validation_plane.bbox[0]-validation_plane.marker_size, validation_plane.bbox[2]+validation_plane.marker_size],
                   [validation_plane.bbox[1]-validation_plane.marker_size, validation_plane.bbox[3]+validation_plane.marker_size]]
    for idx,_ in enumerate(analyzeFrames):
        fix_file = working_dir / f'fixations_interval_{idx+1:02d}.tsv'
        assign_fixations.distance(targets,
                                  fix_file,
                                  working_dir,
                                  filename_stem='fixationAssignment',
                                  iteration=idx,
                                  background_image=(validation_plane.get_ref_image(as_RGB=True),
                                                    np.array([validation_plane.bbox[x] for x in (0,2,3,1)])),
                                  plot_limits=plot_limits)

    utils.update_recording_status(working_dir, utils.Task.Fixation_Assigned, utils.Status.Finished, skip_if_missing=True)