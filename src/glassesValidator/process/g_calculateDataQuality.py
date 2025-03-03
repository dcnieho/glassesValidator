import pathlib

import numpy as np

from glassesTools.validation import config, DataQualityType, compute_offsets

from .. import utils


def process(working_dir, config_dir=None,
            dq_types: list[DataQualityType]=None, allow_dq_fallback=False, include_data_loss=False):
    working_dir  = pathlib.Path(working_dir)
    if config_dir is not None:
        config_dir = pathlib.Path(config_dir)
    if dq_types is None:
        dq_types = []

    print('processing: {}'.format(working_dir.name))
    utils.update_recording_status(working_dir, utils.Task.Data_Quality_Calculated, utils.Status.Running, skip_if_missing=True)

    # get information about gaze target locations
    validationSetup = config.get_validation_setup(config_dir)
    validation_plane = config.plane.ValidationPlane(config_dir, validationSetup)
    targets = {t_id: np.append(validation_plane.targets[t_id].center, 0.) for t_id in validation_plane.targets}   # get centers of targets

    # get interval coded to be analyzed
    analyzeFrames = utils.readMarkerIntervalsFile(working_dir / 'markerInterval.tsv')
    if analyzeFrames is None:
        print('  no marker intervals defined for this recording, skipping')
        return

    compute_offsets.compute(working_dir/'gazePlane.tsv',
                            working_dir/'pose.tsv',
                            working_dir/'fixationAssignment.tsv',
                            analyzeFrames,
                            targets,
                            validation_plane.config['distance']*10.,
                            working_dir,
                            f'dataQuality.tsv',
                            dq_types=dq_types,
                            allow_dq_fallback=allow_dq_fallback,
                            include_data_loss=include_data_loss)

    utils.update_recording_status(working_dir, utils.Task.Data_Quality_Calculated, utils.Status.Finished, skip_if_missing=True)