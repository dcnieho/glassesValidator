# -*- coding: utf-8 -*-

from .tobii_G2 import preprocessData as tobii_G2
from .tobii_G3 import preprocessData as tobii_G3
from .SMI_ETG import preprocessData as SMI_ETG
from .SeeTrue import preprocessData as SeeTrue

def pupil_core(inputDir, outputDir):
    from .pupilLabs import preprocessData

    preprocessData(inputDir, outputDir,'pupilCore')


def pupil_invisible(inputDir, outputDir):
    from .pupilLabs import preprocessData

    preprocessData(inputDir, outputDir,'pupilInvisible')