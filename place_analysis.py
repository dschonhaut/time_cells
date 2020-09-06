"""
place_analysis.py

Author
------
Daniel Schonhaut
Computational Memory Lab
University of Pennsylvania
daniel.schonhaut@gmail.com

Description
-----------
Functions for analyzing firing rate data associated with positional info.

Last Edited
-----------
9/6/20
"""
import sys
import os
from glob import glob
from collections import OrderedDict as od
from time import sleep

import mkl
mkl.set_num_threads(1)
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import pandas as pd
import h5py
import scipy.io as sio

sys.path.append('/home1/dscho/code/general')
import data_io as dio


def head_direction(phi):
    """Return head direction given a phase angle."""
    phi %= 360
    x = [337.5, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5]
    if (phi>=337.5) or (phi<22.5):
        return 'N'
    elif (phi>=22.5) and (phi<67.5):
        return 'NE'
    elif (phi>=67.5) and (phi<112.5):
        return 'E'
    elif (phi>=112.5) and (phi<157.5):
        return 'SE'
    elif (phi>=157.5) and (phi<202.5):
        return 'S'
    elif (phi>=202.5) and (phi<247.5):
        return 'SW'
    elif (phi>=247.5) and (phi<292.5):
        return 'W'
    elif (phi>=292.5) and (phi<337.5):
        return 'NW'
