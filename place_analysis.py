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
9/17/20
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


def area(coords):
    """Return the area of a rectangle in m^2.
    
    Parameters
    ----------
    coords: list of lists
        Rectangle coordinates like [(x1, y1), (x2, y2)]
    """
    coords = np.array(coords)
    return np.abs(np.prod(coords[1, :] - coords[0, :]))


def dist_from_point(coords,
                    point):
    """Return distance between coords and a comparison point.
    
    Distance is in vitural meters.
    
    Parameters
    ----------
    coords: list of lists
        Rectangle coordinates like [(x1, y1), (x2, y2)]
    point: list
        (x, y)
    """
    coords = np.array(coords)
    point = np.array(point)
    coord_center = (coords[0, :] + coords[1, :]) / 2
    return np.linalg.norm(coord_center - point)


def head_direction(phi,
                   n=4):
    """Return head direction given a phase angle."""
    phi %= 360

    if n == 4:
        if (phi>=315) or (phi<45):
            return 'N'
        elif (phi>=45) and (phi<135):
            return 'E'
        elif (phi>=135) and (phi<225):
            return 'S'
        else:
            return 'W'
    elif n == 8:
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
