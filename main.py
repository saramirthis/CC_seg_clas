import sys,os
import copy
import random
path = os.path.abspath('../dev/')
if path not in sys.path:
    sys.path.append(path)

import numpy as np
import scipy as scipy
import scipy.misc as misc
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import genfromtxt
import platform

from sklearn.cluster import DBSCAN
import sklearn.cluster as skl_clus
from sklearn.model_selection import StratifiedKFold, GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing, svm

import bib_mri as FW

def sign_extract(seg, resols): #Function for shape signature extraction
    splines = FW.get_spline(seg,smoothness)

    sign_vect = np.array([]).reshape(0,points) #Initializing temporal signature vector
    for resol in resols:
        sign_vect = np.vstack((sign_vect, FW.get_profile(splines, n_samples=points, radius=resol)))

    return sign_vect

def sign_fit(sig_ref, sig_fit): #Function for signature fitting
    dif_curv = []
    for shift in range(points):
        dif_curv.append(np.abs(np.sum((sig_ref - np.roll(sig_fit[0],shift))**2)))
    return np.apply_along_axis(np.roll, 1, sig_fit, np.argmin(dif_curv))

print "Python version: ", platform.python_version()
print "Numpy version: ", np.version.version
print "Scipy version: ", scipy.__version__
print "Matplotlib version: ", mpl.__version__
