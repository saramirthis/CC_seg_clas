import os
"""
This module contains all the configuration of simulation environment
"""

DIR_BAS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
DIR_MODEL = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "saves"))
DIR_OUTPUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))

if not os.path.exists(DIR_OUTPUT):
  os.makedirs(DIR_OUTPUT)
  print("The output directory is created")

SMOOTHNESS = 700
DEGREE = 5
FIT_RES = 0.35
RESOLS_INF = 0.01
RESOLS_SUP = 0.5
RESOLS_STEP = 0.01
POINTS = 500

#Method to choose cluster representant:
#"min_dist": Element with smallest intra-cluster distance
#"random": Element selected randomly
#"best_acc": Element with best AUC
#"min_dist", "random", "best_acc"
CHOSEN_METHOD = "best_acc"

FL_GRAPH = False
FL_SAVE = True
