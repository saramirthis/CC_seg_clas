import os
"""
This module contains all the configuration of simulation environment
"""

DIR_BAS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
DIR_SAVE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "saves"))

print(DIR_BAS)
print(DIR_SAVE)

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
