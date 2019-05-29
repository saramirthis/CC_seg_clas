"""
Auxiliar functions Module
"""

import math, itertools
import numpy as np
import matplotlib.pyplot as plt

def print_div(string,len_print=50):
    len_prn_str = len_print - len(string) - 2
    assert (len_prn_str > 0), 'Length print greater than expected'
    print('='*len_print)
    print('='*int(math.floor(len_prn_str/2)),string,'='*int(math.ceil(len_prn_str/2)))
    print('='*len_print)

def agreement_matrix(x1,x2):
    x = np.vstack((x1,x2)).T
    y = x.dot(1 << np.arange(x.shape[-1] - 1, -1, -1)).astype('uint8')
    m_ag = np.zeros((4))
    for pos in np.unique(y):
        m_ag[pos] = np.sum(y==pos)
    return m_ag[::-1]

def plot_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #print(cm)

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(list(classes)))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
