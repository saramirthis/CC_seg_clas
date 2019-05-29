import sys,os, platform, random, glob
#import copy
path = os.path.abspath('./aux/')
if path not in sys.path:
    sys.path.append(path)

import numpy as np
import pandas as pd
import nibabel as nib
import scipy as scipy
#import scipy.misc as misc
import matplotlib as mpl
import matplotlib.pyplot as plt
#from numpy import genfromtxt

#from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import StratifiedKFold, GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing, svm

from aux import default_config as df_conf
from aux import func as sign
from aux import aux_fnc as ax
#import bib_mri as FW

ax.print_div('Importing modules')

print('Python version: ', platform.python_version())
print('Numpy version: ', np.version.version)
print('Scipy version: ', scipy.__version__)
print('Matplotlib version: ', mpl.__version__)

ax.print_div('Set list of directories with mask images')

labeled_file = './aux/labels.csv'
all_labels = pd.read_csv(labeled_file,sep=',')

group_opts = ['Freesurfer','Manual','Pardoe']
dirs_all = []
labels_all = []
for group_opt in group_opts:
    str_add = ''
    if group_opt == 'Freesurfer':
        str_add = 'T1_'

    list_opt = all_labels.loc[(all_labels['Label'] != -1) & (all_labels['Group'] == group_opt)]
    dirs_all += ['{}{}/{}{:06.0f}/'.format(df_conf.DIR_BAS,group_opt.lower(),str_add,cod) for cod in list_opt.Subject]
    labels_all += list(list_opt.Label)
#dirs_all, labels_all = dirs_all[:60], labels_all[:60]
print('Found dirs:',len(dirs_all))

ax.print_div('Extracting and fitting Signatures')

resols = np.arange(df_conf.RESOLS_INF,df_conf.RESOLS_SUP,df_conf.RESOLS_STEP)
resols = np.insert(resols,0,df_conf.FIT_RES)
prof_vec = np.empty((len(dirs_all),resols.shape[0],df_conf.POINTS))

for ind, mask_path in enumerate(dirs_all):
    pardoe_seg = glob.glob('{}*.corrected.cc.nii'.format(mask_path))
    if pardoe_seg != []:
        img_mask_msp = nib.load(pardoe_seg[0]).get_data()[::-1,::-1,0]
    else:
        file_mask_reg = '{}mask_reg'.format(mask_path)
        msp_points_reg = '{}msp_points_reg'.format(mask_path)

        in_img_msp = nib.load('{}.nii.gz'.format(msp_points_reg)).get_data()
        msp = np.argmax(np.sum(np.sum(in_img_msp,axis=-1),axis=-1))

        in_img_mask = nib.load('{}.nii.gz'.format(file_mask_reg)).get_data()
        img_mask_msp = in_img_mask[msp]

    refer_temp = sign.sign_extract(img_mask_msp, resols, df_conf.SMOOTHNESS, df_conf.POINTS)
    prof_vec[ind] = refer_temp

    if ind > 0: #Fitting curves using the first one as basis
        prof_ref = prof_vec[0]
        prof_vec[ind] = sign.sign_fit(prof_ref[0], refer_temp, df_conf.POINTS)

print("Segmentations' vector: ", prof_vec.shape)

fl_graph = False

if fl_graph:
    labels_True = np.array(labels_all)==0

    plt.figure()
    plt.plot(prof_vec[labels_True,df_conf.REG_EX].T)
    plt.title("Correct signatures for res: %f"%(resols[df_conf.REG_EX]))
    plt.show()

ax.print_div('Train and test division')

Y_total = np.array(labels_all)
Y_filt = Y_total != 2 #Filtering labels {0:correct,1:error,2:soft_error}
Y_total = Y_total[Y_filt]
X_total = prof_vec[Y_filt,1:,:] #Filtering the fitting resolution
resols = resols[1:] #Filtering the fitting resolution
resols_ref = np.arange(1,len(resols)+1)

print("Arrays' size:",X_total.shape,Y_total.shape)
print('Dic labels count:',np.unique(Y_total, return_counts=True))
print("Initial division:", np.sum(Y_total)/len(Y_total))
split_train = split_valid = 0.3

sss = StratifiedShuffleSplit(n_splits=1, test_size=split_valid, train_size=split_train, random_state=33)
train_index, valid_index = next(sss.split(X_total, Y_total))
test_index = np.ones((len(Y_total))).astype('bool')
test_index[np.concatenate((train_index,valid_index))] = False
test_index = np.arange(len(Y_total))[test_index]
X_train = X_total[train_index]
X_valid = X_total[valid_index]
X_test = X_total[test_index]
Y_train = Y_total[train_index]
Y_valid = Y_total[valid_index]
Y_test = Y_total[test_index]
print('Train set: ({})'.format(split_train),X_train.shape, Y_train.shape, np.sum(Y_train)/len(Y_train))
print('Valid set: ({})'.format(split_valid),X_valid.shape, Y_valid.shape, np.sum(Y_valid)/len(Y_valid))
print('Test set: ({})'.format(1-(split_train+split_valid)),X_test.shape, Y_test.shape,np.sum(Y_test)/len(Y_test))

val_norm = np.amax(np.amax(X_train,axis=0),axis=-1).reshape(1,-1,1)
X_train_norm = X_train/val_norm
X_valid_norm = X_valid/val_norm
X_test_norm = X_test/val_norm

ax.print_div('Training SVM')

tuned_parameters = [{'kernel': ['rbf', 'linear'], 'C': [1, 10, 50, 100]}]
cv_s = StratifiedKFold(5, shuffle=True)

acum_acc = []
acum_pred = np.array([]).reshape(0,Y_valid.shape[0])
d_train = {}
for res in np.arange(resols.shape[0]):
    d_train["string{0}".format(res)] = GridSearchCV(svm.SVC(gamma='scale'), tuned_parameters, iid=False, cv=cv_s, scoring='roc_auc')
    d_train["string{0}".format(res)].fit(X_train_norm[:,res,:], Y_train)
    print("--------------------------------------------------------------------")
    print(d_train["string{0}".format(res)].best_estimator_)

    y_true, y_pred = Y_valid, d_train["string{0}".format(res)].predict(X_valid_norm[:,res,:])
    Matrix_conf = confusion_matrix(y_true, y_pred)
    A_rf_n = (Matrix_conf[0,0]+Matrix_conf[1,1])/np.sum(Matrix_conf).astype('float64')
    print('Final accuracy: ', A_rf_n, ' at res: ', resols[res])
    acum_acc = np.append(acum_acc, A_rf_n)
    acum_pred = np.vstack((acum_pred, y_pred))

acum_acc2 = np.logical_not(np.logical_xor(acum_pred.astype('bool'),Y_valid.astype('bool')))

mm_conf = np.empty((4,acum_acc2.shape[0],acum_acc2.shape[0]))
for i in range(acum_acc2.shape[0]):
    for j in range(acum_acc2.shape[0]):
        mm_conf[:,i,j] = ax.agreement_matrix(acum_acc2[i],acum_acc2[j])

a = mm_conf[0]
b = mm_conf[1]
c = mm_conf[2]
d = mm_conf[3]

DM = (b+c)/(a+b+c+d)
if fl_graph:
    ax.plot_matrix(DM, classes=map(str, resols),title='DM matrix')

ax.print_div('Training ensemble')

choose_el = 'random'#'min_dist', 'random'
tuned_parameters_ens = [{'kernel': ['rbf', 'linear', 'poly'], 'C': [1, 10, 50, 100]}]
num_clusters = np.arange(1,50)
clusters_acc_mean = []
clusters_acc_std = []
for n_cl in num_clusters:
    agg = AgglomerativeClustering(n_clusters=n_cl, affinity='precomputed', linkage='average')
    labels_res = agg.fit_predict(DM)
    #print labels_res
    random_acc = []
    for loop_random in range(15):
        list_clusters = []
        res_chs = []
        labels_list = np.unique(labels_res)
        for clust in labels_list:
            res_clust = np.where(labels_res==clust)[0]
            list_clusters.append(res_clust)
            #print "Set({}): ".format(clust), resols_ref[res_clust]
            if choose_el == 'random':
                res_chs.append(random.choice(res_clust))
            else:
                sum_dist = []
                for r_clust in res_clust:
                    sum_dist.append(np.sum(DM[r_clust,res_clust]))
                res_chs.append(res_clust[np.argmin(sum_dist)])
        #print("Clusters:", len(list_clusters), "Choosen resolutions:", resols_ref[res_chs])

        svm_ind = np.array([]).reshape(0,Y_valid.shape[0])
        for res_ch in res_chs:
            svm_ind = np.vstack((svm_ind, d_train["string{0}".format(res_ch)].predict(X_valid_norm[:,res_ch,:])))
        svm_ind = svm_ind.T
        clf = GridSearchCV(svm.SVC(gamma='scale'), tuned_parameters_ens, iid=False, cv=cv_s, scoring='roc_auc')
        clf.fit(svm_ind, Y_valid)

        svm_ind = np.array([]).reshape(0,Y_test.shape[0])
        for res_ch in res_chs:
            svm_ind = np.vstack((svm_ind, d_train["string{0}".format(res_ch)].predict(X_test_norm[:,res_ch,:])))
        svm_ind = svm_ind.T
        y_true, y_pred = Y_test, clf.predict(svm_ind)
        Matrix_conf = confusion_matrix(y_true, y_pred)
        A_rf_n = (Matrix_conf[0,0]+Matrix_conf[1,1])/np.sum(Matrix_conf).astype('float64')
        #print 'Final accuracy: ', A_rf_n
        random_acc.append(A_rf_n)
    print(n_cl,":", random_acc)
    print("Mean:",np.mean(random_acc),"/ SD:",np.std(random_acc))
    clusters_acc_mean.append(np.mean(random_acc))
    clusters_acc_std.append(np.std(random_acc))

if fl_graph:
    cores = ['darkblue','royalblue']
    plt.figure()
    plt.plot(clusters_acc_mean, color=cores[0])
    plt.plot(np.array(clusters_acc_mean)-np.array(clusters_acc_std), linestyle='dashed', color=cores[1])
    plt.plot(np.array(clusters_acc_mean)+np.array(clusters_acc_std), linestyle='dashed', color=cores[1])
    plt.title("Mean and standard deviation accuracy per cluster size")
    plt.show()

#plt.figure()
#plt.plot(clusters_acc_std, color=cores[1])
#plt.title("Standard deviation accuracy per cluster size")
#plt.show()

max_acc, min_acc = np.amax(clusters_acc_mean), np.amin(clusters_acc_mean)
num_clust_ch = np.where(clusters_acc_mean > min_acc+0.9*(max_acc-min_acc))[0][0]+1
print("Cluster size:", num_clust_ch)

agg = AgglomerativeClustering(n_clusters=num_clust_ch, affinity='precomputed', linkage='complete')
labels_res = agg.fit_predict(DM)
print('Element labels', labels_res)

list_clusters = []
res_chs = []
labels_list = np.unique(labels_res)
for clust in labels_list:
    res_clust = np.where(labels_res==clust)[0]
    list_clusters.append(res_clust)
    print("--------------------------------------------------------------------")
    if len(res_clust) == 1:
        dist_intra = np.amax(DM[res_clust[0],res_clust[0]])
    else:
        dist_intra = np.amax(DM[res_clust[0],res_clust[1:]])
    print("Cluster({}): ".format(clust),resols_ref[res_clust],"Distance intra: ",dist_intra)

    for res_ch in res_clust:
        y_true, y_pred = Y_test, d_train["string{0}".format(res_ch)].predict(X_test_norm[:,res_ch,:])
        Matrix_conf = confusion_matrix(y_true, y_pred)
        A_rf_n = (Matrix_conf[0,0]+Matrix_conf[1,1])/np.sum(Matrix_conf).astype('float64')

        ind_err_rand = np.where(np.logical_xor(y_true, y_pred))
        ind_err = test_index[ind_err_rand]
        print("Element({}): ".format(resols_ref[res_ch]),ind_err, "Acc:({})".format(A_rf_n))

    if choose_el == 'random':
        res_chs.append(random.choice(res_clust))
    else:
        sum_dist = []
        for r_clust in res_clust:
            sum_dist.append(np.sum(DM[r_clust,res_clust]))
        res_chs.append(res_clust[np.argmin(sum_dist)])

print("=====================================================================")
print("Size ensemble:", len(list_clusters), "Choosen resolutions:", resols_ref[res_chs])

svm_ind = np.array([]).reshape(0,Y_valid.shape[0])
for res_ch in res_chs:
    svm_ind = np.vstack((svm_ind, d_train["string{0}".format(res_ch)].predict(X_valid_norm[:,res_ch,:])))
svm_ind = svm_ind.T
clf = GridSearchCV(svm.SVC(gamma='scale'), tuned_parameters_ens, iid=False, cv=cv_s, scoring='roc_auc')
clf.fit(svm_ind, Y_valid)

svm_ind = np.array([]).reshape(0,Y_test.shape[0])
for res_ch in res_chs:
    svm_ind = np.vstack((svm_ind, d_train["string{0}".format(res_ch)].predict(X_test_norm[:,res_ch,:])))
svm_ind = svm_ind.T
y_true, y_pred = Y_test, clf.predict(svm_ind)
Matrix_conf = confusion_matrix(y_true, y_pred)
A_rf_n = (Matrix_conf[0,0]+Matrix_conf[1,1])/np.sum(Matrix_conf).astype('float64')

ind_err_rand = np.where(np.logical_xor(y_true, y_pred))
ind_err = test_index[ind_err_rand]
print("Ensemble: ",ind_err, "Final acc ensemble:({})".format(A_rf_n))
