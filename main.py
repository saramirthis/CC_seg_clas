from ntpath import join
import sys,os, platform, random, glob, sklearn
#from tabulate import tabulate

path = os.path.abspath("./aux/")
if path not in sys.path:
    sys.path.append(path)

import numpy as np
import pandas as pd
import nibabel as nib
import scipy as scipy
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import StratifiedKFold, GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing, svm, metrics

from scipy.cluster.hierarchy import dendrogram, linkage

from joblib import dump

from aux import default_config as df_conf
from aux import func as sign
from aux import aux_fnc as ax

ax.print_div("Importing modules")

print("Python version: ", platform.python_version())
print("Numpy version: ", np.version.version)
print("Scipy version: ", scipy.__version__)
print("Matplotlib version: ", mpl.__version__)
print("Scikit-learn version: ",sklearn.__version__)

ax.print_div("Set list of directories with mask images")

labeled_file = "./aux/labels.csv"
all_labels = pd.read_csv(labeled_file,sep=",")

group_opts = ["Freesurfer","Manual","Pardoe"]
dirs_all = []
labels_all = []
for group_opt in group_opts:
    str_add = ""
    if group_opt == "Freesurfer":
        str_add = "T1_"

    list_opt = all_labels.loc[(all_labels["Label"] != -1) & (all_labels["Group"] == group_opt)]
    dirs_all += ["{}{}/{}{:06.0f}/".format(df_conf.DIR_BAS,group_opt.lower(),str_add,cod) for cod in list_opt.Subject]
    labels_all += list(list_opt.Label)
#dirs_all, labels_all = dirs_all[:60], labels_all[:60]
print("Found dirs:",len(dirs_all))

ax.print_div("Extracting and fitting Signatures")

resols = np.arange(df_conf.RESOLS_INF,df_conf.RESOLS_SUP,df_conf.RESOLS_STEP)
resols = np.insert(resols,0,df_conf.FIT_RES)
prof_vec = np.empty((len(dirs_all),resols.shape[0],df_conf.POINTS))

for ind, mask_path in enumerate(dirs_all):
    pardoe_seg = glob.glob("{}*.corrected.cc.nii".format(mask_path))
    if pardoe_seg != []:
        img_mask_msp = nib.load(pardoe_seg[0]).get_data()[::-1,::-1,0]
    else:
        file_mask_reg = "{}mask_reg".format(mask_path)
        msp_points_reg = "{}msp_points_reg".format(mask_path)

        in_img_msp = nib.load("{}.nii.gz".format(msp_points_reg)).get_data()
        msp = np.argmax(np.sum(np.sum(in_img_msp,axis=-1),axis=-1))

        in_img_mask = nib.load("{}.nii.gz".format(file_mask_reg)).get_data()
        img_mask_msp = in_img_mask[msp]

    refer_temp = sign.sign_extract(img_mask_msp, resols, df_conf.SMOOTHNESS, df_conf.POINTS)
    prof_vec[ind] = refer_temp
    
    if ind > 0: #Fitting curves using the first one as basis
        prof_ref = prof_vec[0]
        prof_vec[ind] = sign.sign_fit(prof_ref[0], refer_temp, df_conf.POINTS)

print("Segmentations' vector: ", prof_vec.shape)

if df_conf.FL_GRAPH:

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(18,5))

    labels_True = np.array(labels_all)==0
    labels_False = np.array(labels_all)==1

    ax1.plot(prof_vec[labels_True,df_conf.REG_EX].T)
    ax1.set_title("Correct signatures for res: %f"%(resols[df_conf.REG_EX]))
    ax2.plot(prof_vec[labels_False,df_conf.REG_EX].T)
    ax2.set_title("Error signatures for res: %f"%(resols[df_conf.REG_EX]))
    plt.show()

ax.print_div("Train and test division")

Y_total = np.array(labels_all)
Y_filt = Y_total != 2 #Filtering labels {0:correct,1:error,2:soft_error}
Y_total = Y_total[Y_filt]
X_total = prof_vec[Y_filt,1:,:] #Filtering the fitting resolution
resols = resols[1:] #Filtering the fitting resolution
resols_ref = np.arange(1,len(resols)+1)

print("Arrays" size:",X_total.shape,Y_total.shape)
print("Dic labels count:",np.unique(Y_total, return_counts=True))
print("Initial division:", np.sum(Y_total)/len(Y_total))
split_train = split_valid = 0.3

sss = StratifiedShuffleSplit(n_splits=1, test_size=split_valid, train_size=split_train, random_state=33)
train_index, valid_index = next(sss.split(X_total, Y_total))
test_index = np.ones((len(Y_total))).astype("bool")
test_index[np.concatenate((train_index,valid_index))] = False
test_index = np.arange(len(Y_total))[test_index]
X_train = X_total[train_index]
X_valid = X_total[valid_index]
X_test = X_total[test_index]
Y_train = Y_total[train_index]
Y_valid = Y_total[valid_index]
Y_test = Y_total[test_index]

val_norm = np.amax(np.amax(X_train,axis=0),axis=-1).reshape(1,-1,1)
X_train_norm = X_train/val_norm
X_valid_norm = X_valid/val_norm
X_test_norm = X_test/val_norm

print("Train set: ({})".format(split_train),X_train_norm.shape, Y_train.shape, np.sum(Y_train)/len(Y_train))
print("Valid set: ({})".format(split_valid),X_valid_norm.shape, Y_valid.shape, np.sum(Y_valid)/len(Y_valid))
print("Test set: ({})".format(1-(split_train+split_valid)), X_test_norm.shape,
      Y_test.shape,np.sum(Y_test)/len(Y_test))

ax.print_div("Training individual SVM"s")

tuned_parameters = [{"kernel": ["rbf", "linear"], "C": [1, 10, 20, 50, 100]}]
cv_s = StratifiedKFold(5, shuffle=True, random_state=2)

acum_pred = np.array([]).reshape(0,Y_valid.shape[0])
d_train = {}
# Each resolution is used as feature shape alogn with one SVM classifier
# Select best hyperparameters for each resolution/classifier using CV
for res in np.arange(resols.shape[0]).astype(int):
    d_train["string{0}".format(res)] = GridSearchCV(svm.SVC(gamma="auto", class_weight="balanced", random_state=4,
                                                            probability=True,),
                                                    tuned_parameters, iid=False, cv=cv_s, scoring="roc_auc")
    d_train["string{0}".format(res)].fit(X_train_norm[:,res,:], Y_train)
    print("--------------------------------------------------------------------")
    print(d_train["string{0}".format(res)].best_estimator_)

    y_true, y_pred = Y_valid, d_train["string{0}".format(res)].predict(X_valid_norm[:,res,:])

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    print("Final AUC: ", metrics.auc(fpr, tpr), " at res: ", resols[res])
    acum_pred = np.vstack((acum_pred, y_pred))

acum_acc = np.logical_not(np.logical_xor(acum_pred.astype("bool"),Y_valid.astype("bool")))

mm_conf = np.empty((4,acum_acc.shape[0],acum_acc.shape[0]))
for i in range(acum_acc.shape[0]):
    for j in range(acum_acc.shape[0]):
        mm_conf[:,i,j] = ax.agreement_matrix(acum_acc[i],acum_acc[j])

a = mm_conf[0]
b = mm_conf[1]
c = mm_conf[2]
d = mm_conf[3]

DM = (b+c)/(a+b+c+d)

if df_conf.FL_GRAPH:
    ax.plot_matrix(DM, classes=map(str, resols),title="DM matrix")

    Z = linkage(DM, "average", optimal_ordering=False)
    fig = plt.figure(figsize=(15, 4))
    dend = dendrogram(Z, labels=np.arange(1,50))
    plt.show()

if df_conf.FL_SAVE:
    ax.print_div("Saving individual SVM's")
    file2save = os.path.join(df_conf.DIR_MODEL, "arr_models_ind.joblib")
    dump(d_train, file2save)

ax.print_div("Tunning ensemble size")

choose_el = "min_dist"#"min_dist", "random"
tuned_parameters_ens = [{"kernel": ["rbf", "linear"], "C": [1, 10, 20, 50, 100],}]
cv_ens = StratifiedKFold(5, shuffle=True, random_state=8)
num_clusters = np.arange(1,50)
clusters_auc = []
# All the possible clusters ranging from 1 (grouping all 49 resolutions) to 49 (one cluster for resolution)
for n_cl in num_clusters:
    agg = AgglomerativeClustering(n_clusters=n_cl, affinity="precomputed", linkage="average")
    labels_res = agg.fit_predict(DM)

    list_clusters = []
    res_chs = []
    labels_list = np.unique(labels_res)
    for clust in labels_list:
        res_clust = np.where(labels_res==clust)[0]
        list_clusters.append(res_clust)
        #print "Set({}): ".format(clust), resols_ref[res_clust]
        if choose_el == "random":
            res_chs.append(random.choice(res_clust))
        else:
            sum_dist = []
            for r_clust in res_clust:
                sum_dist.append(np.sum(DM[r_clust,res_clust]))
            res_chs.append(res_clust[np.argmin(sum_dist)])
    #print("Clusters:", len(list_clusters), "Choosen resolutions:", resols_ref[res_chs])

    svm_ind = np.array([]).reshape(0,Y_valid.shape[0])
    for res_ch in res_chs:
        svm_ind = np.vstack((svm_ind, d_train["string{0}".format(res_ch)].predict_proba(X_valid_norm[:,res_ch,:])[:,1]))
    svm_ind = svm_ind.T
    clf = GridSearchCV(svm.SVC(gamma="auto", class_weight="balanced", random_state=16), tuned_parameters_ens,
                       iid=False, cv=cv_ens, scoring="roc_auc")
    clf.fit(svm_ind, Y_valid)
    svm_ind = np.array([]).reshape(0,Y_test.shape[0])
    for res_ch in res_chs:
        svm_ind = np.vstack((svm_ind, d_train["string{0}".format(res_ch)].predict_proba(X_test_norm[:,res_ch,:])[:,1]))
    svm_ind = svm_ind.T
    y_true, y_pred = Y_test, clf.predict(svm_ind)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    AUC_cl = metrics.auc(fpr, tpr)
    print("--------------------------------------------------------------------")
    print(clf.best_estimator_)
    print("Number of clusters:", n_cl, " / Final AUC: ", AUC_cl)
    clusters_auc.append(AUC_cl)

if df_conf.FL_GRAPH:
    plt.figure()
    plt.plot(clusters_auc, color="darkblue")
    plt.title("AUC per cluster size")
    plt.xlabel("Number of clusters")
    plt.show()

ax.print_div("Asserting ensemble clusters")

max_auc, min_auc = np.amax(clusters_auc), np.amin(clusters_auc)
pc_best_auc_exp = 0.99
num_clust_ch = np.where(clusters_auc > min_auc+pc_best_auc_exp*(max_auc-min_auc))[0][0]+1
print("Best cluster size: {} (with at least {} of best result)".format(num_clust_ch,pc_best_auc_exp))

agg = AgglomerativeClustering(n_clusters=num_clust_ch, affinity="precomputed", linkage="average")
labels_res = agg.fit_predict(DM)
print("Element labels", labels_res)

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
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
        AUC_cl = metrics.auc(fpr, tpr)

        ind_err_rand = np.where(np.logical_xor(y_true, y_pred))
        ind_err = test_index[ind_err_rand]
        print("Element({}): ".format(resols_ref[res_ch]),ind_err, "AUC:({})".format(AUC_cl))

    if choose_el == "random":
        res_chs.append(random.choice(res_clust))
    else:
        sum_dist = []
        for r_clust in res_clust:
            sum_dist.append(np.sum(DM[r_clust,res_clust]))
        res_chs.append(res_clust[np.argmin(sum_dist)])

ax.print_div("Training best ensemble")

print("=========================================================================")
print("Size ensemble:", len(list_clusters), "Chosen resolutions:", resols_ref[res_chs])

svm_ind = np.array([]).reshape(0,Y_valid.shape[0])
for res_ch in res_chs:
    svm_ind = np.vstack((svm_ind, d_train["string{0}".format(res_ch)].predict_proba(X_valid_norm[:,res_ch,:])[:,1]))
svm_ind = svm_ind.T
clf = GridSearchCV(svm.SVC(gamma="auto", class_weight="balanced", random_state=16, probability=True),
                   tuned_parameters_ens, iid=False, cv=cv_ens, scoring="roc_auc")

clf.fit(svm_ind, Y_valid)
print(clf.best_estimator_)

y_pred_probs = clf.predict_proba(svm_ind)[:,1]
__, opt_th = ax.plot_prc(Y_valid, y_pred_probs,fl_plot=df_conf.FL_GRAPH)

print("Best separation threshold: {}".format(opt_th))

if df_conf.FL_SAVE:
    ax.print_div("Saving best ensemble")
    file2save = "{}ensemble_model.joblib".format(df_conf.DIR_MODEL)
    dump(clf, file2save)

    ax.print_div("Saving reference parameters")
    save_sign_refs = {"prof_ref": prof_vec[0:1], "val_norm": val_norm, "res_chs": res_chs, "opt_th": opt_th}
    file2save = os.path.join(df_conf.DIR_MODEL, "sign_refs.joblib")
    dump(save_sign_refs, file2save)

ax.print_div("Testing best ensemble")

svm_ind = np.array([]).reshape(0,Y_test.shape[0])
for res_ch in res_chs:
    svm_ind = np.vstack((svm_ind, d_train["string{0}".format(res_ch)].predict_proba(X_test_norm[:,res_ch,:])[:,1]))
svm_ind = svm_ind.T

y_pred_probs = clf.predict_proba(svm_ind)[:,1]
AUC_cl = ax.plot_roc(y_true, y_pred_probs,fl_plot=df_conf.FL_GRAPH)

y_pred = y_pred_probs > opt_th

mx_conf = confusion_matrix(y_true, y_pred)
if df_conf.FL_GRAPH:
    ax.plot_matrix(mx_conf, classes=np.unique(y_true), fig_size=4,
    title="Confussion matrix", opt_bar=False)

accuracy, recall, precision, f1 = ax.report_metrics(mx_conf)

#print("===== Final Report =====")
#print(tabulate([["AUC", AUC_cl],
#                ["Accuracy", accuracy],
#                ["Recall", recall],
#                ["Precision", precision],
#                ["F1", f1],],
#               ["Metric", "Value"], tablefmt="grid"))
print(AUC_cl,accuracy)
