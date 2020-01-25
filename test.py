import sys, os, platform, random, glob, sklearn, argparse

from pathlib import Path
path = os.path.abspath('./aux/')
if path not in sys.path:
    sys.path.append(path)

import numpy as np
import pandas as pd
import nibabel as nib

from joblib import load

from aux import default_config as df_conf
from aux import func as sign
from aux import aux_fnc as ax

parser = argparse.ArgumentParser(description='Segmentation Quality control Framework')
parser.add_argument('dir_in', type=str, help='Databse Input directory')
parser.add_argument('pattern', type=str, help='Pattern present in all the nifti file names')
parser.add_argument('msp', type=int, help='Slice to be selected on sagittal plane')
parser.add_argument('-opt_th', type=float, help='Threshold to separate classes')
args = parser.parse_args()

ax.print_div('Importing modules')

print('Python version: ', platform.python_version())
print('Numpy version: ', np.version.version)
print('Scikit-learn version: ',sklearn.__version__)

ax.print_div('Set list of directories with mask images')

dirs_all = []
for filename in Path(args.dir_in).rglob('*{}*.nii*'.format(args.pattern)):
    dirs_all.append(filename.resolve())

print('Found dirs:',len(dirs_all))

ax.print_div('Extracting and fitting Signatures')

file2load = '{}sign_refs.joblib'.format(df_conf.DIR_SAVE)
parms_refs = load(file2load)
prof_ref = parms_refs['prof_ref']

resols = np.arange(df_conf.RESOLS_INF,df_conf.RESOLS_SUP,df_conf.RESOLS_STEP)
resols = np.insert(resols,0,df_conf.FIT_RES)
prof_vec = np.empty((len(dirs_all),resols.shape[0],df_conf.POINTS))

for ind, mask_path in enumerate(dirs_all):
    img_mask_msp = nib.load(str(mask_path)).get_data()
    if len(img_mask_msp.shape) == 3:
        img_mask_msp = img_mask_msp[args.msp]
    elif len(img_mask_msp.shape) == 2:
        img_mask_msp = img_mask_msp
    else:
        raise Exception('Please, verify shape of nii file, it must be 2 or 3')

    refer_temp = sign.sign_extract(img_mask_msp, resols, df_conf.SMOOTHNESS, df_conf.POINTS)
    prof_vec[ind] = sign.sign_fit(prof_ref, refer_temp, df_conf.POINTS)

print("Segmentations' vector: ", prof_vec.shape)

X_test = prof_vec[:,1:,:] #Filtering the fitting resolution
resols = resols[1:] #Filtering the fitting resolution
resols_ref = np.arange(1,len(resols)+1)

val_norm = parms_refs['val_norm']
X_test_norm = X_test/val_norm

print('Test set: ', X_test_norm.shape)

ax.print_div("Loading pre-trained models")

file2load = '{}arr_models_ind.joblib'.format(df_conf.DIR_SAVE)
d_train = load(file2load)
file2load = '{}ensemble_model.joblib'.format(df_conf.DIR_SAVE)
clf = load(file2load)
res_chs = parms_refs['res_chs']

ax.print_div("Testing...")

svm_ind = np.array([]).reshape(0,X_test.shape[0])
for res_ch in res_chs:
    svm_ind = np.vstack((svm_ind, d_train["string{0}".format(res_ch)].predict_proba(X_test_norm[:,res_ch,:])[:,1]))
svm_ind = svm_ind.T

if args.opt_th == None:
    threshold = parms_refs['opt_th']
    print('Threshold was not passed.')
    print('Optimal threshold was used instead')
else:
    threshold = args.opt_th
print('Threshold used: {}'.format(threshold))

y_pred_probs = clf.predict_proba(svm_ind)[:,1]
y_pred = y_pred_probs > threshold

ax.print_div("Writing output...")

output = pd.DataFrame([])
output['file'] = dirs_all
output['QC_score'] = y_pred_probs
output['output_label'] = y_pred

output.to_csv('{}output.csv'.format(df_conf.DIR_SAVE),index=False)
ax.print_div("Processed input. OK!")
