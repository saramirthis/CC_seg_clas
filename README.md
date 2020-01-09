# A framework for quality control of corpus callosum segmentation in large-scale studies

## Reproducible paper

This Readme file holds instructions for reproduction of the work **A framework for quality control of corpus callosum segmentation in large-scale studies**

## Environment, used libraries and dependencies

* Python 3.7.1
* Numpy 1.15.4
* Scipy 1.1.0
* Matplotlib 3.0.2
* aux library (My library avaliable on: https://github.com/wilomaku/CC_seg_clas/tree/master/aux)

## Workflow

This framework receives a binary mask, in nifti format (.nii.gz or .nii), and returns a quality score ranging from 0% for completely correct segmentation to 100% for completely incorrect segmentation. The model was trained in 481 corpus callosum segmentations and tested in 207 samples.

![Alt text](figures/workflow_simp.png?raw=true "Title")

## Files structure

This structure tree folder is located in the root of the git repository **wilomaku/CC_seg_clas**:

* aux: Directory with main (func.py) and auxiliar (aux_fnc.py) functions. The default configuration (default_config.py) for train and test the framework is available too. The file labels.csv is only used to train the model and this file is not useful for the user.
* funcs: Empty directory. Only used to compatibility purposes.
* images: Necessary images for notebook visualization. The user should not require to do anything here.
* saves: Saved models used to test the framework. arr_models_ind.joblib has the saved models for the individual classifiers,  ensemble_model.joblib is the ensemble final model and sign_refs.joblib has the partial extracted signatures.

These files are located in the root of the git repository **wilomaku/CC_seg_clas**:

* README.md: File with the repository instructions.
* main.ipynb: Jupyter notebook with the reproducible paper in a step-by-step fashion.
* main.py: Script to train and test the quality control framework.
* test.ipynb: Jupyter notebook to test the quality control framework in a step-by-step fashion.

## Instructions to execute notebook

Please, pay attention to these instructions and follow carefully. Besides Jupyter notebook installed, you must have a work directory with three elements: dataset directory, ipyhton script and library directory with the necessary functions.

Questions? Suggestions? Please write to wjgarciah@unal.edu.co
