# A framework for quality control of corpus callosum segmentation in large-scale studies

## Reproducible paper

This Readme file holds instructions for reproduction of the work **A framework for quality control of corpus callosum segmentation in large-scale studies**

## Environment, used libraries and dependencies

* Python 3.5.4
* Numpy 1.12.1
* Scipy 0.19.1
* Matplotlib 2.0.2
* Scikit-learn 0.19.0
* aux library (My library avaliable on: https://github.com/wilomaku/CC_seg_clas/tree/master/aux)

## Workflow

This framework receives binary mask, in nifti format (.nii.gz or .nii), and returns a quality score ranging from 0% for completely correct segmentation to 100% for completely incorrect segmentation. The model was trained in 481 corpus callosum segmentations and tested in 207 samples.

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
* test.py: Script to test among your segmentations.

## Instructions to execute test script (You have your own segmentations and you want to perform quality control)

Please, pay attention to these instructions and follow carefully. Besides Jupyter notebook installed, you must have a work directory with three elements: dataset directory, ipyhton script and library directory with the necessary functions.

## Instructions to execute train script (You want to train the framework using your dataset)

Please, pay attention to these instructions and follow carefully. Besides Jupyter notebook installed, you must have a work directory with three elements: dataset directory, ipyhton script and library directory with the necessary functions.

## Original publication

We are pleased if you use our framework or this code, we ask you to kindly cite our paper:

Herrera, W. G., Pereira, M., Bento, M., Lapa, A. T., Appenzeller, S., & Rittner, L. (2020). A framework for quality control of corpus callosum segmentation in large-scale studies. Journal of Neuroscience Methods, 108593. (https://doi.org/10.1016/j.jneumeth.2020.108593).

Questions? Suggestions? Please write to wjgarciah@unal.edu.co
MIT License Copyright (c) 2019 William Herrera
