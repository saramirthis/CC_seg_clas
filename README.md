# A framework for quality control of corpus callosum segmentation in large-scale studies

## Reproducible paper

This Readme file holds instructions for reproduction of the work **A framework for quality control of corpus callosum segmentation in large-scale studies**

![Alt text](images/Graphical_abstract.png?raw=true "Title")

## Environment, used libraries and dependencies

* Python 3.5.4
* Numpy 1.12.1
* Scipy 0.19.1
* Matplotlib 2.0.2
* Scikit-learn 0.19.0
* aux library (My library avaliable on: https://github.com/wilomaku/CC_seg_clas/tree/master/aux)

## Workflow

This framework receives binary mask, in nifti format (.nii.gz or .nii), and returns a quality score ranging from 0% for completely correct segmentation to 100% for completely incorrect segmentation. The model was trained in 481 corpus callosum segmentations and tested in 207 samples.

![Alt text](images/Framework_quality.png?raw=true "Title")

## Files structure

The structure of the tree folder is located in the root of the git repository **wilomaku/CC_seg_clas**:

* aux: Directory with main (func.py) and auxiliar (aux_fnc.py) functions. The default configuration (**default_config.py**) for train and test the framework is available too. The file labels.csv was only used to train the model and this file is not longer useful for the user.
* funcs: Empty directory. Only used for compatibility purposes.
* images: Necessary images for notebook visualization. The user should not require to do anything here.
* saves: Saved models used to test the framework. **arr_models_ind.joblib** has the saved models for the individual classifiers,  **ensemble_model.joblib** is the ensemble final model and **sign_refs.joblib** has the partial extracted signatures.

These files are located in the root of the git repository **wilomaku/CC_seg_clas**:

* README.md: File with the repository instructions.
* main.ipynb: Jupyter notebook with the reproducible paper in a step-by-step fashion.
* main.py: Script to train and test the quality control framework. Useful if you have your won dataset to train.
* test.py: Script to test among your segmentations. Useful if you want to test on your own dataset.

## Instructions to use this repository:

Please, pay attention to these instructions and follow carefully.

1. Move to your directory: cd <your_dir>
2. Clone the repository: git clone https://github.com/wilomaku/CC_seg_clas.git
3. If you want to run/train/test any file on this framework, first you need to change the DIR_BAS and DIR_SAVE variables to your paths in **default_config.py**.

### Test script (You want to perform quality control on your own segmentation dataset)

4. You need to have your dataset. The framework only works with binary nifti masks (.nii or .nii.gz are the only extensions accepted). Your masks must be in a folder (<your_test_dir>) either directly in the root of <your_test_dir>:

<your_test_dir>
│   mask1.nii
│   mask2.nii.gz

or every mask in its respective folder:

<your_test_dir>
└───folder1
│   │   mask1.nii
└───folder2
│   │   mask2.nii.gz

Also, it is expected the nifti mask files to be 2D (in sagittal view) or 3D (in which case, the first dimension refers to the sagittal view). Copy the test dataset into your directory: cp <your_dir>/<your_test_dir>

5. Run the test script providing the proper arguments: python 3 test.py <dir_in> <pattern> <msp> <-opt_th>

dir_in: Databse Input directory.
pattern: Pattern present in all the nifti file names. If no particular pattern is present in your name files.
msp: Slice to be selected on sagittal plane, only for 3D nifti masks.
-opt_th: Decision threshold to separate classes (If this value is not passed, the optimal threshold is used instead).

Example: python 3 test.py /home/jovyan/work/dataset/ '' 90 0.5

## Instructions to execute train script (You want to train the framework using your dataset)

Please, pay attention to these instructions and follow carefully. Besides Jupyter notebook installed, you must have a work directory with three elements: dataset directory, ipyhton script and library directory with the necessary functions.

## Instructions to execute in Docker image

Because the model is dependant on the Scikit-learn version, we used a Docker image to guarantee reproducibility:

Install Docker: https://docs.docker.com/install/
Download Docker image: docker pull miykael/nipype_level0 (https://hub.docker.com/r/miykael/nipype_level0)
Run Docker image on Jupyter mode: docker run -p 8889:8888 -v ~/Documents/:/home/jovyan/work -it miykael/nipype_level0
Run Docker image on terminal mode: docker run -p 8889:8888 -v ~/Documents/:/home/jovyan/work -it miykael/nipype_level0 /bin/bash

## Original publication

We are pleased if you use our framework, we ask you to kindly cite our paper:

Herrera, W. G., Pereira, M., Bento, M., Lapa, A. T., Appenzeller, S., & Rittner, L. (2020). A framework for quality control of corpus callosum segmentation in large-scale studies. Journal of Neuroscience Methods, 108593. (https://doi.org/10.1016/j.jneumeth.2020.108593).

Questions? Suggestions? Please write to wjgarciah@unal.edu.co
MIT License Copyright (c) 2019 William Herrera
