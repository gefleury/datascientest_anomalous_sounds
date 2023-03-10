# DataScientest bootcamp project : Anomalous sound detection with machine learning and deep learning

This repository hosts the 120-hour project I have carried out throughout my [Data Scientist training](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/), from May to July 2022. This work has been done in collaboration with my two co-learners, [Sylvain Debieu](https://www.linkedin.com/in/sylvain-debieu-662282125/) and [Quentin Rott](https://www.linkedin.com/in/quentin-rott/), under the mentorship of [Thomas Boehler](https://www.linkedin.com/in/thomas-boehler-ba34a744/) (data scientist @DataScientest).

## Objectives
The project addresses the [2nd task](https://dcase.community/challenge2022/task-low-complexity-acoustic-scene-classification) of the [DCASE2022 Challenge](https://dcase.community/challenge2022/index), entitled *Unsupervised Anomalous Sound Detection for Machine Condition Monitoring Applying Domain Generalization Techniques*. 

## General strategy


## Data 
During this training project, we only used the *development dataset* that can be downloaded [here](https://zenodo.org/record/6355122#.ZAs2YR-ZOUk).  Additional data are available on the [challenge webpage](https://dcase.community/challenge2022/task-low-complexity-acoustic-scene-classification).  
The *development dataset* contains 25200 single-channel 10-second audio clips recorded from 7 machine types (*fan*, *gearbox*, *bearing*, *slider*, *toy car*, *toy train*, and *valve*). Each recording includes both the sounds of the target machine and environmental sounds. 

## Notebooks  
All notebooks in this repo are mine and are organized as follows :
- **Data exploration :**
    - `ASD_dataviz_GF.ipynb` : To analyse the dataset architecture, listen to sound clips, visualize and compare various spectrograms  of sounds.
- **Supervised classification of normal/anomalous sounds using the (labeled) test subset of the development dataset** :
    - `ASD_supervised_clf_sounds_GF.ipynb` : Various techniques of dimensionality reduction and various classifiers (KNN, SVM, random forests, gradient boosting trees)  have been tested.
    - `ASD_supervised_clf_sounds_DL_GF.ipynb` : Same with basic neural networks. Results are very bad. The notebook has been written at the early stage of the training, with no hindsight on deep learning techniques, and can be mostly ignored.
- **Supervised classification of the type of machine from which sounds originate, using the (labeled) train dataset** :
    - `ASD_supervised_clf_machines_GF.ipynb` : Classification is done with gradient boosting trees (xgboost) after dimensionality reduction (PCA). Quick and dirty notebook written to check that deep learning techniques are more efficient for this task (but the comparison has not been pushed to its fullest conclusion).
- **Unsupervised classification of normal/anomalous sounds with deep learning techniques :**
    - `ASD_clf_sounds_DL_from_machine_clf_GF.ipynb` : Various models have been tested and trained to perform first a supervised classification task of machine type. Then, the idea is that only sounds whose machine type has been well classified (with a high enough score) are classified as normal.
    - `ASD_clf_sounds_DL_from_section_clf_GF.ipynb` : Same by pretraining a neural network to classify the section (1, 2 or 3), machine per machine, instead of the machine type.
    - `ASD_clf_sounds_FaceNet_machine_GF.ipynb` : Spectrograms are embedded into vectors of length 128 using a FaceNet approach on the machine type. Then a random forest classifier is used to classify the machine type. The normal/anomalous class is deduced from these classification scores.
    - `ASD_clf_sounds_FaceNet_section_GF.ipynb` : Same by using a FaceNet approach on the section (1, 2 or 3), machine per machine, instead of the machine type.

## Main conclusions




