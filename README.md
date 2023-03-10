# DataScientest bootcamp project : Anomalous sound detection with machine learning and deep learning

This repository hosts the 120-hour project I have carried out throughout my [Data Scientist training](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/), from May to July 2022. This work has been done in collaboration with my two co-learners, [Sylvain Debieu](https://www.linkedin.com/in/sylvain-debieu-662282125/) and [Quentin Rott](https://www.linkedin.com/in/quentin-rott/), under the mentorship of [Thomas Boehler](https://www.linkedin.com/in/thomas-boehler-ba34a744/) (data scientist @DataScientest).

## Objectives
The project addresses the [2nd task](https://dcase.community/challenge2022/task-low-complexity-acoustic-scene-classification) of the [DCASE2022 Challenge](https://dcase.community/challenge2022/index), entitled *Unsupervised Anomalous Sound Detection for Machine Condition Monitoring Applying Domain Generalization Techniques*. The goal is to detect with machine learning and/or deep learning techniques whether a sound recorded from a machine is normal or anomalous, using only normal sounds for training. This is an unsupervised classification problem. This problem is further complicated by the fact that the model should be able to disentangle anomalies from *domain shifts*, *i.e.* from acoustic differences that are not caused by anomalies but by a change of machine parameter regime and/or of environmental noise. 



## Data 
During this training project, we only used the *development dataset* that can be downloaded [here](https://zenodo.org/record/6355122#.ZAs2YR-ZOUk).  Additional data are available on the [challenge webpage](https://dcase.community/challenge2022/task-low-complexity-acoustic-scene-classification).  
The *development dataset* contains **25200 single-channel 10-second audio clips** recorded from 7 machine types (*fan*, *gearbox*, *bearing*, *slider*, *toy car*, *toy train*, and *valve*). Each recording includes both the sounds of the target machine and environmental sounds. For each machine type, the dataset is further split into&nbsp;:
- a *training set* (with normal sounds only) and a *test set* (with normal and anomalous (labeled) sounds) 
- 3 different *sections* (0, 1, and 2). Each section corresponds to a parameter (*e.g.* the rotation velocity for the *bearing*) that has been varied to obtain various recordings within this section. The parameter values are given by the *attributes*, *i.e.* by the suffixes of the file names.  
- a *source domain* and a *target domain*. They correspond to two separated subgroups of *attributes* in each section.

The architecture of the dataset is shown below.
<figure>
    <img src="/images/dev_dataset.png" alt="Structure of the development dataset" style="height: 350px;"/>
    <figcaption>Structure of the <i>development dataset</i>. Figure adapted from the <a target="_blank" href="https://dcase.community/challenge2022/task-low-complexity-acoustic-scene-classification">challenge webpage</a>.</figcaption>
</figure>

## General strategy
Each audio clip is converted to a (mel-)spectrogram using the [librosa](https://librosa.org/doc/latest/index.html#) python package.  

Then, we first address the supervised classification problem, using the test set containing both normal and anomalous (labeled) sounds. Obviously, this task does not follow the challenge rules but is useful to get a first benchmark (and for pedagogical reasons too, the main goal for us was to learn). For that purpose, we use different dimensionality reduction techniques on the spectrograms and then test various standard machine learning techniques (KNN, SVM, random forests, gradient boosting trees) as well as basic neural networks.

Second, we address the unsupervised classification problem using only normal sounds for training. For that purpose, we use an auxiliary task. Since the machine labels are known, we build a supervised classifier to predict those labels or more precisely the probability $p_m$ for a sound to originate from the machine type $m$&nbsp;: The spectrograms are considered as images and computer vision techniques are used. Then, we build an anomaly score using the probabilities $p_m$ and predict that a sound is anomalous if its anomaly score exceeds a given threshold. The same approach is also used by using a section classifier for the auxiliary task.

Note that my co-learners, Quentin and Sylvain, have also tried alternatives anomaly detection approaches based on auto-encoders but such approaches are made difficult by the fact that sounds/spectrograms are very noisy. Their notebooks are not included in this repo. 



## Notebooks  
All notebooks in this repo are mine and are organized as follows&nbsp;:
- **Data exploration&nbsp;:**
    - `ASD_dataviz_GF.ipynb` : To analyse the dataset architecture, listen to sound clips, visualize and compare various spectrograms  of sounds.
- **Supervised classification of normal/anomalous sounds using the (labeled) test subset of the development dataset**&nbsp;:
    - `ASD_supervised_clf_sounds_GF.ipynb` : Various techniques of dimensionality reduction and various classifiers (KNN, SVM, random forests, gradient boosting trees)  have been tested.
    - `ASD_supervised_clf_sounds_DL_GF.ipynb` : Same with basic neural networks. Results are very bad. The notebook has been written at the early stage of the training, with no hindsight on deep learning techniques, and can be mostly ignored.
- **Supervised classification of the type of machine from which sounds originate**&nbsp;:
    - `ASD_supervised_clf_machines_GF.ipynb` : Classification is done with gradient boosting trees (xgboost) after dimensionality reduction (PCA). Quick and dirty notebook written to check that deep learning techniques are more efficient for this task (but the comparison has not been pushed to its fullest conclusion).
- **Unsupervised classification of normal/anomalous sounds with deep learning techniques&nbsp;:**
    - `ASD_clf_sounds_DL_from_machine_clf_GF.ipynb` : Various models have been tested and trained to perform first a supervised classification task of machine type. Then, the idea is that only sounds whose machine type has been well classified (with a high enough score) are classified as normal.
    - `ASD_clf_sounds_DL_from_section_clf_GF.ipynb` : Same by pretraining a neural network to classify the section (0, 1, and 2), machine per machine, instead of the machine type.
    - `ASD_clf_sounds_FaceNet_machine_GF.ipynb` : Spectrograms are embedded into vectors of length 128 using a FaceNet approach on the machine type. Then a random forest classifier is used to classify the machine type. The normal/anomalous class is deduced from these classification scores.
    - `ASD_clf_sounds_FaceNet_section_GF.ipynb` : Same by using a FaceNet approach on the section (0, 1, and 2), machine per machine, instead of the machine type.

## Main conclusions




