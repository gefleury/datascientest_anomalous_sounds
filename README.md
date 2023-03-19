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

The architecture of the dataset is shown below. Data are not uploaded on this repo but for clarity sake, the data folder structure is shown in the folder `data/data/`. 
<br />
<figure>
    <img src="/images/dev_dataset.png" alt="Structure of the development dataset" style="height: 350px;"/>
    <figcaption>Structure of the <i>development dataset</i>. Figure adapted from the <a target="_blank" href="https://dcase.community/challenge2022/task-low-complexity-acoustic-scene-classification">challenge webpage</a>.</figcaption>
</figure>

## General strategy
Each audio clip is converted to a (mel-)spectrogram using the [librosa](https://librosa.org/doc/latest/index.html#) python package. 

Then, we first address the supervised classification problem, using the test set containing both normal and anomalous (labeled) sounds. Obviously, this task does not follow the challenge rules but is useful to get a first benchmark (and for pedagogical reasons too, the main goal for us was to learn). Using [scikit-learn](https://scikit-learn.org/stable/), we test different dimensionality reduction methods on the spectrograms and then various standard machine learning techniques (KNN, SVM, random forests). Gradient boosting trees implemented with [xgboost](https://xgboost.readthedocs.io/en/stable/) turn out to give the best results. Basic neural networks implemented with [tensorflow](https://www.tensorflow.org/overview) are also (quickly) tried.

Second, we address the unsupervised classification problem using only normal sounds for training. For that purpose, we use an auxiliary task. Since the machine labels are known, we build a supervised classifier to predict those labels or more precisely the probability $p_m$ for a sound to originate from the machine type $m$&nbsp;: The spectrograms are considered as images and computer vision techniques (implemented with [tensorflow](https://www.tensorflow.org/overview)) are used. Then, we build an anomaly score using the probabilities $p_m$ and predict that a sound is anomalous if its anomaly score exceeds a given threshold. The same approach is also used by using a section classifier for the auxiliary task.

Note that my co-learners, Quentin and Sylvain, have also tried alternatives anomaly detection approaches based on auto-encoders but such approaches are made difficult by the fact that sounds/spectrograms are very noisy. Their notebooks are not included in this repo. 



## Notebooks  
All notebooks in this repo are mine and are organized as follows&nbsp;:
- **Data exploration&nbsp;:**
    - [`ASD_dataviz.ipynb`](notebooks/ASD_dataviz.ipynb) : To analyse the dataset architecture, listen to sound clips, visualize and compare various spectrograms  of sounds.
- **Preprocessing&nbsp;:**
    - [`ASD_preprocessing_melspectro313x128.ipynb`](notebooks/ASD_preprocessing_melspectro313x128.ipynb) : To build the melspectrograms stored in the folder `data/Features/melspec_313_128` and loaded in the notebooks dedicated to the unsupervised task. In other notebooks, the spectrograms are computed on the fly.
- **Supervised classification of normal/anomalous sounds**&nbsp;:
    - [`ASD_supervised_clf_sounds.ipynb`](notebooks/ASD_supervised_clf_sounds.ipynb) : Various techniques of dimensionality reduction and various classifiers (KNN, SVM, random forests, gradient boosting trees)  have been tested.
    - [`ASD_supervised_clf_sounds_DL.ipynb`](notebooks/ASD_supervised_clf_sounds_DL.ipynb) : Same with basic neural networks. Results are very bad. The notebook has been written at the early stage of the training, with no hindsight on deep learning techniques, and can be mostly ignored.
- **Supervised classification of the machine types**&nbsp;:
    - [`ASD_supervised_clf_machine.ipynb`](notebooks/ASD_supervised_clf_machine.ipynb) : Classification is done with gradient boosting trees (xgboost) after dimensionality reduction (PCA). Quick and dirty notebook written to check that deep learning techniques are more efficient for this task (but the comparison has not been pushed to its fullest conclusion).
- **Unsupervised classification of normal/anomalous sounds with deep learning techniques&nbsp;:**
    - [`ASD_clf_sounds_DL_from_machine_clf.ipynb`](notebooks/ASD_clf_sounds_DL_from_machine_clf.ipynb) : Various models have been tested and trained to perform first a supervised classification task of machine type. Then, the idea is that only sounds whose machine type has been well classified (with a high enough score) are classified as normal.
    - [`ASD_clf_sounds_DL_from_section_clf.ipynb`](notebooks/ASD_clf_sounds_DL_from_section_clf.ipynb) : Same by pretraining a neural network to classify the section (0, 1, and 2), machine per machine, instead of the machine type.
    - [`ASD_clf_sounds_FaceNet_machine.ipynb`](notebooks/ASD_clf_sounds_FaceNet_machine.ipynb) : Spectrograms are embedded into vectors of length 128 using a FaceNet approach on the machine type. Then a random forest classifier is used to classify the machine type. The normal/anomalous class is deduced from these classification scores.
    - [`ASD_clf_sounds_FaceNet_section.ipynb`](notebooks/ASD_clf_sounds_FaceNet_section.ipynb) : Same by using a FaceNet approach on the section (0, 1, and 2), machine per machine, instead of the machine type.

## Main conclusions
The *supervised* classification task of normal/anomalous sounds gives AUC scores in the range $[0.8, 0.98]$ depending on the machine type, with simple machine learning methods. Those scores could be likely improved with model optimization or with the use of more evolved neural networks (only basic ones have been tested for practicing). Note that the *supervised* classification task of the machine types gives better results.   

The *unsupervised* classification task of normal/anomalous sounds is (as expected) much more difficult and we find AUC scores in the range $[0.5, 0.8]$ depending on the machine type. The problem is hindered by *(i)* the presence of noise in the recordings, so as it is often difficult to hear (or to see in a spectrogram) whether a sound is normal or not, and *(ii)* the existence of two (so-called source and target) domains. There is obviously room for improvement, from the pre-processing step (use of filters, of phase spectrograms in addition to amplitude spectograms, tuning of spectrogram parameters, ...) to the optimization of deep learning methods (use of transfer learning, of RNNs, test with other loss functions, ...). Before running long calculations, the methodology itself could be questioned : for instance, it might be interesting to train a model to classify normal/anomalous sounds by using sounds from other machines as anomalous sounds.  

:trophy: Visit the [challenge results webpage](https://dcase.community/challenge2022/task-low-complexity-acoustic-scene-classification-results) for more ideas.

## Reports (in french, written my co-learners)
Project report : [ASD_report.pdf](reports/ASD_report.pdf)  
Defense slides : [ASD_slides.pdf](reports/ASD_slides.pdf)  





