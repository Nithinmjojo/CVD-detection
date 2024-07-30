Cardio-Vascular Disease Detection and Classification Using Convolutional Neural Network and Transfer Learning



Introduction

This repository contains a Python implementation of a Convolutional Neural Network (CNN) model for classifying cardiac arrhythmias. The model leverages transfer learning from the pre-trained VGG19 model and incorporates data augmentation techniques to improve performance.

Dataset

The project utilizes the MIT-BIH Arrhythmia Database, which contains recordings of ECG signals from a wide range of subjects. The dataset is split into training, validation, and testing sets for model development and evaluation.

Model Architecture


The CNN model is based on the VGG19 architecture, with the final layers modified to suit the arrhythmia classification task. Data augmentation techniques, including rescaling, shearing, and rotation, are applied to increase data variability and improve model generalization.

Results

The proposed model achieves an accuracy of 94.87% in classifying six arrhythmia classes: left bundle branch block, normal, premature atrial contraction, premature ventricular contraction, right bundle branch block, and ventricular fibrillation.   

