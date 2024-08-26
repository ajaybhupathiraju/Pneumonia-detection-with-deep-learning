# Pneumonia detection in childrens using deep learning

## Table of Contents
- [Project Overview](#project-overview)
- [Data Sources](#data-sources)
- [Tools and Frameworklib](#tools-and-frameworklib)
- [EDA](#exploratory-data-analysis)
- [Assumptions](#assumptions)
- [Performance metrics](#performance-metrics)
- [Loss function](#loss-function)
- [Choosing a Model](#choosing-a-model)
- [Training the Model](#training-the-model)
- [Making Predictions](#making-predictions)
- [Deployee model](#deployee-model)
- [Recommendations](#recommendations)

## Project Overview
  Pneumonia is an infection in lungs caused by bacteria, viruses or fungi. Pneumonia causes lung tissue to swell (inflammation) and can cause fluid or pus in your lungs. 
  
  Pneumonia is a significant public health issue that often requires timely and accurate diagnosis for effective treatment. This project aims to develop a deep learning model, specifically using Convolutional 
  Neural Networks (CNN), to automatically detect pneumonia from chest X-ray images. The proposed model focuses on efficiently identifying pneumonia, assisting healthcare professionals in diagnosis.

## Data Sources
   Chest X-ray Images from [kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## Tools and Frameworklib

  Programming Language   : Python
  
  Deep Learning Framework: TensorFlow and Keras,Scikit-learn,Matplotlib
  
  Dataset                : Chest X-ray dataset
  
  Tools                  : Jupyter Notebook / Pycharm

## Exploratory Data Analysis

## Assumptions

## Performance metrics

## Loss function

## Choosing a model

we will fit below all models and choose best fit model.

GPU configuration - RTX 4060Ti 16GB with 4352 cuda cores

Model                 | Train accuracy  | Test accuracy |   log_loss    |  area under curve (AUC)   |  Cross Validation  |  Hyperparameters   |
--------------------- | -------------   | ------------- | ------------- |   ---------------------   |  ----------------  | ----------------   | 
VGG16                 |   0.88          |   0.83        |   0.32        |       0.93                |    cv=5            |   K=6              |
Resnet50              |   0.83          |   0.81        |   0.37        |       0.91                |    cv=5            |   alpha : 0.001    |
MobileNet             |   0.82          |   0.81        |   0.38        |       0.9                 |    cv=5            |   alpha=1          |
ResNet50V2            |   0.99          |   0.93        |   0.24        |       0.95                |    cv=5            |   n_estimators=95  |
Custom CNN            |   0.99          |   0.93        |   0.24        |       0.95                |    cv=5            |   n_estimators=95  |


## Training the model
[**ML model code**](ML_Models.ipynb)

## Making predictions
[**ML model code**](ML_Models.ipynb)

## Deployee model

## Recommendations
