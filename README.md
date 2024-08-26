# Pneumonia detection using deep learning

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
1. Since the dataset contains x-ray images of different dimensions to make it consistent, I converted the dimensions to a standard size (256 X 256).
2. It's a huge dataset. Due to limitations of my GPU (RTX 4060Ti 16GB, 128-bit) and time constraints, I could not run algorithms at higher epochs.
3. Reused transfer learning models [ which is alredy trained by large dataset ] to reduce training time and improve better performance.

## Performance metrics
confusion_matrix, accuracy_score,roc_curve

## Loss function
binary crossentropy

## Choosing a model

I will fit below all models and choose best fit model.

Note : GPU configuration - RTX 4060Ti 16GB with 4352 cuda cores

Model                    | Train accuracy  | Test accuracy |   Train loss  | Test/Validation loss |  epochs |    Hyperparameters                      |
------------------------ | -------------   | ------------- | ------------- |  -------------       | --------|  -----------------------------          | 
VGG16 pretrain           |   0.75          |   0.62        |   0.56        |    0.69              |   20    |   optimizer = adam,learning_rate=0.0001 |
Resnet50                 |   0.74          |   0.62        |   0.57        |    0.69              |   20    |   optimizer = adam,learning_rate=0.0001 |
MobileNet                |   0.99          |   0.62        |   0.001       |    157               |   20    |   optimizer = adam,learning_rate=0.0001 |
ResNet50V2               |   0.97          |   0.62        |   0.06        |    5.03              |   20    |   optimizer = adam,learning_rate=0.0001 |
Custom CNN VGG16         |   0.74          |   0.62        |   0.57        |    0.69              |   20    |   optimizer = adam,learning_rate=0.0001 |


## Training the model
[**ML model code**](Pneumonia_Detection.ipynb)

## Making predictions
[**ML model code**](Pneumonia_Detection.ipynb)

## Deployee model

## Recommendations
