# Project Introduction
Accident Detection from CCTV Footage Using CNN
In this project, the goal is to develop a Convolutional Neural Network (CNN) model that can automatically detect accidents from CCTV footage. With the growing number of surveillance cameras in urban areas, there is a significant opportunity to use this data to enhance public safety by quickly identifying and responding to traffic accidents.

# Dataset Overview
The dataset used for this project, available on Kaggle, contains video footage extracted from CCTV cameras. The dataset is labeled to indicate whether an accident has occurred in each video clip.

# Structure
The dataset includes various video frames or sequences, which are either labeled as containing an accident or not.
Classes: There are typically two classes — "Accident" and "No Accident."
Format: The data may consist of video files, which are broken down into frames for processing.
Objective
The primary objective of this project is to build a CNN model that can accurately classify video frames or sequences into "Accident" and "No Accident" categories. This classification can help in real-time monitoring systems to automatically alert authorities or initiate other safety measures in case of an accident.

# Methodology
Data Preprocessing:

Convert video files into frames.
Normalize the pixel values to ensure the model training is stable.
Resize the frames to a consistent size that matches the input requirements of the CNN model (e.g., 224x224 pixels).
Model Architecture:

A CNN architecture will be used to extract spatial features from the frames.
Multiple convolutional layers will capture various aspects of the images.
Dense layers will follow to perform the final classification.
Training:

The model will be trained on labeled data with images marked as containing an accident or not.
Techniques like data augmentation may be used to improve model generalization.
Early stopping, dropout, and batch normalization will be implemented to prevent overfitting.
Evaluation:

The model’s performance will be evaluated using metrics such as accuracy, precision, recall, and F1-score.
A confusion matrix may be used to further understand the model's predictions.
Deployment:

Once trained, the model can be deployed in a Streamlit app or another suitable platform to enable real-time detection of accidents from live CCTV feeds.

Data Source: https://www.kaggle.com/datasets/ckay16/accident-detection-from-cctv-footage

Streamlit app: https://accident-detection-deeplearning-model-gctm55abw2cexnwswghcbp.streamlit.app/
