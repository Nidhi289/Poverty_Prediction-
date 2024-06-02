# Poverty Prediction Using Satellite Imagery

## Overview

This project aims to predict poverty levels using satellite images and additional socio-economic data. The project includes a web application that allows users to upload satellite images and input data to predict poverty levelse prediction is achieved through a convolutional neural network (CNN) built with TensorFlow and TFLearn, and a web interface created using Flask.. The application uses a convolutional neural network (CNN) for image-based predictions and linear regression for predictions based on socio-economic data.

## Table of Contents
1. [Project Structure]
2. [Requirements]
3. [Setup Instructions]
4. [Dataset Preparation]
5. [Training the Model]
6. [Running the Flask Application]
7. [Application Routes]
8. [Model Deatails]
9. [Web Application Usage]
10. [Notes]
11. [Conclusion]
12. [Troubleshooting]
13. [Features]
14. [Authors]
15. [Contact]
16. [Feedback]


## Project Structure

- `app.py`: Main application file containing the Flask web app and prediction logic.
- `templates/`: Directory containing HTML templates for the web pages.
- `static/images/`: Directory where uploaded images are stored.
- `user_data.db`: SQLite database for storing user registration and login information.
- `city_data.csv`: CSV file containing socio-economic data for cities.

## Requirements
- Python 3.7.1
- Required Python packages are listed in `requirements.txt`

## Setup Instructions

1. **Extracting the file :**

Extract the poverty-pridication.zip file into the folder where you want to create and run the application.

2. **Navigate to the project directory :**
cd ./poverty-prediction

3. **Install Dependencies :**

pip install -r requirements.txt

Ensure you have the following libraries installed:
- Python 3.6+
- Flask
- TensorFlow
- TFLearn
- NumPy
- OpenCV
- Matplotlib
- SQLite3
- Pandas
- Scikit-learn
- Keras
- tqdm


## Dataset Preparation

Organize Your Dataset

Place your training images in a directory named training/.

Place your testing images in a directory named testing/.

1. **Create Training Data :**

Run the script to create training data:

python create_train_data.py

This will generate a file named train_data.npy containing the preprocessed training data.

2. **Create Test Data :**

Run the script to create test data:

python process_test_data.py

This will generate a file named test_data.npy containing the preprocessed testing data.

## Training the Model

**Train the Model**

Run the training script:

python train_model.py

This script will:

- Load the training data from train_data.npy.
- Define the CNN architecture.
- Train the model.
- Save the trained model.
- Checkpoints and Logs

During training, model checkpoints and logs will be saved in the log/ directory.

## Running the Flask Application
1. **Start the Flask Server**

Run the Flask application:

python app.py 
This will start the server at http://127.0.0.1:5000/.

2. **Upload and Classify Images :**

- Open your web browser and go to http://127.0.0.1:5000/.
- Upload an image to classify its poverty level.
- The application will display the predicted poverty level and the model's confidence scores.

3. **Usage**

Homepage

- Visit the homepage at http://127.0.0.1:5000/.
- You will see a form to upload an image.
- Upload an Image

- Choose an image file from your local system.
- Click on the "Upload" button.
- View Results

After uploading, the server will process the image and display the predicted poverty level along with confidence scores.

## Application Routes

- `/`: Home page with options to login or register.
- `/home`: Main page after login.
- `/userlog`: User login route.
- `/userreg`: User registration route.
- `/image`: Upload and process satellite images for poverty prediction.
- `/predict`: Predict poverty level based on socio-economic data.
- `/predict2`: Alternative poverty prediction based on predefined rules.


## Model Details
The model architecture consists of:

- Convolutional layers with ReLU activation.
- Max-pooling layers.
- Fully connected layers.
- Dropout for regularization.
- Softmax output layer for classification into three categories.


The model is trained with:

- Adam optimizer.
- Categorical crossentropy loss.
- A learning rate of 1e-3.
- Results

The model will output:

- Predicted poverty level (low, medium, high).
- Confidence scores for each class.

## Web Application Usage

- **Login/Register**: Use the provided forms to either log in or register a new user.
- **Home**: Choose the operation need to be done.
- **Input Methods**: Select the method of predicting poverty.
- **Results**: View the prediction results for the uploaded image.



### Imports
```
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, render_template, url_for, request
import sqlite3
import cv2
import shutil
```

- **NumPy:** For numerical operations.
- **OS:** For interacting with the operating system.
- **Shuffle:** To randomize data.
- **TQDM:** To show progress bars.
- **TFLearn:** For building neural networks.
- **TensorFlow:** Backend for TFLearn.
- **Matplotlib:** For plotting images.
- **Flask:** For creating the web application.
- **SQLite3:** For handling the SQLite database.
- **OpenCV:** For image processing.
- **Shutil:** For file operations.



### Notes

- The model for image-based poverty prediction is loaded from a pre-trained model file (`poverty-{LR}-2conv-basic.model`).
- Ensure that the required data files (`city_data.csv`, `user_data.db`, and any image files for testing) are present in the appropriate directories.

## Conclusion

This project demonstrates the integration of machine learning models with a web application to predict poverty levels using satellite images and socio-economic data. The combination of Flask, TFLearn, and various Python libraries enables the creation of a comprehensive and interactive tool for poverty prediction.




## Troubleshooting
Common Issues

- Ensure all dependencies are installed correctly.
- Verify the structure and naming convention of your dataset.
- Check for correct paths in the script.
- Resetting TensorFlow Graph



## Features

- Different ways of input methods to predict poverty.
- Display the reason for the predicted povert level.


## Authors

Nidarshan N  (1SK20CS028) 

Chandan J P (1SK21CS405)

Praveen Gouda (1SK21CS410)

Venkatesha K P (1SK21CS412)


## Contact 

Contact information
## Feedback

If you have any feedback, please reach out to us at povertypredication@gmail.com

