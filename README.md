🐱🐶 Cats vs Dogs Image Classification using SVM
This project is part of Task 03 of my Machine Learning Internship at SkillCraft Technology. The goal of this task is to build an image classification system that can accurately distinguish between cats and dogs using Support Vector Machines (SVM) and Histogram of Oriented Gradients (HOG) features.

📌 Project Overview
Image classification is a fundamental problem in computer vision. Since traditional machine learning models cannot directly process raw images, this project uses HOG feature extraction to convert images into numerical representations, which are then used to train an SVM classifier.
The project is divided into two main stages:

Training the model
Predicting new images using the trained model
📂 Project Structure
SCT_ML_03/
│
├── task03.py                 # Training script
├── task03_predict.py         # Prediction script
├── svm_cat_dog_model.joblib  # Trained SVM model
│
├── train/
│   └── train/
│       ├── cat.0.jpg
│       ├── dog.0.jpg
│       └── ...
│
├── test1/
│   ├── cat.101.jpg
│   ├── dog.101.jpg
│   └── ...
│
└── test.jpg                  # Image used for prediction
🧠 Approach

1️⃣ Image Preprocessing
Images are resized to 128 × 128 pixels
Converted to grayscale for feature extraction

2️⃣ Feature Extraction
Histogram of Oriented Gradients (HOG) is used to extract edge and texture features

3️⃣ Model Training
A Support Vector Machine (SVM) with a linear kernel is trained
probability=True is enabled to allow confidence-based predictions

4️⃣ Model Saving
The trained model is saved using Joblib for reuse

5️⃣ Prediction
The saved model is loaded
A new image is classified as Cat or Dog
Prediction confidence is visualized

🛠️ Technologies Used
Python
OpenCV
scikit-learn
scikit-image
NumPy
Matplotlib
Joblib

⚙️ Installation
Make sure you are using Python 3.11 (recommended for ML compatibility).
Install required libraries:
pip install opencv-python scikit-image scikit-learn matplotlib joblib numpy
🚀 How to Run
🔹 Train the Model
python task03.py

This will:
Train the SVM model
Save the trained model as svm_cat_dog_model.joblib
🔹 Predict on a New Image
Place an image as test.jpg in the project folder

Run:
python task03_predict.py

The output will display:
Predicted label (Cat / Dog)
Confidence scores
Image with prediction visualization

📊 Output
Classification result (Cat or Dog)
Confidence percentage for each class
Visualization of prediction

🎯 Learning Outcomes
Understanding image preprocessing techniques
Applying HOG feature extraction
Training and evaluating SVM models
Saving and loading ML models
Separating training and inference pipelines

📌 Internship Task
This project was completed as part of the SkillCraft Technology Machine Learning Internship, focusing on applying classical machine learning techniques to real-world image classification problems.
