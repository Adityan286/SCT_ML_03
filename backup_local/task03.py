import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

IMAGE_SIZE = (128, 128)

def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMAGE_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )
    return features

def load_data(folder):
    X, y = [], []
    for file in os.listdir(folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif')):
            path = os.path.join(folder, file)
            X.append(extract_features(path))
            y.append(0 if file.startswith("cat") else 1)
    return np.array(X), np.array(y)

print("Loading training data...")
X_train, y_train = load_data("train/train")

print("Loading testing data...")
X_test, y_test = load_data("test1")

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

svm = SVC(kernel="linear", probability=True)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))

joblib.dump(svm, "svm_cat_dog_model.joblib")
print("Model saved as svm_cat_dog_model.joblib")
