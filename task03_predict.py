import cv2
import numpy as np
from skimage.feature import hog
import joblib
import matplotlib.pyplot as plt
import sys

MODEL_PATH = "svm_cat_dog_model.joblib"
TEST_IMAGE_PATH = r"C:\Users\adhi_\OneDrive\Desktop\Codes\SCT_ML_03\test1\101.jpg"
IMAGE_SIZE = (128, 128)

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    img = cv2.resize(img, IMAGE_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )
    return features.reshape(1, -1), img

svm = joblib.load(MODEL_PATH)

features, img = extract_features(TEST_IMAGE_PATH)
if features is None:
    print("Image not found")
    sys.exit(1)

prob = svm.predict_proba(features)[0]
pred_index = np.argmax(prob)
pred_class = svm.classes_[pred_index]
label = "Cat" if pred_class == 0 else "Dog"

print(f"Prediction: {label}")
print(f"Confidence - Cat: {prob[0]*100:.2f}%, Dog: {prob[1]*100:.2f}%")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax[0].set_title("Input Image")
ax[0].axis("off")

labels = ["Cat", "Dog"]
confidences = prob * 100
ax[1].bar(labels, confidences)
ax[1].set_ylim(0, 100)
ax[1].set_ylabel("Confidence (%)")
ax[1].set_title(f"Prediction: {label}")

plt.tight_layout()
plt.show()
