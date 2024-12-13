import os
from PIL import Image
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.ensemble import IsolationForest

# Preprocess images
def preprocess_images(image_dir, target_size=(32, 32)):
    image_files = sorted(os.listdir(image_dir))
    images = []
    for file in image_files:
        with Image.open(os.path.join(image_dir, file)) as img:
            img_resized = img.resize(target_size)
            img_array = np.array(img_resized) / 255.0
            images.append(img_array)
    return np.array(images), image_files

# Feature extractor
def extract_features(images):
    model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    images_preprocessed = preprocess_input(images)
    features = model.predict(images_preprocessed)
    return features.reshape(len(images), -1)

# Train Isolation Forest
def train_anomaly_model(features):
    clf = IsolationForest(contamination=0.1, random_state=42)
    clf.fit(features)
    return clf

# Save labels
def predict_and_save_labels(model, features, filenames, output_file='labels2.txt'):
    predictions = model.predict(features)
    labels = ["bad" if pred == -1 else "good" for pred in predictions]
    with open(output_file, 'w') as f:
        for label in labels:
            f.write(f"{label}\n")
    print(f"Labels saved to {output_file}")

# Main
if __name__ == "__main__":
    dataset_dir = "C:\\Users\\Jugge\\Downloads\\AnomalyDetectionDataset\\allfrog"
    images, image_filenames = preprocess_images(dataset_dir)
    
    print("Extracting features...")
    features = extract_features(images)
    
    print("Training Isolation Forest...")
    anomaly_model = train_anomaly_model(features)
    
    print("Predicting and saving labels...")
    predict_and_save_labels(anomaly_model, features, image_filenames)
