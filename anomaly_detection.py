import os
from PIL import Image
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.ensemble import IsolationForest

# Step 1: Preprocess Images
def preprocess_images(image_dir, target_size=(32, 32)):
    """
    Load and preprocess images from the given directory.
    Args:
        image_dir (str): Path to the directory containing images.
        target_size (tuple): Desired image size (width, height).
    Returns:
        np.ndarray: Preprocessed image data.
        list: Corresponding image filenames (sorted lexicographically).
    """
    image_files = sorted(os.listdir(image_dir))
    images = []
    for file in image_files:
        file_path = os.path.join(image_dir, file)
        with Image.open(file_path) as img:
            img_resized = img.resize(target_size)  # Resize to target size
            img_array = np.array(img_resized) / 255.0  # Normalize pixels
            images.append(img_array)
    return np.array(images), image_files


# Step 2: Extract Features Using Pre-trained Model
def extract_features(images):
    """
    Extract features from images using a pre-trained VGG16 model.
    Args:
        images (np.ndarray): Preprocessed image data.
    Returns:
        np.ndarray: Flattened feature representations.
    """
    model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    images_preprocessed = preprocess_input(images)  # Preprocess for VGG16
    features = model.predict(images_preprocessed)  # Extract features
    return features.reshape(len(images), -1)  # Flatten features


# Step 3: Train Anomaly Detection Model
def train_anomaly_model(features):
    """
    Train an Isolation Forest model for anomaly detection.
    Args:
        features (np.ndarray): Feature representations of images.
    Returns:
        IsolationForest: Trained anomaly detection model.
    """
    clf = IsolationForest(contamination=0.1, random_state=42)  # Adjust contamination as needed
    clf.fit(features)
    return clf


# Step 4: Predict and Save Labels
def predict_and_save_labels(model, features, filenames, output_file='labels.txt'):
    """
    Predict anomaly labels and save them to a text file.
    Args:
        model (IsolationForest): Trained anomaly detection model.
        features (np.ndarray): Feature representations of images.
        filenames (list): Image filenames (for alignment).
        output_file (str): Path to the output labels file.
    """
    predictions = model.predict(features)  # -1 for anomalies, 1 for normal
    labels = ["bad" if pred == -1 else "good" for pred in predictions]
    
    with open(output_file, 'w') as f:
        for label in labels:
            f.write(f"{label}\n")
    print(f"Labels saved to {output_file}")


# Main Function
if __name__ == "__main__":
    # Define the path to the dataset directory (update this path as needed)
    dataset_dir = "C:\\Users\\Jugge\\Downloads\\AnomalyDetectionDataset\\allfrog"  # Replace with the actual path to the 'allfrog' folder

    # Step 1: Preprocess images
    print("Preprocessing images...")
    images, image_filenames = preprocess_images(dataset_dir)

    # Step 2: Extract features
    print("Extracting features...")
    features = extract_features(images)

    # Step 3: Train anomaly detection model
    print("Training anomaly detection model...")
    anomaly_model = train_anomaly_model(features)

    # Step 4: Predict and save labels
    print("Predicting and saving labels...")
    predict_and_save_labels(anomaly_model, features, image_filenames)
