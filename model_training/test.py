import cv2
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow import keras

folderName = "data/nifti-results"

model = keras.models.load_model("VGGNet.keras", compile=False)

class_folders = [join(folderName, f) for f in listdir(folderName) if not isfile(join(folderName, f))]

image_filenames = []
predicted_labels = []
actual_labels = []

for class_folder in class_folders:
    
    class_label = class_folder.split('/')[-1]  # Extract the class label from the folder name
    class_label = int(''.join(filter(str.isdigit, class_label)))-1
    image_files = [f for f in listdir(class_folder) if isfile(join(class_folder, f))][:10]  # First 10 images

    for image_file in image_files:
        image_path = join(class_folder, image_file)

        frame = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        rows = 128
        cols = 128

        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Replicate the single channel to create three channels (RGB)
        frame_rgb = cv2.merge([frame, frame, frame])

        # Resize and normalize the image
        frame_rgb = cv2.resize(frame_rgb, (rows, cols), interpolation=cv2.INTER_AREA)
        frame_rgb = frame_rgb / 255.0  # Normalize the pixel values

        # Perform prediction using the trained model
        frame_rgb = np.expand_dims(frame_rgb, axis=0)
        y_pred = model.predict(frame_rgb)

        # Convert probabilities to class label (0 or 1)
        predicted_label = np.argmax(y_pred)

        image_filenames.append(image_file)
        predicted_labels.append(predicted_label)
        actual_labels.append(int(class_label))

df = pd.DataFrame({
    "Image Filename": image_filenames,
    "Predicted Label": predicted_labels,
    "Actual Label": actual_labels
})

csv_filename = "predictions.csv"
df.to_csv(csv_filename, index=False)

accuracy = accuracy_score(actual_labels, predicted_labels)
print("Accuracy:", accuracy*100)

print("Classification Report:")
print(classification_report(actual_labels, predicted_labels))

print("Confusion Matrix:")
print(confusion_matrix(actual_labels, predicted_labels))
