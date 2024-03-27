import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Function to preprocess the image
def preprocess_image(image):
    img_resized = cv2.resize(image, (32, 32))
    img_array = np.expand_dims(img_resized, axis=0)  # Add batch dimension
    return img_array / 255.0

# Load the trained model
model = load_model("model.keras")

# Define a dictionary to map class indexes to custom labels
class_labels = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'Ç',
    4: 'D',
    5: 'E',
    6: 'F',
    7: 'G',
    8: 'Ğ',
    9: 'H',
    10: 'I',
    11: 'İ',
    12: 'J',
    13: 'K',
    14: 'L',
    15: 'M',
    16: 'N',
    17: 'O',
    18: 'Ö',
    19: 'P',
    20: 'Q',
    21: 'R',
    22: 'S',
    23: 'Ş',
    24: 'T',
    25: 'U',
    26: 'Ü',
    27: 'V',
    28: 'W',
    29: 'X',
    30: 'Y',
    31: 'Z'
}

def predict_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            # Read the image
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError("Unable to read image at path:", file_path)
            
            # Preprocess the image
            img_array = preprocess_image(img)

            # Perform prediction
            predictions = ""
            for letter_image in img_array:
                prediction = model.predict(np.expand_dims(letter_image, axis=0))
                # Get the predicted class index using argmax
                predicted_index = np.argmax(prediction)
                # Get the probability of the predicted class
                probability = prediction[0][predicted_index] * 100
                # Map the predicted index to a custom label using the class_labels dictionary
                predicted_label = class_labels.get(predicted_index, "Unknown")
                predictions += f"{predicted_label} ({probability:.2f}%)\n"

            prediction_label.config(text="Predicted text:\n" + predictions, fg="green", font=("Arial", 12, "italic"))

        except Exception as e:
            prediction_label.config(text="Error: " + str(e), fg="red", font=("Arial", 12, "bold"))

# Create the main window
root = tk.Tk()
root.title("Image Text Recognizer") # Set application title
root.geometry("400x250") # Set window size

# Create a frame for better organization
frame = tk.Frame(root)
frame.pack(pady=10)

# Create a label for the title
title_label = tk.Label(frame, text="Image Text Recognizer", font=("Helvetica", 16, "bold"), fg="blue")
title_label.grid(row=0, column=0, columnspan=2, pady=10)

# Create a button to select image
select_button = tk.Button(frame, text="Select Image", command=predict_image, bg="lightblue", fg="black", padx=10, pady=5, font=("Arial", 12))
select_button.grid(row=1, column=0, padx=10)

# Create a label to display the prediction
prediction_label = tk.Label(frame, text="", font=("Arial", 14))
prediction_label.grid(row=1, column=1, padx=10)

root.mainloop()
