import cv2
import numpy as np
from tensorflow.keras.models import load_model

#function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (32, 32))
    img_array = np.expand_dims(img_resized, axis=0)  # Add batch dimension
    return img_array / 255.0

#load the trained model
model = load_model("newmodel.keras")

#define a dictionary to map class indexes to custom labels
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

#path to the image to be predicted
image_path= "a.jpg"

#preprocess the image
img_array= preprocess_image(image_path)

#perform prediction
predictions= ""
for letter_image in img_array:
    prediction = model.predict(np.expand_dims(letter_image, axis=0))
    #get the predicted class index using argmax
    predicted_index = np.argmax(prediction)
    #map the predicted index to a custom label using the class_labels dictionary
    predicted_label = class_labels.get(predicted_index, "Unknown")
    predictions += predicted_label

print("Predicted text:", predictions)
