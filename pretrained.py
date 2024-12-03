import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
import numpy as np


# Load the InceptionV3 model with pre-trained weights
model = InceptionV3(weights='imagenet')  # Note: InceptionV3 is a classification model

# Capture image and get file path

a="captured_images/car.jpg"

"""
Predict the class of the image using the InceptionV3 pre-trained model.
Args:
    file_path (str): Path to the image file.

Returns:
    List[Tuple]: Decoded predictions containing the top-3 clas
    
    `ses and their scores.
"""

# Load and preprocess the image
img = image.load_img(a, target_size=(299, 299))  # Resize image to 299x299
img = img.convert('RGB')  # Ensure it's in RGB format
img_array = image.img_to_array(img)  # Convert image to array
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = preprocess_input(img_array)  # Preprocess the image for InceptionV3

# Make a prediction
predictions = model.predict(img_array)

# Decode predictions
decoded_predictions = decode_predictions(predictions, top=3)[0]

for i, (imagenet_id, label, score) in enumerate(decoded_predictions):

    print(f"{i+1}: {label} ({score:.2f})")

   

# Predict using the captured image

# if __name__ == "__main__":
 

#     prediction(k)