#import libraries
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

#load pre-trained VGG16 model
model = VGG16(weights='imagenet')


#load an image for classification
image_path = 'Cruise.png'  # Provide the complete path to your image file
img = image.load_img(image_path, target_size=(224, 224))



# Preprocess the image
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Make predictions on the image
predictions = model.predict(x)

# Decode and display
decoded_predictions = decode_predictions(predictions, top=5)

# Print the result
for i, (imagenet_id, label, score) in enumerate(decoded_predictions[0]):
    print(f"{i + 1}: {label} ({score:.2f})")



