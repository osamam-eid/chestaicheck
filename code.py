import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import urllib.request
import io

# Load the pre-trained model
module_handle_classification = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/classification/2"
module_handle_detection = "https://tfhub.dev/tensorflow/efficientdet/d4/1"
input_size_detection = (1024, 1024)
input_size_classification = (384, 384)

# Create the detection layer
detection_layer = hub.KerasLayer(module_handle_detection, trainable=False)

# Create the classification layer
classification_layer = hub.KerasLayer(module_handle_classification, trainable=False)

# Create the input layer
input_layer = tf.keras.layers.Input(shape=(None, None, 3))

# Apply the detection layer to the input
detection_output = detection_layer(tf.image.convert_image_dtype(input_layer, tf.uint8))


# Resize the input to match the classification input size
resized_input = tf.image.resize(input_layer, input_size_classification)

# Apply the classification layer to the resized input
classification_output = classification_layer(resized_input)

# Create the model
model = tf.keras.Model(inputs=input_layer, outputs=[detection_output, classification_output])

# Define the list of abnormalities that the model can detect
abnormalities = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
                 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

# Function to preprocess the image
def preprocess_image(image):
    img_classification = cv2.resize(image, input_size_classification)
    img_detection = cv2.resize(image, input_size_detection)
    img_detection = np.array(img_detection, dtype=np.float32)
    img_detection /= 255.0
    img_detection = tf.expand_dims(img_detection, axis=0)  # add batch dimension
    img_detection = tf.image.convert_image_dtype(img_detection, tf.uint8)  # convert to tf.uint8
    return img_detection

# Function to display the results
def display_results(image, preds_classification, preds_detection):
    # Display the original image
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.title('Chest X-ray Image')
    plt.show()

    # Display the classifications
    print('Classification results:')
    for i in range(len(abnormalities)):
        print(f'{abnormalities[i]}: {preds_classification[0][i]:.2f}%')

    # Display the detections
    print('Detection results:')
    boxes = preds_detection['detection_boxes'][0]
    scores = preds_detection['detection_scores'][0]
    classes = preds_detection['detection_classes'][0]
    for i in range(len(boxes)):
        if scores[i] > 0.5:  # only show boxes with high confidence
            print(f'Box {i + 1}: {boxes[i]}')
            print(f'Score: {scores[i]:.2f}')
            print(f'Class: {classes[i]}')
            # Draw the box on the image
            image = cv2.rectangle(image, (int(boxes[i][1] * 1024), int(boxes[i][0] * 1024)),
                                  (int(boxes[i][3] * 1024), int(boxes[i][2] * 1024)), (255, 0, 0), 2)

    # Display the image with boxes
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.title('Chest X-ray Image with Detections')
    plt.show()

# Get the image from the URL
url = input("Please enter the URL of the image: ")
while True:
    try:
        with urllib.request.urlopen(url) as url_response:
            image_data = url_response.read()
        image = Image.open(io.BytesIO(image_data))
        image = np.array(image)
        break
    except urllib.error.HTTPError as e:
        if e.code == 305:
            print("Error 305: The server is currently overloaded. Please try again later.")
        else:
            print("Error {}: {}".format(e.code, e.reason))

        choice = input("Do you want to upload another image? (y/n): ")
        if choice.lower() == 'n':
            exit()
        else:
            url = input("Please enter the URL of the image: ")
    # Preprocess the image and make predictions
preprocessed_image = tf.squeeze(preprocess_image(image), axis=0)
resized_image = cv2.resize(image, input_size_classification)  # resize for classification
preds_detection, preds_classification = model.predict(np.expand_dims(preprocessed_image, axis=0))
preds_classification = preds_classification[:len(abnormalities)]  # only keep relevant abnormalities

# Find the index of the abnormality with the highest probability
max_prob_index = np.argmax(preds_classification)
max_prob_abnormality = abnormalities[max_prob_index]

# Print the probabilities for each abnormality
for i, abnormality in enumerate(abnormalities):
    print('{}: {:.2f}%'.format(abnormality, preds_classification[i] * 100))

# Display the results
display_results(resized_image, preds_classification, preds_detection)  # pass in both predictions
