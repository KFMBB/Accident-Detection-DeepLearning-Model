import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import logging
import cv2
# Enable logging
logging.basicConfig(filename='app.log', level=logging.INFO)

# Cache the model using the new caching command
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('Accident_Detection_Model.h5')
    return model

model = load_model()

# Set the title and description
st.title("Accident Detection System")
st.write("Upload an image to detect whether it shows an accident or not.")

# File uploader for image input
file = st.file_uploader("Choose an accident photo from your computer", type=["jpg", "png"])

# Image preprocessing function
def import_and_predict(image_data, model):
    size = (256, 256)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)  # Updated line
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

# If no file is uploaded, prompt the user
if file is None:
    st.text("Please upload an image file.")
else:
    try:
        image = Image.open(file)

        if image.mode != "RGB":
            st.warning("Please upload an RGB image.")
        else:
            st.image(image, use_column_width=True)

            with st.spinner('Processing...'):
                prediction = import_and_predict(image, model)

            class_names = ['Accident', 'No Accident']
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)

            st.success(f"OUTPUT: {predicted_class}")
            st.write(f"Confidence: {confidence:.2f}")

            logging.info(f"User uploaded an image. Prediction: {predicted_class}, Confidence: {confidence:.2f}")

            def grad_cam(input_model, image, layer_name):
                grad_model = tf.keras.models.Model(
                    [input_model.inputs], [input_model.get_layer(layer_name).output, input_model.output]
                )
                with tf.GradientTape() as tape:
                    conv_outputs, predictions = grad_model(np.array([image]))
                    loss = predictions[:, np.argmax(predictions[0])]
                output = conv_outputs[0]
                grads = tape.gradient(loss, conv_outputs)[0]
                gate_f = tf.cast(output > 0, "float32")
                guided_grads = gate_f * grads
                weights = tf.reduce_mean(guided_grads, axis=(0, 1))
                cam = np.dot(output, weights)
                cam = cv2.resize(cam.numpy(), (image.shape[1], image.shape[0]))
                cam = np.maximum(cam, 0)
                heatmap = (cam - cam.min()) / (cam.max() - cam.min())
                return heatmap

            heatmap = grad_cam(model, img_reshape[0], "conv2d")  # Replace "conv2d" with your model's last convolutional layer name
            st.image(heatmap, caption='Grad-CAM Heatmap', use_column_width=True)

    except Exception as e:
        st.error(f"Error loading image: {e}")
        logging.error(f"Error loading image: {e}")

# Feedback form
feedback = st.text_input("Provide feedback:")
if st.button("Submit Feedback"):
    st.success("Thank you for your feedback!")
    logging.info(f"User feedback: {feedback}")
