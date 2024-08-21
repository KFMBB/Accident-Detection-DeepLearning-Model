import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import logging
import cv2

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO)

# Load the model using the new Streamlit caching method
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('best_model.keras')
    return model

model = load_model()

st.title("Accident Detection System")
st.write("Upload an image to detect whether it shows an accident or not.")

file = st.file_uploader("Choose an accident photo from your computer", type=["jpg", "png"])

def import_and_predict(image_data, model):
    try:
        size = (256, 256)  # Image size expected by the model
        image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
        img = np.asarray(image)
        img_reshape = img[np.newaxis, ...]  # Reshape image to match model input shape
        
        prediction = model.predict(img_reshape)

        # Assuming the model uses sigmoid activation for binary classification
        if prediction.shape[-1] == 1:
            predicted_class = "Accident" if prediction[0][0] > 0.5 else "No Accident"
            confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
        else:  # If softmax or multi-class
            class_names = ['Accident', 'No Accident']
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)

        return predicted_class, confidence

    except Exception as e:
        logging.error(f"Error in import_and_predict: {e}")
        return None, None

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
                predicted_class, confidence = import_and_predict(image, model)

            if predicted_class is not None:
                st.success(f"OUTPUT: {predicted_class}")
                st.write(f"Confidence: {confidence:.2f}")
                logging.info(f"User uploaded an image. Prediction: {predicted_class}, Confidence: {confidence:.2f}")

                # Grad-CAM visualization
                img_reshape = np.asarray(ImageOps.fit(image, (256, 256), Image.Resampling.LANCZOS))[np.newaxis, ...]
                heatmap = grad_cam(model, img_reshape[0], "conv2d")  # Replace "conv2d" with the correct layer name
                st.image(heatmap, caption='Grad-CAM Heatmap', use_column_width=True)
            else:
                st.error("Error in making prediction. Please try again.")

    except Exception as e:
        logging.error(f"Error loading image: {e}")
        # st.error("Error processing image. Please try a different image.")

# Feedback section
feedback = st.text_input("Provide feedback:")
if st.button("Submit Feedback"):
    st.success("Thank you for your feedback!")
    logging.info(f"User feedback: {feedback}")
