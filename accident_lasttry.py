# -*- coding: utf-8 -*-
"""Accident_LastTry.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/13-UwGV5HNZOihuZBYuru1MOnR9f9lfNW

<h1> Accident Detection From CCTV Footage </h1>
"""

!kaggle datasets download -d ckay16/accident-detection-from-cctv-footage

!unzip /content/accident-detection-from-cctv-footage.zip

"""<h2>Description :</h2>
<h3>Dataset Description :</h3>
<p> Accident Detection dataset collected from the CCTV footages containing a total of 990 accident and non-accident frames collected from road videos available on YouTube. The 990 files are split in the 791 training frames, 101 test frames and 98 validation frames.
791 (369-accident, 492-non accident) Training, 101 Test and 98 Validation (52-accident, 46-non accident) frames split in Accident and Non-accident frames in all the three folders. </p>

<h3>Problem Analysis: </h3>
<pre>
Input : Images that can be accident/Non Accident
Output : 0(Indicates No Accident)
         1(Indicates Accident)
</pre>

<h1>1. Loading Data</h1>
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Define the directories for training, validation, and testing data
train_dir = os.path.join("/content/data/train")
val_dir = os.path.join("/content/data/val")
test_dir = os.path.join("/content/data/test")

# Create the training dataset
train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(256, 256),
    seed=12332
)
# Create the validation dataset
val_data = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(256, 256),
    seed=12332
)
# Create the testing dataset
test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(256, 256),
    seed=12332
)

# Get the first batch of 32 images and labels from the training dataset
train_data_iterator = train_data.as_numpy_iterator()
tr_batch = train_data_iterator.next()

"""<h2>2.Dataset visualisation </h2>"""

# Function to map label to category name
def label_to_category(label):
    # Assuming you have a dictionary to map label indices to category names
    category_map = {0: 'Category 1', 1: 'Category 2', 2: 'Category 3', 3: 'Category 4'}
    return category_map.get(label, 'Unknown')

# Plotting the full batch of 32 images together
cols = 4
rows = 4
fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 15))
fig.tight_layout(pad=2)

for outer_index, img in enumerate(tr_batch[0][:rows*cols]):
    row = outer_index // cols
    col = outer_index % cols

    ax[row][col].imshow(img.astype(int))
    label = label_to_category(tr_batch[1][outer_index])
    ax[row][col].set_aspect(1)
    ax[row][col].axis("off")
    ax[row][col].title.set_text(label)

plt.savefig("test.png")
plt.show()

"""<h1>2. Preprocessing Data </h1>"""

# Normalizing pixels value between between 0 & 1
train_data = train_data.map(lambda x, y: (x / 255.0, y))

tr_batch = train_data.as_numpy_iterator().next()

# Checking pixel min/max pixel values after normalization
print("Max pixel value : ",tr_batch[0].max())
print("Min pixel value : ",tr_batch[0].min())

#lets see training data after normalization

cols = 4
rows = 4
fig, ax = plt.subplots(nrows= rows , ncols= cols,figsize=(15,15),layout='constrained')
fig.tight_layout(pad=2)




for outer_index, img in enumerate(tr_batch[0][:]):
    if outer_index >= rows*cols:
        break

    if (outer_index % cols == 0):
        for inner_index, img in enumerate(tr_batch[0][outer_index:outer_index+cols]):


            ax[outer_index//cols][inner_index].imshow(img)
            if(tr_batch[1][outer_index + inner_index] == 0):
               label = " Accident"
            else: label = " No Accident"




            ax[outer_index//cols][inner_index].set_aspect(1)

            num_label = tr_batch[1][outer_index + inner_index]

            ax[outer_index//cols][inner_index].axis("off")
            ax[outer_index//cols][inner_index].title.set_text(label)


plt.savefig("test.png")
plt.show()

"""<h2>Loading Validation data for Hyper-parameter Turing</h2>"""

# Normalize pixel values between 0 and 1 for validation data
val_data = val_data.map(lambda x, y: (x / 255.0, y))

# Get the first batch of 32 images and labels from the validation dataset
val_data_iterator = val_data.as_numpy_iterator()
val_batch = val_data_iterator.next()

"""<h1> 3. Model Building   </h1>

"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()

model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
# Adding neural Layer
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

model.summary()

"""<h1> 4.  Training Neural Network </h1>"""

# setting up for logging
logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# Optimize data loading
train_data = train_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_data = val_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

hist = model.fit(train_data, epochs=20, validation_data=val_data, callbacks=[tensorboard_callback])

#hist.history

"""<h2>5.Seeing Training Loss and Accuracy Curve with epochs</h2>"""

fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='training loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.xlabel("Epoch")
plt.ylabel("loss")
plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='training accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

"""<h1>6. Evaluation</h1>"""

test_data = tf.keras.utils.image_dataset_from_directory(test_dir)
test_data_iterator = test_data.as_numpy_iterator()
test_batch = test_data_iterator.next()

pre = tf.keras.metrics.Precision
re = tf.keras.metrics.Recall()

pre = tf.keras.metrics.Precision()
re = tf.keras.metrics.Recall()

for batch in test_data:
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)

def F1_score(precision, recall):
    return (2*precision*recall)/(precision+recall)

print("Model achieved an precision score of {:5f}".format(pre.result()))
print("Model achieved an recall score of {:5f}".format(re.result()))

f1_score = F1_score(pre.result(), re.result())
print("Model achieved an F1-score of {:5f}".format(f1_score))

model.evaluate(train_data)

"""<h1> 7.Test just to see model working </h1>"""

import cv2

# Load random samples from test directory
random_data_dirname = os.path.join("/content/data/test/Accident")
pics = [os.path.join(random_data_dirname, filename) for filename in os.listdir(random_data_dirname)]

# Load first file from samples
sample = cv2.imread(pics[1], cv2.IMREAD_COLOR)
sample = cv2.resize(sample, (256, 256))

# Predict using the trained model
prediction = 1 - model.predict(np.expand_dims(sample / 255.0, axis=0))

# Determine the label based on the prediction
if prediction >= 0.5:
    label = 'Predicted class is Accident'
else:
    label = 'Predicted class is Not Accident'

# Display the image with the prediction label
plt.title(label)
plt.imshow(cv2.cvtColor(sample, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display
plt.axis('off')
plt.show()

model.save('my_model.h5')

!pip install streamlit

# # The following block of code is for the deployment of our model using streamlit
# import streamlit as st
# from PIL import Image
# # Load our model
# model = tf.keras.models.load_model('/content/Accident_Detection_Model.h5')

# # Create our tilte:
# st.title("Accedent Detection Deep learning model")

# # Upload triel image for the deployment:
# uploaded_file = st.file_uploader("/content/data/train/Non Accident/5_8.jpg", type=["jpg"])

# if uploaded_file is not None:
#     # Load the image
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Image of a non accident', use_column_width=True)

#     # Preprocess the image for our model
#     img_array = np.array(image.resize((256, 256))) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     # Make prediction
#     prediction = model.predict(img_array)

#     # Display prediction
#     st.write(f"Prediction: {np.argmax(prediction[0])}")

!streamlit run /usr/local/lib/python3.10/dist-packages/colab_kernel_launcher.py

# @st.cache(allow_output_mutation=True)
# def load_model():
#   model=tf.keras.models.load_model('/content/Accident_Detection_Model.h5')
#   return model
# model=load_model()
# st.write("""
# # Accident Detection System"""
# )
# file=st.file_uploader("Choose accident photo from computer",type=["jpg","png"])
# import cv2
# from PIL import Image,ImageOps
# import numpy as np
# def import_and_predict(image_data,model):
#     size=(256,256)
#     image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
#     img=np.asarray(image)
#     img_reshape=img[np.newaxis,...]
#     prediction=model.predict(img_reshape)
#     return prediction
# if file is None:
#     st.text("Please upload an image file")
# else:
#     image=Image.open(file)
#     st.image(image,use_column_width=True)
#     prediction=import_and_predict(image,model)
#     class_names=['Accident', 'No Accident']
#     string="OUTPUT : "+class_names[np.argmax(prediction)]
#     st.success(string)

# import streamlit as st
# import tensorflow as tf
# from PIL import Image, ImageOps
# import numpy as np
# import logging

# # Enable logging
# logging.basicConfig(filename='app.log', level=logging.INFO)

# # Cache the model to optimize performance
# @st.cache(allow_output_mutation=True)
# def load_model():
#     model = tf.keras.models.load_model('/content/Accident_Detection_Model.h5')
#     return model

# model = load_model()

# # Set the title and description
# st.title("Accident Detection System")
# st.write("Upload an image to detect whether it shows an accident or not.")

# # File uploader for image input
# file = st.file_uploader("Choose an accident photo from your computer", type=["jpg", "png"])

# # Image preprocessing function
# def import_and_predict(image_data, model):
#     size = (256, 256)
#     image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
#     img = np.asarray(image)
#     img_reshape = img[np.newaxis, ...]
#     prediction = model.predict(img_reshape)
#     return prediction

# # If no file is uploaded, prompt the user
# if file is None:
#     st.text("Please upload an image file.")
# else:
#     try:
#         # Load and display the image
#         image = Image.open(file)

#         # Validate image mode (ensure it's RGB)
#         if image.mode != "RGB":
#             st.warning("Please upload an RGB image.")
#         else:
#             st.image(image, use_column_width=True)

#             # Perform prediction
#             with st.spinner('Processing...'):
#                 prediction = import_and_predict(image, model)

#             class_names = ['Accident', 'No Accident']
#             predicted_class = class_names[np.argmax(prediction)]
#             confidence = np.max(prediction)

#             # Display the prediction and confidence
#             st.success(f"OUTPUT: {predicted_class}")
#             st.write(f"Confidence: {confidence:.2f}")

#             # Log the interaction
#             logging.info(f"User uploaded an image. Prediction: {predicted_class}, Confidence: {confidence:.2f}")

#             # Optional: Add Grad-CAM visualization for interpretability (simplified example)
#             def grad_cam(input_model, image, layer_name):
#                 grad_model = tf.keras.models.Model(
#                     [input_model.inputs], [input_model.get_layer(layer_name).output, input_model.output]
#                 )
#                 with tf.GradientTape() as tape:
#                     conv_outputs, predictions = grad_model(np.array([image]))
#                     loss = predictions[:, np.argmax(predictions[0])]
#                 output = conv_outputs[0]
#                 grads = tape.gradient(loss, conv_outputs)[0]
#                 gate_f = tf.cast(output > 0, "float32")
#                 guided_grads = gate_f * grads
#                 weights = tf.reduce_mean(guided_grads, axis=(0, 1))
#                 cam = np.dot(output, weights)
#                 cam = cv2.resize(cam.numpy(), (image.shape[1], image.shape[0]))
#                 cam = np.maximum(cam, 0)
#                 heatmap = (cam - cam.min()) / (cam.max() - cam.min())
#                 return heatmap

#             heatmap = grad_cam(model, img_reshape[0], "conv2d")  # Replace "conv2d" with your model's last convolutional layer name
#             st.image(heatmap, caption='Grad-CAM Heatmap', use_column_width=True)

#     except Exception as e:
#         st.error(f"Error loading image: {e}")
#         logging.error(f"Error loading image: {e}")

# # Feedback form
# feedback = st.text_input("Provide feedback:")
# if st.button("Submit Feedback"):
#     st.success("Thank you for your feedback!")
#     logging.info(f"User feedback: {feedback}")

import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import logging

# Enable logging
logging.basicConfig(filename='app.log', level=logging.INFO)

# Cache the model using the new caching command
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('/content/Accident_Detection_Model.h5')
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
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
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

!streamlit run Accident_LastTry.ipynb

