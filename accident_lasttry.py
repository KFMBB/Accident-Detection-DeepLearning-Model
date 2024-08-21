def import_and_predict(image_data, model):
    try:
        size = (256, 256)
        image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
        img = np.asarray(image)
        img_reshape = img[np.newaxis, ...]
        
        # Assuming a softmax output
        prediction = model.predict(img_reshape)
        
        # Check if output is of shape (1, 2) or (1,)
        if prediction.shape[-1] == 1:
            predicted_class = "Accident" if prediction[0] > 0.5 else "No Accident"
            confidence = prediction[0] if prediction[0] > 0.5 else 1 - prediction[0]
        else:
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)

        return predicted_class, confidence
    except Exception as e:
        logging.error(f"Error in import_and_predict: {e}")
        return None, None

# Rest of your code
# ...

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
                heatmap = grad_cam(model, img_reshape[0], "conv2d")  # Ensure layer name is correct
                st.image(heatmap, caption='Grad-CAM Heatmap', use_column_width=True)
            else:
                st.error("Error in making prediction. Please try again.")

    except Exception as e:
        logging.error(f"Error loading image: {e}")
