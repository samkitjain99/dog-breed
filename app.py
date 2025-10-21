import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

model = load_model("dog.breed.h5")

classes = ['scottish_deerhound', 'maltese_dog', 'bernese_mountain_dog']

st.title("🐶 Dog Breed Prediction App")
st.write("Upload an image of a dog to predict its breed!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)
    st.write("Classifying...")

    img = img.resize((224, 224)) 
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  

    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]



    st.success(f"Prediction: **{predicted_class}** 🐾")
