from streamlit_option_menu import option_menu
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
from tensorflow import keras

# Function to load the pre-trained model
def load_model(model_path):
    model = keras.models.load_model(model_path)
    return model

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = cv2.convertScaleAbs(image)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    resized_image = cv2.resize(rgb_image, (128, 128))
    normalized_image = resized_image / 255.0
    preprocessed_image = np.expand_dims(normalized_image, axis=0)
    st.image(preprocessed_image, caption="Normalized and Preprocessed Image", use_column_width=True)
    
    return preprocessed_image

# Function to make predictions
def make_prediction(model, image):
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    return predicted_class

# Main function to run the prediction app
def prediction_page():
    st.markdown(
    """
    <h2 style="color:#FF204E">Prediction Page</h2>
    """,
    unsafe_allow_html=True)

    # File uploader to upload the image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        original_image = np.array(image)
        st.image(original_image, caption="Uploaded Image", use_column_width=True)
        preprocessed_image = preprocess_image(original_image)

        model_path = os.path.join("model", "VGGNet.keras")
        model = load_model(model_path)

        predicted_class = make_prediction(model, preprocessed_image)

        if predicted_class == 0:
            st.write("Prediction: The person has Autism Spectrum Disorder")
        else:
            st.write("Prediction: The person is Typcally Developing")


st.set_page_config(
    page_title="Autism Spectrum Disorder Detection",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)


st.title('Autism Spectrum Disorder Detection App 🧠')



selected = option_menu(
    menu_title=None,
    options=["Home", "Testing", "Prediction"],
    icons=["window", "globe", "cpu"],
    orientation="horizontal",
    default_index=0,
    styles={
        "nav-link-selected": {"background-color": "#A0153E"},
    }
)


if selected == "Home":
    with st.container():

        target_url = "#"
        image_url = "https://images.unsplash.com/photo-1620230874645-0d85522b20f9?q=80&w=1770&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
        
        st.markdown(
            f'<a href="{target_url}" target="_blank"><img src="{image_url}" alt="Cover Image" width="100%" height="250px"></a>',
            unsafe_allow_html=True
        )

        st.markdown(
            """

            <h2 style="color:#FF204E">Project Overview</h2>

            <p style="color:white">
                The main objective of this project to develop a machine learning model to classify individuals with ASD and those with TD based on structural and functional Magnetic Resonance Imaging (MRI) scans.
            </p>

            <h2 style="color:#FF204E">Data Source</h2>

            <p style="color:white">
                The project utilizes the KKI dataset from the Autism Brain Imaging Data Exchange (ABIDE) repository. This dataset provides MRI scans from both ASD and TD individuals.
            </p>

            <h2 style="color:#FF204E">Data Preprocessing</h2>

            <p style="color:white">
                Raw MRI images were preprocessed using Python's nilearn library. Preprocessing steps likely included:
                <ul>
                    <li><b>Normalization:</b> Ensuring images have consistent intensity values across participants.</li>
                    <li><b>Skull Stripping:</b> Removing non-brain tissue from the image.</li>
                    <li><b>Spatial Smoothing:</b> Blurring the image slightly to reduce noise.</li>
                    <li><b>Independent Component Analysis (ICA) and Dictionary Learning:</b> These techniques might have been used to extract features from the brain scans that are relevant for ASD classification.</li>
                </ul>
            </p>

            <h2 style="color:#FF204E">Data Separation</h2>

            <p style="color:white">
                A separate script likely utilizes phenotypic data (information about the participants' diagnosis) from a CSV file to categorize the preprocessed MRI scans into two folders: one containing scans from individuals with ASD and another containing scans from those with TD.
            </p>

            <h2 style="color:#FF204E">Classification Model</h2>

            <p style="color:white">
                The project employs a Convolutional Neural Network (CNN) architecture called VGGNET16 for classification. CNNs are well-suited for analyzing image data like MRI scans. VGGNET16 is a pre-trained model on a large image dataset, which can then be fine-tuned for the specific task of classifying ASD vs. TD.
            </p>

            <h2 style="color:#FF204E">Current Performance</h2>

            <p style="color:white">
                The model achieves an accuracy of 63.75% at 5 epochs. This indicates that the model can correctly classify ASD or TD in roughly 64 out of every 100 cases after 10 training iterations.
            </p>
            """,
            unsafe_allow_html=True
        )

    

if selected == "Testing":

    st.markdown(
    f"""
    <h2 style="color:#FF204E">Testing Page</h2>


    """,
    unsafe_allow_html=True)


if selected == "Prediction":

    prediction_page()


    