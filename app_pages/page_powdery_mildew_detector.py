import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

from src.data_management import download_dataframe_as_csv
from src.machine_learning.predictive_analysis import (
                                                    load_model_and_predict,
                                                    resize_input_image,
                                                    plot_predictions_probabilities
                                                    )

def page_powdery_mildew_detector_body():
    st.info(
        f"Upload pictures of cherry leaves to identify whether it is healthy or infected with powdery mildew. You can download a report of the examined leaves."
        )

    # Provide a link to download a dataset of infected and healthy leaves for testing
    st.write(
        f"*You can download a set of infected and healthy leaves for live prediction from [here](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves).")

    st.write("---")
    
    st.write(
        f"**Upload a clear picture of a cherry leaf. You may select more than one.**"
        )
    # File uploader widget for users to upload multiple JPEG images
    images_buffer = st.file_uploader(' ', type='jpeg',accept_multiple_files=True)
   
    # Check if any images were uploaded
    if images_buffer is not None:
        # Initialize an empty DataFrame to store results of predictions
        df_report = pd.DataFrame([])
        # Loop through each uploaded image
        for image in images_buffer:

            # Open the image using PIL
            img_pil = (Image.open(image))
            # Display an info message with the name of the image
            st.info(f"Cherry leaf Sample: **{image.name}**")
            # Convert the image to a NumPy array to get its shape
            img_array = np.array(img_pil)
            # Display the uploaded image and its size
            st.image(img_pil, caption=f"Image Size: {img_array.shape[1]}px width x {img_array.shape[0]}px height")

            version = 'v1'
            resized_img = resize_input_image(img=img_pil, version=version)
            # Use the machine learning model to predict the probability and class of the input image
            pred_proba, pred_class = load_model_and_predict(resized_img, version=version)
            # Plot the prediction probabilities for each class
            plot_predictions_probabilities(pred_proba, pred_class)

            # Append the results to the DataFrame
            df_report = df_report.append({"Name":image.name, 'Result': pred_class },
                                        ignore_index=True)
        
        # If the DataFrame is not empty, display the analysis report and provide an option to download it as a CSV file
        if not df_report.empty:
            st.success("Analysis Report")
            st.table(df_report)
            st.markdown(download_dataframe_as_csv(df_report), unsafe_allow_html=True)

    st.write(
        f"For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/JoelChan13/mildew-detector/blob/main/README.md).")