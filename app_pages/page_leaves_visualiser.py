import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread

import itertools
import random

def page_leaves_visualiser_body():
    # Display the title and description of the page
    st.write("### Leaves Visualiser")
    st.info(
        f"A comprehensive study to visually differentiate between healthy cherry leaves and those infected with powdery mildew.")

    # Provide a link to the project's README file for additional information
    st.write(
        f"For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/JoelChan13/mildew-detector/blob/main/README.md).")

    # Display a warning message with detailed information about the symptoms of powdery mildew and the need for image normalization
    st.warning(
        f"We believe cherry leaves affected by powdery mildew show distinct symptoms.\n\n" 
        f" The first symptom is typically a porraceous green, circular lesion on leaves and also fruits," 
        f" which progresses to white or grayish powdery spots.\n\n" 
        f" In order for the user to apply machine learning functionalities," 
        f" certain modifications and preperations have to be done to the images prior use of model for an optimal feature extraction and training.\n\n"
        f" It is essential to normalise the Image dataset before training a Neural Network on it," 
        f" hence, the mean and standard deviation of the entire dataset, that are calculated with a mathematical formula"
        f" which takes into consideration the properties of an image, are essential"
    )
    
    # Specify the version of the output images
    version = 'v1'

    # Checkbox to show the difference between average and variability images
    if st.checkbox("Difference between average and variability image"):
      # Load the images for powdery mildew and healthy leaves
      avg_powdery_mildew = plt.imread(f"outputs/{version}/avg_var_powdery_mildew.png")
      avg_uninfected = plt.imread(f"outputs/{version}/avg_var_healthy.png")

      # Display a warning message about the findings from the average and variability images
      st.warning(
        f"No significant patterns were identified from the average and variability images" 
        f"however, infected leaves tended to display more white stipes at the center.")

      # Display the loaded images with captions
      st.image(avg_powdery_mildew, caption='Affected Cherry Leaf - Average and Variability')
      st.image(avg_uninfected, caption='Healthy Cherry Leaf - Average and Variability')
      st.write("---")

    # Checkbox to show the difference between average images of infected and healthy leaves
    if st.checkbox("Differences between average infected and average healthy leaves"):
          # Load the image showing the difference between average images
          diff_between_avgs = plt.imread(f"outputs/{version}/avg_diff.png")

          # Display a warning message about the inability to differentiate based on the average images
          st.warning(
            f"Through the data obtained from this study, we are unable to "
            f" establish patterns to intuitively differentiate one from another.")
          # Display the image with a caption
          st.image(diff_between_avgs, caption='Difference between average images')

    # Checkbox to show an image montage of selected label
    if st.checkbox("Image Montage"): 
      st.write("To refresh the montage, click on the 'Create Montage' button")
      # Directory where image data is stored
      my_data_dir = 'inputs/cherryleaves_dataset/cherry-leaves'
      # List of labels (directories) in the validation dataset
      labels = os.listdir(my_data_dir+ '/validation')
      # Dropdown menu to select which label to display
      label_to_display = st.selectbox(label="Select label", options=labels, index=0)
      # Button to create the montage
      if st.button("Create Montage"):      
        # Call the image_montage function to display the montage of images
        image_montage(dir_path= my_data_dir + '/validation',
                      label_to_display=label_to_display,
                      nrows=8, ncols=3, figsize=(10,25))
      st.write("---")


def image_montage(dir_path, label_to_display, nrows, ncols, figsize=(15,10)):
  # Set the style of seaborn plots
  sns.set_style("white")
  # Get the list of labels (directories) in the given directory path
  labels = os.listdir(dir_path)

  # Check if the selected label exists in the directory
  if label_to_display in labels:
    # List of images in the selected label's directory
    images_list = os.listdir(dir_path+'/'+ label_to_display)
    # Check if the number of montage spaces (nrows * ncols) is greater than the number of images
    if nrows * ncols < len(images_list):
      # Randomly select a subset of images to display in the montage
      img_idx = random.sample(images_list, nrows * ncols)
    else:
      print(
          f"Decrease nrows or ncols to create your montage. \n"
          f"There are {len(images_list)} in your subset. "
          f"You requested a montage with {nrows * ncols} spaces")
      return
    

    # Create list of axes indices based on nrows and ncols
    list_rows= range(0,nrows)
    list_cols= range(0,ncols)
    plot_idx = list(itertools.product(list_rows,list_cols))


    # Create a Figure and display images
    fig, axes = plt.subplots(nrows=nrows,ncols=ncols, figsize=figsize)
    for x in range(0,nrows*ncols):
      img = imread(dir_path + '/' + label_to_display + '/' + img_idx[x])
      img_shape = img.shape
      # Display the image on the corresponding subplot
      axes[plot_idx[x][0], plot_idx[x][1]].imshow(img)
      axes[plot_idx[x][0], plot_idx[x][1]].set_title(f"Width {img_shape[1]}px x Height {img_shape[0]}px")
      axes[plot_idx[x][0], plot_idx[x][1]].set_xticks([])
      axes[plot_idx[x][0], plot_idx[x][1]].set_yticks([])
    plt.tight_layout()
    
    # Display the figure in Streamlit
    st.pyplot(fig=fig)


  else:
    print("The label you selected doesn't exist.")
    print(f"The existing options are: {labels}")