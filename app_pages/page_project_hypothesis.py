import streamlit as st
import matplotlib.pyplot as plt


def page_project_hypothesis_body():
    st.write("### Hypotesis 1 and validation")

    st.success(
        f"There are distinct visual differences between healthy cherry leaves and those affected by powdery mildew that can be identified through image analysis."
    )
    st.info(
        f"Powdery milder is usually characterised by clear marks over the leaves and their fruits, "
        f" and is typically the first symptom that indicates an infected cherry leaf.\n\n"
        f" The marks can be described as porraceous green lesions on either leaf surface,"
        f" which progresses to white or grayish powdery spots which develop in the infected area on leaves and fruits in the infected area."
    )
    st.write("To visualize a thorough investigation of infected and healthy leaves visit the Leaves Visualiser tab.")
     
    st.warning(
        f""
    )


    st.write("### Hypotesis 2 and validation")

    st.success(
        f"Comparison of Mathematical Functions - The `softmax` function performs better than the `sigmoid` function as an activation function for the CNN output layer. "
    )

    st.info(
        f"Both ```softmax``` and ```sigmoid``` are typically used as functions for binary or multi class classification problems."
        f" How an activation function performs on a model can be evaluated by plotting the model's prediction capacity."
        f" The learning curve shows the accuracy and error rate on the training and validation dataset while the model is training.\n\n"
        f" The model trained using ```softmax``` showed less training/validation sets gap and more"
        f" consistent learning rate after the 5th Epoch compared to the model trained using ```sigmoid```."
    )
    st.warning(
        f"```softmax``` performed better than ```sigmoid```. "
    )
    model_perf_softmax = plt.imread(f"")
    st.image(model_perf_softmax, caption='') 
    model_perf_sigmoid = plt.imread(f"")
    st.image(model_perf_sigmoid, caption='')


    st.write("### Hypotesis 3 and validation")

    st.success(
        f"Use of ```RGB``` images for classification - Cherry leaf images converted from ```grayscale``` to ```RGB``` allow for an improved image classification performance. "
    )
    st.info(
        f"Color digital images are made of pixels, and pixels are made of combinations of primary colors."
        f" Grayscale images, are black-and-white and each pixel is a single sample representing only an amount of light."
        f" A grayscale image due to its nature conveys less information therefore the model is expected to require"
        f" less computational power to train. \n\n However, feeding a model with an RGB image or convert that image to grayscale "
        f" depends on the nature of the images and the information conveyd by the colour."
    )
    st.warning(
        f"Although coloured imges performed better in terms of accuracy, the difference in trainable parameters was marginal,"
        f" hence providing little to no benefits to the computational cost. "
    )
    model_perf_rgb = plt.imread(f"")
    st.image(model_perf_rgb, caption='')
    model_perf_gray = plt.imread(f"")
    st.image(model_perf_gray, caption='') 

    st.write(
        f"For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/JoelChan13/mildew-detector/blob/main/README.md).")