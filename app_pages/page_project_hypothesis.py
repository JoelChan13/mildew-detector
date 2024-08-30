import streamlit as st
import matplotlib.pyplot as plt


def page_project_hypothesis_body():
    st.write("### Hypothesis 1 and validation")

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


    st.write("### Hypothesis 2 and validation")

    st.success(
        f"Comparison of Mathematical Functions - The `` function performs better than the `` function as an activation function for the CNN output layer. "
    )

    st.info(
        f""
    )
    st.warning(
        f"```softmax``` performed better than ```sigmoid```. "
        f"```sigmoid``` performed better than ```softmax```. "
    )
    

    st.write("### Hypothesis 3 and validation")

    st.success(
        f"Use of ```RGB``` images for classification - Cherry leaf images converted from ```grayscale``` to ```RGB``` allow for an improved image classification performance. "
    )
    st.info(
        f""
    )
    st.warning(
        f""
        f""
    )
    

    st.write(
        f"For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/JoelChan13/mildew-detector/blob/main/README.md).")