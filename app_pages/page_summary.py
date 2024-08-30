import streamlit as st
import matplotlib.pyplot as plt


def page_summary_body():

    st.write("### Quick Project Summary")

    st.info(
        f"**General Information**\n\n"
        f" Cherry powdery mildew is a fungal disease caused by the pathogen Podosphaera clandestina, which primarily affects cherry trees."
        f" Infected plants display white to grayish powdery spots, which are the fungal spores and mycelia, however, as the infection progresses, these spots can merge, covering large areas of the leaf surface..\n\n"
        f" This disease thrives in warm, dry conditions and can spread rapidly, especially in humid environments.\n\n"
        f" It typically infects the young leaves, shoots, and fruit of cherry trees, resulting in reduced fruit quality and yield.\n\n"
        f" The mildew fungus lives on the surface of plant tissues and feeds on them by sending tiny filaments into the cells. \n\n"
        f" Several leaves, infected and healthy, were picked and examined."
        f"\nVisual criteria used to detect infected leaves are:\n\n"
        f"* Porraceous green lesions on either leaf surface which progresses to\n "
        f"* white or grayish powdery spots which develop in the infected area on leaves and fruits."
        f" \n\n")

    st.warning(
        f"**Project Dataset**\n\n"
        f"The available dataset contains 2104 healthy leaves and 2104 affected leaves "
        f"individually photographed against a neutral background."
        f"")

    st.success(
        f"The project has three business requirements:\n\n"
        f"1 - A comprehensive study to visually differentiate between healthy cherry leaves and those infected with powdery mildew.\n\n"
        f"2 - Implement a predictive model capable of identifying whether a cherry leaf is healthy or contains powdery mildew based on image analysis, with an accuracy target of 97%. \n\n"
        f"3 - Develop an interactive dashboard that allows users to upload cherry leaf images, receive predictions, and review the analysis results."
        )

    st.write(
        f"For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/JoelChan13/mildew-detector/blob/main/README.md).")