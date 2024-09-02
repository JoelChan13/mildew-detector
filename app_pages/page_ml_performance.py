import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from src.machine_learning.evaluate_clf import load_test_evaluation

def page_ml_performance_metrics():
    version = 'v1'
    st.write("### Images Distribution per Set & Label ")

    # Load and display the image for labels distribution across train, validation, and test sets
    labels_distribution = plt.imread(f"")
    st.image(labels_distribution, caption='Labels Distribution on Train, Validation and Test Sets')

    # Load and display the image for overall sets distribution
    labels_distribution = plt.imread(f"")
    st.image(labels_distribution, caption='Sets Distribution')

    # Warning message explaining the dataset split into train, validation, and test sets
    st.warning(
        f"The cherry leaves dataset was divided into three subsets.\n\n"
        f"Train set (70% of dataset) used to 'fit' the model in order to obtain an initial baseline on which the model will be able to generalise and make predictions.\n\n"
        f"Validation set (10% of dataset) used as a means to refine the model after epochs.\n\n"
        f"The test set (20% of dataset) used to determine the accuracy of the model following the training phase.")
    st.write("---")

    st.write("### Model Performance")

    # Load and display the classification report image
    model_clf = plt.imread(f"outputs/{version}/clf_report.png")
    st.image(model_clf, caption='Classification Report')  

    # Warning message explaining the classification report metrics
    st.warning(
        f"**Classification Report**\n\n"
        f"Precision: Percentage of correct predictions.\n\n"
        f"Recall: Percentage of positive cases detected.\n\n"
        f"F1 Score: Percentage of correct positive predictions.\n\n"
        f"Support: Number of occurences of a particular class in a selected dataset.")

    # Load and display the ROC curve image
    model_roc = plt.imread(f"outputs/{version}/roccurve.png")
    st.image(model_roc, caption='ROC Curve')

    # Warning message explaining what the ROC curve represents
    st.warning(
        f"**ROC Curve**\n\n"
        f"ROC curve is a graphical representation used to evaluate the performance of a binary classification model. "
        f" It plots the True Positive Rate (sensitivity) against the False Positive Rate (1 - specificity) at various threshold settings."
        f" The ROC curve helps visualize how well a model distinguishes between two classes.\n\n"
        f" A model with a curve that rises quickly towards the top-left corner of the graph is considered to have good performance,"
        f" indicating high true positive rates with low false positive rates.")

    # Load and display the confusion matrix image
    model_cm = plt.imread(f"outputs/{version}/confusion_matrix.png")
    st.image(model_cm, caption='Confusion Matrix')

    # Warning message explaining the components of a confusion matrix
    st.warning(
        f"**Confusion Matrix**\n\n"
        f"A confusion matrix is a table used to evaluate the performance of a classification model by comparing its predictions with the actual outcomes.\n\n"
        f"True Positive: The number of correct predictions where the model correctly identifies the positive class.\n\n"
        f"True Negative: The number of correct predictions where the model correctly identifies the negative class.\n\n"
        f"False Positive: The number of incorrect predictions where the model incorrectly identifies the negative class as positive (also known as Type I error).\n\n"
        f"False Negative: The number of incorrect predictions where the model incorrectly identifies the positive class as negative (also known as Type II error).\n\n"
        f"High TP and TN rates, with low FP and FN rates are indicatives of a good model.")

    # Load and display the model performance image
    model_perf = plt.imread(f"")
    st.image(model_perf, caption='Model Performance')  

    # Warning message for model performance 
    st.warning(
        f"**Model Performance**\n\n"
        f"")

    st.write("### Generalised Performance on Test Set")
    st.dataframe(pd.DataFrame(load_test_evaluation(version), index=['Loss', 'Accuracy']))
    
    st.write(
        f"For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/JoelChan13/mildew-detector/blob/main/README.md).")