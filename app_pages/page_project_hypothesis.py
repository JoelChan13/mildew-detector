import streamlit as st
import matplotlib.pyplot as plt


def page_project_hypothesis_body():
    st.write("### Hypothesis 1 & Validation")

    st.success(
        f"There are distinct visual differences between healthy cherry leaves and those affected by powdery mildew that can be identified through image analysis."
    )
    st.info(
        f"Powdery mildew is usually characterised by clear marks over the leaves and their fruits, "
        f" and is typically the first symptom that indicates an infected cherry leaf.\n\n"
        f" The marks can be described as porraceous green lesions on either leaf surface,"
        f" which progresses to white or grayish powdery spots which develop in the infected area on leaves and fruits in the infected area."
    )
    st.write("Visit the Leaves Visualiser tab which contains a detailed investigation of infected and uninfected leaves.")
     
    st.warning(
        f" An effective model develops its predictive capabilities by training on a batch of data without becoming overly dependent on it"
        f" This in turn allows the model to generalize well and make reliable predictions on new data because it did not rely on a set number of connections between features and labels from the training set."
        f" Instead the model captured the overall pattern linking features to labels."
        f" Our model was able to identify differences and learned to distinguish and generalize, leading to accurate predictions."
    )

    st.write("### Hypothesis 2 & Validation")

    st.success(
        f"`softmax` activation function performs better than `sigmoid` activation function for the CNN output layer. "
    )

    st.info(
        f"The sigmoid activation function is typically not used in multi-class, single-label classification models because it outputs a probability between 0 and 1 for each class independently."
        f" In view of this, using sigmoid can lead to predictions where the sum of probabilities across all classes is not equal to 1, which can be problematic for multi-class classification."
        f" Instead, the softmax function is preferred as it normalizes the output probabilities across all classes so that they sum to 1, providing a clear prediction for the single label."
        f" Should the user still opt to use sigmoid, each class would be treated as a separate binary classification, predicting the probability of each class independently."
        f" This approach could be useful in multi-label classification, where more than one class can be true simultaneously."
        f" In a multi-class, single-label scenario, sigmoid activation could lead to ambiguous and less accurate results."
        f" Therefore, softmax is the recommended standard choice for these types of models."
    )
    st.warning(
        f"```softmax``` activation function performed better than ```sigmoid``` in our model"
    )

    st.write("### Hypothesis 3 & Validation")

    st.success(
        f"```RGB``` images perform better than ```grayscale```  in terms of image classification performance."
    )
    st.info(
        f"Converting RGB images to grayscale can simplify image classification by reducing data complexity, which can lead to faster training and simpler models."
        f" It emphasizes pixel intensity, which is useful when color is not crucial, and can reduce color-based noise, improving focus on structural features."
        f" Grayscale images also save memory and storage, which is beneficial for large datasets."
        f" However, this conversion removes color information, which is important for tasks where color differentiation is key."
        f" The loss of features can lead to lower accuracy in color-dependent classifications."
        f" Additionally, relying solely on grayscale might cause the model to overfit to structural features, potentially reducing generalization."
        f" Thus, the decision to convert depends on the specific classification task."
    )
    st.warning(
        f" Although our dataset was significant in size, ```RGB``` images performed marginally better than images converted to ```grayscale```."
        f" Particular consideration should be given if the company decides to increase the number of cherry trees, as opting for grayscale would save money and storage even though ```RGB``` is more accurate."
    )
    

    st.write(
        f"For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/JoelChan13/mildew-detector/blob/main/README.md).")