import streamlit as st
import matplotlib.pyplot as plt


def page_project_hypothesis_body():
    """
    Displays the hypothesis validations and explanations regarding the
    differences between healthy and powdery mildew-infected cherry leaves,
    performance of activation functions, and RGB vs grayscale images.
    """
    st.write("### Hypothesis 1 & Validation")

    st.success(
        f"There are distinct visual differences between healthy cherry "
        f"leaves and those affected by powdery mildew that can be identified "
        f"through image analysis."
    )
    st.info(
        f"Powdery mildew is usually characterised by clear marks over the "
        f"leaves and their fruits, and is typically the first symptom that "
        f"indicates an infected cherry leaf.\n\n"
        f"The marks can be described as porraceous green lesions on either "
        f"leaf surface, which progresses to white or grayish powdery spots "
        f"that develop in the infected area on leaves and fruits."
    )
    st.write(
        f"Visit the Leaves Visualiser tab which contains a detailed "
        f"investigation of infected and uninfected leaves."
    )

    st.warning(
        f"An effective model develops its predictive capabilities by training "
        f"on a batch of data without becoming overly dependent on it. This in "
        f"turn allows the model to generalize well and make reliable "
        f"predictions on new data because it did not rely on a set number of "
        f"connections between features and labels from the training set. "
        f"Instead, the model captured the overall pattern linking features to "
        f"labels. Our model was able to identify differences and learned to "
        f"distinguish and generalize, leading to accurate predictions."
    )

    st.write("### Hypothesis 2 & Validation")

    st.success(
        f"```softmax``` activation function performs better than ```sigmoid```"
        f" activation function for the CNN output layer."
    )

    st.info(
        f"The sigmoid activation function is typically not used in multi-class"
        f", single-label classification models because it outputs a "
        f"probability between 0 & 1 for each class independently. "
        f"In view of this, using sigmoid can lead to predictions where the sum"
        f" of probabilities across all classes is not equal to 1, which can be"
        f" problematic for multi-class classification.\n\n"
        f"Instead, the softmax function is preferred as it normalizes the "
        f"output probabilities across all classes so that they sum to 1, "
        f"providing a clear prediction for the single label. Should the user "
        f"still opt to use sigmoid, each class would be treated as a separate "
        f"binary classification, predicting the probability of each class "
        f"independently. This approach could be useful in multi-label "
        f"classification, where more than 1 class can be true simultaneously."
        f"\n\n"
        f"In a multi-class, single-label scenario, sigmoid activation could "
        f"lead to ambiguous and less accurate results. Therefore, softmax is "
        f"the recommended standard choice for these types of models."
    )
    st.warning(
        f"```softmax``` function performed better than ```sigmoid```"
    )

    st.write("### Hypothesis 3 & Validation")

    st.success(
        f"```RGB``` images perform better than ```grayscale``` "
        f"in terms of image classification performance."
    )
    st.info(
        f"Converting RGB images to grayscale can simplify image classification"
        f" by reducing data complexity, which can lead to faster training & "
        f"simpler models. It emphasizes pixel intensity, which is useful when "
        f"color is not crucial, and can reduce color-based noise, improving "
        f"focus on structural features. Grayscale images also save memory and "
        f"storage, which is beneficial for large datasets.\n\n"
        f"However, this removes color information, which is important "
        f"for tasks where color differentiation is key. The loss of features "
        f"can lead to lower accuracy in color-dependent classifications. "
        f"Additionally, relying solely on grayscale might cause the model to "
        f"overfit to structural features, potentially reducing generalization."
        f"\n\n"
        f"Thus, the decision to convert depends on the specific"
        f" classification task."
    )
    st.warning(
        f"Although our dataset was significant in size, ```RGB``` images"
        f" performed marginally better than ```grayscale``` images."
        f"  Particular consideration should be given if the company"
        f" decides to increase number of cherry trees, as opting for grayscale"
        f"  would save money & storage even if ```RGB``` is more accurate."
    )

    st.write(
        f"For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/JoelChan13/mildew-detector/"
        f"blob/main/README.md)."
    )
