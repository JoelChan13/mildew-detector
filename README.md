# ![Mildew Detector Logo](https://github.com/JoelChan13/mildew-detector/blob/main/readme_images/mildew_detector_logo.png)

## Table of Contents

1. [Dataset Content](#dataset-content)
2. [Business Requirements](#business-requirements)
3. [Hypothesis and Validation](#hypothesis-and-validation)
4. [Rationale for the Model](#rationale-for-the-model)
5. [Trial and Error](#trial-and-error)
6. [The Rationale to Map the Business Requirements to the Data Visualisations and ML Tasks](#the-rationale-to-map-the-business-requirements-to-the-data-visualisations-and-ml-tasks)
7. [ML Business Case](#ml-business-case)
8. [Dashboard Design](#dashboard-design)
9. [CRISP DM Process](#crisp-dm-process)
10. [Bugs](#bugs)
11. [Deployment](#deployment)
12. [Technologies Used](#technologies-used)
13. [Credits](#credits)

### Deployed version at [Mildew Detector](https://p5-mildew-detector-13512f4dba8f.herokuapp.com/)

## Dataset Content

The dataset contains 4208 featured photos of single cherry leaves against a neutral background. The images of the cherry tree leaves were obtained from Farmy & Foods' crop fields and feature images of leaves that are either healthy or infested by powdery mildew. Powdery mildew is a fungal disease that affects a wide range of plants and typically appears as white or grayish powdery spots on the leaves, stems, and sometimes fruits of the infected plants, mainly caused by various species of fungi and typically targets particular types of plants. Notwithstanding the fact that this disease affects many plant species, the client is particularly concerned about their cherry plantation crop, which is one of their finest products in the portfolio.

Dataset Source From [Kaggle](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves).

## Business Requirements

The primary objective of this project is to develop a Machine Learning system that aids Farmy & Foods in addressing the issue of powdery mildew affecting their cherry plantations. Currently, the process is manual verification if a given cherry tree contains powdery mildew. An employee spends around 30 minutes in each tree, taking a few samples of tree leaves and verifying visually if the leaf tree is healthy or has powdery mildew. If there is powdery mildew, the employee applies a specific compound to kill the fungus. The time spent applying this compound is 1 minute. Due to this fact, manually inspecting all plants is not scalable, the client has outlined the following specific business requirements:

1. Visual Differentiation: Conduct a comprehensive study to visually differentiate between healthy cherry leaves and those infected with powdery mildew.
2. Predictive Modeling: Implement a predictive model capable of identifying whether a cherry leaf is healthy or contains powdery mildew based on image analysis, ideally with an accuracy target of not less than 97%.
3. Interactive Dashboard: Develop an interactive dashboard that allows users to upload cherry leaf images, receive predictions, and review the analysis results.

By meeting these requirements, the project will enable Farmy & Foods to maintain product quality and meet market demands effectively through large-scale automated detection.

## Hypothesis and Validation

1. **Hypothesis 1**: Visual Differentiation of Cherry Leaves - There are distinct visual differences between healthy cherry leaves and those affected by powdery mildew that can be identified through image analysis.
   - How to validate: This hypothesis can be tested by conducting a study that involves analyzing average images, variability images, and creating image montages for both healthy and powdery mildew-affected leaves. If the differences are visually apparent and consistent, this would support the hypothesis. The analysis will include identifying specific visual markers or patterns that distinguish healthy leaves from those affected by powdery mildew.<br/>

2. **Hypothesis 2**: Comparison of Mathematical Functions - The `softmax` function performs better than the `sigmoid` function as an activation function for the CNN output layer.
   - How to validate:  This hyptothesis can be tested by understanding the type of problem we are trying to solve, and training and comparing the models, modifying only the activation function in the output layer. This will enable us to identify which model produces more accurate results.<br/>

3. **Hypothesis 3**: ```RGB``` images perform better than ```grayscale```  in terms of image classification performance.
   - How to validate: This hypothesis can be tested by training and comparing the models, modifying only the image colour, which will enable us to identify which model produces more accurate results, along with the margin of difference in accuracy.<br/>

### Hypothesis 1

> There are distinct visual differences between healthy cherry leaves and those affected by powdery mildew that can be identified through image analysis.

**1. Introduction**

We believe cherry leaves infected with powdery mildew have distinct visual traits that set them apart from healthy leaves. Infected leaves typically display a white or grayish powdery coating on their surfaces, caused by the fungal spores. This powdery layer is often found on the upper side of the leaves but can also appear on the underside. Healthy cherry leaves, on the other hand, are usually uniformly green, smooth, and free from any powdery residue or discolouration. The contrast in appearance makes it easier to identify and distinguish between infected and healthy leaves. In order to prove this hypothesis, the above-mentioned distinct traits need to be converted into tensors in order for the ML model to make use of them when training and evaluating.

- Understand the Problem & Mathematical Functions

Normalising images in a dataset before training a neural network is crucial for several reasons. First, it ensures that all pixel values are on a similar scale, typically between 0 and 1 or -1 and 1, which helps the model converge faster during training. Without normalisation, the network might struggle with varying scales of pixel values, leading to inefficient learning. It also prevents any particular feature from dominating the training process, ensuring that the model pays equal attention to all features. Normalisation improves numerical stability, reducing the risk of exploding or vanishing gradients. Additionally, it helps achieve more consistent and reliable performance across different datasets and models, making the training process more robust and effective.

In the context of image datasets, calculating the mean and standard deviation involves considering the four dimensions of an image: B (Batch size), C (Channels, such as RGB), H (Height), and W (Width). These dimensions represent the entire dataset where multiple images are processed simultaneously.

To compute the mean, the pixel values are averaged across all images in the batch, across all color channels, and across every pixel position (height and width). The same applies to the standard deviation, which measures how much the pixel values vary from the mean across these dimensions. This process ensures that the mean and standard deviation represent the overall characteristics of the dataset, accounting for variations in color and spatial information, and allowing for consistent normalisation of all images in the dataset.

**2. Observation**

- Distinct visual traits differentiating a healthy cherry leaf from an infected cherry leaf visible in the below image montage

![healthy](https://github.com/JoelChan13/mildew-detector/blob/main/readme_images/healthy_montage.png)
![powderymildew](https://github.com/JoelChan13/mildew-detector/blob/main/readme_images/powdery_mildew_montage.png)

**3. Conclusion**

A good model develops its predictive abilities by learning from a batch of data without overfitting to it. Overfitting occurs when the model becomes too closely tailored to the training data, memorizing the specific relationships between features and labels rather than understanding the underlying patterns. By avoiding this, the model can generalize its learning, meaning it applies what it has learned to new, unseen data. This generalization allows the model to make reliable predictions on future observations because it has learned the broader patterns that link features to labels, rather than just the specific examples it was trained on. This approach ensures the model remains flexible and effective across different datasets.
Our model was successful in detecting and distinguishing the distinct traits of healthy and infected cherry leaves. In view of this the ML model could be further used to make predictions for further use.

**Sources**:

- [Hands-on Machine Learning with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://books.google.com.mt/books/about/Hands_on_Machine_Learning_with_Scikit_Le.html?id=I6qkDAEACAAJ&source=kp_book_description&redir_esc=y) by Aurélien Géron
- [Powdery mildew: biology, epidemiology, and management of Podosphaera spp. of tree fruit](https://shop.bdspublishing.com/store/bds/detail/product/3-190-9781838798970)

---
### Hypothesis 2

> `softmax` activation function performs better than `sigmoid` activation function for the CNN output layer.

**1. Introduction**

   1. Understand the Problem & Mathematical Functions

The goal is to classify a cherry leaf into one of two distinct categories: healthy or infected. This represents a classification challenge, which can be interpreted as either a binary classification or multi-class classification. In binary classification, the leaf is either classified as healthy or not healthy, whereby in multi-class classification, where each outcome is assigned a single label from multiple classes, the leaf is either classified as healthy or infected.

- Using Sigmoid Activation Function:

The sigmoid activation function can be used effectively for this problem as it is designed for binary classification tasks. It provides an output probability between 0 and 1, which can be interpreted as the likelihood of the cherry leaf being in one of the two classes (healthy or infected). For this specific problem, sigmoid would output a single value representing the probability of the leaf being classified as "infected," with the complement representing "healthy." The model can then use a threshold (commonly 0.5) to decide the final classification. While sigmoid is well-suited for binary classification, it does not naturally handle scenarios where multiple classes could be present, but in this case with only two classes, it functions effectively.

- Using Softmax Activation Function:

Softmax is generally more appropriate for classification problems with multiple classes, even though it can be used for binary classification scenarios. Softmax provides a probability distribution across all classes, ensuring that the output probabilities sum to 1. In the context of binary classification, softmax would output two values, one for each class (healthy and infected), with the values summing to 1. This can be advantageous as it provides a clearer view of the relative probabilities for each class, rather than just a single probability with a threshold. Using softmax allows the model to scale naturally to problems with more than two classes if needed, providing flexibility for future extension or reclassification. For the current binary problem, softmax ensures that the probabilities are well-calibrated and interpretable in the context of multi-class classification principles.

   2. Understand how to evaluate the performance

Learning curves are a valuable diagnostic tool for assessing and comparing the performance of machine learning algorithms, including the choice between softmax and sigmoid activation functions. During training, the model is iteratively exposed to the dataset for a specified number of epochs. An epoch is one complete pass through the training data. Tracking how the model's performance changes over epochs can reveal insights about its learning process and the effectiveness of the activation functions. The accuracy curve shows how the model’s classification performance improves over time for both training and validation sets. For binary classification (healthy vs. infected), this curve helps identify if the model is learning effectively and if the activation function is appropriate. The loss curve tracks the error between the model's predictions and the actual labels. A decreasing loss indicates that the model is learning and improving its predictions. It is crucial to monitor both training and validation loss to understand if the model is generalizing well.
When evaluating learning curves:

   - Training and Validation Loss: A good fit is indicated by both training and validation loss curves decreasing and stabilizing over epochs, with minimal gap between the final values. This suggests that the model is learning effectively and generalizing well to unseen data.

   - Accuracy Trends: Both training and validation accuracy should increase and converge towards similar values. If the validation accuracy is significantly lower than the training accuracy, this might indicate overfitting.

If the learning curves with sigmoid show consistent improvement and convergence with minimal overfitting, it might be suitable. However, watch for issues like vanishing gradients or slow convergence. If using softmax, check if the learning curves demonstrate better convergence and stability. Softmax can provide clearer probabilities for classification and might perform better if it shows reduced validation loss and improved accuracy.

In summary, by analyzing learning curves for both activation functions, you can determine which one better balances learning and generalization. The ideal activation function will show well-aligned training and validation loss and accuracy curves that stabilize with minimal discrepancy.
  
**2. Observation**

```softmax``` showed less training/validation sets gap and more consistent learning rate compared to ```sigmoid```.

   1. Nature of the Output Activation Function:<br/>
      •	Sigmoid: The sigmoid activation function outputs values between 0 and 1 for each class independently. In binary classification, this is used to model probabilities for each class (e.g., class 0 or class 1). However, in your case where you have two mutually exclusive classes (healthy and powdery_mildew), using sigmoid treats these classes as independent, which is inappropriate because the presence of one implies the absence of the other. This can confuse the model.<br/>
      •	Softmax: The softmax function outputs a probability distribution, where the probabilities for all classes sum to 1. In a two-class problem like yours, softmax is appropriate because it treats the classes as mutually exclusive, ensuring that the network only picks one class with the highest confidence.<br/>

   2. Impact on Model Training:<br/>
      •	Sigmoid Output:<br/>
         o	Training Behavior: The sigmoid-based model fails to learn, as shown by the constant loss and accuracy (around 50%) during training and validation. The network gets stuck predicting 50% for both classes, which means it doesn't distinguish between them. This indicates that the model with sigmoid is not learning anything useful.
         o	Performance: The accuracy of the model using sigmoid is only 46.45%, with a precision, recall, and f1-score of 0 for the powdery_mildew class, which means the model predicts everything as healthy.
      •	Softmax Output:<br/>
         o	Training Behavior: The softmax-based model shows progressive learning, with accuracy improving across epochs, reaching 89.45% accuracy with a good balance between precision and recall.<br/>
         o	Performance: The softmax-based model correctly predicts the majority of cases for both classes (healthy and powdery_mildew), with a high f1-score of 0.88 for powdery_mildew. This shows that the model is learning well, predicting the presence of powdery mildew more accurately.<br/>

   3. Mutual Exclusivity in Classification:<br/>
      •	Sigmoid Misinterpretation: Since sigmoid treats each class independently, it is possible for both output nodes to predict high values (both near 1) or low values (both near 0). This would be incorrect in your case since a leaf can only be either healthy or infected. This leads to poor probability outputs, which you see in the 50.16% probability for healthy using sigmoid, implying no strong prediction for either class.<br/>
      •	Softmax Proper Interpretation: Softmax ensures the sum of all class probabilities equals 1. This makes the model focus on differentiating between the two classes (healthy vs. powdery_mildew). In your model, the softmax output gives a high probability for one class over the other, such as the 98.84% probability for powdery_mildew, indicating the model is confident in its predictions.<br/>

   4. Classification Report Insights:<br/>
      •	Sigmoid: The precision, recall, and f1-score for powdery_mildew are all 0, indicating that the sigmoid model fails to predict this class at all. This suggests that the model is simply guessing or defaulting to healthy for all cases.<br/>
      •	Softmax: The model using softmax correctly balances between the two classes, with an f1-score of 0.90 for healthy and 0.88 for powdery_mildew, showing that the model is predicting both classes effectively.<br/>

- Loss/Accuracy of LSTM model trained using Softmax vs Loss/Accuracy of LSTM model trained using Sigmoid:
![softmax_model](https://github.com/JoelChan13/mildew-detector/blob/main/streamlit_images/model_history_softmax_rgb.png)
![sigmoid_model](https://github.com/JoelChan13/mildew-detector/blob/main/streamlit_images/model_history_sigmoid.png)

**3. Conclusion**

Opting for softmax over sigmoid in this case is better because it better handles the mutually exclusive nature of your classification problem, leading to more accurate and reliable predictions. The sigmoid model suffers from poor learning due to its inappropriate treatment of independent probabilities, while the softmax model improves training dynamics and outputs meaningful probabilities for each class.

**Sources**:

- [Hands-on Machine Learning with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://books.google.com.mt/books/about/Hands_on_Machine_Learning_with_Scikit_Le.html?id=I6qkDAEACAAJ&source=kp_book_description&redir_esc=y) by Aurélien Géron

---
### Hypothesis 3

> ```RGB``` images perform better than ```grayscale```  in terms of image classification performance. 

**1. Introduction**

RGB images generally outperform grayscale images in image classification tasks due to their richer feature set and enhanced detail capture. Unlike grayscale images, which only capture intensity information through a single channel, RGB images include three color channels—red, green, and blue—that provide more nuanced data. This additional color information allows models to detect subtle differences between objects, improving their ability to distinguish between classes. The increased dimensionality from RGB channels offers more data for learning, which enhances the model’s feature extraction and generalization capabilities. Color-specific features, such as patterns or markings, are often crucial for accurate classification, making RGB images particularly valuable for complex tasks. Research and benchmarks frequently show that models trained on RGB images achieve higher accuracy compared to those trained on grayscale images, especially in applications where color variations are essential for correct interpretation. Additionally, relying solely on grayscale might cause the model to overfit to structural features, potentially reducing generalization. Thus, the decision to convert depends on the specific classification task. <br/>

**2. Observation**

   1. Model Comparison<br/>
      •	Model Architecture:<br/>
         Both models share an identical architecture, consisting of three convolutional layers, followed by max pooling, a dense layer, dropout, and a final output layer with two classes (healthy or infected). The total number of trainable parameters is the same (3,715,234) for both tests.<br/>
      •	Dataset and Training:<br/>
         o	Both models are trained on the same dataset, which includes equal numbers of images of healthy and powdery mildew-infected leaves.<br/>
         o	The only difference is that one model uses RGB images (3 channels), while the other uses grayscale images (1 channel).<br/>

   2. Key Differences<br/>
      •	Training Time:<br/>
         o	RGB model: Training took about 215 seconds per epoch.<br/>
         o	Grayscale model: Training was faster, taking approximately 188-191 seconds per epoch.<br/>

   Reason:<br/>
   RGB images have three color channels, while grayscale images have only one. The increased number of channels in RGB results in more computations during the forward and backward passes, leading to slightly longer training times.<br/>

      •	Performance:<br/>
         o	Accuracy RGB model: 89.45% test accuracy.<br/>
         o	Accuracy Grayscale model: 95.73% test accuracy.<br/>
         o	Loss RGB model: 0.2222 loss.<br/>
         o	Loss Grayscale model: 0.1173 loss.<br/>
         o	Precision, Recall, and F1-Score for RGB: Healthy: Precision 0.83, Recall 1.00, F1-Score 0.90.<br/>
         o	Precision, Recall, and F1-Score for RGB: Powdery mildew: Precision 0.99, Recall 0.79, F1-Score 0.88.<br/>
         o	Precision, Recall, and F1-Score for RGB: Accuracy: 89%.<br/>
         o	Precision, Recall, and F1-Score for Grayscale: Healthy: Precision 0.94, Recall 0.98, F1-Score 0.96.<br/>
         o	Precision, Recall, and F1-Score for Grayscale: Powdery mildew: Precision 0.98, Recall 0.94, F1-Score 0.96.<br/>
         o	Precision, Recall, and F1-Score for Grayscale: Accuracy: 96%.<br/>

- Loss/Accuracy of LSTM model for Grayscale & Loss/Accuracy of LSTM model for RGB:
![Grayscale](https://github.com/JoelChan13/mildew-detector/blob/main/streamlit_images/model_history_grayscale.png)
![RGB](https://github.com/JoelChan13/mildew-detector/blob/main/streamlit_images/model_history_softmax_rgb.png)

**3. Conclusion**

The grayscale model achieved a higher overall accuracy compared to the RGB model, indicating that it was better at classifying whether a cherry leaf was infected with powdery mildew or not.<br/>
The grayscale model not only achieved better accuracy but also had a lower loss, meaning it was more confident and made fewer mistakes during classification.<br/>
The grayscale model showed better precision and recall across both classes, which led to a higher F1-score. Notably, the grayscale model had a better balance between precision and recall for powdery mildew detection, while the RGB model had a higher precision for powdery mildew but lower recall (fewer infected leaves were correctly identified).<br/>
Notwithstanding this, RGB images contain more detailed color information, which could help in scenarios where color variation is important for classification. RGB images provide more information than grayscale because they contain three color channels (red, green, and blue) instead of just one intensity channel. In the context of plant disease detection, subtle color differences between healthy and infected leaves may be present, even if they were not strongly leveraged in this particular case.<br/>
If powdery mildew or other diseases cause slight color changes in leaves (e.g., a change in greenness or spotting), these changes might be better captured using RGB images. For example, chlorosis (yellowing of the leaves) might accompany powdery mildew, and an RGB model could potentially detect these subtle shifts in hue.<br/>
Color data can sometimes help detect features that are not as easily distinguishable based on texture alone. For example, other types of leaf diseases or conditions might involve more color-based patterns that the RGB model could learn, whereas a grayscale model would miss out on this information.<br/>

**Sources:**

- [Building Machine Learning Pipelines](https://www.google.com.mt/books/edition/Building_Machine_Learning_Pipelines/H6_wDwAAQBAJ?hl=en&gbpv=0) by Hannes Hapke & Catherine Nelson


## Rationale for the Model

### The goal

A good machine learning model is one that generalizes well from the training data to unseen data. This means that the model captures the underlying patterns in the data without overfitting, which occurs when a model performs well on training data but poorly on new, unseen data. Such a model achieves a balance between bias and variance, avoiding being too simple (high bias) or too complex (high variance). It should have a high accuracy on both training and validation datasets, indicating that it can make correct predictions consistently. Additionally, the model should have low error rates, such as low mean squared error for regression tasks or low classification error for classification tasks.

Scalability and efficiency are also important, meaning the model should be able to handle large datasets and make predictions in a reasonable time. Interpretability is another key factor, especially in applications where understanding the model's decisions is critical. A good model is robust, meaning it can handle noisy or missing data and still perform well, and it should be stable, producing consistent results across different subsets of the data or when retrained.

A trial and error approach was chosen in order to determine the hyperparameters, the number of hidden layers and nodes, and which optimizer to opt for. From the trials conducted, it was determined that the model has 1 input layer, 3 hidden layers (2 ConvLayer, 1 FullyConnected), 1 output layer.

### Choosing the hyperparameters

- **Convolutional layer size**

The convolutional layer size in machine learning refers to the dimensions of the filters (kernels) used in a Convolutional Neural Network (CNN). These filters slide over the input data (e.g., an image) to detect patterns, such as edges or textures. The filter size is typically specified as a small matrix, like 3x3 or 5x5, indicating how many pixels the filter covers at once. Smaller filters capture fine details, while larger filters capture broader patterns. The depth of the filter corresponds to the number of channels in the input data, such as the RGB channels in a color image. The output of the convolutional layer is a feature map, which highlights the presence of specific features in different locations. The convolutional layer size impacts the model's ability to detect different levels of detail and affects computational efficiency.

Using a two-dimensional CNN (Conv2D) is more appropriate for pictures of healthy and infected cherry leaves because these images have two spatial dimensions: width and height. A Conv2D layer can effectively capture patterns, like edges, textures, and shapes, by sliding filters over these two dimensions, which is crucial for analyzing the visual features of the leaves. In contrast, a 1D convolution layer is designed for data with a single spatial dimension, like time series or sequential data, where the structure only varies along one axis. Applying 1D convolution to images would ignore the two-dimensional nature of the data, making it ineffective at capturing the spatial relationships necessary for image analysis. Hence, Conv2D was deemed to be the most appropriate choice for processing 2D images like those in our dataset.

- **Convolutional kernel size**

 A 3x3 filter slides over the image in both the x and y directions (width and height), capturing local patterns like edges or textures. The stride of 1 means the filter moves one pixel at a time, ensuring detailed analysis of the image. The choice of a 3x3 filter is preferred over a 2x2 filter because a 2x2 filter would not allow the use of zero padding (adding extra pixels around the edges) without causing issues, especially with even-sized images, potentially leading to a loss of information at the borders. Compared to a 5x5 filter, the 3x3 filter is smaller, allowing it to focus more on fine details, which is important for identifying subtle signs of disease in the leaves. The third dimension of the filter corresponds to the color channels of the image (e.g., RGB), ensuring the filter captures features across all color layers.

- **Number of neurons**

The number of neurons in a machine learning model, particularly in the hidden layers, is determined based on the complexity of the task and the amount of data available. For distinguishing healthy cherry leaves from those infected with powdery mildew, the model needs enough neurons to capture the essential features of the images, such as textures and color patterns. However, too many neurons can lead to overfitting, where the model memorizes the training data but fails to generalize to new data. Conversely, too few neurons might result in underfitting, where the model cannot capture the necessary patterns. A common approach is to start with a small number of neurons and gradually increase them, monitoring the model's performance on a validation set. Powers of two (e.g., 16, 32, 64) are often used due to computational efficiency, which is why it was also chosen for the scope of this ML model.

- **Activation function**:

ReLu was chosen as an activation function due to several benefits in training deep neural networks.ReLU (Rectified Linear Unit) is a popular activation function in deep learning because of its simplicity and efficiency. Unlike the sigmoid activation, which squashes inputs into a small range between 0 and 1, ReLU outputs the input directly if it's positive, and zero otherwise. This non-linear property allows ReLU to introduce non-linearity into the model, enabling it to learn complex patterns. ReLU is computationally efficient since it requires only a simple thresholding operation. Its derivative is 1 for positive inputs and 0 for negative inputs, which helps prevent the vanishing gradient problem—a common issue with sigmoid where gradients become too small, slowing down or stalling learning in deep networks. ReLU’s ability to maintain stronger gradients during backpropagation leads to faster and more reliable convergence, making it highly effective in training deep neural networks.

- **Pooling**

Pooling is a technique used in neural networks to reduce the dimensionality of feature maps, which helps to minimize computational load and control overfitting. It works by summarizing the presence of features within a local region, allowing the network to maintain important information while discarding less critical details. MaxPooling was chosen for the mildew detector because it effectively highlights the most prominent features in an image. In MaxPooling, the pooling layer takes a window (e.g., 3x3) and selects the maximum pixel value within that window. This operation helps retain the most significant features while reducing the image size and computational complexity. Since the background of the cherry leaf images is dark green, MaxPooling focuses on the lighter pixels where the powdery mildew, which is white, appears. By concentrating on these brighter areas, MaxPooling enhances the visibility of the mildew spots relative to the dark background. This approach helps the model more effectively distinguish the disease from the background, improving overall detection accuracy.

- **Output Activation Function**

Output activation functions are crucial in machine learning and deep neural networks as they determine how the model's predictions are scaled and interpreted. For classification tasks, the choice of activation function impacts the model's ability to output probabilities and make accurate predictions.

Softmax is used in multiclass classification problems. It converts raw scores (logits) into probabilities by exponentiating the scores and normalising them so that they sum to 1. This makes it ideal for distinguishing between multiple classes, as it provides a probability distribution across all classes, helping the model to choose the most likely class. Softmax also ensures that the outputs are normalised, which can stabilize training and improve performance in complex classification tasks.

Sigmoid, on the other hand, is used in binary classification. It outputs probabilities between 0 and 1 for a single class, making it suitable for problems where each instance belongs to only one of two classes. However, in a multiclass setting, using sigmoid would require a separate output neuron for each class, each producing an independent probability, which may not effectively capture the mutual exclusivity of the classes.

In our model for detecting powdery mildew on cherry leaves, softmax performed better than sigmoid because it handled the multiclass nature of the problem more effectively. It reduced the gap between training and validation sets and provided a more consistent learning rate by correctly interpreting the multi-class output. Softmax helps in distinguishing between healthy leaves and those with different infection levels by providing a clear probabilistic ranking of each class, which improves the model’s accuracy and robustness in identifying and predicting the presence of mildew.

- **Dropout**

Dropout is a regularization technique used in CNN and other machine learning models to prevent overfitting. During training, dropout randomly deactivates a percentage of neurons in a layer, forcing the model to learn more robust and generalized features by not relying too heavily on any single neuron. This helps in reducing the chance that the model memorizes the training data instead of generalizing from it.

In the context of our project, applying a 20% dropout rate helps to mitigate overfitting, especially given the relatively small number of training samples. This means that during each training iteration, 20% of the neurons in the dropout layers are randomly turned off, which encourages the network to develop multiple redundant representations of the data. As a result, the model becomes more capable of generalizing to new, unseen data.

If dropout layers were not included, the model might overfit to the training data, meaning it would perform well on the training set but poorly on new, unseen data. This is because the model might learn to memorize specific patterns in the training samples rather than learning general features applicable to the broader dataset. Consequently, the absence of dropout could lead to reduced model performance and less reliable predictions in real-world scenarios.

**Source**:

- [Deep Learning and Convolutional Neural Networks for Medical Imaging and Clinical Informatics](https://books.google.com.mt/books/about/Deep_Learning_and_Convolutional_Neural_N.html?id=hM2wDwAAQBAJ&source=kp_book_description&redir_esc=y) by - Le Lu, Xiaosong Wang, Gustavo Carneiro, Lin Yang
- [Hands-on Machine Learning with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://books.google.com.mt/books/about/Hands_on_Machine_Learning_with_Scikit_Le.html?id=I6qkDAEACAAJ&source=kp_book_description&redir_esc=y) by Aurélien Géron

### Hidden Layers

Hidden layers in a neural network are called "hidden" because their internal workings and values are not directly observable from the input or output; they operate between these layers to transform and process the data.

Convolutional Layers are used for feature extraction from images. They apply convolutional filters to input data to detect features such as edges or textures while reducing the dimensionality and computational complexity by sharing parameters across different parts of the image. This approach efficiently captures spatial hierarchies in the data, crucial for identifying patterns like powdery mildew spots on cherry leaves.

Dense (Fully Connected) Layers are used in the final stages of the model. They perform classification by connecting every neuron from the previous layer to each neuron in the dense layer. This layer combines the features extracted by convolutional layers to make predictions. The dense layer's ability to learn complex decision boundaries makes it suitable for classification tasks, where it converts the extracted features into final class probabilities.

The choice of using convolutional layers for feature extraction followed by dense layers for classification is effective because it allows the model to first capture and learn essential patterns from the data and then use these patterns to make precise classifications. This combination leverages the strengths of both types of layers: convolutional layers for spatial feature learning and dense layers for decision-making based on these features.

**Source**:

- [Convolutional Neural Networks: A Comprehensive Guide](https://medium.com/thedeephub/convolutional-neural-networks-a-comprehensive-guide-5cc0b5eae175) by - Jorgecardete

### Model Compilation

Loss functions quantify how far the model's predictions are from the actual values. In multiclass classification, categorical_crossentropy calculates the discrepancy between the predicted probability distribution and the true distribution.
In the model compilation for the powdery mildew detector, categorical_crossentropy was used as the loss function because the problem involves multiclass classification. This loss function, also known as Softmax Loss, measures how well the predicted class probabilities align with the actual class labels. It works with the softmax activation function to produce a probability distribution across all classes, which is essential for accurately classifying the leaves into multiple categories (healthy or infected).

Optimizers adjust the model's weights to minimize the loss function. They use gradients computed during backpropagation to update the weights, thereby improving model accuracy over time. AdaGrad, in particular, is effective for dealing with varying feature scales and sparse data. In view of this, AdaGrad was chosen as the optimizer during the trial and error phase because it adapts the learning rate based on the frequency of parameter updates. This means that parameters which frequently receive updates have their learning rates decreased, while those receiving fewer updates have their learning rates increased. This adaptation helps in efficiently navigating the loss landscape, especially in sparse or noisy data scenarios.

Accuracy is used as a metric to measure the proportion of correct predictions (where y_pred matches y_true) out of the total predictions made. This metric is important because it provides a straightforward measure of how well the model is performing in distinguishing between healthy and infected cherry leaves. Accuracy helps to gauge the overall effectiveness of the model by reflecting its ability to correctly classify instances.

The use of accuracy is particularly relevant when the class distribution is balanced, as it indicates how often the model's predictions align with the true labels. However, in cases of imbalanced classes, where some classes are more frequent than others, accuracy might be misleading. In such scenarios, additional metrics like precision, recall, and F1-score might be used to gain a more nuanced understanding of model performance.

**Source**:

- [Hands-on Machine Learning with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://books.google.com.mt/books/about/Hands_on_Machine_Learning_with_Scikit_Le.html?id=I6qkDAEACAAJ&source=kp_book_description&redir_esc=y) by Aurélien Géron
- [Keras Accuracy Metrics](https://keras.io/api/metrics/accuracy_metrics/#:~:text=metrics.,with%20which%20y_pred%20matches%20y_true%20.)

## Trial and Error

During the trial and error phase of this project, the primary objective was to confirm or deny the hypotheses stated in the hypothesis section and to identify the best hyperparameters for the mildew detector model. This approach allowed for systematic experimentation to fine-tune the model's architecture and parameters. The process was documented in the ModelEvaluation_Trials Jupyter notebook, where different trials were conducted by commenting in and out relevant code to focus on specific tests.

A trial was conducted to test hypothesis #3, which claims that RGB images perform better than grayscale images in terms of image classification accuracy. To validate this hypothesis, models were trained and compared, modifying only the image color. This allowed for the identification of which model produced more accurate results and the margin of difference in accuracy between the two color formats.

Another key focus of the trials was the evaluation of the softmax activation function, tested both with and without early stopping, and then compared to sigmoid, addressing hypothesis #2. Additionally, the model's performance was tested with one convolutional layer removed, helping to refine the convolution layer size.

Optimizers AdaGrad & Adam were chosen as the optimizers during the trial and error phase. Adam was considered as a potential alternative, however, it did not emerge as the ideal optimizer for this project. Adam combines the advantages of two other popular optimizers, AdaGrad and RMSProp. It maintains a per-parameter learning rate and updates based on both the average of recent gradients (like AdaGrad) and the squared gradients (like RMSProp), making it adaptive and efficient for a wide range of tasks. Notwithstanding this, AdaGrad's ability to adapt learning rates to the data's characteristics made it more suitable for this model. AdaGrad adapts the learning rate based on the frequency of parameter updates, decreasing learning rates for frequently updated parameters and increasing them for less frequently updated ones. This adaptation helped efficiently navigate the loss landscape, particularly in the presence of sparse or noisy data.

## The rationale to map the business requirements to the Data Visualisations and ML tasks

### Business Requirement 1: Data Visualisation

>The client is interested in being able to visually differentiate between healthy cherry leaves and those infected with powdery mildew.

- In order to address this business requirement:
   1. We need to provide an interactive dashboard which is easily navigated by the user in order to understand the data.
   2. We need to display the "mean" and "standard deviation" images for healthy and infected leaves.
   3. We need to display the difference between average healthy and infected leaves.
   4. We need to display an image montage for either healthy or infected cells.

### Business Requirement 2: Classification

>The client is interested in employing a predictive model capable of identifying whether a cherry leaf is healthy or contains powdery mildew based on image analysis which is highly accurate.

- In order to address this business requirement:
   1. We need to develop an accurate ML model to predict whether a cherry leaf is healthy or infected.
   2. Our ML model needs to have a dashboard which enables the user to upload cherry leaf images through a widget and obtain a relative prediction statement, indicating whether the leaf is infected or not

### Business Requirement 3: Report

>The client is interested obtaining predictions and review the analysis results through detailed reports. 

- In order to address this business requirement:
   1. We need to develop a dashboard which enables the user to obtain a report with the predicted status from the ML predictions on new leaves after uploading images of cherry leaves.

## ML Business Case

**Visit the Project Handbook on [Mildew Detector Wiki](https://github.com/JoelChan13/mildew-detector/wiki)**

### Mildew Detector in Cherry Leaves

- Farmy & Foods company's current approach at examining and identifying cherry leaves infected with powdery mildew is based on manual inspection, whereby farmers estimate to spend 30 minutes manually inspecting each tree. This lenghty process relies on the farmer's visual abilities and expertise to determine if the trees are healthy or infected, which at times may prove inaccurate or inconsistent since human subjectivity and human error cannot be ruled out.
- In order to address this need, we need to provide farmers with a reliable alternative to detect powdery mildew which is not reliant on manual inspection, which can translate into an ML model able to predict if a leaf is healthy or infected with powdery milder, based on an image database provided by Farmy & Foods company.
- The alternative can be provided using a supervised learning, multi-class, single-label, classification model, whereby a successful model provided an accuracy of 87% or above on the test set.
- Following the upload of a cherry leaf image through the mildew detector, the model output flags the leaf through its respective category, which is either healthy or infected.
- Training data to fit the model was provided by Farmy & Foody company and uploaded on Kaggle in an image dataset containing 4208 images of cherry leaves.

![MildewDetector](https://github.com/JoelChan13/mildew-detector/blob/main/readme_images/mildew_detector.png)

## Dashboard Design

### Page 1: Quick Project Summary

- Quick Project Summary
    - General Information:
        - Cherry powdery mildew is a fungal disease caused by the pathogen Podosphaera clandestina, which primarily affects cherry trees. This disease thrives in warm, dry conditions and can spread rapidly, especially in humid environments. It typically infects the young leaves, shoots, and fruit of cherry trees, resulting in reduced fruit quality and yield. The mildew fungus lives on the surface of plant tissues and feeds on them by sending tiny filaments into the cells.
        - Visual criteria used to detect infected leaves are Porraceous green lesions on either leaf surface which progresses to white or grayish powdery spots which develop in the infected area on leaves and fruits."
- Project Dataset
The available dataset contains 2104 healthy leaves and 2104 affected leaves individually photographed against a neutral background.
- Business Requirements:
    1. A comprehensive study to visually differentiate between healthy cherry leaves and those infected with powdery mildew.
    2. Implement a predictive model capable of identifying whether a cherry leaf is healthy or contains powdery mildew based on image analysis, ideally with an accuracy target of not less than 97%.
    3. Develop an interactive dashboard that allows users to upload cherry leaf images, receive predictions, and review the analysis results. 
- Link to project Readme.md file.

### Page 2: Cherry Leaves Visualiser

This page fulfills the Business Requirement 1 by providing a comprehensive study to visually differentiate between healthy cherry leaves and those infected with powdery mildew.
- Checkbox 1 - Show the difference between average and variability images
- Checkbox 2 - Show the difference between average images of infected and healthy leaves
- Checkbox 3 - Image Montage
- Link to project Readme.md file. 

### Page 3: Powdery Mildew Detector

- A link to download a dataset of infected and healthy leaves for testing for live prediction which can be found on [Kaggle](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves)
- File uploader widget for users to upload multiple JPEG images; this will display the image, a visual representation of the prediction, the prediction statement and its probability.
- Analysis report in table format and download button to produce it in csv format.
- Link to project Readme.md file. 
  
### Page 4: Project Hypothesis and Validation

- Explanation for project hypothesis including statement, explanation, validation and conclusion.
- Link to project Readme.md file. 

### Page 5: ML Performance Metrics

- Image for labels distribution across train, validation, and test sets
- Image for overall sets distribution
- Image for classification report
- Image for model performance - ROC curve
- Image for model accuracy - Confusion matrix
- Image for model performance - Accuracy & Losses of LSTM Model
- Model Evaluation Result on Test Set

## CRISP-DM Process

- CRISP-DM, which stands for Cross-Industry Standard Process for Data Mining, is a widely used methodology for data mining and data science projects. It provides a structured approach to planning and executing a data analysis project and consists of six phases:

   1. Business Understanding: This initial phase focuses on understanding the project objectives and requirements from a business perspective. It involves defining the problem to be solved, setting project goals, and determining the success criteria. A deep understanding of the business context and objectives helps in designing a relevant data analysis strategy.
   2. Data Understanding: In this phase, data is collected and an initial exploration is performed to understand its characteristics. This involves activities such as data collection, data description, data quality assessment, and identifying any data issues like missing values or outliers. The goal is to become familiar with the data and identify potential data quality problems.
   3. Data Preparation: This phase involves preparing the data for modeling. It includes tasks such as cleaning the data, handling missing values, transforming variables, selecting relevant features, and creating new variables if necessary. Data preparation is crucial because the quality of the data directly impacts the performance of the models.
   4. Modeling: In the modeling phase, various modeling techniques are selected and applied to the prepared data to create predictive models. This involves selecting the appropriate algorithms, setting parameters, and training the models. Often, multiple models are built and evaluated to determine which one performs best based on the project's objectives.
   5. Evaluation: After building the models, the evaluation phase assesses the quality and validity of the models to ensure they meet the business objectives. It involves evaluating model performance using appropriate metrics, validating model assumptions, and considering whether the models effectively solve the business problem. If necessary, further refinement or additional data preparation might be performed.
   6. Deployment: The final phase involves deploying the model into the production environment where it will be used to make real-world decisions. This could mean generating reports, integrating the model into existing systems, or simply presenting the results to stakeholders. Deployment also includes setting up monitoring and maintenance procedures to ensure the model continues to perform well over time.

**Source**: [Development Methodologies for Big Data Analytics Systems](https://www.google.com.mt/books/edition/Development_Methodologies_for_Big_Data_A/h0jhEAAAQBAJ?hl=en&gbpv=0)

- The CRISP-DM process is split up into sprints, and further divided into epics, which were documented using a Kanban Board provided by GitHub [@JoelChan13's Mildew Detector in Cherry Leaves Project](https://github.com/users/JoelChan13/projects/8/views/1). 
- Kanban boards are important in CRISP-DM processes because they help manage and visualise the workflow of data mining and data science projects. Here’s how Kanban boards contribute to the efficiency and effectiveness of CRISP-DM:
   1. Visualisation of Tasks and Progress: Kanban boards provide a clear, visual representation of tasks across different stages of the CRISP-DM process. By using columns to represent phases such as "To Do," "In Progress", and "Done," teams can see at a glance where each task is in the workflow. This helps in tracking progress and identifying any bottlenecks or areas that need attention.
   2. Improved Workflow Management: CRISP-DM is an iterative process, meaning that data science teams often need to revisit earlier phases based on findings from later stages. Kanban boards support this flexibility by allowing easy movement of tasks back and forth between columns. This ensures that all necessary steps are completed and that any needed revisions or refinements are tracked and managed efficiently.
   3. Enhanced Communication and Collaboration: Kanban boards facilitate better communication among team members by making the status of tasks visible to everyone involved. This transparency ensures that all team members are on the same page regarding project progress, priorities, and responsibilities. It also encourages collaboration, as team members can quickly see where their input or assistance is needed.
   4. Focus on Continuous Improvement: One of the key principles of Kanban is continuous improvement. By using a Kanban board, teams can regularly review the flow of work, identify inefficiencies or delays, and make adjustments to improve the process. This aligns well with the iterative nature of CRISP-DM, where learning from each phase can lead to enhancements in how subsequent phases are executed.
   5. Adaptability to Changing Requirements: Data science projects often involve uncertainty and changing requirements as new insights are gained. Kanban boards are flexible and can easily adapt to these changes. New tasks can be added, priorities can be shifted, and tasks can be reassigned without disrupting the overall workflow, making them ideal for managing the dynamic environment of CRISP-DM projects.

![KanbanBoard](https://github.com/JoelChan13/mildew-detector/blob/main/readme_images/kanban_board.png)

**Source**: [Kanban: Successful Evolutionary Change for Your Technology Business](https://www.google.com.mt/books/edition/Kanban/RJ0VUkfUWZkC?hl=en)

## Bugs

### Unfixed Bug

- Erratic image predictions were detected on certain occasions, whereby certain shadows and backgrounds ended up misleading the model into erratically classifying certain images as healthy or infected. In order to resolve this issue, the image normalisation process could me retuned in order to ensure that shadows, glares and backgrounds would be taken into considerations, and countered accordingly.

## Deployment

### Heroku

- The App live link is: `https://p5-mildew-detector-13512f4dba8f.herokuapp.com/`
- Set the runtime.txt Python version to a  stack currently supported version.
- The project was deployed to Heroku using the following steps:

1. Create a `requirement.txt` file in GitHub, for Heroku to read, listing the dependencies the program needs in order to run.
2. Set the `runtime.txt` Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
3. Ensure that the recent changes to GitHub have been pushed
4. Access your [Heroku account page](https://id.heroku.com/login) and create and deploy the app running the project. 
5. Click on "CREATE NEW APP", give the app a unique name, and select an ideal geographical region. 
6. Add `heroku/python` buildpack from the _Settings_ tab.
7. From the _Deploy_ tab, choose GitHub as deployment method, and connect to GitHub by selecting the project's repository. 
8. Select branch you want to deploy and click Deploy Branch.
9. Choose to "Deploy Branch" from the _Manual Deploy_ section. 
10. Wait for the logs to run while the dependencies are installed and the app is being built.
11. The mock terminal is then ready and accessible from a link similar to `https://your-projects-name.herokuapp.com/`
12. If the slug size is too large then add large files not required for the app to the `.slugignore` file.

### Forking the Repository

- By forking this GitHub repository, you create a duplicate of the original repository on your GitHub account, allowing you to view or modify it without impacting the original repository. The steps to fork the repository are as follows:

1. Locate the [GitHub Repository](https://github.com/JoelChan13/mildew-detector) of this project and log into your GitHub account.
2. Click on the "Fork" button, on the top right of the page.
3. Choose a destination where to fork the repository.

### Making a local clone

- Cloning a repository downloads a complete copy of all the data in the repository from GitHub.com at that specific time, including every version of each file and folder in the project. The steps to clone a repository are as follows:

1. Locate the [GitHub Repository](https://github.com/JoelChan13/mildew-detector) of this project and log into your GitHub account.
2. Click on the "Code" button.
3. Choose one of the available options: Clone with HTTPS, Open with Git Hub desktop, Download ZIP.

## Technologies Used

### Platforms

- [Heroku](https://en.wikipedia.org/wiki/Heroku) To deploy project
- [Jupiter Notebook](https://jupyter.org/) Edit project code
- [Kaggle](https://www.kaggle.com/) Download project datasets
- [GitHub](https://github.com/): Store project code after pushing from GitPod.
- [Gitpod](https://www.gitpod.io/) Used to write project code and push code to GitHub.

### Languages

- [Python](https://www.python.org/)
- [Markdown](https://www.markdownguide.org/getting-started/)
  
### Main Data Analysis and Machine Learning Libraries

<pre>
- tensorflow-cpu 2.6.0  used to create the model
- numpy 1.19.2          used to convert to array 
- scikit-learn 0.24.2   used to evaluate the model
- streamlit 0.85.0      used to create the dashboard
- pandas 1.1.2          used to create/save as dataframe
- matplotlib 3.3.1      used to plot the sets' distribution
- keras 2.6.0           used to set model hyperparameters
- plotly 4.12.0         used to plot model's learning curve 
- seaborn 0.11.0        used to plot model's confusion matrix
- protobuf==3.20        used to used to encode data into a compact binary format
</pre>


## Credits

### Content

- The leaves dataset was linked from [Kaggle](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves) and created by [Code Institute](https://www.kaggle.com/codeinstitute)
- The powdery mildew description was taken from [garden design](https://www.gardendesign.com/how-to/powdery-mildew.html) and [almanac](https://www.almanac.com/pest/powdery-mildew)
- The [CRISP DM](https://www.datascience-pm.com/crisp-dm-2/) steps adopted in the [GitHub project](https://github.com/cla-cif/Cherry-Powdery-Mildew-Detector/projects?query=is%3Aopen) were modeled on [Introduction to CRISP-DM](https://www.ibm.com/docs/en/spss-modeler/saas?topic=guide-introduction-crisp-dm) articles from IBM.

### Media

- The logo image was generated using [DALL·E 3](https://openai.com/index/dall-e-3/)

### Code

-  The template used for this project, along with jupyter notebooks for data collection and data visualisation were obtained from [Code Institute Walkthrough Project 1 - Malaria Detector](https://github.com/Code-Institute-Solutions/WalkthroughProject01).
- The cherry powdery milew detector project compiled by Claudia Cifaldi was also used as a reference when producing the mildew detecor project [GitHub - Claudia Cifaldi](https://github.com/cla-cif/Cherry-Powdery-Mildew-Detector/tree/main).

### Acknowledgements

- Thanks to the Student Care & Tutor Assistance teams at [Code Institute](https://codeinstitute.net/global/) for their occasional inputs whenever I encountered any issues which I was unable to solve on my own.
- I would also like to thank my mentor, Mr. Mo Shami, for his straight-forward approach, for encouraging me to challenge myself, and for sharing his knowledge in the field.

### Deployed version at [Mildew Detector](https://p5-mildew-detector-13512f4dba8f.herokuapp.com/)