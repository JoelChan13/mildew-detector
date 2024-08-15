# ![CI logo](https://codeinstitute.s3.amazonaws.com/fullstack/ci_logo_small.png)

## Table of Contents
1. [Dataset Content](#dataset-content)
2. [Business Requirements](#business-requirements)
3. [Hypothesis and validation](#hypothesis-and-validation)
4. [Rationale for the model](#the-rationale-for-the-model)
5. [Trial and error](#trial-and-error)
6. [Implementation of the Business Requirements](#the-rationale-to-map-the-business-requirements-to-the-data-visualizations-and-ml-tasks)
7. [ML Business case](#ml-business-case)
8. [Dashboard design](#dashboard-design-streamlit-app-user-interface)
9. [CRISP DM Process](#the-process-of-cross-industry-standard-process-for-data-mining)
10. [Bugs](#bugs)
11. [Deployment](#deployment)
12. [Technologies used](#technologies-used)
13. [Credits](#credits)

### Deployed version at [cherry-powdery-mildew-detector.herokuapp.com](https://cherry-powdery-mildew-detector.herokuapp.com/)

## Dataset Content

The dataset contains 4208 featured photos of single cherry leaves against a neutral background. The images of the cherry tree leaves were obtained from Farmy & Foods' crop fields and feature images of leaves that are either healthy or infested by powdery mildew. Powdery mildew is a fungal disease that affects a wide range of plants and typically appears as white or grayish powdery spots on the leaves, stems, and sometimes fruits of the infected plants, mainly caused by various species of fungi and typically targets particular types of plants. Notwithstanding the fact that this disease affects many plant species, the client is particularly concerned about their cherry plantation crop is one of their finest products in the portfolio.

Dataset Source From [Kaggle](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves).

## Business Requirements

The primary objective of this project is to develop a Machine Learning system that aids Farmy & Foods in addressing the issue of powdery mildew affecting their cherry plantations. Currently, the process is manual verification if a given cherry tree contains powdery mildew. An employee spends around 30 minutes in each tree, taking a few samples of tree leaves and verifying visually if the leaf tree is healthy or has powdery mildew. If there is powdery mildew, the employee applies a specific compound to kill the fungus. The time spent applying this compound is 1 minute. Due to this fact, manually inspecting all plants is not scalable, the client has outlined the following specific business requirements:

1. Visual Differentiation: Conduct a comprehensive study to visually differentiate between healthy cherry leaves and those infected with powdery mildew.
2. Predictive Modeling: Implement a predictive model capable of identifying whether a cherry leaf is healthy or contains powdery mildew based on image analysis, with an accuracy target of 97%.
3. Interactive Dashboard: Develop an interactive dashboard that allows users to upload cherry leaf images, receive predictions, and review the analysis results.

By meeting these requirements, the project will enable Farmy & Foods to maintain product quality and meet market demands effectively through large-scale automated detection.

## Hypothesis and how to validate?

1. **Hypothesis 1**: Visual Differentiation of Cherry Leaves - There are distinct visual differences between healthy cherry leaves and those affected by powdery mildew that can be identified through image analysis.
   - __How to validate__: This hypothesis can be tested by conducting a study that involves analyzing average images, variability images, and creating image montages for both healthy and powdery mildew-affected leaves. If the differences are visually apparent and consistent, this would support the hypothesis. The analysis will include identifying specific visual markers or patterns that distinguish healthy leaves from those affected by powdery mildew.<br/>

2. **Hypothesis 2**: Predictive Capability of Machine Learning Model - A machine learning model (e.g., Convolutional Neural Network) can be trained to accurately predict whether a cherry leaf is healthy or affected by powdery mildew, with a target accuracy of at least 97%.
   - __How to validate__: This hypothesis will be validated by training a machine learning model on the cherry leaf images and then evaluating its performance on a separate test set. The model's accuracy, precision, recall, and F1-score will be used as metrics to determine its effectiveness. If the model achieves an accuracy of 97% or higher on the test set, the hypothesis will be considered validated. <br/>

3. **Hypothesis 3**: Generalization Across Cherry Plantations - The trained machine learning model can generalize well across cherry leaves from different farms and plantations, detecting powdery mildew consistently despite variations in environmental conditions.  
   - __How to validate__: This can be tested by ensuring the training dataset includes diverse samples from various farms, and then evaluating the model's performance across these different sources. If the model maintains high accuracy and performs consistently across different test samples, this hypothesis will be supported.

## The rationale to map the business requirements to the Data Visualisations and ML tasks

- List your business requirements and a rationale to map them to the Data Visualisations and ML tasks.

## ML Business Case

- In the previous bullet, you potentially visualised an ML task to answer a business requirement. You should frame the business case using the method we covered in the course.

## Dashboard Design

- List all dashboard pages and their content, either blocks of information or widgets, like buttons, checkboxes, images, or any other items, that your dashboard library supports.
- Finally, during the project development, you may revisit your dashboard plan to update a given feature (for example, at the beginning of the project, you were confident you would use a given plot to display an insight, but later, you chose another plot type).

## Unfixed Bugs

- You will need to mention unfixed bugs and why they were unfixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable for consideration, paucity of time and difficulty understanding implementation is not a valid reason to leave bugs unfixed.

## Deployment

### Heroku

- The App live link is: `https://YOUR_APP_NAME.herokuapp.com/`
- Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
- The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large, then add large files not required for the app to the .slugignore file.

## Main Data Analysis and Machine Learning Libraries

- Here, you should list the libraries used in the project and provide an example(s) of how you used these libraries.

## Credits

- In this section, you need to reference where you got your content, media and from where you got extra help. It is common practice to use code from other repositories and tutorials. However, it is necessary to be very specific about these sources to avoid plagiarism.
- You can break the credits section up into Content and Media, depending on what you have included in your project.

### Content

- The text for the Home page was taken from Wikipedia Article A.
- Instructions on how to implement form validation on the Sign-Up page were taken from [Specific YouTube Tutorial](https://www.youtube.com/).
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/).

### Media

- The photos used on the home and sign-up page are from This Open-Source site.
- The images used for the gallery page were taken from this other open-source site.

## Acknowledgements (optional)

- Thank the people who provided support throughout this project.
