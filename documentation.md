# HEART DISEASE PREDICTION USING MACHINE LEARNING

## Final Project Documentation

**Project Name:** Heart Disease Prediction Using Machine Learning  
**Programming Language:** Python  
**Machine Learning Framework:** Scikit-learn  
**Web Application Framework:** Streamlit  
**Dataset:** Heart Disease Dataset (`heart.csv`)  
**Source Code:** GitHub  
**Project Date:** 26 November 2026

---

## 1. Introduction

Heart disease is one of the major health concerns worldwide. Early identification of potential heart disease risk can help individuals seek appropriate medical evaluation.

The objective of this project is to develop a machine learning-based prediction system that analyzes patient health information and predicts the possibility of heart disease based on the provided dataset.

The project uses Python and machine learning techniques to develop a classification model and integrates the model into an interactive Streamlit web application.

The application allows users to enter patient health information and view prediction results, probabilities, and explanations through a simple interface.

This project is developed for educational and experimental purposes. Its predictions are not a replacement for professional medical diagnosis.

## 2. Project Objectives

The main objectives of this project are:

1. Understand and analyze the heart disease dataset.
2. Perform data cleaning and preprocessing.
3. Develop a machine learning classification model.
4. Evaluate the model using classification metrics.
5. Create an interactive Streamlit application.
6. Implement dynamic input handling and validation.
7. Display prediction results and confidence information.
8. Provide understandable explanations of model predictions.
9. Improve the application's user interface and usability.
10. Document implementation challenges and possible future improvements.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Dataset loading and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning model and preprocessing |
| Matplotlib | Data visualization |
| Streamlit | Interactive web application |
| HTML and CSS | Custom application styling |
| GitHub | Source code management |

## 4. Dataset Description

The project uses a Heart Disease dataset stored in a CSV file named `heart.csv`.

The dataset contains health-related attributes used to predict the presence or absence of heart disease.

The features include:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Resting electrocardiographic results
- Maximum heart rate
- Exercise-induced angina
- ST depression
- Slope
- Number of major vessels
- Thalassemia category

The target column represents the heart disease classification.

The dataset is loaded using Pandas and examined to understand its structure, data types, missing values, and duplicate records.

## 5. Data Preprocessing

Data preprocessing is an important step in preparing the dataset for machine learning.

The following operations are included in the project:

1. Load the dataset using Pandas.
2. Inspect the dataset structure and feature columns.
3. Remove unnecessary index columns.
4. Identify and remove duplicate records.
5. Separate input features and the target variable.
6. Handle missing values using median imputation.
7. Convert numerical values into compatible formats.
8. Prepare the dataset for model training and prediction.

These steps help maintain consistent input data and improve the reliability of the prediction workflow.

## 6. Machine Learning Model

The current Streamlit application uses a Random Forest Classifier.

Random Forest is an ensemble machine learning algorithm that combines predictions from multiple decision trees to perform classification.

The configuration used in the current application is:

| Parameter | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Number of estimators | 150 |
| Maximum depth | 8 |
| Random state | 42 |
| Class weighting | Balanced |

The model is trained using the available heart disease dataset.

When users submit patient information, the application processes the input and passes it to the trained model to generate a prediction.

**Model implementation note:** The current application trains the model from the CSV dataset when the application loads. A separately saved model artifact has not yet been integrated into a deployment API.

## 7. Model Evaluation

Machine learning classification performance can be evaluated using the following metrics:

### 7.1 Accuracy

Accuracy measures the percentage of correctly classified observations.

### 7.2 Precision

Precision measures the proportion of positive predictions that are actually positive.

### 7.3 Recall

Recall measures the proportion of actual positive observations correctly identified by the model.

### 7.4 F1-Score

The F1-score combines precision and recall into a single metric.

### 7.5 Confusion Matrix

A confusion matrix presents the classification results using:

- True Positive
- True Negative
- False Positive
- False Negative

These metrics are useful for understanding the model's classification performance.

**Experimental results:** Numerical evaluation results should be added from the actual model evaluation output. No unverified performance values are reported in this documentation.

## 8. Streamlit Application Development

An interactive web application was developed using Streamlit to make the machine learning model accessible through a user-friendly interface.

The application allows users to enter patient health information and receive a prediction.

### 8.1 User Input Handling

The application provides input controls for the required patient features.

The implementation includes:

- Numerical input fields.
- Male and Female selection for the sex field.
- Support for decimal values where applicable.
- Dynamic input handling.
- Validation of user-provided values.

### 8.2 Input Validation and Error Handling

Input validation helps prevent incorrect values from reaching the machine learning model.

The application checks user inputs and provides error messages when invalid values are entered.

It also handles missing or unsuitable inputs and guides users toward acceptable values.

### 8.3 Prediction Display

After the user submits the input form, the application displays the model's prediction.

The result indicates whether the input is classified into the heart disease positive or negative category.

### 8.4 Prediction Confidence

The application displays class probabilities and confidence information to help users understand the model output.

A bar chart is used to present the probability distribution visually.

### 8.5 Prediction Explanation

The application provides a human-readable explanation of the prediction.

It also includes feature-effect information to help users understand how changes in input features relate to the model's output.

These explanations represent model behavior and should not be interpreted as medical conclusions.

### 8.6 User Interface Design

Custom CSS was implemented to improve the appearance and usability of the application.

The interface includes:

- Custom styling.
- Responsive layout.
- Improved input presentation.
- Clear prediction result display.
- Readable colors and text.
- Consistent spacing and visual organization.

### 8.7 Application Testing

The Streamlit application was successfully launched and tested.

The application loads the dataset, accepts patient information, and generates prediction results through its interface.

## 9. Application Deployment

The Streamlit application was deployed using Streamlit Community Cloud.

The project's source code and dataset are maintained in GitHub.

**GitHub Repository:**

https://github.com/KrishPatel-1384/Heart_Disease

The deployed application was tested to confirm that the interface loads and the prediction functionality works.

The application uses the dataset stored alongside the project files in the repository.

### Deployment Status

| Component | Status |
|---|---|
| GitHub repository | Available |
| Dataset | Available in repository |
| Streamlit application | Deployed |
| Prediction functionality | Working |
| FastAPI service | Not implemented |
| AWS deployment | Not completed |
| Docker containerization | Not completed |

## 10. Implementation Challenges and Solutions

During development, several technical challenges were encountered.

| Challenge | Solution |
|---|---|
| Dataset not found | Updated the application to locate the CSV file relative to the application directory |
| Missing Python packages | Corrected the dependency installation configuration |
| Incorrect numerical data types | Converted prediction inputs into compatible numerical formats |
| Decimal input errors | Updated input handling to support floating-point values |
| Sex field displaying numerical values | Added Male and Female selection options |
| Invalid user inputs | Implemented validation and error messages |
| Prediction result disappearing | Used Streamlit session state to retain prediction results |
| Application styling issues | Improved the interface using custom CSS |

These solutions helped improve the application's functionality and usability.

## 11. API Development and Performance Optimization: Future Work

A separate FastAPI prediction service has not yet been created.

The following activities are planned for a future development stage:

1. Develop a REST API using FastAPI.
2. Create health-check and prediction endpoints.
3. Implement API-key authentication.
4. Add request validation and error handling.
5. Measure request-response latency.
6. Identify processing bottlenecks.
7. Implement prediction caching.
8. Investigate model optimization and quantization where supported.
9. Perform concurrent-request load testing.

Since the API has not been implemented, no API latency or optimization results are claimed in this report.

## 12. AWS Deployment: Future Work

The application has not been deployed to AWS.

AWS deployment is planned as a future improvement to make the prediction service accessible through cloud infrastructure.

The proposed deployment components are:

- Amazon ECR for storing Docker images.
- Amazon ECS with Fargate for running application containers.
- Application Load Balancer for handling incoming requests.
- IAM for access management.
- Amazon CloudWatch for application monitoring.
- HTTPS for secure communication.

These services have not yet been configured or deployed for this project.

## 13. Limitations

The current project has several limitations:

1. Prediction quality depends on the dataset used for training.
2. The model may not generalize to all patient populations.
3. The model's predictions are not medical diagnoses.
4. The application currently trains the model when it loads.
5. A separately deployed REST API is not yet available.
6. AWS cloud deployment and scalability have not been tested.
7. API performance optimization has not yet been performed.
8. Further validation using larger and more diverse datasets would be beneficial.

## 14. Future Improvements

The following improvements can be implemented in future versions:

- Develop and integrate a FastAPI prediction service.
- Save and version the trained machine learning model.
- Deploy the API using AWS services.
- Implement authentication and secure communication.
- Profile API request-response latency.
- Introduce caching for repeated predictions.
- Conduct load testing with concurrent requests.
- Investigate model quantization and other optimization techniques.
- Improve model interpretability using SHAP.
- Evaluate additional machine learning algorithms.
- Implement automated CI/CD workflows.
- Monitor data quality and model performance over time.

## 15. Conclusion

The Heart Disease Prediction Using Machine Learning project demonstrates the practical application of machine learning techniques in health-related data classification.

The project includes dataset preprocessing, Random Forest model development, prediction generation, and integration with an interactive Streamlit application.

The completed application provides dynamic patient input handling, validation, prediction results, class probabilities, confidence information, and human-readable explanations.

The application has been successfully deployed using Streamlit Community Cloud and its prediction functionality has been tested.

FastAPI development, AWS deployment, and API performance optimization remain future development activities.

Overall, the project provides practical experience in Python programming, machine learning, data preprocessing, application development, model integration, and deployment.

---

**End of Project Documentation**
