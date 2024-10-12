# Building Classification using ML & Deep Learning

## Project Overview
This project aims to classify buildings into five distinct categories (A, B, C, D, and S) using a combination of Machine Learning and Deep Learning techniques. We have utilized traditional ML classifiers (SVM, Logistic Regression, Random Forest, XGBoost) alongside a Neural Network model to achieve robust classification. The project also involves feature extraction, PCA dimensionality reduction, and ensemble techniques to enhance performance.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Approach](#approach)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Observations and Findings](#observations-and-findings)
- [Future Work](#future-work)
- [Contributions](#contributions)
- [License](#license)

## Dataset
The dataset contains grayscale images of various buildings belonging to five categories (A, B, C, D, S). Each image has a size of (300, 400), with a total of 2466 images for training and testing. The dataset is split into training and test sets with the following distribution:
- A: 289 images
- B: 352 images
- C: 721 images
- D: 904 images
- S: 200 images

## Approach
Our approach includes the following key steps:
1. **Exploratory Data Analysis (EDA):** Initial analysis of the dataset, class distribution, and image properties.
2. **Feature Extraction:** Extracting features from images using pre-trained efficientNet model.
3. **Dimensionality Reduction:** Applying PCA to reduce feature dimensions while retaining 95% variance.
4. **Model Training:** Training classifiers including SVM, Logistic Regression, Random Forest, XGBoost, and a Neural Network.
5. **Model Ensembling:** Combining the best models using soft voting for improved accuracy.
6. **Evaluation:** Using a test set for final performance evaluation, including confusion matrices and classification reports.

## Installation
To run this project locally, follow the steps below:

### 1. Clone the repository:
```bash
git clone https://github.com/your-username/building-classification.git
cd building-classification
```

### 2. Install the dependencies:
Use the provided requirements.txt file to install all necessary packages:
```
pip install -r requirements.txt
```

### 3. Download the dataset:
You can find the dataset here: https://drive.google.com/drive/folders/1RqHqUA83mm-CdqWZxiJ_JrlthtF7EUkX?usp=sharing

### 4. Run the Jupyter Notebook:
You can view the complete analysis and training process in the provided Jupyter notebook:
```
jupyter notebook building-classification.ipynb
```
## Usage
### 1. Training Models:
The models are trained using PCA-reduced features. You can modify hyperparameters in the notebook or directly run the cells to train the models.

### 2. Making Predictions:
To predict the category of a building image, use the pre-trained models: (as shown at the end of the notebook)
```
# Load the voting classifier and predict on new images
predicted_label, individual_predictions = predict_image(image_path, pca, classifiers, transforms, device)
print(f"Predicted label: {predicted_label}")
```
### 3. Evaluation:
Evaluate the ensemble model using:
```
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
```

## Results 
(more results and visualizations are inside 'results' folder)
Our ensemble model achieved X% accuracy on the test dataset, with a detailed classification report provided below. The confusion matrix highlights the model's performance across all categories.

### Confusion Matrix:
![image](https://github.com/user-attachments/assets/a6592252-8a23-4b31-ae6b-5645a5b40715)

### Classification Report:
![image](https://github.com/user-attachments/assets/0c2d07b0-504a-44d3-b757-97a24b81707d)

## Observations and Findings
-	EfficientNet was the most suitable model for our small dataset, particularly with data augmentation and advanced loss functions.
-	Handling Imbalanced Data: Using Focal Loss and stratified sampling was crucial for dealing with imbalanced classes.
-	PCA allowed us to uncover the non-linear structure of the data, leading to significant performance improvements with non-linear classifiers.

## Future Work
- **Hyperparameter Tuning:** Further tuning of the Neural Network and other models could yield even better results.
- **Deploying the Model:** Building a web application or API to serve the trained models for real-time predictions.

## Contributions
Feel free to open issues or pull requests to contribute to the project. All contributions are welcome!

## License
This project is licensed under the MIT License.



