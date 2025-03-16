# 🏥 Cancer Prediction using Machine Learning

## 📌 Project Overview

This project is designed to predict whether a tumor is malignant or benign based on a dataset containing various medical features. Using machine learning algorithms like Support Vector Machine (SVM), Random Forest, and XGBoost, the model learns patterns from historical data and makes accurate predictions. The project also includes data preprocessing, exploratory data analysis (EDA), hyperparameter tuning, and model evaluation to ensure high performance and reliability.
This project aims to predict cancer diagnoses based on medical data using various machine learning techniques. The dataset used contains features extracted from breast cancer cell images, and the goal is to classify them as malignant or benign.

## ✨ Features of the Project

- 📊 Data preprocessing (handling missing values, encoding categorical data, feature selection)
- 🔍 Exploratory Data Analysis (EDA) with heatmaps, box plots, and distribution plots
- 🤖 Model training using multiple algorithms:
  - 🔹 Support Vector Machine (SVM)
  - 🌳 Random Forest Classifier
  - 🚀 XGBoost Classifier
- ⚙️ Hyperparameter tuning using GridSearchCV
- 📈 Model evaluation using accuracy, confusion matrix, and classification report
- ⚖️ Data balancing using SMOTE (Synthetic Minority Over-sampling Technique)
- 💾 Saving and loading models using Pickle

## 📦 Requirements

The following libraries are required to run the project:

```python
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
pickle
imblearn
```

## 📥 Installation

Use the following command to install the required dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn
```

## 📂 Dataset

The dataset is stored as `data.csv` and contains:

- 🆔 **ID**: Unique identifier
- 🔬 **Diagnosis**: Malignant (1) or Benign (0)
- 📊 **Feature Columns**: Various numerical measurements related to tumor characteristics

## 🚀 Usage

### 1️⃣ Load Dataset

```python
import pandas as pd
data = pd.read_csv('data.csv')
```

### 2️⃣ Data Preprocessing

- 🔄 Convert categorical variables to numerical using label encoding
- ❌ Check for null values and handle them if needed
- 📉 Perform feature selection based on importance ranking

### 3️⃣ Model Training

```python
from sklearn.model_selection import train_test_split
x = data.drop(['diagnosis', 'id'], axis=1)
y = data['diagnosis']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
```

#### 🌳 Train a Random Forest Model

```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=500, random_state=10)
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)
print(f'Accuracy: {accuracy}')
```

### 4️⃣ Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators': [100, 500, 1000]}
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=10, scoring='accuracy')
grid.fit(x_train, y_train)
print(grid.best_params_)
```

### 5️⃣ Save and Load Model

```python
import pickle
pickle.dump(clf, open('random_forest_model.pkl', 'wb'))
loaded_model = pickle.load(open('random_forest_model.pkl', 'rb'))
```

## 📊 Results

- 📈 Model evaluation metrics include accuracy, confusion matrix, and classification report.
- 🏆 The best performing model is selected based on the highest accuracy score.

## 🎯 Conclusion

This project successfully implements machine learning models to predict cancer diagnoses. The use of different classifiers and feature selection techniques enhances prediction accuracy. Future improvements can involve deep learning models and more advanced data preprocessing techniques.

## 👨‍💻 Author

Developed by **Sumit** 🚀

