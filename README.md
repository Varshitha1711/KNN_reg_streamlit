
## 2. California House Price Prediction (KNN Regressor)

```md
# California House Price Prediction using KNN Regressor

This project predicts California house prices using a K-Nearest Neighbors (KNN) Regressor with hyperparameter tuning and a Streamlit web application.

## Features

- California Housing Dataset
- Data preprocessing and scaling
- Hyperparameter tuning using GridSearchCV
- KNN Regression model
- Interactive Streamlit dashboard

## Project Structure

project/
│
├── data/
│ ├── housing.csv
│ └── cleaned_housing.csv
│
├── models/
│ ├── knn_model.pkl
│ └── scaler.pkl
│
├── app.py
├── eda.py
├── training.py
└── requirements.txt

## Installation

```bash
pip install -r requirements.txt
```
## Train the Model
python training.py

## Run the Application
streamlit run app.py

## Technologies Used
Python
Pandas
NumPy
Scikit-Learn
Streamlit

## Model
K-Nearest Neighbors Regressor
GridSearchCV Hyperparameter Tuning
StandardScaler