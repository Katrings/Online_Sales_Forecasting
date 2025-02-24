# Online_sales_forecasting using XGBoost

## Project Overview:
This project aims to develop a predictive model for **Total Revenue** from online sales using machine learning techniques. The model predicts revenue based on input features such as `Units Sold`, `Unit Price`, and `Product Category`. The **XGBoost Regressor** is employed for this regression task, and model optimization is performed using **GridSearchCV** for hyperparameter tuning.

## Project Objective:
The primary objective of this project is to predict total revenue generated from online sales transactions based on several features. We leverage **XGBoost**, a powerful gradient boosting model, to provide accurate revenue predictions. The model's performance is evaluated using **Mean Squared Error (MSE)**.

## Installation Requirements:
To execute the code and run the project, the following Python libraries must be installed:
- **pandas**: For data manipulation and analysis.
- **scikit-learn**: For machine learning models and utilities.
- **xgboost**: For building the **XGBoost** regression model.
- **joblib**: For saving and loading the trained model.

You can install these dependencies by running the following command:

pip install pandas scikit-learn xgboost joblib

## How to Run:
1. Clone or download the repository to your local machine.
2. Open the **Jupyter notebook** `online_sales_forecasting_xgboost.ipynb`.
3. Execute the notebook cells in order to:
   - Load the dataset
   - Preprocess the data (handle missing values and encode categorical variables)
   - Train the model using **XGBoost**
   - Tune the hyperparameters with **GridSearchCV**
   - Evaluate the model’s performance and save the trained model.

4. To use the trained model for making future predictions, follow these steps:

```python
import joblib

# Load the trained XGBoost model
model = joblib.load('best_xgb_model.pkl')

# Predict total revenue for new input data
new_data = [[100, 20]]  # Example: Units Sold = 100, Unit Price = 20
prediction = model.predict(new_data)
print("Predicted Revenue:", prediction)
Project Structure:
The project is organized as follows:
online_sales_forecasting_xgboost.ipynb  # Jupyter notebook with the code for data preprocessing, model training, and evaluation
best_xgb_model.pkl                       # Saved XGBoost model for future use
README.md                               # Project documentation with instructions

##Model Evaluation:
The XGBoost model, after hyperparameter tuning, achieved excellent results. The Mean Squared Error (MSE) for the best model was found to be 720.32, indicating the model’s accuracy in predicting total revenue.

Best Model Hyperparameters:

learning_rate: 0.2
max_depth: 3
n_estimators: 300
Model Performance (MSE): 720.32

## Next Steps:
Incorporate additional features such as Product Category, Payment Method, and other potentially relevant factors to improve model accuracy.
Test alternative regression models, such as LightGBM or Support Vector Machines (SVM), for comparison.
Consider deploying the model to make real-time predictions for new sales data.

## Conclusion:
This project demonstrates how XGBoost can be leveraged to predict Total Revenue from online sales data. Through proper hyperparameter optimization with GridSearchCV, the model's performance has been significantly improved. The resulting model is capable of making accurate revenue predictions, and its further refinement can be achieved with additional features and experimentation with other algorithms.

