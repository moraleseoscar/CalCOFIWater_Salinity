
import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import flask
import joblib

model_path = 'project/output/model.pkl'
# Load the model
pipeline = joblib.load(model_path)
# Load the data
bottle = pd.read_csv('project/CLEAN/bottle_clean.csv')
# Select columns of interest
bottle_df = bottle[['Salnty', 'T_degC', 'Depthm', 'O2ml_L', 'STheta', 'O2Sat']]
# Drop rows with missing values
bottle_df.dropna(inplace=True)
# Split the data into features and target
X = bottle_df[['T_degC', 'Depthm', 'O2ml_L', 'STheta', 'O2Sat']]

# Predict first row of data
X_predict = X.head(1)
y_predict = pipeline.predict(X_predict)

print(y_predict)