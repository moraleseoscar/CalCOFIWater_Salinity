

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import os


# Load the data
bottle = pd.read_csv('project/CLEAN/bottle_clean.csv')
bottle.head(2)

# Select columns of interest
bottle_df = bottle[['Salnty', 'T_degC', 'Depthm', 'O2ml_L', 'STheta', 'O2Sat']]

# Drop rows with missing values
bottle_df.dropna(inplace=True)

# Split the data into features and target
X = bottle_df[['T_degC', 'Depthm', 'O2ml_L', 'STheta', 'O2Sat']]
y = bottle_df['Salnty']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with scaling and the MLPRegressor
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42))
])

# Establishing a grid of hyperparameters to test
options_grid = {
    'hidden_layer_sizes': [(100, 50), (50, 50), (50, 100, 50)],
    'activation': ['tanh', 'relu'],
    'alpha': [0.0001, 0.05, 0.000001]
}

# Perform a grid search with the options in options_grid
grid_search = GridSearchCV(pipeline, options_grid, cv=5, n_jobs=-1)

# Fit the grid search
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)

