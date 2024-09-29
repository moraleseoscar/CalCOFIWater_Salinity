
import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import flask


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
    'mlp__hidden_layer_sizes': [(100, 50), (50, 50), (50, 100, 50)],
    'mlp__activation': ['tanh', 'relu'],
    'mlp__alpha': [0.0001, 0.05, 0.000001]
}

# Perform a grid search with the options in options_grid
grid_search = GridSearchCV(pipeline, options_grid, cv=5, n_jobs=-1)

# Fit the grid search
grid_search.fit(X_train, y_train)

# Print the best parameters
print(grid_search.best_params_)

# Predict on the test set
y_pred = grid_search.predict(X_test)

# Save the predicted values to a new df
results = pd.DataFrame({
		'Actual': y_test,
		'Predicted': y_pred
})
# Save the results to a csv file
results.to_csv('project/output/predictions.csv', index=False)

# Calculate the mean squared error
new_mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {new_mse}')

# Calculate the R-squared value
new_r2 = r2_score(y_test, y_pred)
print(f'R-squared: {new_r2}')

# Save the model
import joblib
joblib.dump(grid_search, 'project/output/model.pkl')

# Expose the model as a REST API
app = flask.Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
		data = flask.request.json
		prediction = grid_search.predict(data)
		return flask.jsonify({'prediction': list(prediction)})