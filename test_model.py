# test_model.py
import unittest
import pandas as pd
import joblib
import numpy as np

class TestModelPrediction(unittest.TestCase):
    def test_prediction_accuracy(self):
        # Load the model
        model = joblib.load('project/output/model.pkl')
        
        # Load sample data
        bottle = pd.read_csv('project/CLEAN/bottle_clean.csv')
        bottle_df = bottle[['Salnty', 'T_degC', 'Depthm', 'O2ml_L', 'STheta', 'O2Sat']].dropna()
        X_test = bottle_df[['T_degC', 'Depthm', 'O2ml_L', 'STheta', 'O2Sat']].head(1)
        y_true = bottle_df['Salnty'].head(1).values

        # Make prediction
        y_pred = model.predict(X_test)
        # Check if the prediction is close to the true value
        np.testing.assert_allclose(y_pred, y_true, rtol=0.1)

if __name__ == '__main__':
    unittest.main()