import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from unittest.mock import patch
# import the necessary modules from your project
import sys
sys.path.append('c:/Users/Danie/Desktop/MachineMax Tech_Test/Codebase')
from statistical_models import LinearRegressionPredictor

class TestLinearRegressionPredictor(unittest.TestCase):

    def setUp(self):
        self.model = LinearRegression()
        self.predictor = LinearRegressionPredictor(self.model)

    def test_calculate_slope(self):
        '''
        This test will pass if the slope is calculated correctly
        '''
        series = pd.Series([1, 2, 3, 4, 5])
        expected_slope = 1.0
        result_slope = self.predictor.calculate_slope(series)
        self.assertEqual(result_slope, expected_slope)

    def test_compute_metrics(self):
        '''
        This test will pass if the metrics are computed correctly
        '''
        y_tests = np.array([0, 0, 1, 1])
        y_preds = np.array([0, 1, 1, 1])
        metrics = self.predictor.compute_metrics(y_tests, y_preds)
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('confusion_matrix', metrics)
        self.assertIn('classification_report', metrics)
        
    @patch('matplotlib.pyplot.show')
    def test_plot_roc_auc(self, plt_mock):
        '''
        This test will pass if the ROC AUC is plotted correctly
        '''
        y_tests = np.array([0, 0, 2, 1])
        y_preds = np.array([0, 2, 1, 1])
        auc = self.predictor.plot_roc_auc(y_tests, y_preds)
        self.assertEqual(len(auc), len(np.unique(y_tests)))

    @patch('matplotlib.pyplot.show')
    def test_plot_confusion_matrix(self, plt_mock):
        '''
        This test will pass if the confusion matrix is plotted correctly
        '''
        cm = np.array([[2, 1], [1, 2]])
        self.predictor.plot_confusion_matrix(cm)

    @patch('matplotlib.pyplot.show')
    def test_predict(self, plt_mock):
        '''
        This test will pass if the predict method return the correct metrics
        '''
        df = pd.DataFrame({
            'rms': [1.0, 1.1, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
            'state_encoded': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        })
        engine_state, metrics = self.predictor.predict(df)
        self.assertEqual(len(engine_state), len(df))
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('confusion_matrix', metrics)
        self.assertIn('roc_auc', metrics)
        self.assertIn('average_roc_auc', metrics)

if __name__ == "__main__":
    unittest.main()