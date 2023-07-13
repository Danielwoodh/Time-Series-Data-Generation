import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import sys
sys.path.append('c:/Users/Danie/Desktop/MachineMax Tech_Test/Codebase')
from statistical_models import LinearRegressionPredictor

def test_calculate_slope():
    model = LinearRegression()
    predictor = LinearRegressionPredictor(model)
    series = pd.Series([1, 2, 3, 4, 5])
    slope = predictor.calculate_slope(series)
    assert np.isclose(slope, 1.0)

def test_compute_metrics():
    model = LinearRegression()
    predictor = LinearRegressionPredictor(model)
    y_tests = np.array([0, 1, 2, 0, 1, 2])
    y_preds = np.array([0, 1, 2, 1, 0, 2])
    metrics = predictor.compute_metrics(y_tests, y_preds)
    assert np.isclose(metrics['accuracy'], 0.5)
    assert np.isclose(metrics['precision'], 0.3333333333333333)
    assert np.isclose(metrics['recall'], 0.3333333333333333)
    assert np.isclose(metrics['f1_score'], 0.3333333333333333)
    assert np.array_equal(metrics['confusion_matrix'], np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]]))

def test_predict():
    model = LinearRegression()
    predictor = LinearRegressionPredictor(model)
    df = pd.DataFrame({'rms': [1, 2, 3, 4, 5], 'state_encoded': [0, 1, 2, 0, 1]})
    engine_state, metrics = predictor.predict(df)
    assert np.array_equal(engine_state, np.array([1, 1, 1, 1, 1]))
    assert np.isclose(metrics['accuracy'], 0.4)
    assert np.isclose(metrics['precision'], 0.3333333333333333)
    assert np.isclose(metrics['recall'], 0.3333333333333333)
    assert np.isclose(metrics['f1_score'], 0.3333333333333333)
    assert np.array_equal(metrics['confusion_matrix'], np.array([[0, 2, 0], [0, 1, 1], [0, 1, 0]]))
    assert np.isclose(metrics['roc_auc'][0], 0.5)
    assert np.isclose(metrics['roc_auc'][1], 0.5)
    assert np.isclose(metrics['roc_auc'][2], 0.5)
    assert np.isclose(metrics['average_roc_auc'], 0.5)