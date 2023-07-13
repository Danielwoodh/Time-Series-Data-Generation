import unittest
import sys
sys.path.append('c:/Users/Danie/Desktop/MachineMax Tech_Test/Codebase') # add the path to the Codebase module
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from models import TimeSeriesModelEvaluator
import numpy as np

class TestTimeSeriesModelEvaluator(unittest.TestCase):
    def setUp(self):
        self.model = LogisticRegression()
        self.time_series_model_evaluator = TimeSeriesModelEvaluator(self.model)
        self.X = np.random.rand(100, 1)  # 100 instances, 1 feature
        self.y = np.random.randint(0, 3, 100)  # 100 instances, 3 classes (0, 1, 2)

    def test_compute_metrics(self):
        y_preds = np.random.randint(0, 3, 100)
        metrics = self.time_series_model_evaluator.compute_metrics(self.y, y_preds)
        self.assertTrue('accuracy' in metrics)
        self.assertTrue('precision' in metrics)
        self.assertTrue('recall' in metrics)
        self.assertTrue('f1_score' in metrics)
        self.assertTrue('confusion_matrix' in metrics)
        self.assertTrue('classification_report' in metrics)

    def test_evaluate_no_cv(self):
        metrics = self.time_series_model_evaluator.evaluate_no_cv(self.X, self.y)
        self.assertTrue('accuracy' in metrics)
        self.assertTrue('precision' in metrics)
        self.assertTrue('recall' in metrics)
        self.assertTrue('f1_score' in metrics)
        self.assertTrue('confusion_matrix' in metrics)
        self.assertTrue('classification_report' in metrics)

    def test_evaluate_cv(self):
        cv = TimeSeriesSplit(n_splits=5)
        metrics = self.time_series_model_evaluator.evaluate_cv(self.X, self.y, cv)
        self.assertTrue('accuracy' in metrics)
        self.assertTrue('precision' in metrics)
        self.assertTrue('recall' in metrics)
        self.assertTrue('f1_score' in metrics)
        self.assertTrue('confusion_matrix' in metrics)
        self.assertTrue('classification_report' in metrics)

if __name__ == '__main__':
    unittest.main(verbosity=2)