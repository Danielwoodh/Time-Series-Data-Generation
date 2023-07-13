import unittest
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
# import the necessary modules from your project
import sys
sys.path.append('c:/Users/Danie/Desktop/MachineMax Tech_Test/Codebase')
from models import TimeSeriesModelEvaluator

class TestTimeSeriesModelEvaluator(unittest.TestCase):
    def setUp(self):
        self.X = np.array([
            [1, 8],
            [2, 5],
            [3, 1],
            [4, 9],
            [5, 6],
            [6, 0],
            [7, 7],
            [8, 4],
            [9, 0],
            [10, 8],
            [11, 8],
            [12, 5],
            [13, 1],
            [14, 9],
            [15, 6],
            [16, 0],
            [17, 7],
            [18, 4],
            [19, 0],
            [20, 8]
        ])
        self.y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        self.model = LogisticRegression(multi_class='multinomial', random_state=42)
        self.hyperparams = {'C': [0.1, 1, 10]}
        self.evaluator = TimeSeriesModelEvaluator(self.model, self.hyperparams)

    def test_compute_metrics(self):
        # Given
        y_test = np.array([0, 1, 2, 2, 1])
        y_pred = np.array([0, 1, 2, 1, 1])

        # When
        metrics = self.evaluator.compute_metrics(y_test, y_pred)

        # Then
        self.assertEqual(metrics['accuracy'], 0.8)

    def test_plot_confusion_matrix(self):
        # Given
        cm = np.array([[2, 0, 0], [0, 2, 1], [0, 0, 0]])

        # When
        # This test will pass if it doesn't throw an exception
        self.evaluator.plot_confusion_matrix(cm)

    def test_evaluate_no_cv(self):
        # When
        metrics = self.evaluator.evaluate_no_cv(self.X, self.y)

        # Then
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('confusion_matrix', metrics)
        self.assertIn('classification_report', metrics)
        self.assertIn('roc_auc', metrics)
        self.assertIn('average_roc_auc', metrics)

    def test_evaluate_cv(self):
        # Given
        cv = TimeSeriesSplit(n_splits=3)

        # When
        metrics = self.evaluator.evaluate_cv(self.X, self.y, cv)

        # Then
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('confusion_matrix', metrics)
        self.assertIn('classification_report', metrics)
        self.assertIn('roc_auc', metrics)
        self.assertIn('average_roc_auc', metrics)

    def test_evaluate(self):
        # When
        metrics = self.evaluator.evaluate(self.X, self.y)

        # Then
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('confusion_matrix', metrics)
        self.assertIn('classification_report', metrics)
        self.assertIn('roc_auc', metrics)
        self.assertIn('average_roc_auc', metrics)

    def test_evaluate_error(self):
        # When/Then
        with self.assertRaises(ValueError):
            self.evaluator.evaluate(self.X, self.y, 'InvalidSplit', n_splits=5)


if __name__ == '__main__':
    unittest.main()
