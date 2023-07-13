import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, accuracy_score, auc, classification_report
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

class LinearRegressionPredictor:
    '''
    This class is used to predict the engine state using linear regression.

    Args:
        model (sklearn.linear_model): The model to use for prediction
    '''
    def __init__(self, model: LinearRegression):
        """
        Initialize an instance of LinearRegressionPredictor.
        """
        self.model = model

    def calculate_slope(self, series: pd.Series) -> float:
        """
        This function is used to calculate the slope of a series using linear regression.

        Args:
            series (pd.Series): The series to calculate the slope

        Returns:
            float: The slope of the series
        """
        y = series.values.reshape(-1,1)
        x = np.array(range(len(series))).reshape(-1,1)
        self.model.fit(x, y)
        return self.model.coef_[0][0]

    def compute_metrics(self, y_tests: np.ndarray, y_preds: np.ndarray) -> dict:
        """
        This function is used to compute various metrics given true and predicted labels.

        Args:
            y_tests (np.ndarray): The true labels
            y_preds (np.ndarray): The predicted labels

        Returns:
            dict: A dictionary of calculated metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_tests, y_preds),
            'precision': precision_score(y_tests, y_preds, average='macro', zero_division=0),
            'recall': recall_score(y_tests, y_preds, average='macro', zero_division=0),
            'f1_score': f1_score(y_tests, y_preds, average='macro'),
            'confusion_matrix': confusion_matrix(y_tests, y_preds),
            'classification_report': classification_report(y_tests, y_preds)
        } 
        return metrics

    def plot_roc_auc(self, y_tests: np.ndarray, y_preds: np.ndarray) -> None:
        """
        This function is used to plot the ROC AUC curve.

        Args:
            y_tests (np.ndarray): The true labels
            y_preds (np.ndarray): The predicted labels
        """
        classes = np.unique(y_tests)
        y_true = label_binarize(y_tests, classes=classes)
        y_pred = label_binarize(y_preds, classes=classes)

        n_classes = len(classes)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure()
        colors = ['blue', 'red', 'green']
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label='ROC curve of class {0} (area = {1:0.2f})'
                    ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.show()

        return roc_auc

    def plot_confusion_matrix(self, cm: np.ndarray) -> None:
        """
        This function is used to plot the confusion matrix.

        Args:
            cm (np.ndarray): The confusion matrix
        """
        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title('Confusion matrix')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')

    def predict(self, df: pd.DataFrame) -> tuple:
        """
        This function is used to predict the engine state based on the RMS values.

        Args:
            df (pd.DataFrame): A DataFrame with 'rms' and 'state_encoded' columns

        Returns:
            tuple: A tuple containing the predicted engine state and the metrics used to evaluate the model
        """
        rolling_slope = df['rms'].rolling(window=10).apply(self.calculate_slope)
        engine_state = np.where(rolling_slope > 0, 0, 1)
        engine_state = np.where(df['rms'] < 2, 2, engine_state)

        metrics = self.compute_metrics(df['state_encoded'], engine_state)
        self.plot_confusion_matrix(metrics['confusion_matrix'])
        roc_auc = self.plot_roc_auc(df['state_encoded'], engine_state)
        metrics['roc_auc'] = roc_auc
        metrics['average_roc_auc'] = np.mean(list(roc_auc.values()))

        return engine_state, metrics
