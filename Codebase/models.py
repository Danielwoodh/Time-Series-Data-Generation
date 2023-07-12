import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, accuracy_score, auc, classification_report
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from Codebase.utils import States
import shap

class TimeSeriesModelEvaluator:
    '''
    This class is used to evaluate time series models.
    '''
    def __init__(
        self,
        model,
        hyperparams: dict = None
    ):
        self.model = model
        self.hyperparams = hyperparams
    
    def evaluate(self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5
    ):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        if self.hyperparams:
            self.model = GridSearchCV(self.model, self.hyperparams, cv=tscv)
            self.model.fit(X, y)
        y_preds = []
        y_probs = []
        y_tests = []
        for i, (train_index, test_index) in enumerate(tscv.split(X)):
            print(f'Fold {i+1}')
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.model.fit(X_train, y_train)
            if len(np.unique(y_train)) == len(np.unique(y_test)):
                y_pred = self.model.predict(X_test)
                y_pred_proba = self.model.predict_proba(X_test)
                y_preds.append(y_pred)
                y_probs.append(y_pred_proba)
                y_tests.append(y_test)
        if self.hyperparams:
            print("Best parameters:", self.model.best_params_)
        y_preds = np.concatenate(y_preds)
        y_probs = np.concatenate(y_probs, axis=0)
        y_tests = np.concatenate(y_tests)
        lb = LabelBinarizer()
        lb.fit(y_tests)
        y_test_bin = lb.transform(y_tests)
        y_pred_bin = lb.transform(y_preds)
        roc_auc = roc_auc_score(y_test_bin, y_probs, multi_class='ovr')
        metrics = {
            'accuracy': accuracy_score(y_tests, y_preds),
            'precision': precision_score(y_tests, y_preds, average='macro', zero_division=0),
            'recall': recall_score(y_tests, y_preds, average='macro', zero_division=0),
            'f1_score': f1_score(y_tests, y_preds, average='macro'),
            'roc_auc_score': roc_auc,
            'confusion_matrix': confusion_matrix(y_tests, y_preds),
            'classification_report': classification_report(y_tests, y_preds)
        }
        return metrics