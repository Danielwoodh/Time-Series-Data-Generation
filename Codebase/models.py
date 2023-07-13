import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, accuracy_score, auc, classification_report
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.preprocessing import MinMaxScaler
from Codebase.utils import States
import shap

class TimeSeriesModelEvaluator:
    '''
    This class is used to evaluate time series models.

    Args:
        model (sklearn.model): The model to evaluate
        hyperparams (dict): The hyperparameters to use for tuning the model
    '''
    def __init__(
        self,
        model,
        hyperparams: dict = None
    ):
        self.model = model
        self.hyperparams = hyperparams

    def compute_metrics(
        self,
        y_tests: np.ndarray,
        y_preds: np.ndarray
    ) -> dict:
        metrics = {
            'accuracy': accuracy_score(y_tests, y_preds),
            'precision': precision_score(y_tests, y_preds, average='macro', zero_division=0),
            'recall': recall_score(y_tests, y_preds, average='macro', zero_division=0),
            'f1_score': f1_score(y_tests, y_preds, average='macro'),
            'confusion_matrix': confusion_matrix(y_tests, y_preds),
            'classification_report': classification_report(y_tests, y_preds)
        } 
        return metrics

    def plot_confusion_matrix(
        self,
        cm: np.ndarray
    ) -> None:
        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title('Confusion matrix')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')

    def plot_roc_auc(self, y_tests: np.ndarray, y_preds: np.ndarray) -> None:
        # Binarize the output
        classes = np.unique(y_tests)
        y_bin = label_binarize(y_tests, classes=classes)
        y_pred_bin = label_binarize(y_preds, classes=classes)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(len(classes)):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_pred_bin[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure()
        colors = ['blue', 'red', 'green']
        for i, color in zip(range(len(classes)), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label='ROC curve of class {0} (area = {1:0.2f})'
                    ''.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC_AUC Curve for Logistic Regression Model')
        plt.legend(loc="lower right")
        plt.show()

        return roc_auc

    def evaluate_no_cv(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        metrics = self.compute_metrics(y_test, y_pred)

        # Plot confusion matrix and ROC AUC
        self.plot_confusion_matrix(metrics['confusion_matrix'])
        metrics['roc_auc'] = self.plot_roc_auc(y_test, y_pred)
        metrics['average_roc_auc'] = np.mean(list(metrics['roc_auc'].values()))

        return metrics

    def evaluate_cv(self, X, y, cv):
        y_preds = []
        y_tests = []
        for i, (train_index, test_index) in enumerate(cv.split(X)):
            print(f'Fold {i+1}')
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            y_preds.append(y_pred)
            y_tests.append(y_test)
        
        # Concatenate all predictions and true labels
        y_preds = np.concatenate(y_preds)
        y_tests = np.concatenate(y_tests)
        metrics = self.compute_metrics(y_tests, y_preds)

        # Plot confusion matrix and ROC AUC
        self.plot_confusion_matrix(metrics['confusion_matrix'])
        metrics['roc_auc'] = self.plot_roc_auc(y_test, y_pred)
        metrics['average_roc_auc'] = np.mean(list(metrics['roc_auc'].values()))
        return metrics
    
    def evaluate(self,
        X: np.ndarray,
        y: np.ndarray,
        split_type: str = 'train_test_split',
        n_splits: int = 5
    ):
        '''
        This function evaluates the model using the specified split type.

        Args:
            X (np.ndarray): The features
            y (np.ndarray): The labels
            split_type (str): The type of split to use. Either 'TimeSeriesSplit' or 'train_test_split'
            n_splits (int): The number of splits to use for cross validation
        '''
        match split_type:
            case 'TimeSeriesSplit':
                cv = TimeSeriesSplit(n_splits=n_splits)
            case 'train_test_split':
                cv = None
            case _:
                raise ValueError("Invalid split type. Must be either 'TimeSeriesSplit' or 'train_test_split'")
         
        if self.hyperparams is not None:
            self.model = RandomizedSearchCV(self.model, self.hyperparams, cv=cv)

        if cv is None:
            metrics = self.evaluate_no_cv(X, y)
        else:
            metrics = self.evaluate_cv(X, y, cv)

        if self.hyperparams is not None:
            print(f'Best hyperparameters: {self.model.best_params_}')

        return metrics