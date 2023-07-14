'''
This script outlines how to use the TimeSeriesForest model from sktime with TimeSeriesSplit cross-validation.

Note that TimeSeriesSplit cross-validation only works if each segment of data contains all classes.
'''

from sktime.classification.interval_based import TimeSeriesForest
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from utils import generate_time_series_data

# Generate time-series data
start_date = # insert start date
end_date = # insert end date
freq = '10S'
time_series_df = generate_time_series_data(start_date, end_date, freq)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    time_series_df.drop(columns=['timestamp', 'rms_smoothed']),  # features
    time_series_df['state'],  # target variable
    test_size=0.2,
    random_state=0
)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, 30],
    'min_interval': [2, 3, 4],
    'n_jobs': [-1]
}
# Set up time series cross-validation, this only works if each segment of data contains all classes
tscv = TimeSeriesSplit(n_splits=5)
# Fit TimeSeriesForest model with GridSearchCV
tsf = TimeSeriesForest(random_state=0)
# Set up grid search
grid_search = GridSearchCV(tsf, param_grid, cv=tscv, n_jobs=-1)
grid_search.fit(X_train, y_train)

# This should fix it, pseudo hard-coded grid-search
for g in ParameterGrid(grid):
    rf.set_params(**g)
    rf.fit(X,y)
    # save if best
    if rf.roc_auc_score_ > best_score:
        best_score = rf.roc_auc_score_
        best_grid = g

# Evaluate model
accuracy = grid_search.score(X_test, y_test)
print(f"Accuracy: {accuracy}")
print(f"Best parameters: {grid_search.best_params_}")

