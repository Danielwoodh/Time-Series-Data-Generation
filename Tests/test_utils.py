import pytest
from datetime import datetime
from pandas.api.types import CategoricalDtype
from utils import States, TimeSeriesGenerator

@pytest.fixture
def time_series_generator():
    return TimeSeriesGenerator()

def test_generate_time_series_data(time_series_generator):
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 1, 2)
    freq = '10S'
    time_series_df = time_series_generator.generate_time_series_data(start_date, end_date, freq)

    # Check that the dataframe has the correct columns
    assert set(time_series_df.columns) == {'timestamp', 'state', 'rms', 'rms_smoothed'}

    # Check that the 'state' column has the correct data type
    assert time_series_df['state'].dtype == CategoricalDtype(categories=[States.OFF, States.IDLE, States.RUNNING], ordered=True)

    # Check that the 'timestamp' column has the correct data type
    assert time_series_df['timestamp'].dtype == 'datetime64[ns]'

    # Check that the 'rms' column has the correct data type
    assert time_series_df['rms'].dtype == 'float64'

    # Check that the 'rms_smoothed' column has the correct data type
    assert time_series_df['rms_smoothed'].dtype == 'float64'

    # Check that the dataframe has the correct number of rows
    expected_rows = int((end_date - start_date).total_seconds() / 10)
    assert len(time_series_df) == expected_rows

    # Check that the 'state' column has the correct values
    expected_states = [States.OFF, States.IDLE, States.RUNNING] * expected_rows
    assert (time_series_df['state'].cat.categories[time_series_df['state'].cat.codes] == expected_states).all()

    # Check that the 'rms' column has the correct values
    assert time_series_df['rms'].min() >= 0
    assert time_series_df['rms'].max() <= 1

    # Check that the 'rms_smoothed' column has the correct values
    assert time_series_df['rms_smoothed'].min() >= 0
    assert time_series_df['rms_smoothed'].max() <= 1