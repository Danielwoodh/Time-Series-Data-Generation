import pytest
from datetime import datetime
from pandas._testing import assert_frame_equal
from codebase.utils_alternative import DataGenerator, States


@pytest.fixture
def data_generator():
    return DataGenerator()


def test_generate_time_series_data(data_generator):
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 1, 2)
    freq = '10S'
    expected_columns = ['rms', 'state', 'timestamp', 'state_change', 'rms_smoothed']
    expected_states = [States.OFF, States.IDLE, States.RUNNING, States.IDLE, States.OFF]
    expected_rms = [0.0, 0.5, 1.0, 0.5, 0.0]
    expected_rms_smoothed = [0.0, 0.166666, 0.5, 0.833333, 0.0]

    result = data_generator.generate_time_series_data(start_date, end_date, freq)

    assert set(result.columns) == set(expected_columns)
    assert result.shape[0] == 87  # 8 intervals * 10 seconds per interval + 1
    assert result['state'].tolist() == expected_states
    assert result['rms'].tolist() == expected_rms
    assert result['rms_smoothed'].tolist() == pytest.approx(expected_rms_smoothed, rel=1e-3)


def test_create_dataframe(data_generator):
    time_series_data = [
        (0.0, States.OFF, datetime(2022, 1, 1, 0, 0, 0)),
        (0.5, States.IDLE, datetime(2022, 1, 1, 0, 0, 10)),
        (1.0, States.RUNNING, datetime(2022, 1, 1, 0, 0, 20)),
        (0.5, States.IDLE, datetime(2022, 1, 1, 0, 0, 30)),
        (0.0, States.OFF, datetime(2022, 1, 1, 0, 0, 40)),
    ]
    expected_columns = ['rms', 'state', 'timestamp', 'state_change']

    result = data_generator._create_dataframe(time_series_data)

    assert set(result.columns) == set(expected_columns)
    assert_frame_equal(result, result.sort_values('timestamp').reset_index(drop=True))


def test_apply_kalman_filter(data_generator):
    time_series_data = [
        (0.0, States.OFF, datetime(2022, 1, 1, 0, 0, 0)),
        (0.5, States.IDLE, datetime(2022, 1, 1, 0, 0, 10)),
        (1.0, States.RUNNING, datetime(2022, 1, 1, 0, 0, 20)),
        (0.5, States.IDLE, datetime(2022, 1, 1, 0, 0, 30)),
        (0.0, States.OFF, datetime(2022, 1, 1, 0, 0, 40)),
    ]
    time_series_df = data_generator._create_dataframe(time_series_data)

    result = data_generator._apply_kalman_filter(time_series_df)

    assert result.tolist() == pytest.approx([0.0, 0.166666, 0.5, 0.833333, 0.0], rel=1e-3)