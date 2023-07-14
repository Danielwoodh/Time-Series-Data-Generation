import sys
import unittest
from datetime import datetime, timedelta
from pandas._testing import assert_frame_equal
import pandas as pd
# import the necessary modules from your project
sys.path.append('c:/Users/Danie/Desktop/MachineMax Tech_Test/Codebase')
from utils import States, StateGenerator, RMSGenerator, IntervalGenerator, DataGenerator
from pykalman import KalmanFilter

# Initialize the StateGenerator
states = [States.ACTIVE, States.IDLE, States.OFF]
# Define the transition probabilities
transition_probabilities = {
    States.OFF: {
        States.OFF: 0.5,
        States.IDLE: 0.5,
        States.ACTIVE: 0.0
    },
    States.IDLE: {
        States.OFF: 0.2,
        States.IDLE: 0.5,
        States.ACTIVE: 0.3
    },
    States.ACTIVE: {
        States.OFF: 0.0001,
        States.IDLE: 0.18,
        States.ACTIVE: 0.82
    }
}

DEFAULT_RMS_RANGES = {
    States.OFF: ('uniform', 0, 2),
    States.IDLE: ('normal', 250, 75),
    States.ACTIVE: ('lognormal', 6.2, 0.2)
}

# Initialize the IntervalGenerator
state_duration_map = {
    States.OFF: (181, 481),
    States.IDLE: (60, 211),
    States.ACTIVE: (30, 481)
}

class TestDataGenerator(unittest.TestCase):
    def setUp(self):
        self.state_generator = StateGenerator(
            states=states,
            transition_probabilities=transition_probabilities,
            random_seed=1
        )
        self.rms_generator = RMSGenerator(
            DEFAULT_RMS_RANGES,
            random_seed=1
        )
        self.interval_generator = IntervalGenerator(
            state_duration_map,
            random_seed=1
        )
        self.kalman_filter = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=0,
            initial_state_covariance=1,
            observation_covariance=.5,
            transition_covariance=10
        )

        self.data_generator = DataGenerator(
            state_generator=self.state_generator,
            rms_generator=self.rms_generator,
            interval_generator=self.interval_generator,
            kalman_filter=self.kalman_filter,
            random_seed=1
        )

    def test__validate_inputs(self):
        start_date = datetime.now()
        end_date = start_date + timedelta(days=1)

        # Test when start_date is greater than end_date
        with self.assertRaises(ValueError):
            self.data_generator._validate_inputs(end_date, start_date, '10S')
        
        # Test when start_date and end_date are not datetime objects
        with self.assertRaises(ValueError):
            self.data_generator._validate_inputs("2023-07-13", end_date, '10S')
        
        # Test when freq is not a valid pandas date frequency string
        with self.assertRaises(ValueError):
            self.data_generator._validate_inputs(start_date, end_date, '10X')

        # Test when freq is less than 1 second
        with self.assertRaises(ValueError):
            self.data_generator._validate_inputs(start_date, end_date, '500ms')
        
        # Test when freq is greater than the difference between start_date and end_date
        with self.assertRaises(ValueError):
            self.data_generator._validate_inputs(start_date, end_date, '2D')

    def test_generate_time_series_data(self):
        start_date = datetime.now()
        end_date = start_date + timedelta(days=1)
        time_series_df = self.data_generator.generate_time_series_data(start_date, end_date, '10S')

        # Check if the generated dataframe has the required columns
        self.assertListEqual(list(time_series_df.columns), ['rms', 'state', 'timestamp', 'state_change', 'rms_smoothed'])
        
        # Check if the data types of the columns are correct
        self.assertEqual(time_series_df['rms'].dtype, 'float64')
        self.assertEqual(time_series_df['state'].dtype, 'object')  # or another type depending on your implementation
        self.assertEqual(time_series_df['timestamp'].dtype, 'datetime64[ns]')
        self.assertEqual(time_series_df['state_change'].dtype, 'bool')
        self.assertEqual(time_series_df['rms_smoothed'].dtype, 'float64')

        # Check if the timestamps are in ascending order
        self.assertTrue(time_series_df['timestamp'].is_monotonic_increasing)

if __name__ == '__main__':
    unittest.main()