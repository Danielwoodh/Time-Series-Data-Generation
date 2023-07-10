import random
from enum import Enum
from typing import List, Dict, Tuple, Any
from datetime import datetime
# import timesynth as ts
import numpy as np
import pandas as pd
from pykalman import KalmanFilter

# Constants
class States(Enum):
    ACTIVE = 'ACTIVE'
    IDLE = 'IDLE'
    OFF = 'OFF'

DEFAULT_RMS_RANGES = {
    States.OFF: (0, 1), 
    States.IDLE: (2, 300), 
    States.ACTIVE: (301, 600)
}

DEFAULT_MIN_DURATION = 1
DEFAULT_MAX_DURATION = 60

# Column names
TIMESTAMP = 'timestamp'
STATE = 'state'
RMS = 'rms'
RMS_SMOOTHED = 'rms_smoothed'
STATE_CHANGE = 'state_change'

class StateGenerator:
    '''
    This class generates random states for the time series dataset.
    '''
    def __init__(self, states: List[States]):
        if not states:
            raise ValueError("States list cannot be empty.")
        self.states = states

    def generate_state(self) -> States:
        '''
        This function generates a random state.

        Outputs:
            current_state (States): The current state.
        '''
        return random.choice(self.states)


class IntervalGenerator:
    '''
    This class generates the intervals for the time series data.
    '''
    def __init__(self, min_duration: int, max_duration: int):
        self.min_duration = min_duration
        self.max_duration = max_duration

    def calculate_steps(self, interval: int, freq: str) -> int:
        '''
        This function calculates the number of steps for a given interval and frequency.

        Args:
            interval (int): The interval in seconds.
            freq (str): The frequency of the dataset.

        Outputs:
            steps (int): The number of steps for the given interval.
        '''
        return int(interval / pd.Timedelta(freq).total_seconds())


class RMSGenerator:
    '''
    This class generates random RMS values for the time series dataset.
    '''
    def __init__(self, rms_ranges: Dict[States, Tuple[float, float]]):
        self.rms_ranges = rms_ranges

    def calculate_rms(self, current_state: States) -> float:
        '''
        This function generates a random RMS value for the current state.

        Args:
            current_state (States): The current state.

        Outputs:
            current_rms (float): The current RMS value.
        '''
        rms_range = self.rms_ranges.get(current_state)
        if rms_range is None:
            raise ValueError(f"No range defined for state {current_state}.")
        return np.random.uniform(rms_range[0], rms_range[1])

# class RMSGenerator:
#     '''
#     This class generates random RMS values for the time series dataset.
#     '''
#     def __init__(self, rms_ranges: Dict[States, Tuple[float, float]], timeseries_class=ts.TimeSampler):
#         self.rms_ranges = rms_ranges
#         self.timeseries = timeseries_class()

#     def calculate_rms(self, current_state: States) -> float:
#         '''
#         This function generates a random RMS value for the current state.

#         Args:
#             current_state (States): The current state.

#         Outputs:
#             current_rms (float): The current RMS value.
#         '''
#         rms_range = self.rms_ranges.get(current_state)
#         if rms_range is None:
#             raise ValueError(f"No range defined for state {current_state}.")
#         # Set the parameters for the time series generation
#         self.timeseries.sample_time(start_time=rms_range[0], end_time=rms_range[1])
#         # Use a sinusoidal process
#         sinusoid = ts.signals.Sinusoidal(frequency=0.25)
#         # Use Gaussian noise
#         white_noise = ts.noise.GaussianNoise(std=0.3)
#         timeseries = ts.TimeSeries(signal_generator=sinusoid, noise_generator=white_noise)
#         samples, signals, errors = timeseries.sample(self.timeseries)
#         # Return a random sample
#         return random.choice(samples)


class DataGenerator:
    '''
    This class generates time series data for a given set of states and rms ranges.
    '''
    def __init__(
        self,
        state_generator: StateGenerator,
        rms_generator: RMSGenerator,
        interval_generator: IntervalGenerator,
        kalman_filter: KalmanFilter
    ):
        self.state_generator = state_generator
        self.rms_generator = rms_generator
        self.interval_generator = interval_generator
        self.kalman_filter = kalman_filter

    def _generate_data_for_interval(self, current_state: States, current_timestamp: datetime, freq: str, interval: int) -> List[Tuple[float, States, datetime]]:
        '''
        Generate data for a single interval.

        Args:
            current_state (States): The current state.
            current_timestamp (datetime): The current timestamp.
            freq (str): The frequency of the dataset.
            interval (int): The interval in seconds.

        Outputs:
            interval_data (List[Tuple[float, States, datetime]]): List of tuples with RMS values, states and timestamps for the interval.
        '''
        interval_data = []
        steps = self.interval_generator.calculate_steps(interval, freq)
        for _ in range(steps):
            current_rms = self.rms_generator.calculate_rms(current_state)
            interval_data.append((current_rms, current_state, current_timestamp))
            current_timestamp += pd.Timedelta(freq)

        return interval_data

    def generate_time_series_data(
        self,
        start_date: datetime,
        end_date: datetime,
        freq: str = '1S',
    ) -> pd.DataFrame:
        '''
        This function generates a time series dataset with the following columns:
        - timestamp: datetime
        - state: str
        - rms: float
        - rms_smoothed: float#

        Args:
            start_date (datetime): The start date of the dataset.
            end_date (datetime): The end date of the dataset.
            freq (str): The frequency of the dataset.

        Outputs:
            time_series_df (pd.DataFrame): The generated dataset.
        '''
        time_series_data = []

        current_state = self.state_generator.generate_state()
        current_timestamp = start_date

        while current_timestamp < end_date:
            interval = np.random.uniform(self.interval_generator.min_duration, self.interval_generator.max_duration)
            interval_data = self._generate_data_for_interval(current_state, current_timestamp, freq, interval)
            time_series_data.extend(interval_data)
            current_timestamp = interval_data[-1][-1]
            current_state = self.state_generator.generate_state()

        time_series_df = self._create_dataframe(time_series_data)
        return time_series_df