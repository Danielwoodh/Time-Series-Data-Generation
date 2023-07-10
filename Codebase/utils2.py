import random
from enum import Enum
from typing import List, Dict, Tuple, Any
from datetime import datetime
import numpy as np
import pandas as pd
from pykalman import KalmanFilter

# Constants
class States(Enum):
    ACTIVE = 'ACTIVE'
    IDLE = 'IDLE'
    OFF = 'OFF'

DEFAULT_RMS_RANGES = {
    States.OFF: (0, 3), 
    States.IDLE: (50, 300), 
    States.ACTIVE: (250, 600)
}

DEFAULT_MIN_DURATION = 1
DEFAULT_MAX_DURATION = 200

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

    def generate_interval(self, current_state: States) -> int:
        '''
        This function generates a random interval for the current state.

        Args:
            current_state (States): The current state.
            default_interval_ranges (Dict[States, Tuple[int, int]]): The default interval ranges for each state.

        Outputs:
            interval (int): The interval in seconds.
        '''
        interval = random.randint(self.min_duration, self.max_duration)
        return interval

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
            interval = random.randint(self.interval_generator.min_duration, self.interval_generator.max_duration)
            interval_data = self._generate_data_for_interval(current_state, current_timestamp, freq, interval)
            time_series_data.extend(interval_data)
            current_timestamp = interval_data[-1][-1]
            current_state = self.state_generator.generate_state()

        time_series_df = self._create_dataframe(time_series_data)
        return time_series_df

    def _create_dataframe(self, time_series_data: List[Tuple[float, States, datetime]]) -> pd.DataFrame:
        '''
        This function creates a dataframe and applies the Kalman filter to the 'rms' column of the DataFrame to smooth the data.

        Args:
            time_series_data (List[Tuple[float, States, datetime]]): List of tuples with RMS values, states and timestamps.

        Outputs:
            time_series_df (pd.DataFrame): DataFrame with the following columns: timestamp, state, rms and rms_smoothed
        '''
        time_series_df = pd.DataFrame(time_series_data, columns=[RMS, STATE, TIMESTAMP])
        time_series_df[STATE_CHANGE] = time_series_df[STATE] != time_series_df[STATE].shift()
        time_series_df[RMS_SMOOTHED] = self._apply_kalman_filter(time_series_df)
        return time_series_df

    def _apply_kalman_filter(self, time_series_df: pd.DataFrame) -> pd.Series:
        '''
        This function applies a Kalman filter to the 'rms' column of the DataFrame to smooth the data.

        Args:
            time_series_df (pd.DataFrame): The DataFrame containing the 'rms' column.

        Outputs:
            rms_smoothed (pd.Series): The smoothed 'rms' values.
        '''
        # Generate a unique group id for each continuous state
        time_series_df['group'] = (time_series_df[STATE_CHANGE]).cumsum()

        # Pad each group with a small buffer of data from the neighboring group
        buffer_size = 2  # The size of the buffer, adjust as needed
        time_series_df[RMS] = time_series_df.groupby('group')[RMS].transform(lambda x: x.rolling(buffer_size, min_periods=1).mean())

        # Apply the Kalman filter to each group separately
        rms_smoothed = time_series_df.groupby('group').apply(lambda group: self._apply_kalman_filter_to_group(group[RMS]))

        # Flatten the MultiIndex
        rms_smoothed = rms_smoothed.reset_index(level=0, drop=True)

        return rms_smoothed

    # def _apply_kalman_filter(self, time_series_df: pd.DataFrame) -> List[float]:
    #     '''
    #     This function applies a Kalman filter to the 'rms' column of the DataFrame to smooth the data.

    #     Args:
    #         time_series_df (pd.DataFrame): The DataFrame containing the 'rms' column.

    #     Outputs:
    #         rms_smoothed (List[float]): The smoothed 'rms' values.
    #     '''
    #     time_series_df[STATE_CHANGE] = time_series_df[STATE] != time_series_df[STATE].shift()
    #     state_means, _ = self.kalman_filter.em(time_series_df[RMS].values).smooth(time_series_df[RMS].values)
    #     return [item for sublist in state_means for item in sublist]

    def _apply_kalman_filter_to_group(self, group_rms: pd.Series) -> pd.Series:
        '''
        This function applies a Kalman filter to a single group of 'rms' values.

        Args:
            group_rms (pd.Series): The 'rms' values for a single group.

        Outputs:
            group_rms_smoothed (pd.Series): The smoothed 'rms' values for the group.
        '''
        if len(group_rms) > 1:
            state_means, _ = self.kalman_filter.em(group_rms.values).smooth(group_rms.values)
            return pd.Series([item for sublist in state_means for item in sublist], index=group_rms.index)
        else:
            return group_rms