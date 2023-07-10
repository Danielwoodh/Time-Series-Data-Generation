from datetime import datetime, timedelta
import pandas as pd
import random
from tqdm import tqdm
from pykalman import KalmanFilter
from typing import List, Tuple

class TimeSeriesDataGenerator:
    '''
    This class generates time series data for a given set of states and rms ranges.
    '''
    def __init__(
        self,
        kalman_filter: KalmanFilter,
        states: List[str] = ['ACTIVE', 'IDLE', 'OFF'],
        rms_ranges: dict = {'OFF': (0, 1), 'IDLE': (2, 300), 'ACTIVE': (301, 600)},
        min_duration: int = 1
    ):
        self.states = states
        self.rms_ranges = rms_ranges
        self.kalman_filter = kalman_filter
        self.min_duration = min_duration

    def verify_inputs(
        self,
        start_date: datetime,
        end_date: datetime,
        max_duration: int,
        freq: str
    ) -> None:
        '''
        This function verifies the inputs to the generate_time_series_data function.

        Args:
            start_date (datetime): The start date of the dataset.
            end_date (datetime): The end date of the dataset.
            max_duration (int): The maximum duration of a state in seconds.
            freq (str): The frequency of the dataset.

        Raises:
            ValueError: If start_date >= end_date.
            ValueError: If max_duration < self.min_duration.
            ValueError: If max_duration < pd.Timedelta(freq).total_seconds().
            ValueError: If self.rms_ranges['OFF'][0] != 0 or self.rms_ranges['OFF'][1] != 1.
            ValueError: If self.rms_ranges['IDLE'][0] <= 1.
            ValueError: If self.rms_ranges['ACTIVE'][0] <= self.rms_ranges['IDLE'][1].
        '''
        if start_date >= end_date:
            raise ValueError("Start date must be before end date.")

        if max_duration < self.min_duration:
            raise ValueError("Maximum duration must be greater than minimum duration.")

        if max_duration < pd.Timedelta(freq).total_seconds():
            raise ValueError("Frequency must be less than the maximum duration.")

        if self.rms_ranges['OFF'][0] != 0 or self.rms_ranges['OFF'][1] != 1:
            raise ValueError("'OFF' state range must be strictly between 0 and 1.")

        if self.rms_ranges['IDLE'][0] <= 1:
            raise ValueError("'IDLE' state range must start from a value greater than 1.")

        if self.rms_ranges['ACTIVE'][0] <= self.rms_ranges['IDLE'][1]:
            raise ValueError("'ACTIVE' state range must start from a value greater than the end of 'IDLE' range.")

    def calculate_rms(self, state: str) -> float:
        '''
        This function randomises the rms value for a given state

        Args:
            state (str): The current state.

        Outputs:
            rms (float): The randomised rms value.
        '''
        return random.uniform(*self.rms_ranges[state])

    def calculate_steps(self, interval: int, freq: str) -> int:
        '''
        This function calculates the number of steps in a given interval.

        Args:
            interval (int): The interval in seconds.
            freq (str): The frequency of the data.

        Outputs:
            steps (int): The number of steps in the interval.
        '''
        steps_per_second = pd.Timedelta("1s") / pd.Timedelta(freq)
        return int(interval * steps_per_second)

    def update_progress_bar(self, pbar, steps: int) -> None:
        '''
        This function updates the progress bar.

        Args:
            pbar (tqdm): The progress bar.
            steps (int): The number of steps to update the progress bar.
        '''
        pbar.update(steps)

    def append_generated_data(
        self,
        states: List[str],
        rms_values: List[float],
        current_state: str,
        steps: int
    ) -> None:
        '''
        This function appends the generated data to the states and rms_values lists.

        Args:
            states (List[str]): The list of states.
            rms_values (List[float]): The list of rms values.
            current_state (str): The current state.
            steps (int): The number of steps to append to the lists.
        '''
        for _ in range(steps):
            current_rms = self.calculate_rms(current_state)
            rms_values.append(current_rms)
            states.append(current_state)

    def apply_kalman_filter(self, df: pd.DataFrame) -> List[float]:
        '''
        This function applies a Kalman filter to the 'rms' column of the DataFrame to smooth the data.

        Args:
            df (pd.DataFrame): The DataFrame containing the 'rms' column.

        Outputs:
            rms_smoothed (List[float]): The smoothed 'rms' values.
        '''
        # Compare the current state to the previous state
        df['state_change'] = df['state'] != df['state'].shift()
        # Get the indices where state has changed
        change_indices = df[df['state_change']].index.tolist()
        # Add the starting and ending index to change_indices
        change_indices = change_indices + [len(df)]

        pairs = [[change_indices[i], change_indices[i+1]] for i in range(len(change_indices)-1)]

        rms_smoothed = []
        # Loop over pairs of indices in change_indices
        for pair in pairs:
            # Select the data in the current interval
            interval_data = df['rms'][pair[0]:pair[1]]    
            # Ensure there is more than one data point in the interval
            if len(interval_data) > 1:
                # Apply the Kalman filter to the data in the current interval
                state_means, state_covariances = self.kalman_filter.em(interval_data).smooth(interval_data)
                # Replace the 'rms' values in the DataFrame with the filtered values
                rms_smoothed.append(state_means.flatten())
            else:
                rms_smoothed.append(interval_data)
        
        return [item for sublist in rms_smoothed for item in sublist]

    def generate_time_series_data(
        self,
        start_date: datetime,
        end_date: datetime,
        freq: str = '1S',
        max_duration: int = 60
    ) -> pd.DataFrame:
        '''
        This function generates a time series dataset with the following columns:
        - timestamp: datetime
        - state: str
        - rms: float
        - rms_smoothed: float

        Args:
            start_date (datetime): The start date of the dataset.
            end_date (datetime): The end date of the dataset.
            freq (str, optional): The frequency of the dataset. Defaults to '1S'.
            max_duration (int, optional): The maximum duration of a state. Defaults to 60.

        Outputs:
            df (pd.DataFrame): The generated time series dataset.
        '''
        self.verify_inputs(start_date, end_date, max_duration, freq)
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        data = {'timestamp': date_range}
        df = pd.DataFrame(data)

        states = []
        rms_values = []
        current_step = 0

        pbar = tqdm(total=len(date_range), desc="Generating Data", ncols=80)

        while current_step < len(date_range):
            current_state = random.choice(self.states)
            interval = random.randint(self.min_duration, max_duration)

            steps = self.calculate_steps(interval, freq)

            if current_step + steps > len(date_range):
                steps = len(date_range) - current_step

            self.append_generated_data(states, rms_values, current_state, steps)

            current_step += steps
            self.update_progress_bar(pbar, steps)

        pbar.close()

        df['state'] = states
        df['rms'] = rms_values

        df['rms_smoothed'] = self.apply_kalman_filter(df)

        return df

def generate_rms_ranges(
    min_value_1: int = 2,
    max_value_1: int = 300,
    max_value_2: int = 600
) -> dict:
    '''
    This function generates a dictionary of rms ranges for the 'OFF', 'IDLE', and 'ACTIVE' states.

    Args:
        min_value_1 (int, optional): The minimum value for the 'IDLE' state. Defaults to 2.
        max_value_1 (int, optional): The maximum value for the 'IDLE' state. Defaults to 300.
        max_value_2 (int, optional): The maximum value for the 'ACTIVE' state. Defaults to 600.
    
    Outputs:
        rms_ranges (dict): A dictionary of rms ranges for the 'OFF', 'IDLE', and 'ACTIVE' states.
    '''
    value_1 = random.uniform(min_value_1, max_value_1)
    value_2 = random.uniform(value_1 - 50, max_value_2)
    rms_ranges = {
        'OFF': (0, 1),
        'IDLE': (2, value_1),
        'ACTIVE': (value_1 - 50, value_2)
    }
    return rms_ranges