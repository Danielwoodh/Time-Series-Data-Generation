from datetime import datetime, timedelta
import pandas as pd
import random
from tqdm import tqdm
from pykalman import KalmanFilter
from dataclasses import dataclass

@dataclass
class TimeSeriesDataGenerator:
    STATES = ['ACTIVE', 'IDLE', 'OFF']
    RMS_RANGES = {'OFF': (0, 1), 'IDLE': (2, 300), 'ACTIVE': (301, 600)}
    MIN_DURATION = 10

    def verify_inputs(
        self,
        start_date: datetime,
        end_date: datetime,
        max_duration: int,
        freq: str,
        rms_ranges: dict
    ):
        if start_date >= end_date:
            raise ValueError("Start date must be before end date.")
        elif max_duration < self.MIN_DURATION:
            raise ValueError("Maximum duration must be greater than minimum duration.")
        elif max_duration < pd.Timedelta(freq).total_seconds():
            raise ValueError("Frequency must be less than the maximum duration.")
        
        # Verify rms_ranges
        if rms_ranges['OFF'][0] != 0 or rms_ranges['OFF'][1] != 1:
            raise ValueError("'OFF' state range must be strictly between 0 and 1.")
        elif rms_ranges['IDLE'][0] <= 1:
            raise ValueError("'IDLE' state range must start from a value greater than 1.")
        elif rms_ranges['ACTIVE'][0] <= rms_ranges['IDLE'][1]:
            raise ValueError("'ACTIVE' state range must start from a value greater than the end of 'IDLE' range.")

    def calculate_rms(self, state, rms_ranges) -> int:
        return random.uniform(*rms_ranges[state])

    def calculate_steps(self, interval, freq) -> int:
        steps_per_second = pd.Timedelta("1s") / pd.Timedelta(freq)
        return int(interval * steps_per_second)

    def update_progress_bar(self, pbar, steps) -> None:
        pbar.update(steps)

    def append_generated_data(
        self,
        states: list,
        rms_values: list,
        current_state: str,
        steps: int,
        rms_ranges: dict
    ) -> None:
        for _ in range(steps):
            current_rms = self.calculate_rms(current_state, rms_ranges)
            rms_values.append(current_rms)
            states.append(current_state)

    def kalman_filter(self, df: pd.DataFrame) -> list:
        # Compare the current state to the previous state
        df['state_change'] = df['state'] != df['state'].shift()
        # Get the indices where state has changed
        change_indices = df[df['state_change']].index.tolist()
        # Add the starting and ending index to change_indices
        change_indices = change_indices + [len(df)]

        pairs = [[change_indices[i], change_indices[i+1]] for i in range(len(change_indices)-1)]

        # Create a Kalman Filter
        kf = KalmanFilter(transition_matrices=[1],
                        observation_matrices=[1],
                        initial_state_mean=0,
                        initial_state_covariance=1,
                        observation_covariance=0.7,
                        transition_covariance=1)
        rms_smoothed = []
        # Loop over pairs of indices in change_indices
        for pair in pairs:
            # Select the data in the current interval
            interval_data = df['rms'][pair[0]:pair[1]]    
            # Ensure there is more than one data point in the interval
            if len(interval_data) > 1:
                # Apply the Kalman filter to the data in the current interval
                state_means, state_covariances = kf.em(interval_data).smooth(interval_data)
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
    max_duration: int = 60,
    rms_ranges: dict = RMS_RANGES
    ) -> pd.DataFrame:
        self.verify_inputs(start_date, end_date, max_duration, freq, rms_ranges)
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        data = {'timestamp': date_range}
        df = pd.DataFrame(data)

        states = []
        rms_values = []
        current_step = 0

        pbar = tqdm(total=len(date_range), desc="Generating Data", ncols=80)

        while current_step < len(date_range):
            current_state = random.choice(self.STATES)
            interval = random.randint(self.MIN_DURATION, max_duration)

            steps = self.calculate_steps(interval, freq)

            if current_step + steps > len(date_range):
                steps = len(date_range) - current_step

            self.append_generated_data(states, rms_values, current_state, steps, rms_ranges)

            current_step += steps
            self.update_progress_bar(pbar, steps)

        pbar.close()

        df['state'] = states
        df['rms'] = rms_values

        df['rms_smoothed'] = self.kalman_filter(df)

        return df

def generate_rms_ranges(
    min_value_1: int = 2,
    max_value_1: int = 300,
    max_value_2: int = 600
):
    value_1 = random.uniform(min_value_1, max_value_1)
    value_2 = random.uniform(value_1+1, max_value_2)
    rms_ranges = {
        'OFF': (0, 1),
        'IDLE': (2, value_1),
        'ACTIVE': (value_1 + 1, value_2)
    }
    return rms_ranges