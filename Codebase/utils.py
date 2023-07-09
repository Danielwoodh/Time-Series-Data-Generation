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
    MIN_DURATION = 2

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

    def kalman_filter(self, rms_values: list) -> list:
        # Create a Kalman Filter
        kf = KalmanFilter(transition_matrices=[1],
                        observation_matrices=[1],
                        initial_state_mean=0,
                        initial_state_covariance=1,
                        observation_covariance=1,
                        transition_covariance=.01)

        # Use the observations y to get a rolling mean and variance
        rms_means, rms_covariances = kf.em(rms_values).smooth(rms_values)
        return rms_means
        
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

        return df
