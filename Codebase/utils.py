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
    def __init__(
        self,
        states: List[States],
        transition_probabilities: Dict[States, Dict[States, float]],
        random_seed: int = None
    ):
        self._validate_inputs(states, transition_probabilities)
        self.states = states
        self.transition_probabilities = transition_probabilities
        self.has_been_active = False
        if random_seed is not None:
            random.seed(random_seed)

    @staticmethod
    def _validate_inputs(
        states: List[States],
        transition_probabilities: Dict[States, Dict[States, float]]
    ) -> None:
        '''
        This function validates the inputs to the StateGenerator class.

        Args:
            states (List[States]): The list of states.
            transition_probabilities (Dict[States, Dict[States, float]]): The transition probabilities.
        '''
        if not all(isinstance(state, States) for state in States):
            raise ValueError('All states must be instances of the States Enum.')
        if not all(isinstance(state, States) and isinstance(prob_dict, dict) 
                    and all(isinstance(s, States) and isinstance(p, float) for s, p in prob_dict.items()) 
                    for state, prob_dict in transition_probabilities.items()
        ):
            raise ValueError("Transition probabilities must be a dictionary of States mapped to dictionary of states and probabilities.")

    def generate_state(self, current_state: States) -> States:
        '''
        This function generates the next state based on the current state and transition probabilities.

        Args:
            current_state (States): The current state.

        Outputs:
            next_state (States): The next state.
        '''
        transition_probs = self.transition_probabilities.get(current_state)
        if transition_probs is None:
            raise ValueError(f"No transition probabilities defined for state {current_state}.")
        
        match current_state:
            case States.IDLE:
                transition_probs = self._handle_idle_state(transition_probs)
            case States.ACTIVE:
                self.has_been_active = True  # set flag
            case States.OFF:
                self.has_been_active = False # reset flag
            case _:
                pass

        return random.choices(
                population=list(transition_probs.keys()),
                weights=list(transition_probs.values())
            )[0]

    def _handle_idle_state(self, transition_probs: Dict[States, float]) -> Dict[States, float]:
        '''
        This function handles the idle state.

        Args:
            transition_probs (Dict[States, float]): The transition probabilities.

        Outputs:
            transition_probs (Dict[States, float]): The transition probabilities.
        '''
        if self.has_been_active:
            return transition_probs
        else:
            return {state: prob for state, prob in transition_probs.items() if state != States.OFF}


class IntervalGenerator:
    '''
    This class generates the intervals for the time series data.
    '''
    def __init__(
        self,
        state_duration_map: Dict[States, Tuple[int, int]],
        random_seed: int = None
    ):
        self._validate_inputs(state_duration_map)
        self.state_duration_map = state_duration_map
        if random_seed is not None:
            random.seed(random_seed)

    @staticmethod
    def _validate_inputs(state_duration_map: Dict[States, Tuple[int, int]]) -> None:
        '''
        This function validates the inputs to the IntervalGenerator class.

        Args:
            state_duration_map (Dict[States, Tuple[int, int]]): The state duration map.
        '''
        if not all(isinstance(state, States) and isinstance(duration, tuple) 
                    and len(duration) == 2 and all(isinstance(d, int) for d in duration) 
                    for state, duration in state_duration_map.items()):
            raise ValueError("State duration map must be a dictionary of States mapped to tuples of two integers.")

    def get_duration_for_state(self, state: States) -> int:
        '''
        This function returns a duration for the given state based on the state duration map.

        Args:
            state (States): The current state.

        Outputs:
            duration (int): The duration for the state.
        '''
        duration_range = self.state_duration_map.get(state)
        if duration_range is None:
            raise ValueError(f"No duration range defined for state {state}.")
        return int(np.random.uniform(duration_range[0], duration_range[1]))

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

    Args:
        rms_ranges (Dict[States, Tuple[str, float, float]]): The RMS ranges and distribution type for each state,
        note that if uniform or lognormal distributions are chosen, the (min, max) values become (mu, sigma).
    '''
    def __init__(
        self,
        rms_ranges: Dict[States, Tuple[str, float, float]],
        random_seed: int = None
    ):
        self._validate_inputs(rms_ranges)
        self.rms_ranges = rms_ranges
        if random_seed is not None:
            random.seed(random_seed)

    @staticmethod
    def _validate_inputs(rms_ranges: Dict[States, Tuple[str, float, float]]) -> None:
        '''
        This function validates the inputs to the RMSGenerator class.

        Args:
            rms_ranges (Dict[States, Tuple[str, float, float]]): The RMS ranges and distribution type for each state.
        '''
        if not all(isinstance(state, States) and isinstance(rms_info, tuple) 
                    and len(rms_info) == 3 and isinstance(rms_info[0], str) 
                    and all(isinstance(num, (int, float)) for num in rms_info[1:]) 
                    for state, rms_info in rms_ranges.items()):
            raise ValueError("""RMS ranges must be a dictionary of States mapped to tuples containing a string (distribution-type)
                and two numbers (either (min, max) or (mu, sigma) depending on distribution type)."""
            )

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
        match rms_range[0]:
            case 'normal':
                return np.random.normal(rms_range[1], rms_range[2])
            case 'uniform':
                return np.random.uniform(rms_range[1], rms_range[2])
            case 'lognormal':
                return np.random.lognormal(rms_range[1], rms_range[2])
            case _:
                raise ValueError(f'RMS Generator failed for state {current_state}.')


class DataGenerator:
    '''
    This class generates time series data for a given set of states and rms ranges.

    Args:
        state_generator (StateGenerator): The state generator.
        rms_generator (RMSGenerator): The RMS generator.
        interval_generator (IntervalGenerator): The interval generator.
        kalman_filter (KalmanFilter): The Kalman filter to smooth the RMS values.
    '''
    def __init__(
        self,
        state_generator: StateGenerator,
        rms_generator: RMSGenerator,
        interval_generator: IntervalGenerator,
        kalman_filter: KalmanFilter = None,
        random_seed: int = None
    ):
        self.state_generator = state_generator
        self.rms_generator = rms_generator
        self.interval_generator = interval_generator
        self.kalman_filter = kalman_filter
        if random_seed is not None:
            random.seed(random_seed)

    @staticmethod
    def _validate_inputs(
        start_date: datetime,
        end_date: datetime,
        freq: str
    ) -> None:
        '''
        This function verifies the inputs to the generate_time_series_data function.

        Args:
            start_date (datetime): The start date of the dataset.
            end_date (datetime): The end date of the dataset.
            freq (str): The frequency of the dataset.

        Raises:
            ValueError: If start_date >= end_date.
            ValueError: If start_date or end_date is not a datetime object.
            ValueError: If freq is not a valid pandas date frequency.
        '''
        if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
            raise ValueError("Start date and end date must be datetime objects.")

        if start_date >= end_date:
            raise ValueError("Start date must be before end date.")
        
        try:
            pd.Timedelta(freq)
        except ValueError:
            raise ValueError("Frequency must be a valid pandas date frequency string.")

        if pd.Timedelta(freq).total_seconds() < 1:
            raise ValueError("Frequency must be at least 1 second.")
        
        if pd.Timedelta(freq) > end_date - start_date:
            raise ValueError("Frequency must be less than the time between start date and end date.")

    def _generate_data_for_interval(
        self,
        current_state: States,
        current_timestamp: datetime,
        freq: str, interval: int
    ) -> List[Tuple[float, States, datetime]]:
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
        freq: str = '10S',
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
            freq (str): The frequency of the dataset.

        Outputs:
            time_series_df (pd.DataFrame): The generated dataset.
        '''
        self._validate_inputs(start_date, end_date, freq)
        
        time_series_data = []
        current_state = States.OFF
        current_state = self.state_generator.generate_state(current_state)
        current_timestamp = start_date

        # Generate data for each interval
        while current_timestamp < end_date:
            # Generate the current interval for the state
            interval = self.interval_generator.get_duration_for_state(current_state)
            # Generate the data for the interval
            interval_data = self._generate_data_for_interval(current_state, current_timestamp, freq, interval)
            # Extend the time series data with the interval data
            time_series_data.extend(interval_data)
            # Update the current timestamp and state
            current_timestamp = interval_data[-1][-1] + pd.Timedelta(freq)
            # Generate a new state
            current_state = self.state_generator.generate_state(current_state)

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
        if self.kalman_filter:
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
        buffer_size = 5  # The size of the buffer, adjust as needed
        time_series_df[RMS] = time_series_df.groupby('group')[RMS].transform(lambda x: x.rolling(buffer_size, min_periods=1).mean())

        # Apply the Kalman filter to each group separately
        rms_smoothed = time_series_df.groupby('group').apply(lambda group: self._apply_kalman_filter_to_group(group[RMS]))
        print(f'rms_smoothed_initial: {rms_smoothed.head()}')
        print(type(rms_smoothed))

        rms_smoothed = rms_smoothed.reset_index(level=0, drop=True)
        print(f'rms_smoothed_reset: {rms_smoothed.head()}')
        if isinstance(rms_smoothed, pd.Series):
            return rms_smoothed
        elif len(rms_smoothed.columns) > 1:
            # If the Series has a MultiIndex, drop the first level
            rms_smoothed = rms_smoothed.droplevel(0)
            print(f'rms_smoothed: {rms_smoothed}')

        return rms_smoothed

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