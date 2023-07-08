from datetime import datetime, timedelta
import pandas as pd
import random

STATES = ['OFF', 'IDLE', 'ACTIVE']

def generate_time_series_data(start_date, end_date, freq='1S', min_duration=5, max_duration=60):
    """
    Generate time series data for machine states with corresponding acceleration magnitudes.
    Args:
        start_date (str): Start date for the time series data in the format 'YYYY-MM-DD HH:MM:SS'.
        end_date (str): End date for the time series data in the format 'YYYY-MM-DD HH:MM:SS'.
        freq (str): Frequency for the time series data. Default is '1S' (i.e., one data point per second).
        min_duration (int): Minimum duration for each state in seconds. Default is 5 seconds.
        max_duration (int): Maximum duration for each state in seconds. Default is 60 seconds.
    Returns:
        df (pd.DataFrame): DataFrame containing the time series data.
    """
    start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
    end_date = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    data = {'timestamp': date_range}
    df = pd.DataFrame(data)

    # Generate machine states and corresponding acceleration magnitudes
    states = []
    rms_values = []
    current_time = start_date
    while current_time < end_date:
        current_state = random.choice(STATES)
        current_rms = 0 if current_state == 'OFF' else (random.uniform(1, 300) if current_state == 'IDLE' else random.uniform(301, 600))
        state_end_time = current_time + timedelta(seconds=random.randint(min_duration, max_duration))
        states.append(current_state)
        rms_values.append(current_rms)
        current_time = state_end_time

    df['state'] = states
    df['rms'] = rms_values
    
    return df