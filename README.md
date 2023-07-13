[![Build Status](https://travis-ci.org/Danielwoodh/MachineMax.svg?branch=time-series-dan)](https://travis-ci.org/Danielwoodh/MachineMax)
[![Coverage Status](https://coveralls.io/repos/github/Danielwoodh/MachineMax/badge.svg?branch=time-series-dan)](https://coveralls.io/github/Danielwoodh/MachineMax?branch=time-series-dan)

# Class Descriptions

This document provides a comprehensive overview of the following classes: `DataGenerator`, `StateGenerator`, `IntervalGenerator`, and `RMSGenerator`.

## `DataGenerator` Class

The `DataGenerator` class is a robust tool designed to generate time series data based on various user-defined states, intervals, and RMS (Root Mean Square) values. The generated data is ideal for data analysis, machine learning model training, or any application that relies on time series data.

The `DataGenerator` class accomplishes this by taking instances of three generators and a filter:

1. `StateGenerator` - Generates the various states.
2. `RMSGenerator` - Generates the RMS values.
3. `IntervalGenerator` - Determines the duration of each state.
4. `KalmanFilter` - Optional filter used for smoothing the RMS values.

A unique feature of this class is the use of a random seed to ensure the repeatability of the generated data, which can be vital in research and testing.

### Core Methods

Here are the main functions used within this class:

- `generate_time_series_data`: This is the primary method for generating the complete time series dataset based on user-specified start and end dates and a frequency. The generated data is then transformed into a pandas DataFrame for easy manipulation and analysis.

- `_validate_inputs`: A helper function to verify the correctness of the inputs to the `generate_time_series_data` function. Raises an exception if the inputs do not meet the specified conditions.

- `_generate_data_for_interval`: A helper function for generating data for each interval.

- `_create_dataframe`: A function for converting the generated data into a pandas DataFrame. If a Kalman filter is provided, it applies this filter to smooth the data in the 'rms' column of the DataFrame.

- `_apply_kalman_filter`: A function for applying the Kalman filter to the 'rms' column of the DataFrame. It applies the filter separately to each continuous state in the DataFrame.

- `_apply_kalman_filter_to_group`: A helper function for applying the Kalman filter to a single group of 'rms' values.

## `StateGenerator` Class

The `StateGenerator` class is responsible for determining the state for each step in the time series data. The state transition is achieved by randomly picking the next state based on the current state and pre-defined transition probabilities. A key feature of this class is the handling of IDLE states - if the system was ACTIVE before it went IDLE, it can't transition to OFF directly, adding a level of realism to the data generated.

## `IntervalGenerator` Class

The `IntervalGenerator` class determines the duration for each state in the time series data. It generates a random duration within a pre-defined range for each state. The generated duration is then translated into the number of time steps based on the frequency of the time series data, ensuring a diverse and dynamic generation of time-series data.

## `RMSGenerator` Class

The `RMSGenerator` class handles the generation of RMS values for each state of the system. The RMS values are generated randomly within a pre-defined range for each state, using a specified distribution (normal, uniform, or lognormal). This allows the `RMSGenerator` to simulate different types of fluctuations in RMS values depending on the state of the system, enhancing the flexibility and usefulness of the data generated.

# Ideas

- Rather than having randomly selected values, have a probability distribution so that long extended periods of a state are less likely
- This maybe should only be applied to IDLE and ACTIVE states?

- Add randomness to the 'ACTIVE' state, at each time-step, do a check to see if it switches to 'IDLE' based
on a probability, if it does, determine a random interval to be at 'IDLE' (should be low, 2-10 time-steps)
before returning to 'ACTIVE'. Probability of switching should be low (0.1% ?).

# Next Steps:

1. Fix the ROC_AUC curve (plot 1 line per class) (DONE)
2. Add Descriptions for each model, and add limitations + improvements for each (1/2 DONE)
3. Transfer continuous detection to models.py
4. Write unit-tests for the models (1/2 DONE)
5. Complete Stretch goals (1/2 DONE)