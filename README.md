# Ideas

- Rather than having randomly selected values, have a probability distribution so that long extended periods of a state are less likely
- This maybe should only be applied to IDLE and ACTIVE states?

- Add randomness to the 'ACTIVE' state, at each time-step, do a check to see if it switches to 'IDLE' based
on a probability, if it does, determine a random interval to be at 'IDLE' (should be low, 2-10 time-steps)
before returning to 'ACTIVE'. Probability of switching should be low (0.1% ?).