import numpy as np
from sklearn.linear_model import LinearRegression

# Provided series
series = np.array([2586.47, 5396.37, 5055.03, 1737.02, 8213.81, 20707.11, 10535.09, 16540.57, 17619.01, 36261.43, 15119.69, 28173.5, 11052.07])

# Provided functions
def slope_to_angle(slope):
    return np.degrees(np.arctan(slope))

def calculate_trend(series):
    """Calculate the slope of the trend line for the series."""
    time = np.arange(len(series)).reshape(-1, 1)

    model = LinearRegression().fit(time, series)
    return model.coef_[0]

# Calculate slope and then convert to angle
slope = calculate_trend(series)
angle = slope_to_angle(slope)

print(angle)