# regression_model.py

from typing import List, Tuple
import numpy as np
from sklearn.linear_model import LinearRegression


class TravelTimeRegressor:
    """
    Linear Regression model to predict travel time from distance.
    You can train it with (distance, time) pairs.
    """

    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False

    def train(self, data: List[Tuple[float, float]]):
        """
        data: list of (distance, time)
        """
        if not data:
            raise ValueError("No data provided for training.")

        X = np.array([[d] for d, _t in data])  # distance as feature
        y = np.array([t for _d, t in data])    # time as target

        self.model.fit(X, y)
        self.is_trained = True

    def predict_time(self, distance: float) -> float:
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")
        return float(self.model.predict(np.array([[distance]]))[0])
