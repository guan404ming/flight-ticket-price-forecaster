import numpy as np


class EchoRegressor:
    """
    Dummpy predictor
    """
    def __init__(self, lags: int, y_steps: int, series_num: int = 2) -> None:
        self.lags = lags
        self.y_steps = y_steps
        self.series_num = series_num

    def predict(self, X: np.ndarray):
        y_pred = np.zeros([X.shape[0], self.y_steps * self.series_num], dtype=np.float32)
        for i in range(self.series_num):
            y_pred[:, i*self.y_steps:(i+1)*self.y_steps] += X[:, (i+1)*self.lags - 1].reshape(-1, 1)

        return y_pred
