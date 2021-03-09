import numpy as np


class VolatilityExtractor:
    name = "volatility"

    def __init__(
        self,
        dataset: np.ndarray,
        price_changes: np.ndarray,
        period: int = 60,
    ) -> None:
        self.dataset = dataset
        self.period = period

        number_of_samples = len(price_changes)

        volatilities = np.zeros((number_of_samples,))

        for i in range(number_of_samples):
            start = max(0, i - self.period)
            volatilities[i] = np.var(price_changes[start:i])
        volatilities[0] = volatilities[1]

        self.feature_data = volatilities