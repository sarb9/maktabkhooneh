class MidPriceExtractor:
    name = "mid_price"

    def __init__(self, dataset) -> None:
        self.dataset = dataset

        self.feature_data = (self.dataset[:, 1, 0, 0] + self.dataset[:, 0, 0, 0]) / 2
