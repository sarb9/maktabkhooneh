import numpy as np

from dataset.raw_dataset import RawDataset
from feature_extraction import (
    VolatilityExtractor,
    MidPriceExtractor,
    PriceChangeExtractor,
)
from dataset.trainable_dataset import TrainableDataset
from settings import NORMALIZATION_NUMBER_OF_BUCKETS
from normalization.price_normalizer import PriceNormalizer

if __name__ == "__main__":
    raw_dataset = RawDataset()
    size = raw_dataset.load()

    mid_price_extractor = MidPriceExtractor(raw_dataset.books)
    price_change_extractor = PriceChangeExtractor(
        raw_dataset.books,
        mid_price_extractor.feature_data,
    )
    volatility_extractor = VolatilityExtractor(
        raw_dataset.books,
        price_change_extractor.feature_data,
    )

    labels = volatility_extractor.feature_data

    normalizer = PriceNormalizer(raw_dataset.books, NORMALIZATION_NUMBER_OF_BUCKETS)

    features = normalizer.bucket_books.reshape(normalizer.bucket_books.shape[0], -1)

    trainable_dataset = TrainableDataset(
        features=features,
        label=labels,
        input_width=10,
        label_width=1,
        shift=2,
        train_portion=0.8,
        validation_portion=0.1,
        test_portion=0.1,
    )

    trainable_dataset.plot()
