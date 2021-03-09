import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from dataset.raw_dataset import RawDataset
from feature_extraction import (
    MidPriceExtractor,
    PriceChangeExtractor,
    SpreadExtractor,
    VolatilityExtractor,
    AskVolumeExtractor,
    BidVolumeExtractor,
    VolumeDifferenceExtractor,
)


def plot(features):
    plt.figure(figsize=(20, 6))

    ax = sns.violinplot(data=features)
    ax.set_xticklabels(
        [
            MidPriceExtractor.name,
            PriceChangeExtractor.name,
            SpreadExtractor.name,
            VolatilityExtractor.name,
            AskVolumeExtractor.name,
            BidVolumeExtractor.name,
            VolumeDifferenceExtractor.name,
        ],
        rotation=90,
    )

    plt.show()


if __name__ == "__main__":
    raw_dataset = RawDataset()
    size = raw_dataset.load()

    mid_price_extractor = MidPriceExtractor(raw_dataset.books)
    price_change_extractor = PriceChangeExtractor(
        raw_dataset.books,
        mid_price_extractor.feature_data,
    )
    spread_extractor = SpreadExtractor(
        raw_dataset.books, mid_price_extractor.feature_data
    )
    volatility_extractor = VolatilityExtractor(
        raw_dataset.books,
        price_change_extractor.feature_data,
    )

    ask_volume_extractor = AskVolumeExtractor(raw_dataset.books)
    bid_volume_extractor = BidVolumeExtractor(raw_dataset.books)
    voluem_difference_extractor = VolumeDifferenceExtractor(
        raw_dataset.books,
        ask_volume_extractor.feature_data,
        bid_volume_extractor.feature_data,
    )

    features = np.vstack(
        (
            mid_price_extractor.feature_data,
            price_change_extractor.feature_data,
            spread_extractor.feature_data,
            volatility_extractor.feature_data,
            ask_volume_extractor.feature_data,
            bid_volume_extractor.feature_data,
            voluem_difference_extractor.feature_data,
        )
    ).transpose()

    plot(features)

    features_mean = np.mean(features, axis=1)
    features_std = np.std(features, axis=1)

    features = (features - features_mean[:, np.newaxis]) / features_std[:, np.newaxis]

    plot(features)
