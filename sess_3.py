from numpy import number
from settings import NORMALIZATION_NUMBER_OF_BUCKETS
from dataset.raw_dataset import RawDataset
from normalization.price_normalizer import PriceNormalizer


if __name__ == "__main__":
    raw_dataset = RawDataset()
    size = raw_dataset.load()

    number_of_buckets = NORMALIZATION_NUMBER_OF_BUCKETS
    normalizer = PriceNormalizer(raw_dataset.books, number_of_buckets)

    normalizer.plot_buckets()

    normalizer.sample_plot()
