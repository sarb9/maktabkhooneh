import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import indices

from tqdm import tqdm


class PriceNormalizer:
    bins = 10000

    def __init__(self, books: np.ndarray, bucket_count: int) -> None:
        self.books = books
        self.bucket_count = bucket_count

        self.mid_prices = (self.books[:, 0, 0, 0] + self.books[:, 1, 0, 0]) / 2

        self.normalized_price_books = self.books
        self.normalized_price_books[:, :, 0, :] /= self.mid_prices[
            :,
            np.newaxis,
            np.newaxis,
        ]

        self.buckets = self._calc_buckets(bucket_count)

        self.convert()

    def _calc_buckets(self, bucket_count: int) -> np.ndarray:

        price_volume_pairs = np.zeros((2, 2, self.books.shape[0] * self.books.shape[3]))

        for i in range(2):
            for j in range(self.books.shape[0]):
                for k in range(self.books.shape[3]):
                    price_volume_pairs[i, :, j * self.books.shape[3] + k] = np.array(
                        [
                            self.normalized_price_books[j, i, 0, k],
                            self.normalized_price_books[j, i, 1, k],
                        ]
                    )

        sorted_array = np.zeros((2, 2, self.books.shape[0] * self.books.shape[3]))

        print("Sorting prices...")
        sorted_array[0, :, :] = price_volume_pairs[
            0, :, np.argsort(price_volume_pairs[0, 0, :])
        ].transpose()
        sorted_array[1, :, :] = price_volume_pairs[
            1, :, np.argsort(price_volume_pairs[1, 0, :])
        ].transpose()
        print("Sorting done.")

        cumulative_array = np.copy(sorted_array)
        cumulative_array[:, 1, :] = np.cumsum(sorted_array[:, 1, :], axis=1)
        cumulative_array[:, 1, :] /= cumulative_array[:, 1, -1][:, np.newaxis]

        split_points = (1 + np.arange(bucket_count)) / bucket_count

        cut_indices = np.zeros((2, self.bucket_count), dtype=np.int64)
        cut_indices[0, :] = np.searchsorted(cumulative_array[0, 1, :], split_points)
        cut_indices[1, :] = np.searchsorted(cumulative_array[1, 1, :], split_points)

        cut_ratios = np.zeros((2, self.bucket_count))
        cut_ratios[0] = cumulative_array[0, 0, cut_indices[0]]
        cut_ratios[1] = cumulative_array[1, 0, cut_indices[1]]

        self.buckets = cut_ratios

        return cut_ratios

    def plot_buckets(self) -> None:
        bucket_length = np.diff(self.buckets, axis=1)

        fig, ax = plt.subplots(1, 1)

        plt.xlim(
            -self.bucket_count - 1,
            self.bucket_count + 1,
        )
        ax.bar(
            range(-self.bucket_count + 1, 0),
            bucket_length[0][::-1],
        )
        ax.bar(
            range(1, self.bucket_count),
            bucket_length[0],
        )

        plt.show()

    def convert_to_bucket_book(self, book):

        volume_cum = np.cumsum(book[:, 1, :], axis=1)

        indices = np.zeros((2, self.bucket_count), dtype=np.int64)
        indices[0] = np.searchsorted(book[0, 0, :][::-1], self.buckets[0], side="left")
        indices[1] = np.searchsorted(book[1, 0, :], self.buckets[1], side="left")

        indices[indices == 100] = 99

        bucket_volumes = np.zeros((2, self.bucket_count))
        bucket_volumes[0] = volume_cum[0, indices[0]]
        bucket_volumes[1] = volume_cum[1, indices[1]]

        bucket_volumes = np.diff(bucket_volumes, prepend=0, axis=1)

        return bucket_volumes

    def convert(self):
        bucket_books = np.zeros((self.books.shape[0], 2, self.bucket_count))

        for i, book in tqdm(enumerate(self.normalized_price_books)):
            bucket_books[i] = self.convert_to_bucket_book(book)

        self.bucket_books = bucket_books

    def sample_plot(self, sample_size=1000, draw_buckets=False) -> None:
        books = np.copy(self.books)
        ax_rand = np.random.choice(books.shape[0], size=sample_size)

        books = books[ax_rand, ...]

        plt.ylim((0, 400))
        plt.xlim((0.992, 1.008))

        plt.scatter(
            books[:, 0, 0, :].reshape(sample_size * books.shape[3]),
            books[:, 0, 1, :].reshape(sample_size * books.shape[3]),
            color="green",
            label="Bid",
            s=0.01,
        )

        plt.scatter(
            books[:, 1, 0, :].reshape(sample_size * books.shape[3]),
            books[:, 1, 1, :].reshape(sample_size * books.shape[3]),
            color="red",
            label="Bid",
            s=0.01,
        )

        plt.show()
