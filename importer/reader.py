from pathlib import Path
from typing import Tuple, List
import json

from tqdm import tqdm
import numpy as np

from settings import DATASET_RAW_DATA_DIRECTORY, DATASET_RAW_DATA_FILE


class DataReader:
    def __init__(self) -> None:
        self.size = 0
        self.books = np.array([], dtype=np.float32)

    def load(self) -> int:
        file_path = Path(DATASET_RAW_DATA_DIRECTORY) / DATASET_RAW_DATA_FILE

        def transform(orders) -> Tuple[List[float], List[float]]:
            prices = [order["price"] for order in orders]
            quantities = [order["quantity"] for order in orders]
            return prices, quantities

        size = 0
        books = []

        with open(file_path) as data:
            for line in tqdm(data):
                row = json.loads(line)

                bids = transform(row["bids"])
                asks = transform(row["asks"])
                snapshot = [bids, asks]

                try:
                    books.append(snapshot)
                except ValueError as e:
                    print("Error reading order-book number")
                    books.append(books[-1])

                size += 1

        self.size = size
        self.books = np.array(books, dtype=np.float32)

        return size