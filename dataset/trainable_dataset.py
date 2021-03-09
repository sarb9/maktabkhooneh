import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


class TrainableDataset:
    def __init__(
        self,
        features: np.ndarray,
        label: np.ndarray,
        input_width: int,
        label_width: int,
        shift: int,
        train_portion: float,
        validation_portion: float,
        test_portion: float,
    ) -> None:

        self.features = features
        self.labels = label
        import pdb

        pdb.set_trace()
        self.dataset = np.hstack((self.labels[:, np.newaxis], self.features))

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        assert train_portion + validation_portion + test_portion == 1

        self.train_portion = train_portion
        self.validation_portion = validation_portion
        self.test_portion = test_portion

        self.size = len(self.dataset)

        self.split_data()
        self.create_window_slices()

    def split_data(self):

        train_start = 0
        validation_start = int(self.size * self.train_portion)
        test_start = int(self.size * (self.train_portion + self.validation_portion))

        self.train_data_frame = self.dataset[train_start:validation_start]
        self.val_data_frame = self.dataset[validation_start:test_start]
        self.test_data_frame = self.dataset[test_start:]

    def create_window_slices(self):

        self.total_window_size = self.input_width + self.shift
        window = np.arange(self.total_window_size)

        self.input_slice = slice(0, self.input_width)
        self.input_indices = window[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = window[self.labels_slice]

    def plot(self, model=None, max_subplots=4, plot_col=0):
        inputs, labels = self.example

        mpl.rcParams["axes.grid"] = True
        plt.figure(figsize=(12, 8))

        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f"{plot_col} [normed]")

            plt.plot(
                self.input_indices,
                inputs[n, :, plot_col],
                label="Inputs",
                marker=".",
                zorder=-10,
            )

            self.label_columns = None
            if self.label_columns:
                label_col = None if plot_col not in self.label_columns else plot_col
            else:
                label_col = plot_col

            if label_col is None:
                continue

            plt.scatter(
                self.label_indices,
                labels[n, :, label_col],
                edgecolors="k",
                label="Labels",
                c="#2ca02c",
                s=64,
            )

            if model is not None:
                predictions = model(inputs)
                plt.scatter(
                    self.label_indices,
                    predictions[n, :, label_col],
                    marker="X",
                    edgecolors="k",
                    label="Predictions",
                    c="#ff7f0e",
                    s=64,
                )

            if n == 0:
                plt.legend()

        plt.xlabel("Time [h]")
        mpl.rcParams["axes.grid"] = False

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]

        labels = features[:, self.labels_slice, 0]
        labels = tf.expand_dims(labels, -1)

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def create_tensorflow_dataset(self, data, stride=1, batch_size=64):
        data = np.array(data, dtype=np.float32)
        dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=stride,
            shuffle=True,
            batch_size=batch_size,
        )

        dataset = dataset.map(self.split_window)

        return dataset

    @property
    def train(self):
        return self.create_tensorflow_dataset(self.train_data_frame)

    @property
    def val(self):
        return self.create_tensorflow_dataset(self.val_data_frame)

    @property
    def test(self):
        return self.create_tensorflow_dataset(self.test_data_frame, stride=60)

    @property
    def plot_data(self):
        data = np.array(self.val_data_frame, dtype=np.float32)

        dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=128,
        )

        dataset = dataset.map(self.split_window)

        return dataset

    @property
    def example(self):
        result = getattr(self, "_example", None)
        if result is None:
            result = next(iter(self.train))
            self._example = result
        return result