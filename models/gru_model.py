import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from settings import (
    TRAIN_MAX_EPOCH,
    MODEL_CHECKPOINT_DIRECTORY,
    MODEL_CHECKPOINT_FILE_FORMAT,
)
from dataset.feature_extraction import (
    bucket_book,
    price_change,
    spread,
    volatility,
    volume,
    activity,
)


class GruModel:

    checkpoint_path = MODEL_CHECKPOINT_DIRECTORY + MODEL_CHECKPOINT_FILE_FORMAT

    def __init__(self, dataset, load=False) -> None:
        self.dataset = dataset

        self.tfmodel = tf.keras.models.Sequential(
            [
                # Shape [batch, time, features] => [batch, time, lstm_units]
                tf.keras.layers.GRU(128, return_sequences=False),
                # Shape => [batch, time, features]
                tf.keras.layers.Dense(units=1),
            ]
        )

        if load:
            self.load_model()

    def fit(self, patience=2):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, mode="min"
        )

        self.tfmodel.compile(
            loss=tf.losses.MeanSquaredError(),
            optimizer=tf.optimizers.Adam(),
            metrics=[
                tf.metrics.MeanAbsoluteError(),
                tf.metrics.RootMeanSquaredError(),
            ],
        )

        history = self.tfmodel.fit(
            self.dataset.train,
            epochs=TRAIN_MAX_EPOCH,
            validation_data=self.dataset.val,
            callbacks=[
                early_stopping,
            ],
        )
        return history

    @classmethod
    def feature_columns(self):
        return (
            bucket_book.BucketBookExtractor.features()
            + price_change.PriceChangeExtractor.features()
            + spread.SpreadExtractor.features()
            + volatility.VolatilityExtractor.features()
            + volume.AskVolumeExtractor.features()
            + volume.BidVolumeExtractor.features()
            + volume.VolumeDifferenceExtractor.features()
            + activity.ActivityExtractor.features()
        )

    @classmethod
    def label_column(self):
        return volatility.VolatilityExtractor.features()[0]

    def __call__(self, input):
        return self.tfmodel(input).numpy().flatten()

    def save_model(self):
        self.tfmodel.save(self.checkpoint_path)

    def load_model(self):
        self.tfmodel = tf.keras.models.load_model(self.checkpoint_path)

    def evaluate(self):
        return self.tfmodel.evaluate(self.dataset.test)

    def plot_prediction_series(self, subplots=3, length=100):
        predictions = [0] * 60
        volatilities = self.dataset.val_data_frame[self.dataset.label_column]

        for element in self.dataset.plot_data.as_numpy_iterator():
            input, label = element
            predictions += list(self.tfmodel(input))

        plt.figure(figsize=(12, 8))

        start = 0
        for n in range(subplots):
            plt.subplot(3, 1, n + 1)
            plt.ylabel(f"{self.dataset.label_column} [normed]")

            plt.plot(
                range(length),
                predictions[start : start + length],
                label="predictions",
                zorder=-10,
            )

            plt.plot(
                range(length),
                volatilities[start : start + length],
                label="target",
                c="#ff7f0e",
            )

            start += length
            if n == 0:
                plt.legend()

        plt.xlabel("Time [h]")
        plt.show()

    def plot_prediction_distribution(self):
        predictions = np.array([])
        for sample in self.test:
            prediction = self.model(sample).numpy()
            predictions = np.append(predictions, prediction)
        labels = np.array([])
        for label in self.test:
            labels = np.append(labels, label[1].numpy())

        plt.plot(predictions)
        plt.plot(labels)
        plt.show()

        bins = np.linspace(-1, 3, 200)
        alpha = 0.7
        plt.hist(labels, bins=bins, alpha=alpha)
        plt.hist(predictions, bins=bins, alpha=alpha)
        plt.show()
