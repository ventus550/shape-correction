import keras
import numpy as np
import tensorflow as tf
from pathlib import Path
from shapely.geometry import Polygon


def save(model, path: Path | str, metadata={}, frozen=False):
    path = Path(path)
    if frozen:
        metadata["frozen"] = True
        model = keras.models.Model(inputs=model.input, outputs=model.output)
        model.trainable = False
    weights = model.get_weights()
    config = model.get_config()
    config["name"] = {"name": config["name"], **metadata}
    model = tf.keras.models.Model.from_config(config)
    model.set_weights(weights)
    model.save(path)
    return model


def load(path: Path | str):
    path = Path(path)
    model = keras.models.load_model(path)
    model.meta = model.get_config()["name"]
    model._name = path.name
    return model


def IoU(label, pred, nb_vertices=4):
    y_polygon = Polygon(label.reshape(nb_vertices, 2))
    pred_polygon = Polygon(pred.reshape(nb_vertices, 2))

    I = y_polygon.intersection(pred_polygon).area
    U = y_polygon.union(pred_polygon).area
    return I / U


def dice(label, pred, nb_vertices=4):
    y_polygon = Polygon(label.reshape(nb_vertices, 2))
    pred_polygon = Polygon(pred.reshape(nb_vertices, 2))

    I = y_polygon.intersection(pred_polygon).area
    return 2 * I / (y_polygon.area + pred_polygon.area)


def draw_data_point(x, y, p, axs, size=70):
    y *= size
    p *= size
    axs.imshow(
        x,
        cmap="gray",
    )
    axs.scatter(y[:, 0], y[:, 1])
    axs.scatter(p[:, 0], p[:, 1])


class DataGenerator(keras.utils.Sequence):
    def __init__(
        self,
        data: list[np.ndarray, np.ndarray] | Path | str,
        transform=lambda x, y: (x, y),
        batch_size: int = 32,
        shuffle: bool = True,
    ):
        data_array = data
        if isinstance(data, Path) or isinstance(data, str):
            data_array = np.load(data).values()

        X, Y = data_array
        assert len(X) == len(Y)

        self.data = data_array
        self.transform = lambda x: transform(*x)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(X))
        self.on_epoch_end()

    def __len__(self):
        return len(self.data[0]) // self.batch_size

    def __getitem__(self, index):
        pointer = index * self.batch_size
        batch = self.indices[pointer : pointer + self.batch_size]
        return self.__get_data(batch)

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __get_data(self, batch):
        X, Y = self.data
        XB, YB = X[batch], Y[batch]
        XB, YB = list(zip(*map(self.transform, zip(XB, YB))))
        return np.array(XB), np.array(YB)
