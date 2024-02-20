import json
import pickle
import math
import random
from typing import Optional


class CNN:
    image_width: int
    image_height: int
    input_layer_size: int
    final_objects: list[str]
    final_objects_count: int
    layers_count: int
    layers_nodes_count: list[int]

    learning_rate: float
    weights: list[list[list[float]]]
    values: list[list[float]]

    dataset_dir: str
    out_file: str

    def __init__(self,
                 config_file: str,
                 dataset_dir: str,
                 out_file: str,
                 weights_file: Optional[str] = None):
        with open(config_file, "r") as file:
            data = json.load(file)["config"]
            image_width: int = data["image_width"]
            image_height: int = data["image_width"]
            layers_nodes_count: list[int] = data["layers_nodes_count"]
            final_objects: list = data["final_objects"]
            self.learning_rate = data["learning_rate"]

        if not (28 <= image_width <= 720 and 28 <= image_height <= 720):
            raise Exception("Image width and height should be in range [28, 720].")

        if any(list(map(lambda x: x < 1, layers_nodes_count))):
            raise Exception("There must be at least one neuron in each layer.")

        if len(final_objects) < 3:
            raise Exception("Insufficient number of recognized objects (minimum 3).")

        self.image_width = image_width
        self.image_height = image_height
        self.input_layer_size = image_width * image_height

        self.final_objects = final_objects
        self.final_objects_count = len(final_objects)

        self.layers_count = len(layers_nodes_count) + 2
        self.layers_nodes_count = [
            self.input_layer_size,
            *layers_nodes_count,
            self.final_objects_count
        ]

        self.values = [[0 for _ in range(self.layers_nodes_count[layer])] for layer in range(self.layers_count)]
        self.dataset_dir = dataset_dir
        self.out_file = out_file

        if weights_file:
            self.__load_weights(weights_file=weights_file)
        else:
            self.__fill_weights()

        self.__save_weights(self.out_file)

    def __fill_weights(self):
        self.weights = list(list(list(random.uniform(0, 1)
                                      for _ in range(self.layers_nodes_count[layer + 1]))
                                 for _ in range(self.layers_nodes_count[layer]))
                            for layer in range(self.layers_count - 1))

    def __load_weights(self, weights_file: str):
        with open(weights_file, "rb") as file:
            self.weights = pickle.load(file)

    def __save_weights(self, weights_file: str, after_epoch: Optional[int] = None):
        if after_epoch:
            weights_file = "".join(weights_file.split(".")[:-1]) + f"_epoch{after_epoch}.pickle"
        with open(weights_file, "wb") as file:
            pickle.dump(self.weights, file)

    @staticmethod
    def __activation_f(x: float) -> float:
        return 1 / (1 + math.e ** -x)

    @staticmethod
    def __activation_f_der(x: float) -> float:
        return CNN.__activation_f(x) * (1 - CNN.__activation_f(x))

    @staticmethod
    def __mse(expected: list[float]):
        return sum(list(map(lambda x: (1 - x) ** 2, expected))) / len(expected)

    def __feed_forward(self):
        for layer in range(1, self.layers_count):
            for relation in range(self.layers_nodes_count[layer - 1]):
                for node in range(self.layers_nodes_count[layer]):
                    self.values[layer][node] += (
                        self.__activation_f(self.values[layer - 1][relation] *
                                            self.weights[layer - 1][relation][node]))

    def __back_prop(self):
        pass

    def __iteration(self):
        pass

    def train(self):
        pass

    def __str__(self):
        return ""

    def __repr__(self):
        return ""

    def __call__(self):
        return ""
