import json
import csv
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

    weights: list[list[list[float]]]
    values: list[[float], [float]]

    def __init__(self,
                 config_file: str,
                 weights_file: Optional[str] = None):
        with open(config_file, "r") as file:
            data = json.load(file)["config"]
            image_width: int = int(data["image_width"])
            image_height: int = int(data["image_width"])
            layers_nodes_count: list[int] = list(map(lambda x: int(x), data["layers_nodes_count"]))
            final_objects: list = list(data["final_objects"])

        if not (32 <= image_width <= 3840 and 32 <= image_height <= 3840):
            raise Exception("Image width and height should be in range [32, 3840].")

        if any(list(map(lambda x: x < 1, layers_nodes_count))):
            raise Exception("There must be at least one neuron in each layer.")

        if len(final_objects) < 3:
            raise Exception("Insufficient number of recognized objects (minimum 3).")

        self.image_width = image_width
        self.image_height = image_height
        self.input_layer_size = image_width * image_height  # input layer dimension depends on the size of an image

        self.final_objects = final_objects
        self.final_objects_count = len(final_objects)

        self.layers_count = len(layers_nodes_count) + 2
        self.layers_nodes_count = [
            self.input_layer_size,
            *layers_nodes_count,
            self.final_objects_count
        ]

        if weights_file:
            self.__load_weights(weights_file=weights_file)
        else:
            self.__fill_weights()

    def __fill_weights(self):
        self.weights = list(list(list(random.uniform(0, 1)
                                      for _ in range(self.layers_nodes_count[layer + 1]))
                                 for _ in range(self.layers_nodes_count[layer]))
                            for layer in range(self.layers_count - 1))

    def __load_weights(self, weights_file: str):
        with open(weights_file, "r") as file:
            reader = csv.reader(file)
            for line in reader:
                print(line)

        self.weights = list(list(list(random.uniform(0, 1)
                                      for _ in range(self.layers_nodes_count[layer + 1]))
                                 for _ in range(self.layers_nodes_count[layer]))
                            for layer in range(self.layers_count - 1))

    @staticmethod
    def __activation_f(x: float) -> float:
        return 1 / (1 + math.e ** -x)

    @staticmethod
    def __activation_f_der(self, x: float) -> float:
        return self.activation_f(x) * (1 - self.activation_f(x))

    def __str__(self):
        return '#' * 80

    def __repr__(self):
        return "rdfgdfgvdf"

    def __call__(self):
        # check weights and recognize
        pass

    def train(self):
        pass
