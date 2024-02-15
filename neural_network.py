import math
import os.path
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
    weights: list[[[int], ...], ...]
    values: list[[int], [int]]

    def __init__(self,
                 image_width: int,
                 image_height: int,
                 layers_nodes_count: list[int],
                 final_objects: list[str],
                 weights_file: Optional[str]):
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

        if not weights_file:
            self.__fill_weights()

        if not os.path.isfile(weights_file):
            raise Exception("Weights file not found.")

        self.__load_weights()

    def __fill_weights(self):
        for layer in range(self.layers_count):
            for node in range(self.layers_nodes_count[layer]):
                self.weights[layer][node] = random.uniform(0, 1)

    def __load_weights(self):
        pass

    @staticmethod
    def __activation_f(x: float) -> float:
        return 1 / (1 + math.e ** -x)

    @staticmethod
    def __activation_f_der(self, x: float) -> float:
        return self.activation_f(x) * (1 - self.activation_f(x))

    def __str__(self):
        # represent
        pass

    def __call__(self):
        # check weights and recognize
        pass

    def train(self):
        pass
