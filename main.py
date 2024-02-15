from typing import Any


class CNN:
    image_width: int
    image_height: int
    input_layer_size: int
    final_objects: list[Any]
    final_objects_count: int
    layers_count: int
    layers_nodes_count: list[int]
    weights: list[[[int], ...], ...]
    values: list[[int], [int]]

    def __init__(self,
                 image_width: int,
                 image_height: int,
                 layers_nodes_count: list[int],
                 final_objects: list[Any]):
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

    def __call__(self):
        # check weights
        pass

    """def __str__(self):
        # represent
        pass"""

    def __repr__(self):
        # to save in file
        pass

    @staticmethod
    def activation_f(x: float) -> float:
        return 1 / (1 + math.e ** -x)

    @staticmethod
    def activation_f_der(self, x: float) -> float:
        return self.activation_f(x) * (1 - self.activation_f(x))

    def fill_weights(self):
        for layer in range(self.layers_count):
            for node in range(self.layers_nodes_count[layer]):
                self.weights[layer][node] = random.uniform(0, 1)

    def load_weights(self, input: str):
        with open(input, 'r') as file:
            aux = ""
            for param in self.__dict__.keys():
                while temp := file.read(1) != "\n":
                    aux += temp

                if aux != "":
                    param = int(aux)

            for layer in self.weights:
                for node in layer:
                    aux = ""
                    while temp := file.read(1) != "\n":
                        aux += temp

                    if aux != "":
                        self.weights[layer][node] = int(aux)

    def save(self, out: str):
        with open(out, 'w') as file:
            for param in self.__dict__.values():
                file.write(str(param) + "\n")

            for layer in self.weights:
                for node in layer:
                    file.write(str(node) + "\n")

    def train(self):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", default="files")
    parser.add_argument("-l", "--load", default=None)
    parser.add_argument("-o", "--out", default="out.dat")
    named_args = parser.parse_args(sys.argv[2:])

    params = {
        "image_height": 64,
        "image_width": 64,
        "layers_nodes_count": [
            2048,
            1024,
            512,
            256,
            128,
            128,
            64,
            64,
            32,
            32
        ],
        "final_objects_count": 10
    }
    neural_network = CNN(*params.values())

    dataset_dir = named_args.directory
    if not os.path.isdir(dataset_dir):
        exit("Dataset not found.")

    out_file = named_args.out
    if named_args.load:
        if os.path.isfile(named_args.load):
            neural_network.load_weights()
        else:
            exit("File with weights not found.")
    else:
        neural_network.fill_weights()


# python neural_network.py
if __name__ == '__main__':
    import os
    import sys
    import argparse
    import math
    import random
    main()
