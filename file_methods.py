import os
import csv
import json
from neural_network import CNN


def load_config(file_name: str) -> CNN:
    if not os.path.isfile(file_name):
        raise Exception("Config file not found.")

    with open(file_name, "r") as file:
        config = json.load(file)

    return CNN(
        image_width=config["image_width"],
        image_height=config["image_height"],
        layers_nodes_count=config["layers_nodes_count"],
        final_objects=config["final_objects"]
    )
