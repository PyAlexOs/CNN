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
    from neural_network import CNN
    main()
