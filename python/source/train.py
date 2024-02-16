def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config/config.json",
                        help="file with neural network configuration")
    parser.add_argument("-w", "--weights", default=None,
                        help="file with weights that will be used as the initial")
    parser.add_argument("-d", "--dataset", default="../dataset",
                        help="directory with the training sample")
    parser.add_argument("-o", "--out", default="config/weights",
                        help="directory where files with weights will be saved during the training")
    named_args = parser.parse_args(sys.argv[1:])

    config_file: str = named_args.config
    if not (config_file.endswith(".json") and os.path.isfile(config_file)):
        raise Exception("Incorrect config file name given.")

    weights_file = named_args.weights
    if weights_file:
        weights_file = str(weights_file)
        if not (weights_file.endswith(".csv") and os.path.isfile(weights_file)):
            raise Exception("Incorrect weights file name given.")

    dataset_dir: str = named_args.dataset
    if not os.path.isdir(dataset_dir):
        raise Exception("Dataset directory not found.")

    out = named_args.out
    if not (out.endswith(".csv")):
        raise Exception("Incorrect out file name given.")

    neural_network = CNN(
        config_file=config_file,
        weights_file=weights_file
                         )


# python source/train.py
if __name__ == '__main__':
    import os
    import sys
    import argparse
    from neural_network import CNN
    main()
