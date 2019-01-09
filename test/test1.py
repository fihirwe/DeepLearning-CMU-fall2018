#! /usr/bin/python3
import hw1 
import numpy as np
#import matplotlib.pyplot as plt
import os
import argparse
import contextlib


@contextlib.contextmanager
def numpy_print_options(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


def make_one_hot(labels_idx):
    labels = np.zeros((labels_idx.shape[0], 10))
    labels[np.arange(labels_idx.shape[0]), labels_idx] = 1
    return labels


def process_dset_partition(dset_partition, normalize=True):
    data, labels_idx = dset_partition
    mu, std = data.mean(), data.std() if normalize else (0, 1)
    return (data - mu) / std, make_one_hot(labels_idx)


def visualize(outpath):
    # Configure the training visualization process below
    # Change these hyperparameters around to experiment with your implementation
    epochs = 100
    batch_size = 100
    thisdir = os.path.dirname(__file__)
    savepath = outpath
    train_data_path = os.path.join(thisdir, "../data/train_data.npy")
    train_labels_path = os.path.join(thisdir, "../data/train_labels.npy")

    val_data_path = os.path.join(thisdir, "../data/val_data.npy")
    val_labels_path = os.path.join(thisdir, "../data/val_labels.npy")

    test_data_path = os.path.join(thisdir, "../data/test_data.npy")
    test_labels_path = os.path.join(thisdir, "../data/test_labels.npy")

    dset = (
            process_dset_partition((np.load(train_data_path), np.load(train_labels_path))),
            process_dset_partition((np.load(val_data_path), np.load(val_labels_path))),
            process_dset_partition((np.load(test_data_path), np.load(test_labels_path))))

    mlp = hw1.hw1.MLP(784, 10, [32, 32, 32], [hw1.hw1.Sigmoid(), hw1.hw1.Sigmoid(), hw1.hw1.Sigmoid(), hw1.hw1.Identity()],
                 hw1.hw1.random_normal_weight_init, hw1.hw1.zeros_bias_init,
                 hw1.hw1.SoftmaxCrossEntropy(),
                 1e-3)
    """
    mlp = nn.MLP(784, 10, [32, 32, 32], [nn.Sigmoid(), nn.Sigmoid(), nn.Sigmoid(), nn.Identity()],
                 nn.random_normal_weight_init, nn.zeros_bias_init,
                 nn.SoftmaxCrossEntropy(),
                 1e-3)
    """
    visualize_training_statistics(mlp, dset, epochs, batch_size, savepath)
    print("Saved output to {}".format(savepath))




def visualize_training_statistics(mlp, dset, epochs, batch_size, savepath=None):
    training_losses, training_errors, validation_losses, validation_errors, confusion_matrix = nn.get_training_stats(
        mlp, dset, epochs, batch_size)

    

def parse_args():
    parser = argparse.ArgumentParser(description="11-785 HW1P1 Visualizer")
    parser.add_argument('--outpath', type=str, default=None,
                        help='Path to output')
    return parser.parse_args()


def main(arglist):
    visualize(outpath=arglist.outpath)
    print("Done :)")


if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)
