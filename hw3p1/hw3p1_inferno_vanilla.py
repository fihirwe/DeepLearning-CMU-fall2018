"""
This is the script that trains the model.
"""

import sys

from hw3.vanilla_model import parse_args, train_model


def main(argv):
    args = parse_args(argv)
    train_model(args)


if __name__ == '__main__':
    main(sys.argv[1:])
