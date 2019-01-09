import argparse
import os
import sys

import numpy as np
import torch
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset

from hw2.all_cnn import all_cnn_module
from hw2.preprocessing import cifar_10_preprocess


def initializer(m):
    """
    Simple initializer
    """
    if hasattr(m, 'bias'):
        m.bias.data.zero_()
    if hasattr(m, 'weight'):
        torch.nn.init.xavier_uniform(m.weight.data)


def make_loaders(args):
    """
    Run preprocessing and make loaders
    Take some data from the train set to make a validation set
    """
    kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
    x = np.load(os.path.join(args.dataset, 'train_feats.npy')).astype(np.float32)
    y = np.load(os.path.join(args.dataset, 'train_labels.npy'))
    valid = 1000
    x, xval = x[:-valid], x[-valid:]
    y, yval = y[:-valid], y[-valid:]
    xtest = np.load(os.path.join(args.dataset, 'test_feats.npy')).astype(np.float32)
    ytest = np.zeros((xtest.shape[0],), dtype=y.dtype)
    xt = np.concatenate([xval, xtest], axis=0)
    x, xt = cifar_10_preprocess(x, xt)
    xval, xtest = xt[:valid], xt[valid:]
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                              batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(xval), torch.from_numpy(yval)),
                            batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(xtest), torch.from_numpy(ytest)),
                             batch_size=args.batch_size, shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader


def write_predictions(args, model, loader):
    """
    Write submission file
    """
    model.eval()
    with open('predictions.txt', 'w') as f:
        for x, y in loader:
            if args.cuda:
                x = x.cuda()
            ypred = torch.max(model(x), 1)[1].data.cpu().numpy()
            for yp in ypred:
                f.write("{}\n".format(np.asscalar(yp)))


def train_model(args):
    """
    Perform training then call prediction
    """
    model = all_cnn_module()
    model.apply(initializer)
    train_loader, validate_loader, test_loader = make_loaders(args)
    # Build trainer
    savepath = os.path.join(args.save_directory, Trainer()._checkpoint_filename)
    if os.path.exists(savepath):
        trainer = Trainer().load(from_directory=args.save_directory)
        if args.cuda:
            trainer.cuda()
    else:
        trainer = Trainer(model) \
            .build_criterion('CrossEntropyLoss') \
            .build_metric('CategoricalError') \
            .save_every((1, 'epochs')) \
            .validate_every((1, 'epochs')) \
            .save_to_directory(args.save_directory) \
            .set_max_num_epochs(args.epochs) \
            .build_logger(TensorboardLogger(log_scalars_every=(1, 'iteration'),
                                            log_images_every='never'),
                          log_directory=args.save_directory)

        # These are the params from the paper
        trainer.build_optimizer('SGD', lr=0.01, momentum=0.9, weight_decay=0.001)
        # Also works with Adam and default settings
        # trainer.build_optimizer('Adam')
        trainer.bind_loader('train', train_loader)
        trainer.bind_loader('validate', validate_loader)

        if args.cuda:
            trainer.cuda()

        # Go!
        trainer.fit()
        trainer.save()
    write_predictions(args, trainer.model, test_loader)


def main(argv):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch HW2 Part 1 Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size')
    parser.add_argument('--save-directory', type=str, default='output/inferno/all-cnn', help='output directory')
    parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of epochs')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--dataset', type=str, default='../../dataset', help='path to dataset')
    args = parser.parse_args(argv)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    train_model(args)


if __name__ == '__main__':
    main(sys.argv[1:])
