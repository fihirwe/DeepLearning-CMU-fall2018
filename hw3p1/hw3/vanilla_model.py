import argparse
import os
import time
from collections import Counter

import numpy as np
import torch
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.base import Callback
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader


class W2DataLoader(DataLoader):
    """
    Dataset yields features
    """

    def __init__(self, name, args):
        self.data = np.load(os.path.join(args.data_directory, "wiki.{}.npy".format(name)))
        self.batch_size = args.batch_size
        self.cuda = args.cuda

    def __iter__(self):
        words = np.concatenate(np.random.permutation(self.data))
        word_count = words.shape[0]
        l = (word_count - 1) // self.batch_size
        n = l * self.batch_size
        x = words[0:n].reshape((self.batch_size, l)).T
        y = words[1:n + 1].reshape((self.batch_size, l)).T
        x = torch.from_numpy(x).long()
        y = torch.from_numpy(y).long()
        if self.cuda:
            x = x.cuda()
            y = y.cuda()

        pos = 0
        L = 50
        while pos + L < l:
            yield x[pos:pos + L], y[pos:pos + L]
            pos += L


def sample_gumbel(shape, eps=1e-10, out=None):
    """
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    return - torch.log(eps - torch.log(U + eps))


class W2Model(nn.Module):
    """
    The model itself
    """

    def __init__(self, args, vocab_size, projection_bias=None):
        super(W2Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 400)
        self.rnns = nn.ModuleList()
        self.rnns.append(nn.LSTM(400, 1150))
        self.rnns.append(nn.LSTM(1150, 1150))
        self.rnns.append(nn.LSTM(1150, 400))
        self.projection = nn.Linear(400, vocab_size)
        if projection_bias is not None:
            self.projection.bias.data = projection_bias
        self.projection.weight = self.embedding.weight

    def forward(self, features, future=0, stochastic=False):
        # Model
        h = features
        # Embedding layer
        h = self.embedding(h)
        # RNNs
        states = []
        for l in self.rnns:
            h, state = l(h)
            states.append(state)
        # Projection layer
        logits = self.projection(h)
        if stochastic:
            logits = logits + Variable(sample_gumbel(logits.size(), out=logits.data.new()))
        if future > 0:
            outputs = []
            h = logits[-1, :, :]
            h = torch.max(h, dim=1)[1]
            h = torch.unsqueeze(h, 0)
            for i in range(future):
                h = self.embedding(h)
                for j, (rnn, state) in enumerate(zip(self.rnns, states)):
                    h, state = rnn(h, state)
                    states[j] = state
                h = self.projection(h)
                if stochastic:
                    gumbel = Variable(sample_gumbel(shape=h.size(), out=h.data.new()))
                    h = h + gumbel
                outputs.append(h)
                h = torch.max(h, dim=2)[1]
            logits = torch.cat([logits] + outputs, dim=0)
        return logits


class EpochTimer(Callback):
    """
    Callback that prints the elapsed time per epoch
    """

    def __init__(self):
        super(EpochTimer, self).__init__()
        self.start_time = None

    def begin_of_training_run(self, **_kwargs):
        self.start_time = time.time()

    def begin_of_epoch(self, **_kwargs):
        self.start_time = time.time()

    def end_of_epoch(self, epoch_count, **_kwargs):
        assert self.start_time is not None
        end_time = time.time()
        elapsed = end_time - self.start_time
        print("Epoch {} elapsed: {}".format(epoch_count, elapsed))
        self.start_time = None


class SeqCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    Reshape inputs for CrossEntropyLoss
    """

    def forward(self, input, target):
        return super(SeqCrossEntropyLoss, self).forward(input.view(-1, input.size(2)), target.view(-1))


def smoothed_unigram(vocab_size, args):
    # Calculate smoothed unigram distribution
    words = np.concatenate(np.load(os.path.join(args.data_directory, 'wiki.test.npy')))
    counter = Counter(words)
    word_counts = np.array([counter[i] for i in range(vocab_size)], dtype=np.float32)
    word_count = np.sum(word_counts)
    p = word_counts / word_count
    smoothing = 0.1
    phat = (p * (1. - smoothing)) + (smoothing / vocab_size)
    return phat


def train_model(args):
    """
    Performs the training
    """
    if os.path.exists(os.path.join(args.save_directory, Trainer()._checkpoint_filename)):
        # Skip training if checkpoint exists
        return
    vocab = np.load(os.path.join(args.data_directory, 'vocab.npy'))
    vocab_size = vocab.shape[0]
    unigram_dist = torch.from_numpy(np.log(smoothed_unigram(vocab_size=vocab_size, args=args))).float()
    model = W2Model(args, vocab_size=vocab_size, projection_bias=unigram_dist)
    train_loader = W2DataLoader('train', args)
    validate_loader = W2DataLoader('valid', args)
    # Build trainer
    trainer = Trainer(model) \
        .build_criterion(SeqCrossEntropyLoss) \
        .build_optimizer('Adam') \
        .validate_every((1, 'epochs')) \
        .save_every((1, 'epochs')) \
        .save_to_directory(args.save_directory) \
        .set_max_num_epochs(args.epochs) \
        .build_logger(TensorboardLogger(log_scalars_every=(1, 'iteration'),
                                        log_images_every='never'),
                      log_directory=args.save_directory)

    trainer.register_callback(EpochTimer)
    # Bind loaders
    trainer.bind_loader('train', train_loader)
    trainer.bind_loader('validate', validate_loader)
    if args.cuda:
        trainer.cuda()

    # Go!
    trainer.fit()
    trainer.save()
    torch.save(trainer.model.state_dict(), os.path.join(args.save_directory, 'model.t5'))


def parse_args(argv):
    # Training settings
    parser = argparse.ArgumentParser(description='Homework 3 Part 1 Baseline')
    parser.add_argument('--batch-size', type=int, default=40, metavar='N', help='input batch size')
    parser.add_argument('--save-directory', type=str, default='output/simple/v4', help='output directory')
    parser.add_argument('--data-directory', type=str, default='../../dataset', help='data directory')
    parser.add_argument('--epochs', type=int, default=2, metavar='N', help='number of epochs to train')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

    args = parser.parse_args(argv)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
