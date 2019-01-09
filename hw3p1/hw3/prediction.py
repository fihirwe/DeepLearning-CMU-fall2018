import os

import torch
from torch.autograd import Variable

from .vanilla_model import W2Model, parse_args

VOCAB_SIZE = 33278


def prediction(inp):
    """
    Input is a text sequences. Produce scores for the next word in the sequence.
    :param inp: array of words (batch size, sequence length) [0-labels]
    :return: array of scores for the next word (batch size, labels)
    """
    args = parse_args([])
    model = W2Model(args=args, vocab_size=VOCAB_SIZE)
    model_path = os.path.abspath(os.path.join(__file__, '../../model.t5'))
    state = torch.load(model_path)
    model.load_state_dict(state)
    model.eval()
    x = Variable(torch.from_numpy(inp.T)).long()  # (len, n)
    logits = model(x)
    ret = logits[-1, :, :].cpu().data.numpy()
    return ret
