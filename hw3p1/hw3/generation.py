import os

import numpy as np
import torch
from torch.autograd import Variable

from .vanilla_model import W2Model, parse_args

VOCAB_SIZE = 33278


def generation(inp, forward):
    """
    Generate a string
    :param inp:
    :param forward: number of words to generate
    :return:
    """
    args = parse_args([])
    model = W2Model(args=args, vocab_size=VOCAB_SIZE)
    model_path = os.path.abspath(os.path.join(__file__, '../../model.t5'))
    state = torch.load(model_path)
    model.load_state_dict(state)
    model.eval()
    x = Variable(torch.from_numpy(inp.T)).long()
    logits = model(x, stochastic=True, future=forward - 1)
    logits = logits.cpu().data.numpy()
    logits = np.transpose(logits, (1, 0, 2))
    logits = logits[:, inp.shape[1] - 1:, :]
    outputs = np.argmax(logits, axis=2)  # (n, forward)
    return outputs
