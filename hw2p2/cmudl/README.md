# cmudl

Utilities for 11-785 @ CMU

## Setup


You can simply install the python package from this repository.

```bash
$ git clone https://github.com/cmudeeplearning11785/cmudl
$ cd cmudl
$ pip install -e .
```

## Utilities

### Process submission for hw2p2

To process your hw2p2 predictions for submission to the Kaggle competition. This assumes your submission file is a
serialized ndarray (npy format) with shape (num_predictions,).

```
$ bin/cmudl hw2p2 -s /path/to/scores.npy
```

This command will send your predictions to our server and we will return a csv file that will be saved in the current
directory as `submission.csv`
