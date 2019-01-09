import time
import requests
import os

URL_BASE = 'https://cmu-11785-hw2p2.herokuapp.com/'


def submit(filepath, outpath=None):
    if not os.path.exists(filepath):
        print("No such path exists: {}".format(filepath))
        exit(1)

    print("Submitting predictions: {}".format(filepath))

    url = URL_BASE + 'score'
    r = requests.post(url, files={'submission': open(filepath, 'rb')})

    if r.status_code != 200:
        print("Error, service not available. Try again in a little while.")
        exit(1)

    outpath = "submission.csv" if outpath is None else outpath
    with open(outpath, 'w') as f:
        f.write(r.text)

    print("Wrote submission to {}".format(outpath))
