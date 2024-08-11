import os
import pickle

import numpy as np
from reservoirpy.datasets import japanese_vowels

from config import DATA_DIR


def save(path):
    dataset = japanese_vowels(repeat_targets=True)
    with open(path, "wb") as f:
        pickle.dump(dataset, f)
    print("Saved japanese_vowels dataset to {}".format(path))


if __name__ == "__main__":
    path = os.path.join(DATA_DIR, "japanese_vowels.pickle")
    save(path)

