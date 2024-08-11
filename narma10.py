import os
import pickle

import numpy as np
import scipy.io
from reservoirpy.datasets import narma

from config import DATA_DIR


def save(path, timesteps, chunks, order=10, a1=0.3, a2=0.04, b=1.5, c=0.1, seed=None):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 0.5, timesteps).reshape(-1, 1)
    y = narma(timesteps, order, a1, a2, b, c, seed=seed)
    dataset = np.array_split(x, chunks), np.array_split(y, chunks)
    ext = os.path.splitext(path)[1]
    if ext == ".pickle":
        with open(path, "wb") as f:
            pickle.dump(dataset, f)
    elif ext == ".mat":
        scipy.io.savemat(path, {"x": dataset[0], "y": dataset[1]})
    print("Saved narma{} dataset to {}".format(order, path))


if __name__ == "__main__":
    path = os.path.join(DATA_DIR, "narma10.pickle")
    save(path, 2000, 8, seed=1)
