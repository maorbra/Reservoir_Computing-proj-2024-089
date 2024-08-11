import argparse
import os
import sys
import pickle

import numpy as np

from config import DATA_DIR


def save(path, bits, adc=False, low=0, seed=None):
    N = 2 ** bits
    x = np.full((N, 1, bits), low)
    digits = list(range(N))
    for d in digits:
        bit_seq = np.array(list(bin(d)[2:]), dtype=int)
        bit_seq[bit_seq == 0] = low
        x[d][0][-len(bit_seq):] = bit_seq
    y = np.repeat(digits, bits).reshape(-1, 1, bits)
    if adc:
        x, y = y.reshape(-1, bits, 1), x.reshape(-1, bits, 1)
    rng = np.random.default_rng(seed)
    idx = rng.choice(digits, N, replace=False)
    dataset = x, y, x[idx], y[idx]
    with open(path, "wb") as f:
        pickle.dump(dataset, f)
    print("Saved dataset to {}".format(path))
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bits", type=int, default=4)
    parser.add_argument("--adc", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    name = "adc" if args.adc else "dac"
    path = os.path.join(DATA_DIR, "{}{}bit.pickle".format(name, args.bits))
    x, y, xtest, ytest = save(path, args.bits, args.adc, seed=1)
