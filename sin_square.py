import os
import pickle
import numpy as np
import scipy.signal
import scipy.io


from config import DATA_DIR


def save(path, fmin=0.5, fmax=2, nt=50, df=0.5):
    t = np.linspace(0, 1, nt)
    f = np.arange(fmin, fmax, df)
    x_sin = [np.sin(2 * np.pi * _ * t).reshape(-1, 1) for _ in f]
    y_sin = [np.ones((nt, 1)) for _ in f]
    x_square = [scipy.signal.square(2 * np.pi * _ * t).reshape(-1, 1) for _ in f]
    y_square = [np.zeros((nt, 1)) for _ in f]
    x = x_sin + x_square
    y = y_sin + y_square
    dataset = x, y
    ext = os.path.splitext(path)[1]
    if ext == ".pickle":
        with open(path, "wb") as f:
            pickle.dump(dataset, f)
    elif ext == ".mat":
        scipy.io.savemat(path, {"x": x, "y": y})
    print("Saved sin/square dataset to {}".format(path))


if __name__ == "__main__":
    save(os.path.join(DATA_DIR, "sin_square.pickle"), fmin=0.5, fmax=3.5, df=0.25)
