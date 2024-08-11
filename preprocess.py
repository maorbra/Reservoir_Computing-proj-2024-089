import sys
import os
import pickle
import glob
from collections import defaultdict
import librosa
from lyon.calc import LyonCalc
from joblib import Parallel, delayed
import scipy.signal
from scipy.io import wavfile
import numpy as np
import tqdm

from config import DATA_DIR

os.chdir(os.path.dirname(__file__))

FSDS_DIR = os.path.join(DATA_DIR, "free-spoken-digit-dataset")
RECORDINGS_DIR = os.path.join(FSDS_DIR, "recordings")


def extract_features(f, model, pbar=None, **kwargs):
    sample_rate, signal = wavfile.read(f)
    result = dict(sr=sample_rate)
    result['label'] = int(os.path.basename(f)[0])
    if model in ['spectrogram', 'logspectrogram', 'tanhspectrogram']:
        f, t, coeffs = scipy.signal.spectrogram(
            signal, sample_rate, nperseg=128
        )
        if model == 'logspectrogram':
            coeffs = np.log10(1 + coeffs)
        elif model == 'tanhspectrogram':
            coeffs = np.tanh(coeffs / np.mean(coeffs))
        result['x'] = coeffs.T
    elif model == 'mfcc':
        coeffs = librosa.feature.mfcc(
            y=signal.astype(np.double),
            sr=sample_rate,
            **kwargs,
        )
        result['x'] = coeffs.T
    elif model == 'lyon':
        calc = LyonCalc()
        coeffs = calc.lyon_passive_ear(
            signal=signal.astype(np.double),
            sample_rate=sample_rate,
            **kwargs
        )
        result['x'] = coeffs
    else:
        raise ValueError("Unknown model")

    result['duration'] = len(result['x'])
    try:
        pbar.update(1)
    except AttributeError:
        sys.stderr.write('.')
    return result


def save_data(
        dataset_path,
        target_path,
        model,
        speaker="all",
        fraction=1,
        n_jobs=1,
        low=-1,
        seed=None,
        **kwargs
):
    """
    Extract MFCCs or Lyon cochlear model coefficients
    from the free spoken digit dataset
    and saves them into a file along with the digit labels.
    """

    metadata = defaultdict(list)
    file_mask = "*_{}_*.wav".format(speaker.replace("all", "*"))
    files = glob.glob(os.path.join(dataset_path, file_mask))
    rs = np.random.RandomState(seed)
    idx = rs.permutation(len(files))[:int(fraction * len(files))]

    with tqdm.tqdm(total=len(idx)) as pbar:
        try:
            data = Parallel(n_jobs=n_jobs)(
                delayed(extract_features)(files[i], model, pbar=pbar, **kwargs)
                for i in idx
            )
        except Exception as e:
            raise e
        finally:
            pbar.close()
    x = []
    y = []
    for i, d in enumerate(data):
        x.append(d.pop('x'))
        y.append(np.full((x[i].shape[0], 10), low))
        y[-1][:, d['label']] = 1
        for key in d:
            metadata[key].append(d[key])

    if target_path.endswith(".npz"):
        np.savez(
            target_path, 
            **{f'x{i}': xi for i, xi in enumerate(x)},
            **{f'y{i}': yi for i, yi in enumerate(y)},
            **metadata
        )
    else:
        with open(target_path, "wb") as f:
            dataset = (x, y)
            pickle.dump(dataset, f)
            print("Dataset saved to {}".format(target_path))


def load_data(npz_path):
    x = []
    y = []
    metadata = dict()
    with np.load(npz_path) as f:
        for key in f:
            metadata[key] = f[key]
    labels = metadata['label']
    for i in range(len(labels)):
        x.append(metadata.pop(f'x{i}'))
        y.append(metadata.pop(f'y{i}'))
    return x, y, metadata


def list2array(x: list, continue_last=False):
    """Convert list of uneven lists to array
    padding shorter lists with trailing zeros"""

    def key(_: np.ndarray) -> int:
        return _.shape[0]

    xarr = np.zeros((len(x), *(max(x, key=key)).shape))
    for i in range(len(x)):
        xarr[i, 0:x[i].shape[0]] = x[i]
        if continue_last:
            xarr[i, x[i].shape[0]:] = x[i][-1]

    return xarr


def join_items(x: list, batch: int):
    xnew = []
    for i in range(0, len(x), batch):
        xnew.append(np.concatenate(x[i:i + batch]))
    return xnew


if __name__ == "__main__":
    model = "lyon"
    speaker = 'all'
    fraction = 0.1
    if len(sys.argv) > 4:
        raise TypeError("Invalid number of arguments")
    if len(sys.argv) == 2:
        speaker = sys.argv[1]
    elif len(sys.argv) == 3:
        model, speaker = sys.argv[1:]
    elif len(sys.argv) == 4:
        model, speaker, fraction = sys.argv[1:]
    filename = 'data/digits_{}_{}_{}.pickle'.format(model, speaker, fraction)
    print("Writing data to {}".format(filename))
    save_data(
        RECORDINGS_DIR, filename, model,
        speaker=speaker,
        decimation_factor=50,
        fraction=0.1,
        seed=42
    )
