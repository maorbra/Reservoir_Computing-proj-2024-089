import logging
import os
import sys
import pickle
import time
import numpy as np
from pymeasure.experiment import (
        Parameter,
        IntegerParameter,
        FloatParameter,
        ListParameter,
        Results,
        Worker,
        unique_filename
)
import reservoirpy.observables
from reservoirpy.nodes import Ridge

from config import RESULTS_DIR, DATA_DIR
from procedure_base import ProcedureBase, log
from preprocess import save_data, load_data, RECORDINGS_DIR


def idx_train_test(n, k, seed):
    idx_all = list(range(n))
    rs = np.random.RandomState(seed)
    idx_test = rs.choice(idx_all, size=k, replace=False)
    idx_train = np.array(list(set(idx_all).difference(set(idx_test))))
    return idx_train, idx_test


def word_error_rate(ytrue, ypred):
    return np.mean(np.argmax(ytrue, axis=-1) != np.argmax(ypred, axis=-1))


def ridge_fit(x_train, x_test, y_train, y_test, ridge, readout_bias):
    readout = Ridge(ridge=ridge, input_bias=readout_bias)
    readout.fit(x_train, y_train)
    y_pred = readout.run(x_test)
    nmse = np.mean((y_test - y_pred) ** 2) / np.var(y_test)
    wer = word_error_rate(y_test, y_pred)
    return nmse, wer, readout.Wout, readout.bias


class Train(ProcedureBase):
    
    DATA_COLUMNS = ["Iteration", "Execution Time (s)", "Power (mW)", "nrmse", "rsquare", "wer"]

    dataset = Parameter("Dataset", default='data/sin_square.pickle')
    iterations = IntegerParameter("Iterations", default=1)
    test_size = FloatParameter("Test Size", default=0.1)
    n_cross_validations = IntegerParameter("Number of cross-validations", default=10)
    dump_path = Parameter("Dump Path", default="data/train.pickle")
    verbosity = IntegerParameter("Console Log Verbosity", default=logging.DEBUG)

    def load_dataset(self):
        if self.dataset == "digits":
            self.dataset = RecognizeDigits.generate_dataset(self)
        if self.dataset.endswith(".npz"):
            dataset = load_data(self.dataset)[:2]
        else:
            with open(self.dataset, "rb") as f:
                dataset = pickle.load(f)
        return dataset

    def train(self, dataset):
        if len(dataset) not in [2, 4]:
            raise TypeError("Unsupported format of the dataset")
        if len(dataset) == 2:
            x, y = dataset
        else:
            x_train, y_train, x_test, y_test = dataset
            x = x_train + x_test
            y = y_train + y_test
            idx_train = range(len(x_train))
            idx_test = range(len(x_test))

        states = self.reservoir.run(x)
        transients = self.reservoir.transients.copy()
        metrics = ["nmse", "nrmse", "rsquare", "wer"]
        trials = {
            "idx_train": [],
            "idx_test": [],
            "ridge": [],
            "Wout": [],
            "bias": [],
        }
        for metric in metrics:
            trials[metric] = []

        for i in range(self.n_cross_validations):
            if len(dataset) == 2:
                idx_train, idx_test = idx_train_test(len(x), int(self.test_size * len(x)), self.seed + i)
            x_train = np.concatenate([states[_] for _ in idx_train])
            y_train = np.concatenate([y[_] for _ in idx_train])
            x_test = np.concatenate([states[_] for _ in idx_test])
            y_test = np.concatenate([y[_] for _ in idx_test])
            ridge = self.reservoir.readout.ridge
            input_bias = self.reservoir.readout.input_bias
            nmse, wer, Wout, bias = ridge_fit(
                x_train, x_test, y_train, y_test, ridge, input_bias
            )
            nrmse = np.sqrt(nmse)
            rsquare = 1 - nmse
            log.debug(
                "Cross-validation %d/%d: nmse=%g, nrmse=%g, ridge=%g, wer=%g",
                i + 1, self.n_cross_validations, nmse, nrmse, ridge, wer
            )
            trials["idx_train"].append(list(idx_train))
            trials["idx_test"].append(list(idx_test))
            trials["nmse"].append(nmse)
            trials["nrmse"].append(nrmse)
            trials["rsquare"].append(rsquare)
            trials["ridge"].append(ridge)
            trials["wer"].append(wer)
            trials["Wout"].append(Wout)
            trials["bias"].append(bias)

        result = dict()
        for key, val in self.reservoir.calibration.items():
            result[key] = val
        for k in metrics:
            result[k] = np.mean(trials[k])
        log.debug("Training results:")
        for k in metrics:
            log.debug("%s = (%g±%g  mean±std, %g min, %g max)", k, result[k], np.std(trials[k]), min(trials[k]), max(trials[k]))
        if self.dump_path:
            log.debug("Saving reservoir training data to %s", self.dump_path)
            dump = dict(
                x=x,
                y=y,
                states=states,
                transients=transients,
                Win=self.reservoir.Win,
                hypers=self.reservoir.hypers,
                trials=trials,
                **result
            )
            with open(self.dump_path, "wb") as f:
                pickle.dump(dump, f)
        return result

    def execute(self, params=None):
        if not params:
            params = dict()
        self.dataset = self.load_dataset()
        self.reservoir = self.init_reservoir(**params)
        for i in range(self.iterations):
            tic = time.time()
            power = np.nan
            result = self.train(self.dataset)
            toc = time.time()
            result[self.DATA_COLUMNS[0]] = i
            result[self.DATA_COLUMNS[1]] = toc - tic
            result[self.DATA_COLUMNS[2]] = power
            self.emit("results", result)
            self.emit("progress", 100 * i / self.iterations)
            if self.should_stop():
                log.debug("Caught the stop flag in the procedure")
                break

    @classmethod
    def make_argparser(cls):
        parser = super().make_argparser()
        parser.add_argument("dataset")
        parser.add_argument("--test-size", type=float,
                            default=Train.test_size.default)
        parser.add_argument("--ridge", type=float,
                            default=Train.ridge.default)
        parser.add_argument("--n-cross-validations", type=float,
                            default=Train.n_cross_validations.default)
        return parser


class RecognizeDigits(Train):
    dump_path = Parameter("Dump Path", default="data/recognize_digits.pickle")
    dataset_fraction = FloatParameter("Dataset Fraction", default=0.01,
                                      decimals=2, step=0.01, minimum=0.01, maximum=1)
    speaker = Parameter("Speaker", default="all")
    audio_model = ListParameter(
        "Audio Model",
        choices=["spectrogram", "logspectrogram", "tanhspectrogram", "lyon", "mfcc"],
        default="logspectrogram"
    )

    def load_dataset(self):
        x, y, *metadata = load_data(self.dataset)
        return x, y

    @staticmethod
    def generate_dataset(procedure):
        dataset = "digits_{}_{}_{}.npz".format(
            procedure.audio_model, procedure.speaker, procedure.dataset_fraction
        )
        dataset = os.path.join(DATA_DIR, dataset)
        save_data(
            RECORDINGS_DIR,
            dataset,
            procedure.audio_model,
            procedure.speaker,
            procedure.dataset_fraction
        )
        return dataset

    def execute(self, params=None):
        log.debug("Running spoken-digit recognition task")
        return super().execute(params)

    @classmethod
    def make_argparser(cls):
        parser = super().make_argparser()
        parser.add_argument("--dataset-fraction", type=float,
                            default=RecognizeDigits.dataset_fraction.default)
        parser.add_argument("--speaker",
                            default=RecognizeDigits.speaker.default)
        parser.add_argument("--audio-model",
                            choices=RecognizeDigits.audio_model.choices,
                            default=RecognizeDigits.audio_model.default)
        return parser

    @classmethod
    def init_procedure(cls, args):
        procedure = super().init_procedure(args)
        procedure.dataset = RecognizeDigits.generate_dataset(procedure)
        return procedure


if __name__ == '__main__':
    parser = Train.make_argparser()
    args = parser.parse_args()

    if len(sys.argv) > 1:
        procedure = Train.init_procedure(args)

        if not os.path.exists(RESULTS_DIR):
            os.mkdir(RESULTS_DIR)
        results_path = unique_filename(RESULTS_DIR, "train")
        results = Results(procedure, results_path)
        log.debug("Constructing the Worker")
        worker = Worker(results)
        worker.start()
        log.debug("Started the Worker")

        log.debug("Joining with the worker in at most 1 hr")
        worker.join(timeout=3600)
    else:
        from pymeasure.display.Qt import QtWidgets
        from main_window import MainWindow
        app = QtWidgets.QApplication(sys.argv)
        inputs = (
            "scope_type",
            "dataset",
            "test_size",
            "n_cross_validations",
            "nodes",
            "neuron_duration",
            "logging_rate",
            "fir_rate",
            "fir_length",
            "optical_power",
            "phi0",
            "input_scaling",
            "batch",
            "input_connectivity",
        )
        window = MainWindow(
            Train,
            inputs=inputs,
            displays=inputs,
            sequencer_inputs=[
                "optical_power",
                "nodes",
                "neuron_duration",
                "phi0",
                "input_scaling",
                "connectivity",
                "logging_rate",
            ],
            filename_prefix="train",
        )
        window.show()
        sys.exit(app.exec())
