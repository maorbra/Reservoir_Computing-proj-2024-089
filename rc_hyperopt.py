import argparse
import gc
from collections import defaultdict
import json
import logging
import os
import pickle
import shutil
import sys
import time
import warnings

import hyperopt
import numpy as np
from pymeasure.experiment import (
    ListParameter,
    Parameter,
    IntegerParameter,
    BooleanParameter,
    Results,
    Worker,
    unique_filename
)
from hyperopt import STATUS_OK, STATUS_FAIL

from config import DATA_DIR, RESULTS_DIR
from procedure_base import ProcedureBase
from train import Train
from procedure_base import log


def load_results(exp):
    report_path = os.path.join(exp, "results")
    results = []
    for file in os.listdir(report_path):
        if os.path.isfile(os.path.join(report_path, file)):
            with open(os.path.join(report_path, file), "r") as f:
                results.append(json.load(f))
    return results


def research(objective, dataset, config_path, report_path=None, trials=None):
    """
    Wrapper for hyperopt fmin function. Will run hyperopt fmin on the
    objective function passed as argument, on the data stored in the
    dataset argument.

    Note
    ----

        Installation of :mod:`hyperopt` is required to use this function.

    Parameters
    ----------
    objective : Callable
        Objective function defining the function to
        optimize. Must be able to receive the dataset argument and
        all parameters sampled by hyperopt during the search. These
        parameters must be keyword arguments only without default value
        (this can be achieved by separating them from the other arguments
        with an empty starred expression. See examples for more info.)
    dataset : tuple or lists or arrays of data
        Argument used to pass data to the objective function during
        the hyperopt run. It will be passed as is to the objective
        function : it can be in whatever format.
    config_path : str or Path
        Path to the hyperopt experimentation configuration file used to
        define this run.
    report_path : str, optional
        Path to the directory where to store the results of the run. By default,
        this directory is set to be {name of the experiment}/results/.
    """
    from glob import glob
    import hyperopt as hopt
    from reservoirpy.hyper._hypersearch import _get_conf_from_json
    from reservoirpy.hyper._hypersearch import _get_report_path

    config = _get_conf_from_json(config_path)
    report_path = _get_report_path(config["exp"], report_path)

    def objective_wrapper(kwargs):

        try:
            start = time.time()

            returned_dict = objective(dataset, config, **kwargs)

            end = time.time()
            duration = end - start

            returned_dict["status"] = hopt.STATUS_OK
            returned_dict["start_time"] = start
            returned_dict["duration"] = duration

            save_file = f"{returned_dict['loss']:.7f}_hyperopt_results"

        except Exception as e:
            raise e
            start = time.time()

            returned_dict = {
                "status": hopt.STATUS_FAIL,
                "start_time": start,
                "error": str(e),
            }

            save_file = f"ERR{start}_hyperopt_results"

        try:
            json_dict = {"returned_dict": returned_dict, "current_params": kwargs}
            save_file = os.path.join(report_path, save_file)
            nb_save_file_with_same_loss = len(glob(f"{save_file}*"))
            save_file = f"{save_file}_{nb_save_file_with_same_loss+1}call.json"
            with open(save_file, "w+") as f:
                json.dump(json_dict, f)
        except Exception as e:
            warnings.warn(
                "Results of current simulation were NOT saved "
                "correctly to JSON file."
            )
            warnings.warn(str(e))

        return returned_dict

    search_space = config["hp_space"]

    if trials is None:
        trials = hopt.Trials()

    if config.get("seed") is None:
        rs = np.random.default_rng()
    else:
        rs = np.random.default_rng(config["seed"])

    best = hopt.fmin(
        objective_wrapper,
        space=search_space,
        algo=config["hp_method"],
        max_evals=config["hp_max_evals"],
        trials=trials,
        rstate=rs,
    )

    return best, trials


class Hyperoptimize(Train):
    config_path = Parameter("Hyperparameter space path", default="search_config.json")
    inst_per_trial = IntegerParameter("Instances per Trial", default=1)
    hp_max_evals = IntegerParameter("Hyperopt Num. Evals", default=100)
    hp_method = ListParameter("Hyperopt method", choices=["tpe", "random"], default="tpe")
    loss_metric = ListParameter("Loss Metric", choices=["wer", "rsquare", "nmse", "nrmse"],
                                default="nrmse")
    calibrate_every = IntegerParameter("Calibrate Every # Iterations", default=10)
    continue_search = BooleanParameter("Continue Search from Previous Run", default=True)

    DATA_COLUMNS = [
        "execution_time",
        "Pmax",
        "phi0",
        "nodes",
        "neuron_duration",
        "fir_in_gain",
        "fir_rate",
        "input_connectivity",
        "input_scaling",
        "ridge"
    ] + list(loss_metric.choices)
    evals = None
    trials = None
    config = None

    @property
    def trials_fname(self):
        return os.path.join(DATA_DIR, self.config["exp"] + ".trials.pickle")

    def objective(
        self,
        dataset,
        config,
        *,
        pmax,
        phi0,
        nodes,
        neuron_duration,
        logging_mode,
        logging_rate,
        fir_in_gain,
        fir_rate,
        fir_length,
        input_connectivity,
        input_scaling,
        rescale_input,
        ridge,
        readout_bias,
    ):
        inst_per_trial = config["inst_per_trial"]
        nodes = int(nodes)
        fir_length = int(fir_length)
        neuron_duration = int(neuron_duration)

        result = defaultdict(list)
        params = {
            "optical_power": pmax,
            "phi0": phi0,
            "nodes": nodes,
            "seed": self.seed,
            "neuron_duration": neuron_duration,
            "logging_mode": logging_mode,
            "logging_rate": logging_rate,
            "fir_in_gain": fir_in_gain,
            "fir_rate": fir_rate,
            "fir_length": fir_length,
            "input_connectivity": input_connectivity,
            "input_scaling": input_scaling,
            "rescale_input": rescale_input,
            "ridge": ridge,
            "readout_bias": readout_bias,
        }
        self.log.debug("Initializing reservoir with params:\n%s", str(params))
        tic = time.time()
        self.reservoir = self.init_reservoir(**params)
        if self.calibrate_every > 0 and self.evals % self.calibrate_every == 0:
            self.calibrate()
        for n in range(inst_per_trial):
            temp_seed = self.seed + n
            input_dim = dataset[0][0].shape[-1]
            self.reservoir.init_Win(input_dim, temp_seed)
            self.reservoir.init_coeffs(temp_seed)
            try:
                temp_result = self.train(dataset)
            except Exception as e:
                self.log.debug("Exception occured in train(): %s(%s)", type(e).__name__, e)
                result["status"] = STATUS_FAIL
                raise e
            for key, val in temp_result.items():
                result[key].append(val)

        for key, val in result.items():
            result[key] = np.min(val)
        procedure_result = result.copy()
        toc = time.time()
        exec_time = toc - tic
        procedure_result.update(**{
            Hyperoptimize.DATA_COLUMNS[0]: exec_time,
            Hyperoptimize.DATA_COLUMNS[1]: pmax,
            Hyperoptimize.DATA_COLUMNS[2]: phi0,
            Hyperoptimize.DATA_COLUMNS[3]: nodes,
            Hyperoptimize.DATA_COLUMNS[4]: neuron_duration,
            Hyperoptimize.DATA_COLUMNS[5]: fir_in_gain,
            Hyperoptimize.DATA_COLUMNS[6]: fir_rate,
            Hyperoptimize.DATA_COLUMNS[9]: input_connectivity,
            Hyperoptimize.DATA_COLUMNS[10]: input_scaling,
            Hyperoptimize.DATA_COLUMNS[11]: ridge,
        })
        self.emit("results", procedure_result)
        self.evals += 1
        self.emit("progress", 100 * self.evals / self.hp_max_evals)
        result["loss"] = result[config["loss_metric"]]
        result["status"] = STATUS_OK

        losses = [l for l in self.trials.losses() if l is not None]
        if len(losses) > 0 and result["loss"] < min(losses):
            dst = os.path.join(
                DATA_DIR,
                self.config["exp"] + "_{}.pickle".format(result["loss"])
            )
            shutil.copyfile(self.dump_path, dst)
        with open(self.trials_fname, "wb") as f:
            self.log.debug("Saving Trials to %s", self.trials_fname)
            pickle.dump(self.trials, f)

        gc.collect()

        return result

    def execute(self):
        log.debug("Hypersearch configuration:")
        hp_space = self.config["hp_space"]
        log.debug("Hyperparameter space:")
        for k, v in hp_space.items():
            log.debug("%s: %s", k, v)
            setattr(self, k, v)
        if not os.path.exists(RESULTS_DIR):
            os.mkdir(RESULTS_DIR)
        dataset = self.load_dataset()
        self.evals = 0
        if self.continue_search and os.path.isfile(self.trials_fname):
            self.log.debug("Loading trials from %s", self.trials_fname)
            with open(self.trials_fname, "rb") as f:
                self.trials = pickle.load(f)
        else:
            self.trials = hyperopt.Trials()
        research(
            self.objective,
            dataset,
            self.config_path,
            os.path.dirname(self.trials_fname),
            self.trials
        )

    @classmethod
    def make_argparser(cls):
        parser = ProcedureBase.make_argparser()
        parser.add_argument(
            "--config", default=Hyperoptimize.config_path.default
        )
        parser.add_argument(
            "--gui", action="store_true"
        )
        return parser

    @classmethod
    def init_procedure(cls, args):
        procedure = super().init_procedure(args)
        procedure.config_path = procedure.config
        with open(procedure.config_path, "r") as f:
            procedure.config = json.load(f)
        for k, v in procedure.config.items():
            if k == "hp_space":
                continue
            log.debug("%s: %s", k, v)
            setattr(procedure, k, v)
        return procedure


if __name__ == '__main__':
    parser = Hyperoptimize.make_argparser()
    args = parser.parse_args()

    if args.gui:
        from pymeasure.display.Qt import QtWidgets
        from main_window import MainWindow

        app = QtWidgets.QApplication(sys.argv)
        inputs = (
            "simulation",
            "hp_max_evals",
            "hp_method",
            "loss_metric",
            "seed",
            "inst_per_trial",
            "test_size",
            "batch",
        )
        window = MainWindow(
            Hyperoptimize,
            inputs=inputs,
            displays=inputs,
            filename_prefix="hyperoptimize",
        )
        window.show()
        sys.exit(app.exec())
    else:
        procedure = Hyperoptimize.init_procedure(args)
        if not os.path.exists(RESULTS_DIR):
            os.mkdir(RESULTS_DIR)
        results_path = unique_filename(RESULTS_DIR, "hyperoptimize")
        results = Results(procedure, results_path)
        worker = Worker(results)
        worker.start()
        duration = procedure.duration * 3600
        worker.join(duration)
