import os
import sys
import numpy as np
from pymeasure.log import console_log
from pymeasure.experiment import (
        IntegerParameter,
        Parameter,
        ListParameter,
        Worker,
        Results,
        unique_filename
)

from config import DATA_DIR, RESULTS_DIR
from procedure_base import log
from train import Train, RecognizeDigits
from rc_hyperopt import Hyperoptimize, load_results


class RunBestReservoir(Train):

    exp_path = Parameter("Experiment")
    metric = ListParameter("Loss Metric", choices=Hyperoptimize.loss_metric.choices, default="wer")
    rank = IntegerParameter("Rank", default=0)

    def execute(self):
        results = load_results(os.path.join(DATA_DIR, self.exp_path))
        scores = [result["returned_dict"][self.metric] for result in results]
        idx = np.argsort(scores)
        result = results[idx[self.rank]]
        returned_dict = result["returned_dict"]
        best_params = result["current_params"]
        for key in ["nodes", "neuron_duration", "fir_length"]:
            best_params[key] = int(best_params.pop(key))
        best_params["optical_power"] = best_params.pop("pmax")
        self.set_parameters(best_params)
        for key in self.parameter_values():
            log.info("%s=%s", key, getattr(self, key))
        log.info("Running reservoir with params: %s", best_params)
        log.info("Retured dict: %s", returned_dict)
        if self.simulation:
            self.dump_path = None
        else:
            self.dump_path = os.path.join(
                DATA_DIR, "{}_{}_of_{}_{}.pickle".format(self.exp_path, self.rank + 1, len(results), self.metric)
            )
        super().execute(params=best_params)

    @classmethod
    def make_argparser(cls):
        parser = RecognizeDigits.make_argparser()
        parser.add_argument("--exp", default=RunBestReservoir.exp_path.default)
        parser.add_argument("--metric", default=RunBestReservoir.metric.default)
        parser.add_argument("--rank", default=RunBestReservoir.rank.default, type=int)
        return parser

    @classmethod
    def init_procedure(cls, args):
        procedure = super().init_procedure(args)
        procedure.exp_path = args.exp
        procedure.metric = args.metric
        procedure.rank = args.rank
        return procedure


if __name__ == "__main__":
    if len(sys.argv) > 1:

        parser = RunBestReservoir.make_argparser()
        args = parser.parse_args()
        procedure = RunBestReservoir.init_procedure(args)

        if not os.path.exists(RESULTS_DIR):
            os.mkdir(RESULTS_DIR)
        results_path = unique_filename(RESULTS_DIR, "recognize_digits")
        results = Results(procedure, results_path)
        log.debug("Constructing the Worker")
        worker = Worker(results)
        worker.start()
        log.debug("Started the Worker")

        log.info("Joining with the worker in at most 1 hr")
        worker.join(timeout=3600)
    else:
        from pymeasure.display.Qt import QtWidgets
        from main_window import MainWindow
        app = QtWidgets.QApplication(sys.argv)
        inputs = (
            "exp_path",
            "simulation",
            "metric",
            "rank",
            "dataset_fraction",
            "batch",
            "log_duration",
            "calibrate_mzm",
            "calibrate_att",
        )
        window = MainWindow(
            RunBestReservoir,
            inputs=inputs,
            displays=inputs,
            filename_prefix="recognize_digits",
        )
        window.show()
        sys.exit(app.exec())
