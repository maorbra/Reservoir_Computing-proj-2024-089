import os.path
import logging
import sys
import time

import numpy as np
from pymeasure.log import console_log
from pymeasure.experiment import (
    BooleanParameter,
    Results,
    Worker,
    unique_filename
)
from reservoirpy.observables import rsquare

from config import RESULTS_DIR
from procedure_base import ProcedureBase

log = logging.getLogger('')
log.addHandler(logging.NullHandler())


def calc_rsquare_memoizer(model, dataset, i, run=True, states=None):
    T = i + 1
    x = []
    y = []
    for j in range(len(dataset)):
        x.append(dataset[j][T:])
        y.append(dataset[j][:-T])
    if run:
        s = model.run(x)
    else:
        s = [states[i][T:] for i in range(len(states))]
    readout = model.readout_capacity
    readout.fit(s, y)
    sumR2 = 0
    for j in range(len(dataset)):
        prediction = readout.Wout.T.dot(s[j].T).T + readout.bias
        sumR2 += rsquare(y[j], prediction)
    R2 = sumR2 / len(dataset)
    return R2


def mem_capacity(model, dataset, tmin, run, states=None):
    tmax = len(min(dataset, key=lambda _: len(_))) - tmin
    return sum(calc_rsquare_memoizer(model, dataset, i, run, states) for i in range(tmax))


class MeasureCapacity(ProcedureBase):

    run_each_iteration = BooleanParameter("Run Each Iteration", default=False)
    DATA_COLUMNS = ["Iteration", "R2", "Capacity", "Execution Time", "Power (mW)"]

    def execute(self):
        self.x, self.y = self.load_dataset()
        reservoir = self.init_reservoir()
        log.info(f"Starting reservoir's memory capacity measurement")
        capacity = 0
        # TODO: add integer procedure parameter self.nmin
        nmin = 1
        maxiter = len(min(self.x, key=lambda _: len(_))) - nmin

        states = None
        if not self.run_each_iteration:
            states = reservoir.run(self.x)

        for i in range(min(self.iterations, maxiter)):
            power = np.nan
            if self.measure_power and not self.simulation:
                self.wfg.output = True
                self.moku.enable_output(1, False)
                self.wfg.ask("*OPC?")
                power = np.mean(
                    self.moku.get_data()[self.signal_channel_moku]
                ) / self.gain * 1e3
                self.wfg.output = False
                self.moku.enable_output(1, True)
                self.wfg.ask("*OPC?")

            tic = time.time()
            r2 = calc_rsquare_memoizer(reservoir, self.x, i, self.run_each_iteration, states)
            toc = time.time()
            capacity += r2
            exec_time = (toc - tic)

            result = {
                self.DATA_COLUMNS[0]: i,
                self.DATA_COLUMNS[1]: r2,
                self.DATA_COLUMNS[2]: capacity,
                self.DATA_COLUMNS[3]: exec_time,
                self.DATA_COLUMNS[4]: power,
            }
            self.emit('results', result)
            self.emit('progress', 100 * i / self.iterations)
            if self.should_stop():
                log.info("Caught the stop flag in the procedure")
                break


if __name__ == '__main__':
    console_log(log)

    if len(sys.argv) > 1:
        parser = make_argparser(MeasureCapacity)
        parser.add_argument("--run", action="store_true",
                            default=MeasureCapacity.run_each_iteration.default)
        args = parser.parse_args()
        procedure = init_procedure(MeasureCapacity, args)
        procedure.run_each_iteration = args.run

        if not os.path.exists(RESULTS_DIR):
            os.mkdir(RESULTS_DIR)
        results_path = unique_filename(RESULTS_DIR, "measure_capacity")
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
            "iterations",
            "run_each_iteration",
            "nodes",
            "optical_power",
            "phi0",
            "input_scaling",
            "neuron_duration",
            "logging_rate",
            "fir_rate",
            "fir_length",
            "dataset_fraction",
            "batch",
            "connectivity",
            "input_connectivity",
            "calibrate_mzm",
            "calibrate_att",
        )
        window = MainWindow(
            MeasureCapacity,
            inputs=inputs,
            displays=inputs,
            sequencer_inputs=[
                "optical_power",
                "nodes",
                "neuron_duration",
                "phi0",
                "input_scaling",
            ],
            filename_prefix="measure_capacity"
        )
        window.show()
        sys.exit(app.exec())
