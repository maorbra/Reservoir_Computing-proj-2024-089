import sys
import os
import logging

from pymeasure.experiment import (
    Results,
    Worker,
    unique_filename
)
from pymeasure.log import console_log

from procedure_base import ProcedureBase, log
from config import RESULTS_DIR


class Calibrate(ProcedureBase):

    def execute(self):
        self.reservoir = self.init_reservoir()
        self.calibrate()


if __name__ == '__main__':
    console_log(log)
    parser = Calibrate.make_argparser()
    args = parser.parse_args()
    procedure = Calibrate.init_procedure(args)

    if not os.path.exists(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)
    results_path = unique_filename(RESULTS_DIR, "calibrate")
    results = Results(procedure, results_path)
    worker = Worker(results)
    worker.start()
    worker.join(timeout=300)
